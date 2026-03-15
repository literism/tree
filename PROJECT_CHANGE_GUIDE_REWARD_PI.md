## 项目功能修改说明（基于新 reward 与 oracle 策略 π）

本文档用于指导你后续修改代码：
- 现在项目把 `summary_based_classifier/` 重构为按功能分组的 package；
- 结合我们讨论并写入证明的 **reward（可行性 + -∞）** 与 **oracle 策略 π / greedy**，明确要改哪些模块、在哪里改。

---

### 0. 目录结构（已整理）

`summary_based_classifier/` 现在按功能拆分：

- `summary_based_classifier/cli/`：运行入口（pipeline orchestrator）
  - `run_pipeline.py`
- `summary_based_classifier/core/`：核心数据结构与状态
  - `topic_state.py`
- `summary_based_classifier/core/pipeline/`：在线处理主流程
  - `builder.py`
  - `sequential_article_processor.py`
- `summary_based_classifier/core/trajectory/`：轨迹/并行轨迹处理与存储
  - `trajectory_sampler.py`
  - `trajectory_storage.py`
  - `parallel_trajectory_processor.py`
- `summary_based_classifier/llm/`：分类/更新模型与 prompt
  - `classify_generator.py`
  - `updater.py`
  - `prompts.py`
  - `prompt_pool.py`
  - `generate_summaries.py`
- `summary_based_classifier/models/`：模型客户端/worker
  - `model_clients.py`
  - `pooled_model_client.py`
  - `model_workers.py`
  - `merge_adapter.py`
- `summary_based_classifier/reward/`：reward 相关
  - `reward_calculator.py`（历史实现）
- `summary_based_classifier/data/`：数据准备/标注
  - `data_split.py`
  - `prepare_dataset.py`
  - `prepare_dataset_dpo.py`
  - `batch_labeler.py`
- `summary_based_classifier/training/`：训练脚本
  - `train_classify_generator.py`
  - `train_updater.py`
  - `train_dpo.py`
  - `trajectory_dpo_trainer.py`
- `summary_based_classifier/inference/`：推理
  - `inference.py`
  - `inference_parallel.py`
  - `parallel_inference_processor.py`
- `summary_based_classifier/evaluation/`：评估/可视化
  - `evaluate.py`
  - `evaluate_labeling.py`
  - `visualize_trees.py`
  - `browse_structure.py`

运行入口：

```bash
cd /home/literism/tree
python3 -m summary_based_classifier.cli.run_pipeline --help
```

---

### 1. 新 reward 的代码落点（需要新增/替换的模块）

我们新 reward 的核心是：

- **可行性**：只要出现任意一对文章 `(i,j)` 满足 `R_T(i,j) > R_{T*}(i,j)`，则该状态/动作判为 **-∞（不可行）**。
- **有限势函数**：在可行时，
  - `R_T(i,j) = -1` 表示同叶同类
  - 否则 `R_T(i,j) = LCA_depth(i,j)`（跨叶）
  - `Φ(T,S) = Σ w_ij * R_T(i,j)`
- **总 reward**：`r_t = Φ(T_t,S_t) - Φ(T_{t-1},S_{t-1})`，因此最大化最终 `Φ(T_n,S_n)` 等价于最大化总 reward（telescoping）。

#### 1.1 建议新增的实现文件

建议新增：
- `summary_based_classifier/reward/feasible_reward.py`

包含：
- `compute_R_T(i,j, tree_state)`：返回 -1 或 LCA 深度
- `is_feasible(T, S, T_star_or_oracle)`：判定是否违反 `R_T > R_*`
- `phi(T,S)`：返回 `-inf` 或有限值
- `step_reward(prev_T, prev_S, new_T, new_S)`：计算增量

#### 1.2 需要接入的位置

- **在线处理主流程**：
  - `summary_based_classifier/core/pipeline/builder.py`
  - `summary_based_classifier/core/pipeline/sequential_article_processor.py`

这些模块决定了：
- action 空间是什么（CreateLeaf / InsertParentPath / UpdateSummary）
- 何时记录 action 与 state（用于后续训练）

你需要把“可行性检查/不可行即丢弃”接入：
- 采样轨迹时：过滤掉不可行 action（避免把 -∞ 轨迹写进训练集，除非你想用于“强负例”）
- 执行 InsertParentPath 前：做 TC 检查（等价于可行性局部检查）

---

### 2. oracle 策略 π / greedy 的代码落点

你现在的实验主线更像 imitation learning：
- 用 gold 树 `T*` 产生 oracle 动作（π）
- 用 π 监督训练 classify 模型与结构动作选择

#### 2.1 建议新增模块

- `summary_based_classifier/core/policy/oracle_policy.py`（新增目录 `core/policy/`）

输出：
- `oracle_classify_action(article, current_node, T_star)`：应该走哪个子节点 / 是否 NEW
- `oracle_insert_parent_action(new_leaf_x, parent_v, siblings, T_star)`：是否 InsertParentPath、选哪个 `y`

并提供一个统一数据结构：
- `OracleAction`（可 JSON 序列化）

#### 2.2 接入位置

- `summary_based_classifier/core/pipeline/builder.py`
  - 在做模型采样前/后，记录 π 的“正确动作”，用于：
    - 行为克隆（SFT）
    - 或 DPO 偏好对（π 动作 vs 非 π 动作）

---

### 3. Summary 更新训练：为什么主 reward 训练不到，怎么补

你已经发现：主 reward 只看结构关系矩阵，summary 更新的效果大多延迟体现。

建议把 summary 更新训练拆成：

- **SFT/监督**（teacher summary）
  - 文件：`summary_based_classifier/llm/updater.py`
  - 数据生成：`summary_based_classifier/data/prepare_dataset.py`（新增“生成 updater 监督数据”的阶段）

- **proxy reward / 规则判定**（即时可计算）
  - 规则检查：
    - 不超出父类范围（entailment / NLI / judge）
    - 不与兄弟重复（overlap / NLI / judge）
    - 覆盖新证据
  - 下游分类增益：更新前后对 classifier 输出 margin 的提升

落点：
- `summary_based_classifier/training/train_updater.py`：支持用规则/偏好数据做 DPO 或 rerank

---

### 4. DPO/轨迹训练需要调整的点

如果你仍保留 DPO（对 classify/updater 做偏好训练），建议调整为：

- **对 classify/结构动作**：用 oracle π 生成偏好：
  - `chosen = π action`，`rejected = 其它可行 action`
- **对 updater**：用规则+下游分类增益生成偏好（而不是主结构 reward）

主要修改文件：
- `summary_based_classifier/training/train_dpo.py`
- `summary_based_classifier/training/trajectory_dpo_trainer.py`
- `summary_based_classifier/core/trajectory/trajectory_sampler.py`

---

### 5. 需要你后续真正动刀的代码点（清单）

- **实现新 reward**：新增 `summary_based_classifier/reward/feasible_reward.py`，并在在线处理/采样处接入。
- **实现 oracle π**：新增 `summary_based_classifier/core/policy/oracle_policy.py`，并在构建/采样时记录。
- **改 InsertParentPath 触发与约束**：在 `builder.py` 中把“只在 CreateLeaf 时发生、x 必须参与归拢”写成硬规则，并把 (TC) 作为可行性检查。
- **数据集生成**：在 `data/prepare_dataset.py` 与 `data/prepare_dataset_dpo.py` 里加入“π 监督数据/偏好数据”的导出。
- **训练入口**：
  - `training/train_classify_generator.py`：接入 π 的监督 label
  - `training/train_updater.py`：接入规则/teacher summary
  - `training/train_dpo.py`：把结构 reward 替换为“π 偏好”或新 reward

---

### 6. 迁移注意事项

- 现在 `summary_based_classifier` 是 package：建议用 `python -m ...` 方式运行。
- 旧入口兼容：`summary_based_classifier/run.py` 作为 wrapper 调用 `cli/run_pipeline.py`。
- 如果你在外部脚本里直接 `python summary_based_classifier/xxx.py`，可能会因为包内绝对 import 报错；统一改成 `python -m summary_based_classifier.xxx`。
