## 一、符号说明

### 1.1 对象

- 树：$T=(V,E)$，根为 $r$
- 深度：$d_T(v)$ 表示节点 $v$ 到根的距离，根深度取 $0$
- 子节点集合：$\mathrm{Ch}_T(v)$
- 最低公共祖先：$\mathrm{LCA}_T(u,v)$
- 文章集合：按顺序 $d_1,d_2,\dots,d_n$
- 已处理集合：$S_t=\{d_1,\dots,d_t\}$
- 叶子绑定：$\ell_T(d)\in V$ 为文章 $d$ 当前所在叶子
- 目标树：gold 树 $T^*$

### 1.2 文章对关系 $R_T(i,j)$

对任意两篇文章 $d_i,d_j$，定义它们在 $T$ 上的关系：

$$
R_T(i,j)=
\begin{cases}
-1, & \text{if }\ \ell_T(d_i)=\ell_T(d_j)\\
L_T(i,j), & \text{if }\ \ell_T(d_i)\ne\ell_T(d_j)
\end{cases}
$$

其中

$$
L_T(i,j)=d_T\big(\mathrm{LCA}_T(\ell_T(d_i),\ell_T(d_j))\big)\in\{0,1,2,\dots\}.
$$

目标值：$R_{T^*}(i,j)$，以及 $L^*(i,j)=L_{T^*}(i,j)$。

### 1.3 子树包含的文章集合

对树上的任意节点 $x$，定义

$$
\mathrm{Docs}_T(x)=\{d\in S_t|\ \ell_T(d)\ \text{位于}\ x\ \text{的子树中}\}.
$$

---

## 二、单篇文章处理流程

输入：上一时刻树 $T_{t-1}$，新文章 $d_t$。
输出：新树 $T_t$ 与叶子绑定 $\ell_{T_t}(d_t)$。

允许的操作：
- **ClassifyDoc**：将文章分到已有节点
- **CreateLeaf**：在某节点下创建新叶子
- **InsertParentPath**：在某节点处插入新父节点并收拢两棵兄弟子树
- **UpdateSummary**：更新节点 summary

### 2.1 Top-down：为新文章找到（或创建）叶子

从根 $r$ 开始沿树向下分类：
- 若命中某个已有叶子 $u$，令 $\ell(d_t)=u$
- 若判定需要新类，则在当前节点 $v$ 下执行 **CreateLeaf** 创建新叶子 $x$，并令 $\ell(d_t)=x$

### 2.2 局部重挂：只在 CreateLeaf 后，最多执行一次 InsertParentPath

当且仅当本次走到 **CreateLeaf**（创建了新叶子 $x$）时，允许**最多执行一次**局部收拢：

- **InsertParentPath(T, v, x, y)**
  - $v$ 是新叶子 $x$ 的父节点（创建时 $x\in\mathrm{Ch}_T(v)$），且 $d_T(v)=k$
  - 选择兄弟子树根 $y\in\mathrm{Ch}_T(v)\setminus\{x\}$
  - 创建新内部节点 $p$ 作为 $v$ 的新孩子，并令 $\mathrm{Ch}_T(p)=\{x,y\}$，同时把 $x,y$ 从 $\mathrm{Ch}_T(v)$ 移除

**效果**：设 $T'$ 为操作后树。则对任意
$d_i,d_j\in \mathrm{Docs}_{T'}(p)$ 且 $\ell_{T'}(d_i)\ne \ell_{T'}(d_j)$，都有
$$
R_{T'}(i,j)=R_T(i,j)+1.
$$

同一叶子内的对始终为 $-1$，不受影响。

---

## 三、势函数和 reward

势函数同时惩罚：
- **推错/overshoot**：当前关系比目标更深
- **分类错误（目标同类但你分开）**：目标 $R_{T^*}=-1$，但当前 $R_T\ge 0$

这两者都可以统一为同一个“违反不等式”的事件：$R_T(i,j)>R_{T^*}(i,j)$。

### 3.1 带 $-\infty$ 的势函数

定义

$$
\Phi(T,S)=
\begin{cases}
-\infty, & \exists\ i,j\in S\ \text{s.t.}\ R_T(i,j)>R_{T^*}(i,j)\\
\sum_{\{i,j\}\subseteq S} w_{ij}\cdot R_T(i,j), & \text{otherwise}
\end{cases}
\qquad (w_{ij}\ge 0)
$$

直观：只要出现一次“推错”或“目标同类却被分开”，整体直接判为 $-\infty$。

### 3.2 单步 reward

$$
r_t=\Phi(T_t,S_t)-\Phi(T_{t-1},S_{t-1}).
$$

---

## 四、可行性

### 4.1 可行性定义

称 $T$ 对 $S$ **可行**，当且仅当对所有 $i,j\in S$ 都满足

$$
R_T(i,j)\le R_{T^*}(i,j).
$$

于是有等价关系：
- $T$ 对 $S$ 可行 $\Longleftrightarrow \Phi(T,S)$ 有限
- 不可行 $\Longleftrightarrow \Phi(T,S)=-\infty$

### 4.2 (TC) InsertParentPath 的可行性检查

对一次 **InsertParentPath(T, v, x, y)**（其中 $d_T(v)=k$）：它会让 $\mathrm{Docs}(p)$ 内、不同叶子之间的 $R$ 全部 $+1$。

因此该动作 **可行当且仅当**：对所有
$d_i,d_j\in \mathrm{Docs}_{T'}(p)$ 且 $\ell_{T'}(d_i)\ne \ell_{T'}(d_j)$，都有

$$
R_{T^*}(i,j)\ge k+1.
$$

---

## 五、证明greedy策略最终一定得到全局最优

##### 证明思路：

- 定义一个策略$\pi^*$，它依赖于目标树$T^*$
- 证明greedy策略的每一步等同于采用策略$\pi^*$
- 证明策略$\pi^*$最终达到全局最优上界

##### 5.1 全局最优上界

令全集 $S_n=\{d_1,\dots,d_n\}$。对任何最终可行的树 $T$，都有逐对不等式 $R_T(i,j)\le R_{T^*}(i,j)$，权重 $w_{ij}\ge 0$，因此

$$
\Phi(T,S_n)=\sum_{\{i,j\}\subseteq S_n} w_{ij}\,R_T(i,j)
\le
\sum_{\{i,j\}\subseteq S_n} w_{ij}\,R_{T^*}(i,j)
=\Phi(T^*,S_n).
$$

所以 $\Phi(T^*,S_n)$ 是所有可行策略最终得分的共同上界。

$$
\sum_{t=1}^n r_t = \Phi(T_n,S_n)-\Phi(T_0,S_0),
$$

初始项是常数，所以**最大化最终 $\Phi(T_n,S_n)$ 等价于最大化总和 reward**。

##### 5.2 定义策略 $\pi^*$

策略 $\pi^*$ 使用目标树 $T^*$ 来决定：
- Top-down 路由应当走向哪条目标路径；
- 若 CreateLeaf 后要做 InsertParentPath，应选择哪个兄弟 $y$。

###### 5.2.1 目标簇标识

对任意深度 $k\ge 0$，定义目标树第 $k$ 层簇标识：
- $\mathrm{anc}_k^*(d)$：文章 $d$ 在 $T^*$ 上深度为 $k$ 的祖先。

###### 5.2.2 $\pi^*$ 的具体规则

处理 $d_t$：

1) **oracle Top-down**：总是把 $d_t$ 放入当前树中包含其正确同类文章的叶子结点；若需要新类则在正确路径父节点 $v$ 下 CreateLeaf 得到新叶子 $x$。

2) **若本步发生 CreateLeaf@v**：令 $k=d_T(v)$。

- 在兄弟集合 $\mathrm{Ch}_T(v)\setminus\{x\}$ 中寻找唯一的（若存在）兄弟子树 $y$，满足

$$
\forall d\in \mathrm{Docs}_T(y),\quad \mathrm{anc}_{k+1}^*(d)=\mathrm{anc}_{k+1}^*(d_t).
$$

- 若存在这样的 $y$，则执行 InsertParentPath(T, v, x, y)；否则不执行结构动作。

直观解释：只把“目标第 $k+1$ 层应当同簇”的新叶子与一个已存在兄弟簇收拢起来。

##### 5.3 引理 A：$\pi^*$ 的动作等同于单步 reward 最大

**证明**：

归纳法，当$t = 1$时，策略$\pi^*$会在根节点下创建节点，显然等同于reward最大。

当$t>1$时，$T_{t-1}$由$\pi^*$ 得到，那么考虑时刻$t,d_t$：

- 若不存在任何通过 (TC) 的 InsertParentPath，只能执行分类或者创建新类，则任何可行策略都不能改变原有结构；此时 $\pi^*$ 的选择显然是单步最优。
- 若存在通过 (TC) 的 InsertParentPath：
  - 一旦选择某个 $y$ 不能满足目标第 $k+1$ 层同簇， $\forall d_a \in Docs_T(y)$ 满足 $R_T(d_a,d_t)=k+1$ 但 $R_{T^*}(a,b)\le k$，导致不可行（$-\infty$），因此这种动作不在可行动作集合里。
  - 假设有多个$y$满足目标第 $k+1$ 层同簇，假设它们是${v_1,...,v_n}$，那么$\forall v_i,v_j, d_a\in Doc_T(v_i), d_b\in Doc_T(v_j), \mathrm{anc}_{k+1}^*(d)=\mathrm{anc}_{k+1}^*(d_t)$，这与“$T_{t-1}$由$\pi^*$ 得到”矛盾。因此最多只有一个$y$满足目标第 $k+1$ 层同簇。
  - $\pi^*$在此时会将新类$x$与唯一的$y$归拢，此时得到的新类$p$下所有文章之间的$R' = R+1, \Delta\Phi>0$，因此$\pi^*$等价于reward最大。

##### 5.4 引理 B：$\pi^*$ 最终达到上界

对目标树 $T^*$ 的任意节点 $u^*$，定义它覆盖的文章集合：

$$
\mathrm{Docs}^*(u^*) = \{ d\in S_n | d \text{在 }T^*\text{ 的叶子位于 }u^*\text{ 的子树中}\}.
$$

在时刻 $t$，定义 $C_t$ 为“已经在当前树中被显式表示为某个连通子树的目标节点集合”。
存在一一映射 $f_t:C_t\to V(T_t)$ 使得
$$
\forall u^*\in C_t,\quad \mathrm{Docs}_{T_t}(f_t(u^*))=\mathrm{Docs}^*(u^*)\cap S_t\\
若有多个u^*_{1\backsim n}，满足\\
\mathrm{Docs}_{T_t}(f_t(u^*_1)) = ... = \mathrm{Docs}_{T_t}(f_t(u^*_n))\\
则u^*_{1\backsim n}必然有继承关系，则取最深的那一个加入C_t
$$

**引理 B**：每个目标内部节点$u^*$都会被“补齐”，$\forall u^* \in V(T^*),u^*\in C_n$

**证明**：

对任意目标结构树内部节点 $u^*$（非根），令 $t_0$ 为这样一个最小时间：在前缀 $S_{t_0}$ 中，$u^*$ 的子树里已经出现了至少两篇文章，且它们属于 $u^*$ 的两个不同孩子子树。
则在时间 $t_0$ 这一步，策略 $\pi^*$ 必然会在  $u^*$对应的当前结点的父节点处执行一次 InsertParentPath，把新叶子与一个已存在兄弟簇收拢，从而使得 $u^*$ 成为一个显式簇（加入 $C_t$）。
由此可得：沿着任意目标叶子的路径，每一层缺失父类都会在其“第二个孩子分支出现”的那一刻被补齐，因此$\forall u^* \in V(T^*),u^*\in C_n$。


**还需要证明：不会构造多余的节点（Soundness / No-extra-nodes）**。

我们证明：每一个由 InsertParentPath 构造出来的内部节点 $p\in V(T_n)$，都对应目标树中的一个唯一节点 $u^*$，并且这种对应是单射（不会重复）。

- **定义（内部节点的目标标签）**：对任意当前树中的内部节点 $p$（满足 $\mathrm{Docs}_{T_n}(p)\ne\varnothing$），令

$$
\ell^*(p):=\mathrm{anc}_{d_{T_n}(p)}^*(d)\quad \text{其中 } d\in\mathrm{Docs}_{T_n}(p)\text{ 任取一篇文章。}
$$

我们接下来证明这个定义良好（与选哪篇 $d$ 无关），并且

$$
\mathrm{Docs}_{T_n}(p)=\mathrm{Docs}^*(\ell^*(p))\cap S_n.
$$

- **引理 B1（定义良好且不跨簇）**：对任意内部节点 $p$，$\ell^*(p)$ 与选取的 $d\in\mathrm{Docs}(p)$ 无关。

  **理由**：$p$ 是某次 InsertParentPath 在某个深度 $k=d(p)-1$ 的父节点 $v$ 下，把新叶 $x$ 与兄弟 $y$ 收拢得到的。由 $\pi^*$ 的选择规则，
  $$
  \forall d'\in\mathrm{Docs}(y),\ \mathrm{anc}_{k+1}^*(d')=\mathrm{anc}_{k+1}^*(d_t),
  $$
  且 $x$ 子树内的文章也都满足同一个 $\mathrm{anc}_{k+1}^*$（否则会违反可行性/纯度）。因此 $p$ 子树内所有文章在目标第 $d(p)=k+1$ 层的祖先一致，$\ell^*(p)$ 定义良好。

- **引理 B2（每个内部节点都对应某个目标节点）**：对任意内部节点 $p$，都有
  $$
  \mathrm{Docs}_{T_n}(p)=\mathrm{Docs}^*(\ell^*(p))\cap S_n.
  $$

  **理由**：由上一个引理，$p$ 子树里所有文章在目标第 $d(p)$ 层祖先都等于同一个目标节点 $u^*=\ell^*(p)$；另一方面，$\pi^*$ 的 Top-down oracle 分类保证不会把任何不属于该目标簇的文章路由进 $p$ 的子树，因此 $\mathrm{Docs}(p)$ 恰好等于该目标簇与已出现集合的交。

- **引理 B3（单射：不会重复构造同一个目标节点）**：若 $p_1\ne p_2$ 且 $d(p_1)=d(p_2)$，则 $\ell^*(p_1)\ne \ell^*(p_2)$。

  **理由**：同一深度的两个不同目标簇在 $T^*$ 上互不相交。由引理 B2，$\mathrm{Docs}(p_1)$ 与 $\mathrm{Docs}(p_2)$ 分别等于两个不同目标簇与 $S_n$ 的交，因此不可能相同。反过来若 $\ell^*(p_1)=\ell^*(p_2)$ 则两者覆盖文章集合相同，必然违反树的连通/唯一父结构（或者与“选择最深的那一个加入 $C_t$”的约定矛盾）。

综上，当前树中由 InsertParentPath 产生的每个内部节点都能唯一对应到目标树中的一个节点（同深度层级下为单射），因此**不会产生多余的内部节点**。

再结合前面已经证明的“每个目标内部节点最终都会被补齐”（Completeness），得到：最终当前树与目标树在文章视角下的内部节点集合是一一对应的（同构），不会少也不会多。

于是对任意文章对 $(i,j)$，其目标最深共同祖先的所有缺失层都被补齐，最终得到 $R_{T_n}(i,j)=R_{T^*}(i,j)$，从而

$$
\Phi(T_n,S_n)=\Phi(T^*,S_n).
$$

结合 5.1 的上界可知：$\pi^*$ 达到上界，因此总 reward 全局最大。