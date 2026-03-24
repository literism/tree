"""
训练分类生成系统
- 类式训练器
- SFTTrainer + LoRA
- prompt/completion 数据格式
"""

import argparse
import faulthandler
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from summary_based_classifier.config import SummaryBasedConfig, TrainingConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

faulthandler.enable()


def _dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"不支持的dtype配置: {dtype_str}")


def _keep_prompt_completion_only(dataset_dict):
    for split in dataset_dict.keys():
        keep_cols = {"prompt", "completion"}
        cols_to_remove = [c for c in dataset_dict[split].column_names if c not in keep_cols]
        if cols_to_remove:
            dataset_dict[split] = dataset_dict[split].remove_columns(cols_to_remove)
    return dataset_dict


def _validate_dataset_columns(dataset_dict):
    for split in dataset_dict.keys():
        cols = set(dataset_dict[split].column_names)
        if "prompt" not in cols or "completion" not in cols:
            raise ValueError(
                f"{split} 数据缺少必要列。当前列: {sorted(cols)}；需要包含: ['prompt', 'completion']"
            )


class ClassifyGeneratorTrainer:
    def __init__(
        self,
        base_model: str,
        train_data: str,
        val_data: str,
        output_dir: str,
        training_config: TrainingConfig,
        max_samples: Optional[int] = None,
    ):
        self.base_model = base_model
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_config = training_config
        self.max_samples = max_samples

        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if torch.cuda.is_available() and self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)

    def _print_header(self):
        print("=" * 80)
        print("训练分类生成系统")
        print("=" * 80)
        print(f"基础模型: {self.base_model}")
        print(f"训练数据: {self.train_data}")
        print(f"验证数据: {self.val_data}")
        print(f"输出目录: {self.output_dir}")
        print(f"分布式: world_size={self.world_size}, local_rank={self.local_rank}")

        tc = self.training_config
        print("\n训练配置:")
        print(f"  - LoRA rank: {tc.lora_r}")
        print(f"  - LoRA alpha: {tc.lora_alpha}")
        print(f"  - Learning rate: {tc.learning_rate}")
        print(f"  - Epochs: {tc.num_epochs}")
        print(f"  - Batch size: {tc.batch_size}")
        print(f"  - Gradient accumulation: {tc.gradient_accumulation_steps}")
        print(f"  - Load 4bit: {tc.load_in_4bit}")

    def _load_datasets(self):
        cache_dir = "/mnt/literism/.cache/huggingface/datasets"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = cache_dir

        print("\n加载数据集...")
        dataset = load_dataset(
            "json",
            data_files={"train": self.train_data, "validation": self.val_data},
            cache_dir=cache_dir,
        )

        if self.max_samples is not None and self.max_samples > 0:
            print(f"限制训练集到: {self.max_samples}")
            if len(dataset["train"]) > self.max_samples:
                dataset["train"] = dataset["train"].select(range(self.max_samples))
            max_val_samples = max(int(self.max_samples * 0.1), 10)
            if len(dataset["validation"]) > max_val_samples:
                dataset["validation"] = dataset["validation"].select(range(max_val_samples))

        dataset = _keep_prompt_completion_only(dataset)
        _validate_dataset_columns(dataset)

        print(f"  - 训练集: {len(dataset['train'])}")
        print(f"  - 验证集: {len(dataset['validation'])}")
        return dataset

    def _load_model_and_tokenizer(self):
        print("\n加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            # 两卡单进程训练时，flash_sdp 在部分驱动/模型组合下可能触发底层崩溃
            # 仅关闭 flash，保留 mem_efficient/math，尽量兼顾稳定性和速度
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                print("  - SDPA后端: flash=False, mem_efficient=True, math=True")
            except Exception as e:
                print(f"  - SDPA后端配置跳过: {e}")

        tc = self.training_config
        quantization_config = None
        if tc.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_dtype_from_str(tc.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=tc.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=tc.bnb_4bit_quant_type,
            )

        print("\n加载基础模型...")
        visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        model_parallel_enabled = visible_gpu_count > 1 and self.world_size == 1
        device_map = "auto" if model_parallel_enabled else None
        print(f"  - 可见GPU数量: {visible_gpu_count}")
        print(f"  - 模型并行: {'启用' if model_parallel_enabled else '关闭'}")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device_map,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if tc.bf16 else torch.float16,
        )
        model.config.use_cache = False

        if tc.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        elif torch.cuda.is_available() and self.local_rank == -1 and not model_parallel_enabled:
            model = model.to("cuda")

        return model, tokenizer

    def _build_peft_config(self) -> LoraConfig:
        tc = self.training_config
        print("\n配置LoRA...")
        return LoraConfig(
            r=tc.lora_r,
            lora_alpha=tc.lora_alpha,
            lora_dropout=tc.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def _build_sft_config(self) -> SFTConfig:
        tc = self.training_config
        max_length = int(tc.max_length)

        print("\n配置SFT参数...")
        print(f"  - max_length: {max_length}")
        print("  - completion_only_loss: True")
        print(f"  - per_device_train_batch_size: {tc.batch_size}")
        print(f"  - per_device_eval_batch_size: {tc.batch_size}")
        print(f"  - dataloader_num_workers: {tc.dataloader_num_workers}")

        return SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.batch_size,
            per_device_eval_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            warmup_ratio=tc.warmup_ratio,
            logging_steps=tc.logging_steps,
            save_steps=tc.save_steps,
            eval_steps=tc.eval_steps,
            save_total_limit=tc.save_total_limit,
            bf16=tc.bf16,
            gradient_checkpointing=tc.gradient_checkpointing,
            dataloader_num_workers=tc.dataloader_num_workers,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=True,
            max_length=max_length,
            packing=False,
            completion_only_loss=True,
            assistant_only_loss=False,
            ddp_find_unused_parameters=False,
        )

    def train(self):
        self._print_header()
        dataset = self._load_datasets()
        model, tokenizer = self._load_model_and_tokenizer()
        peft_config = self._build_peft_config()
        sft_config = self._build_sft_config()

        print("\n创建SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            peft_config=peft_config,
            processing_class=tokenizer,
        )

        print("\n" + "=" * 80)
        print("开始训练...")
        print("=" * 80)
        trainer.train()

        print("\n" + "=" * 80)
        print("保存模型...")
        print("=" * 80)

        if not trainer.is_world_process_zero():
            print("非主进程跳过保存。")
            return

        adapter_path = self.output_dir / "adapter"
        final_model_path = self.output_dir / "final_model"

        print(f"  - 保存LoRA adapter: {adapter_path}")
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        print(f"  - 合并并保存完整模型: {final_model_path}")
        merged_model = trainer.model.merge_and_unload()

        num_added = len(tokenizer.get_added_vocab())
        true_vocab_size = tokenizer.vocab_size + num_added
        if merged_model.get_input_embeddings().weight.shape[0] != true_vocab_size:
            merged_model.resize_token_embeddings(true_vocab_size)
        merged_model.config.vocab_size = true_vocab_size

        merged_model.save_pretrained(final_model_path, safe_serialization=True)
        tokenizer.save_pretrained(final_model_path)
        print(f"\n训练完成！模型已保存到: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="训练分类生成系统")
    parser.add_argument("--base_model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据文件")
    parser.add_argument("--val_data", type=str, required=True, help="验证数据文件")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大训练样本数")
    args = parser.parse_args()

    if args.config:
        cfg = SummaryBasedConfig.from_json(args.config)
        training_config = cfg.training
    else:
        training_config = TrainingConfig()

    trainer = ClassifyGeneratorTrainer(
        base_model=args.base_model,
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        training_config=training_config,
        max_samples=args.max_samples,
    )
    trainer.train()


if __name__ == "__main__":
    main()
