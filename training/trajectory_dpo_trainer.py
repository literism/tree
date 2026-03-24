"""
Trajectory-level DPO训练器
实现轨迹级别的Direct Preference Optimization
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
from tqdm import tqdm


@dataclass
class PreferencePair:
    """偏好对"""
    positive_trajectory: List[Tuple[str, str]]  # [(prompt, completion), ...]
    negative_trajectory: List[Tuple[str, str]]  # [(prompt, completion), ...]
    margin: float  # R_pos - R_neg


class TrajectoryDPODataset(Dataset):
    """Trajectory-level DPO数据集"""
    
    def __init__(
        self,
        preference_pairs: List[PreferencePair],
        tokenizer,
        max_length: int = 16384
    ):
        """
        Args:
            preference_pairs: 偏好对列表
            tokenizer: tokenizer
            max_length: 最大长度
        """
        self.preference_pairs = preference_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]
        
        # 使用新的PreferencePair格式（chosen_prompt/completion, rejected_prompt/completion）
        # Tokenize chosen (positive)
        chosen_text = pair.chosen_prompt + pair.chosen_completion
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Label: -100 for prompt tokens, actual tokens for completion
        chosen_prompt_tokens = self.tokenizer(
            pair.chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        chosen_labels = chosen_tokens['input_ids'].clone()
        chosen_labels[:, :chosen_prompt_tokens['input_ids'].shape[1]] = -100
        
        # Tokenize rejected (negative)
        rejected_text = pair.rejected_prompt + pair.rejected_completion
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        rejected_prompt_tokens = self.tokenizer(
            pair.rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        rejected_labels = rejected_tokens['input_ids'].clone()
        rejected_labels[:, :rejected_prompt_tokens['input_ids'].shape[1]] = -100
        
        return {
            'pos_inputs': [chosen_tokens['input_ids']],
            'pos_labels': [chosen_labels],
            'neg_inputs': [rejected_tokens['input_ids']],
            'neg_labels': [rejected_labels],
            'margin': pair.reward_diff  # 使用reward_diff作为margin
        }


class TrajectoryDPOTrainer:
    """Trajectory-level DPO训练器"""
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        device: str = 'cuda'
    ):
        """
        Args:
            model: 要训练的模型
            ref_model: 参考模型（冻结）
            tokenizer: tokenizer
            beta: DPO的beta参数
            learning_rate: 学习率
            device: 设备
        """
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
    
    def compute_trajectory_logprob(
        self,
        model,
        inputs_list: List[torch.Tensor],
        labels_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算整条轨迹的log概率（所有actions的log概率之和）
        
        Args:
            model: 模型
            inputs_list: 输入列表（轨迹中的每个action）
            labels_list: 标签列表
            
        Returns:
            log概率（标量）
        """
        total_logprob = 0.0
        
        for inputs, labels in zip(inputs_list, labels_list):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=inputs,
                    labels=labels,
                    return_dict=True
                )
                
                # 获取logits
                logits = outputs.logits
                
                # 计算log概率
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # 只计算非-100的位置
                mask = (shift_labels != -100)
                
                if mask.sum() > 0:
                    # 计算log softmax
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    
                    # 提取对应的log概率
                    selected_log_probs = torch.gather(
                        log_probs,
                        dim=-1,
                        index=shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    # 只累加非-100位置的log概率
                    token_logprobs = selected_log_probs * mask.float()
                    total_logprob += token_logprobs.sum()
        
        return total_logprob
    
    def compute_dpo_loss(
        self,
        pos_inputs: List[torch.Tensor],
        pos_labels: List[torch.Tensor],
        neg_inputs: List[torch.Tensor],
        neg_labels: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算DPO损失
        
        DPO Loss = -log(sigmoid(β * (log π_θ(y_w|x) - log π_θ(y_l|x) 
                                       - log π_ref(y_w|x) + log π_ref(y_l|x))))
        
        对于trajectory-level，log π_θ(trajectory) = sum of log π_θ(action_i)
        """
        # 计算正轨迹的log概率
        pos_logprob_policy = self.compute_trajectory_logprob(
            self.model, pos_inputs, pos_labels
        )
        pos_logprob_ref = self.compute_trajectory_logprob(
            self.ref_model, pos_inputs, pos_labels
        )
        
        # 计算负轨迹的log概率
        neg_logprob_policy = self.compute_trajectory_logprob(
            self.model, neg_inputs, neg_labels
        )
        neg_logprob_ref = self.compute_trajectory_logprob(
            self.ref_model, neg_inputs, neg_labels
        )
        
        # 计算DPO损失
        logits = (pos_logprob_policy - neg_logprob_policy) - \
                 (pos_logprob_ref - neg_logprob_ref)
        
        loss = -F.logsigmoid(self.beta * logits)
        
        return loss
    
    def train_step(self, batch: Dict) -> float:
        """训练一个batch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_dpo_loss(
            pos_inputs=batch['pos_inputs'],
            pos_labels=batch['pos_labels'],
            neg_inputs=batch['neg_inputs'],
            neg_labels=batch['neg_labels']
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        dataset: TrajectoryDPODataset,
        num_epochs: int = 1,
        batch_size: int = 1,
        logging_steps: int = 10,
        save_steps: int = 500,
        output_dir: str = './output'
    ):
        """
        训练模型
        
        Args:
            dataset: 数据集
            num_epochs: epoch数
            batch_size: batch大小（建议为1，因为轨迹长度不同）
            logging_steps: 日志步数
            save_steps: 保存步数
            output_dir: 输出目录
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        global_step = 0
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            pbar = tqdm(dataloader, desc="Training")
            for batch in pbar:
                loss = self.train_step(batch)
                total_loss += loss
                global_step += 1
                
                if global_step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    total_loss = 0.0
                
                if global_step % save_steps == 0:
                    self.save_model(output_dir, global_step)
        
        # 最终保存
        self.save_model(output_dir, 'final')
    
    def save_model(self, output_dir: str, step):
        """保存模型"""
        save_path = os.path.join(output_dir, f'checkpoint-{step}')
        os.makedirs(save_path, exist_ok=True)
        
        # 如果是PEFT模型，只保存adapter
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), os.path.join(save_path, 'model.pt'))
        
        self.tokenizer.save_pretrained(save_path)
        print(f"\nModel saved to {save_path}")
    
    def _collate_fn(self, batch):
        """Collate function（batch_size=1时直接返回）"""
        if len(batch) == 1:
            return batch[0]
        # 如果batch_size>1，需要padding（这里简化处理）
        return batch[0]


def load_model_and_tokenizer(
    model_path: str,
    load_in_4bit: bool = False,
    device_map: str = 'auto'
):
    """
    加载模型和tokenizer
    
    Args:
        model_path: 模型路径
        load_in_4bit: 是否使用4-bit量化
        device_map: 设备映射
        
    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
    
    return model, tokenizer


def setup_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None
):
    """
    设置LoRA
    
    Args:
        model: 基础模型
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: 目标模块
        
    Returns:
        PEFT模型
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'gate_proj', 'up_proj', 'down_proj']
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias='none',
        task_type='CAUSAL_LM'
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model

