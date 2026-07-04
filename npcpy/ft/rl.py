from dataclasses import dataclass, field
from typing import List

from datetime import datetime
import glob
import json
import os
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

try:
    from peft import LoraConfig, PeftModel
except ImportError:
    LoraConfig = None
    PeftModel = None

try:
    from trl import DPOTrainer, DPOConfig
except (ImportError, RuntimeError):
    DPOTrainer = None
    DPOConfig = None

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as mlx_opt
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.trainer import TrainingArgs as MLXTrainingArgs, train as mlx_train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    nn = None

import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
from npcpy.ft.sft import _resolve_mlx_model, _num_lora_layers

@dataclass
class RLConfig:
    base_model_name: str = "Qwen/Qwen3-0.6B"
    adapter_path: str = "./rl_adapter"
    device: str = "cpu"
    max_iterations: int = 8
    min_reward_gap: float = 0.4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    beta: float = 0.5
    max_length: int = 512
    max_prompt_length: int = 256
    use_4bit: bool = False
    use_8bit: bool = False
    fp16: bool = False
    bf16: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    max_pairs: int = 200
    group_size: int = 4
    clip_eps: float = 0.2
    warmup_steps: int = 5
    logging_steps: int = 5
    save_steps: int = 20

class TaskExecutor:

    def __init__(
        self,
        agent: NPC,
        max_iterations: int = 8
    ):
        self.agent = agent
        self.max_iterations = max_iterations

    def execute_task(
        self,
        task_prompt: str
    ) -> Dict[str, Any]:

        messages = [
            {
                "role": "system",
                "content": self.agent.primary_directive
            }
        ]

        raw_responses = []
        current_prompt = task_prompt

        for i in range(self.max_iterations):
            response_obj = self.agent.get_llm_response(
                current_prompt,
                messages=messages,
                auto_process_tool_calls=True
            )

            raw_responses.append(response_obj)
            messages = response_obj.get('messages', messages)

            last_content = messages[-1].get('content', '')

            if self._is_complete(last_content):
                return {
                    "raw_responses": raw_responses,
                    "final_output": last_content,
                    "total_iterations": i + 1,
                    "completed": True
                }

            current_prompt = (
                "Continue or provide final answer."
            )

        return {
            "raw_responses": raw_responses,
            "final_output": messages[-1].get('content', ''),
            "total_iterations": self.max_iterations,
            "completed": False
        }

    def _is_complete(self, content: str) -> bool:

        completion_markers = [
            "final answer:",
            "conclusion:",
            "result:",
            "therefore",
            "in summary"
        ]
        content_lower = content.lower()
        return any(
            marker in content_lower
            for marker in completion_markers
        )

def collect_traces(
    tasks: List[Dict[str, Any]],
    agents: List[NPC],
    reward_fn: Callable[[Dict], float],
    config: Optional[RLConfig] = None
) -> List[Dict[str, Any]]:

    if config is None:
        config = RLConfig()

    traces = []

    for task in tasks:
        task_prompt = task.get('prompt', task.get('input', ''))

        for agent in agents:
            executor = TaskExecutor(
                agent,
                max_iterations=config.max_iterations
            )

            result = executor.execute_task(task_prompt)

            trace = {
                "agent_name": agent.name,
                "task_prompt": task_prompt,
                "final_output": result['final_output'],
                "total_iterations": result['total_iterations'],
                "completed": result['completed'],
                "task_metadata": task
            }

            trace['reward'] = reward_fn(trace)

            traces.append(trace)

            print(f"Agent {agent.name}: Reward={trace['reward']:.2f}")

    return traces

def create_preference_pairs(
    traces: List[Dict[str, Any]],
    min_reward_gap: float = 0.4
) -> Dataset:

    df = pd.DataFrame(traces)
    df = df[df['reward'] > -1.0].copy()

    if len(df) < 2:
        return None

    df = df.sort_values('reward', ascending=False)

    top_quantile = df['reward'].quantile(
        0.8,
        interpolation='higher'
    )
    low_quantile = df['reward'].quantile(
        0.2,
        interpolation='lower'
    )

    high_traces = df[df['reward'] >= top_quantile]
    low_traces = df[df['reward'] <= low_quantile]

    pairs = []

    for _, high_trace in high_traces.iterrows():
        for _, low_trace in low_traces.iterrows():
            reward_gap = (
                high_trace['reward'] - low_trace['reward']
            )

            if reward_gap >= min_reward_gap:
                pairs.append({
                    "prompt": str(high_trace['task_prompt']),
                    "chosen": str(high_trace['final_output']),
                    "rejected": str(low_trace['final_output'])
                })

    if len(pairs) < 5:
        print(f"Warning: Only {len(pairs)} pairs found. May overfit.")

    return Dataset.from_list(pairs)

def _train_dpo_mlx(
    pairs: List[Dict[str, str]],
    config: RLConfig,
) -> str:
    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx and mlx-lm. pip install mlx mlx-lm")

    os.makedirs(config.adapter_path, exist_ok=True)

    mlx_model_name = _resolve_mlx_model(config.base_model_name)
    model, tokenizer = mlx_load(mlx_model_name)

    lora_cfg = {
        "rank": config.lora_r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "scale": config.lora_alpha / config.lora_r,
    }
    linear_to_lora_layers(model, _num_lora_layers(config.lora_r), lora_cfg)

    processed = []
    for p in pairs:
        text = f"{p['prompt']}\n{p['chosen']}"
        tokens = tokenizer.encode(text)
        if tokens[-1] != tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)
        processed.append((tokens, 0))

    class _ProcessedDataset:
        def __init__(self, data):
            self._data = data
        def __getitem__(self, idx):
            return self._data[idx]
        def __len__(self):
            return len(self._data)

    train_dataset = _ProcessedDataset(processed)

    iters_per_epoch = max(1, len(pairs) // config.per_device_train_batch_size)
    total_iters = iters_per_epoch * config.num_train_epochs

    adapter_file = os.path.join(config.adapter_path, "adapters.safetensors")

    training_args = MLXTrainingArgs(
        batch_size=config.per_device_train_batch_size,
        iters=total_iters,
        val_batches=0,
        steps_per_report=config.logging_steps,
        steps_per_eval=0,
        steps_per_save=config.save_steps,
        max_seq_length=config.max_length,
        adapter_file=adapter_file,
        grad_checkpoint=True,
        grad_accumulation_steps=config.gradient_accumulation_steps,
    )

    optimizer = mlx_opt.AdamW(learning_rate=config.learning_rate)

    print(f"MLX DPO (SFT on chosen): {mlx_model_name}, {len(pairs)} pairs, {total_iters} iters")

    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
    )

    adapter_config = {
        "model": mlx_model_name,
        "fine_tune_type": "lora",
        "num_layers": _num_lora_layers(config.lora_r),
        "lora_parameters": {
            "rank": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "scale": config.lora_alpha / config.lora_r,
        },
    }
    with open(os.path.join(config.adapter_path, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"MLX adapter saved to {config.adapter_path}")
    return config.adapter_path

def _train_dpo_torch(
    traces: List[Dict[str, Any]],
    config: RLConfig,
) -> str:

    preference_dataset = create_preference_pairs(
        traces,
        min_reward_gap=config.min_reward_gap
    )

    if preference_dataset is None or len(preference_dataset) == 0:
        print("No valid preference pairs. Cannot train.")
        return None

    if config.max_pairs and len(preference_dataset) > config.max_pairs:
        preference_dataset = preference_dataset.select(range(config.max_pairs))

    print(f"Training with {len(preference_dataset)} preference pairs (device={config.device})")

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }

    if config.device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    if config.use_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 4-bit. pip install bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        print("Using 4-bit quantization")
    elif config.use_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 8-bit. pip install bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print("Using 8-bit quantization")
    else:
        if config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif config.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules
    )

    if config.use_4bit or config.use_8bit:
        optim = "paged_adamw_8bit"
    else:
        optim = "adamw_torch"

    use_fp16 = config.fp16 or config.use_4bit
    use_bf16 = config.bf16
    if config.device == "cpu":
        use_fp16 = False
        use_bf16 = False

    training_args = DPOConfig(
        output_dir="./dpo_results",
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=0.1,
        beta=config.beta,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        remove_unused_columns=False,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        dataloader_num_workers=0,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optim,
        warmup_steps=config.warmup_steps,
        save_strategy="steps",
        save_total_limit=2,
        no_cuda=(config.device == "cpu"),
    )

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=preference_dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    print("Starting DPO training...")
    trainer.train()

    os.makedirs(config.adapter_path, exist_ok=True)
    trainer.save_model(config.adapter_path)
    print(f"Adapter saved to {config.adapter_path}")

    return config.adapter_path

def train_with_dpo(
    traces: List[Dict[str, Any]],
    config: Optional[RLConfig] = None
) -> str:

    if config is None:
        config = RLConfig()

    if config.device == "mlx":
        preference_dataset = create_preference_pairs(
            traces,
            min_reward_gap=config.min_reward_gap
        )
        if preference_dataset is None or len(preference_dataset) == 0:
            print("No valid preference pairs. Cannot train.")
            return None

        if config.max_pairs and len(preference_dataset) > config.max_pairs:
            preference_dataset = preference_dataset.select(range(config.max_pairs))

        pairs = [
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in preference_dataset
        ]
        return _train_dpo_mlx(pairs, config)
    else:
        return _train_dpo_torch(traces, config)


def run_rl_training(
    tasks: List[Dict[str, Any]],
    agents: List[NPC],
    reward_fn: Callable[[Dict], float],
    config: Optional[RLConfig] = None,
    save_traces: bool = True
) -> str:

    if config is None:
        config = RLConfig()

    print(f"Collecting traces from {len(tasks)} tasks...")
    traces = collect_traces(
        tasks,
        agents,
        reward_fn,
        config
    )

    if save_traces:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        traces_file = f"rl_traces_{timestamp}.csv"
        df = pd.DataFrame(traces)
        df.to_csv(traces_file, index=False)
        print(f"Traces saved to {traces_file}")

    print("Training with DPO...")
    adapter_path = train_with_dpo(traces, config)

    return adapter_path

def load_rl_model(
    base_model_id: str,
    adapter_path: str,
    device: str = "cpu",
    use_4bit: bool = False,
    use_8bit: bool = False,
    merge_adapter: bool = True
):
    if device == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError("MLX backend requires mlx and mlx-lm. pip install mlx mlx-lm")
        mlx_model = _resolve_mlx_model(base_model_id)
        if adapter_path and os.path.exists(adapter_path):
            model, tokenizer = mlx_load(mlx_model, adapter_path=adapter_path)
        else:
            model, tokenizer = mlx_load(mlx_model)
        return model, tokenizer

    print(f"Loading base model: {base_model_id}")

    model_kwargs = {
        "trust_remote_code": True
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    if use_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 4-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif use_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 8-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter and not (use_4bit or use_8bit):
            model = model.merge_and_unload()

    return model, tokenizer


def _load_mlx_model_with_lora(config: RLConfig):
    """Load base MLX model, optionally resume from adapter, inject LoRA."""
    mlx_name = _resolve_mlx_model(config.base_model_name)
    if config.adapter_path and os.path.exists(os.path.join(config.adapter_path, "adapters.safetensors")):
        model, tokenizer = mlx_load(mlx_name, adapter_path=config.adapter_path)
    else:
        model, tokenizer = mlx_load(mlx_name)
    lora_cfg = {
        "rank": config.lora_r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "scale": config.lora_alpha / config.lora_r,
    }
    linear_to_lora_layers(model, _num_lora_layers(config.lora_r), lora_cfg)
    return model, tokenizer, mlx_name


def _save_adapter_config(path: str, mlx_name: str, config: RLConfig):
    os.makedirs(path, exist_ok=True)
    adapter_config = {
        "model": mlx_name,
        "fine_tune_type": "lora",
        "num_layers": _num_lora_layers(config.lora_r),
        "lora_parameters": {
            "rank": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "scale": config.lora_alpha / config.lora_r,
        },
    }
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)


def _tokenize_for_ppo(tokenizer, prompt, response, max_len=2048):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}"
    tokens = tokenizer.encode(text)
    if tokens[-1] != tokenizer.eos_token_id:
        tokens.append(tokenizer.eos_token_id)
    return tokens[:max_len]


def _compute_log_probs(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    batch_size, seq_len = targets.shape
    flat_targets = targets.reshape(-1)
    flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
    token_log_probs = flat_log_probs[mx.arange(flat_targets.size), flat_targets]
    token_log_probs = token_log_probs.reshape(batch_size, seq_len)
    steps = mx.arange(1, seq_len + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    token_log_probs = token_log_probs * mask
    per_example = token_log_probs.sum(axis=1)
    return per_example


def _batch_compute_log_probs(model, tokenizer, records, batch_size=16, max_len=1024):
    """Compute reference log-probs for all records in batches."""
    tokenized = []
    for rec in records:
        tokens = _tokenize_for_ppo(tokenizer, rec["instruction"], rec["response"], max_len=max_len)
        tokenized.append(tokens)

    indexed = sorted(enumerate(tokenized), key=lambda x: len(x[1]))
    ref_log_probs = [0.0] * len(records)

    for i in range(0, len(indexed), batch_size):
        batch = indexed[i:i + batch_size]
        lengths = [len(t) for _, t in batch]
        max_l = min(max(lengths), max_len)
        pad_to = 32
        padded_len = 1 + pad_to * ((max_l + pad_to - 1) // pad_to)
        padded_len = min(padded_len, max_len)

        batch_arr = np.zeros((len(batch), padded_len), np.int32)
        batch_lengths = []
        for j, (orig_idx, tokens) in enumerate(batch):
            trunc = min(len(tokens), max_len)
            batch_arr[j, :trunc] = tokens[:trunc]
            batch_lengths.append((0, trunc))

        batch_mx = mx.array(batch_arr)
        lengths_mx = mx.array(batch_lengths)
        lps = _compute_log_probs(model, batch_mx, lengths_mx)
        mx.eval(lps)
        lps_list = [float(x) for x in lps.tolist()]
        for j, (orig_idx, _) in enumerate(batch):
            ref_log_probs[orig_idx] = lps_list[j]

    return ref_log_probs


def _tokenize_grpo_torch(tokenizer, prompt, response, max_len=2048):
    text = f"<|im_start|>user\n{prompt}\n<|im_start|>assistant\n{response}"
    tokens = tokenizer.encode(text)
    if tokens[-1] != tokenizer.eos_token_id:
        tokens.append(tokenizer.eos_token_id)
    return tokens[:max_len]


def _load_torch_model_with_lora(config: RLConfig):
    """Load base torch model, optionally resume adapter, inject LoRA."""
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("train_with_grpo on torch requires transformers")
    if LoraConfig is None:
        raise ImportError("train_with_grpo on torch requires peft")

    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if config.device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    if config.use_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 4-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.use_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 8-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        if config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif config.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.adapter_path and os.path.isdir(config.adapter_path) and os.path.isfile(
        os.path.join(config.adapter_path, "adapter_config.json")
    ):
        print(f"Loading adapter: {config.adapter_path}")
        model = PeftModel.from_pretrained(model, config.adapter_path)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )
    try:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
    except Exception:
        pass

    return model, tokenizer


def _grpo_loss_torch(model, input_ids, advantages, attention_mask=None):
    """Compute advantage-weighted cross-entropy loss for GRPO."""
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        token_log_probs = token_log_probs * mask
        loss = -(token_log_probs * advantages.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
    else:
        loss = -(token_log_probs * advantages.unsqueeze(-1)).mean()
    return loss


class _GRPODataSet:
    def __init__(self, data):
        self._data = data
    def __len__(self):
        return len(self._data)
    def __getitem__(self, idx):
        return self._data[idx]


def _grpo_collate_torch(batch, pad_token_id):
    tokens_list, advantages = zip(*batch)
    lengths = [len(t) for t in tokens_list]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        input_ids[i, :lengths[i]] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[i, :lengths[i]] = 1
    return input_ids, attention_mask, torch.tensor(advantages, dtype=torch.float)


def _train_with_grpo_torch(groups: List[Dict[str, Any]], config: RLConfig) -> str:
    if torch is None:
        raise ImportError("train_with_grpo on torch requires torch")

    model, tokenizer = _load_torch_model_with_lora(config)
    os.makedirs(config.adapter_path, exist_ok=True)

    processed = []
    for group in groups:
        prompt = group["prompt"]
        responses = group["responses"]
        if len(responses) < 2:
            continue
        rewards = [r for _, r in responses]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-6
        for response, reward in responses:
            advantage = (reward - mean_r) / std_r
            tokens = _tokenize_grpo_torch(tokenizer, prompt, response, max_len=config.max_length)
            processed.append((tokens, float(advantage)))

    if len(processed) < 10:
        print("Not enough GRPO data.")
        return None

    from torch.utils.data import DataLoader

    dataset = _GRPODataSet(processed)
    loader = DataLoader(
        dataset,
        batch_size=config.group_size,
        shuffle=True,
        collate_fn=lambda batch: _grpo_collate_torch(batch, tokenizer.pad_token_id),
    )

    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    iters_per_epoch = max(1, len(processed) // config.group_size)
    total_steps = iters_per_epoch * config.num_train_epochs

    model.train()
    global_step = 0
    for epoch in range(config.num_train_epochs):
        for batch_idx, (input_ids, attention_mask, advantages) in enumerate(loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            advantages = advantages.to(device)

            loss = _grpo_loss_torch(model, input_ids, advantages, attention_mask)
            loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % config.logging_steps == 0:
                print(f"GRPO | epoch={epoch} step={global_step}/{total_steps} loss={loss.item():.4f}")

            if global_step % config.save_steps == 0:
                ckpt_dir = os.path.join(config.adapter_path, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    model.save_pretrained(config.adapter_path)
    tokenizer.save_pretrained(config.adapter_path)
    print(f"GRPO adapter saved to {config.adapter_path}")
    return config.adapter_path


def _train_with_grpo_mlx(
    groups: List[Dict[str, Any]],
    config: Optional[RLConfig] = None
) -> str:
    """Train with GRPO on MLX.

    groups = [{"prompt": str, "responses": [(response_str, reward), ...]}, ...]
    """
    if config is None:
        config = RLConfig()
    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx and mlx-lm")

    model, tokenizer, mlx_name = _load_mlx_model_with_lora(config)
    os.makedirs(config.adapter_path, exist_ok=True)

    processed = []
    for group in groups:
        prompt = group["prompt"]
        responses = group["responses"]
        if len(responses) < 2:
            continue
        rewards = [r for _, r in responses]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-6
        for response, reward in responses:
            advantage = (reward - mean_r) / std_r
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}"
            tokens = tokenizer.encode(text)
            if tokens[-1] != tokenizer.eos_token_id:
                tokens.append(tokenizer.eos_token_id)
            processed.append((tokens, 0, float(advantage)))

    if len(processed) < 10:
        print("Not enough GRPO data.")
        return None

    class _GRPOBatch:
        def __init__(self, data):
            self._data = data
        def __getitem__(self, idx):
            return self._data[idx]
        def __len__(self):
            return len(self._data)

    dataset = _GRPOBatch(processed)
    iters_per_epoch = max(1, len(processed) // config.group_size)
    total_iters = iters_per_epoch * config.num_train_epochs
    adapter_file = os.path.join(config.adapter_path, "adapters.safetensors")

    def grpo_loss(model, batch, lengths, advantages):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        batch_size, seq_len = targets.shape
        flat_targets = targets.reshape(-1)
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        token_log_probs = flat_log_probs[mx.arange(flat_targets.size), flat_targets]
        token_log_probs = token_log_probs.reshape(batch_size, seq_len)
        steps = mx.arange(1, seq_len + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        adv_per_token = advantages[:, mx.newaxis] * mask
        weighted_ce = -token_log_probs * adv_per_token
        loss = weighted_ce.sum() / (mask.sum() + 1e-8)
        return loss, mask.sum()

    def iterate_grpo_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, train=False, comm_group=None):
        idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
        if len(dataset) < batch_size:
            raise ValueError(f"Need >= {batch_size} examples, got {len(dataset)}")
        batch_idx = [
            idx[i:i + batch_size]
            for i in range(0, len(idx) - batch_size + 1, batch_size)
        ]
        if seed:
            np.random.seed(seed)
        while True:
            indices = np.random.permutation(len(batch_idx))
            for i in indices:
                batch_data = [dataset[j] for j in batch_idx[i]]
                tokens_list, offsets, advantages = zip(*batch_data)
                lengths = [len(t) for t in tokens_list]
                max_len = min(max(lengths), max_seq_length)
                pad_to = 32
                max_length_in_batch = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
                max_length_in_batch = min(max_length_in_batch, max_seq_length)
                batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)
                for j in range(batch_size):
                    trunc = min(lengths[j], max_seq_length)
                    batch_arr[j, :trunc] = tokens_list[j][:trunc]
                    lengths[j] = trunc
                yield mx.array(batch_arr), mx.array(list(zip(offsets, lengths))), mx.array(advantages)
            if not loop:
                break

    training_args = MLXTrainingArgs(
        batch_size=config.group_size,
        iters=total_iters,
        val_batches=0,
        steps_per_report=max(1, total_iters // 20),
        steps_per_eval=0,
        steps_per_save=max(1, total_iters // 5),
        max_seq_length=min(config.max_length, 1024),
        adapter_file=adapter_file,
        grad_checkpoint=True,
        grad_accumulation_steps=config.gradient_accumulation_steps,
    )
    optimizer = mlx_opt.AdamW(learning_rate=config.learning_rate)

    print(f"GRPO: {mlx_name}, {len(groups)} groups, {len(processed)} examples, {total_iters} iters, max_len={training_args.max_seq_length}")

    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        val_dataset=None,
        args=training_args,
        loss=grpo_loss,
        iterate_batches=iterate_grpo_batches,
    )

    _save_adapter_config(config.adapter_path, mlx_name, config)
    print(f"GRPO adapter saved to {config.adapter_path}")
    return config.adapter_path


def train_with_grpo(
    groups: List[Dict[str, Any]],
    config: Optional[RLConfig] = None
) -> str:
    """Train with GRPO on MLX or torch.

    groups = [{"prompt": str, "responses": [(response_str, reward), ...]}, ...]
    """
    if config is None:
        config = RLConfig()

    if config.device == "mlx":
        return _train_with_grpo_mlx(groups, config)
    else:
        return _train_with_grpo_torch(groups, config)


def train_with_ppo(
    records: List[Dict[str, Any]],
    config: Optional[RLConfig] = None
) -> str:
    """Train with PPO on MLX.

    records = [{"instruction": str, "response": str, "reward": float}, ...]
    """
    if config is None:
        config = RLConfig()
    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx and mlx-lm")

    os.makedirs(config.adapter_path, exist_ok=True)
    mlx_name = _resolve_mlx_model(config.base_model_name)

    print("Computing reference log-probs...")
    ref_model, tokenizer = mlx_load(mlx_name)
    ref_log_probs_list = _batch_compute_log_probs(ref_model, tokenizer, records, batch_size=1, max_len=1024)
    del ref_model
    mx.clear_cache()
    print(f"Reference log-probs computed for {len(records)} traces")

    policy_model, tokenizer = mlx_load(mlx_name)
    linear_to_lora_layers(
        policy_model,
        _num_lora_layers(config.lora_r),
        {"rank": config.lora_r, "alpha": config.lora_alpha, "dropout": config.lora_dropout, "scale": config.lora_alpha / config.lora_r},
    )

    rewards = [r["reward"] for r in records]
    baseline = sum(rewards) / len(rewards)

    processed = []
    for rec, ref_lp in zip(records, ref_log_probs_list):
        tokens = _tokenize_for_ppo(tokenizer, rec["instruction"], rec["response"])
        advantage = rec["reward"] - baseline
        processed.append((tokens, 0, float(advantage), float(ref_lp)))

    class _PPOBatch:
        def __init__(self, data):
            self._data = data
        def __getitem__(self, idx):
            return self._data[idx]
        def __len__(self):
            return len(self._data)

    dataset = _PPOBatch(processed)

    def ppo_loss(model, batch, lengths, advantages, ref_log_probs):
        policy_log_probs = _compute_log_probs(model, batch, lengths)
        ratio = mx.exp(policy_log_probs - ref_log_probs)
        clipped_ratio = mx.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        policy_loss = -mx.minimum(surrogate1, surrogate2).mean()
        kl = (policy_log_probs - ref_log_probs).mean()
        loss = policy_loss + config.beta * kl
        return loss, mx.array(len(processed))

    def iterate_ppo_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, train=False, comm_group=None):
        idx = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
        if len(dataset) < batch_size:
            raise ValueError(f"Need >= {batch_size} examples, got {len(dataset)}")
        batch_idx = [
            idx[i:i + batch_size]
            for i in range(0, len(idx) - batch_size + 1, batch_size)
        ]
        if seed:
            np.random.seed(seed)
        while True:
            indices = np.random.permutation(len(batch_idx))
            for i in indices:
                batch_data = [dataset[j] for j in batch_idx[i]]
                tokens_list, offsets, advantages, ref_lps = zip(*batch_data)
                lengths = [len(t) for t in tokens_list]
                max_len = min(max(lengths), max_seq_length)
                pad_to = 32
                max_length_in_batch = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
                max_length_in_batch = min(max_length_in_batch, max_seq_length)
                batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)
                for j in range(batch_size):
                    trunc = min(lengths[j], max_seq_length)
                    batch_arr[j, :trunc] = tokens_list[j][:trunc]
                    lengths[j] = trunc
                yield (
                    mx.array(batch_arr),
                    mx.array(list(zip(offsets, lengths))),
                    mx.array(advantages),
                    mx.array(ref_lps),
                )
            if not loop:
                break

    iters_per_epoch = max(1, len(processed) // config.group_size)
    total_iters = iters_per_epoch * config.num_train_epochs
    adapter_file = os.path.join(config.adapter_path, "adapters.safetensors")

    training_args = MLXTrainingArgs(
        batch_size=config.group_size,
        iters=total_iters,
        val_batches=0,
        steps_per_report=max(1, total_iters // 20),
        steps_per_eval=0,
        steps_per_save=max(1, total_iters // 5),
        max_seq_length=min(config.max_length, 1024),
        adapter_file=adapter_file,
        grad_checkpoint=True,
        grad_accumulation_steps=config.gradient_accumulation_steps,
    )
    optimizer = mlx_opt.AdamW(learning_rate=config.learning_rate)

    print(f"PPO: {mlx_name}, {len(records)} traces, {total_iters} iters, beta={config.beta}, clip={config.clip_eps}, max_len={training_args.max_seq_length}")

    mlx_train(
        model=policy_model,
        optimizer=optimizer,
        train_dataset=dataset,
        val_dataset=None,
        args=training_args,
        loss=ppo_loss,
        iterate_batches=iterate_ppo_batches,
    )

    _save_adapter_config(config.adapter_path, mlx_name, config)
    print(f"PPO adapter saved to {config.adapter_path}")
    return config.adapter_path
