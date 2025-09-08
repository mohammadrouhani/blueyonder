from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

@dataclass
class PolicyConfig:
    model_name: str
    trust_remote_code: bool = True
    load_in_4bit: bool = True
    fp16: bool = True
    bf16: bool = False
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    device: str = "auto"

class MoEPolicy:
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32)

        quant_args = {}
        if cfg.load_in_4bit:
            quant_args.update(dict(load_in_4bit=True, bnb_4bit_compute_dtype=dtype))

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
            device_map="auto" if cfg.device == "auto" else None,
            **quant_args,
        )

        if hasattr(self.model.config, "add_router_probs"):
            self.model.config.add_router_probs = True

        if cfg.use_lora:
            assert PEFT_AVAILABLE, "peft is required for LoRA; install with `pip install peft`"
            if cfg.load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            peft_cfg = LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules), bias="none", task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, peft_cfg)
        self.model.train()

        self.gen_cfg = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=(cfg.temperature > 0),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def parameters(self):
        return self.model.parameters()

    def generate(self, prompts: list[str]) -> list[str]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, generation_config=self.gen_cfg)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def forward_logprobs(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs, getattr(outputs, "router_logits", None)

    def expert_stats(self, router_logits) -> Optional[Dict[str, Any]]:
        if router_logits is None: return None
        stats = []
        for layer_logits in router_logits:
            expert_ids = layer_logits.argmax(dim=-1)
            stats.append(expert_ids.float().mean().item())
        return {"mean_expert_index_per_layer": stats}
