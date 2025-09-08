from dataclasses import dataclass
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

@dataclass
class PPOConfig:
    batch_size: int = 8
    mini_batch_size: int = 2
    update_epochs: int = 2
    clip_range: float = 0.2
    kl_coef: float = 0.02
    entropy_coef: float = 0.01
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.0
    grad_accum_steps: int = 8
    fp16: bool = True

class PPOTrainer:
    def __init__(self, policy, ref_policy, cfg: PPOConfig):
        self.policy = policy
        self.ref_policy = ref_policy
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy.model.parameters()),
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and torch.cuda.is_available())

    def _prepare_inputs(self, tokenizer, prompts: List[str], responses: List[str]):
        inputs, labels = [], []
        for p, r in zip(prompts, responses):
            pr = tokenizer(p, add_special_tokens=False)["input_ids"]
            rr = tokenizer(r, add_special_tokens=False)["input_ids"]
            if len(pr) > 640: pr = pr[-640:]
            if len(rr) > 256: rr = rr[:256]
            ids = pr + rr
            lbl = [-100]*len(pr) + rr
            inputs.append(torch.tensor(ids, dtype=torch.long))
            labels.append(torch.tensor(lbl, dtype=torch.long))
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        label_mask = (labels != -100).long()
        return input_ids, attention_mask, labels, label_mask

    def step(self, tokenizer, prompts: List[str], responses: List[str], rewards: torch.Tensor, old_logps: torch.Tensor, step_grad: int):
        input_ids, attention_mask, labels, label_mask = self._prepare_inputs(tokenizer, prompts, responses)
        device = next(self.policy.model.parameters()).device
        input_ids = input_ids.to(device); attention_mask = attention_mask.to(device)
        labels = labels.to(device); label_mask = label_mask.to(device).float()

        with torch.cuda.amp.autocast(enabled=self.cfg.fp16 and torch.cuda.is_available()):
            logprobs, _ = self.policy.forward_logprobs(input_ids, attention_mask)
            with torch.no_grad():
                logprobs_ref, _ = self.ref_policy.forward_logprobs(input_ids, attention_mask)

            token_logps = logprobs.gather(-1, labels.unsqueeze(-1).clamp_min(0)).squeeze(-1)
            token_logps_ref = logprobs_ref.gather(-1, labels.unsqueeze(-1).clamp_min(0)).squeeze(-1)

            lp = (token_logps * label_mask).sum(dim=1) / (label_mask.sum(dim=1) + 1e-6)
            lp_ref = (token_logps_ref * label_mask).sum(dim=1) / (label_mask.sum(dim=1) + 1e-6)

            ratios = (lp - old_logps.to(device)).exp()
            advantages = rewards.to(device)

            clipped = torch.clamp(ratios, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * advantages
            obj = torch.min(ratios * advantages, clipped).mean()

            kl = (lp - lp_ref).mean()
            entropy = - (token_logps.exp() * token_logps * label_mask).sum() / (label_mask.sum() + 1e-6)

            loss = -obj + self.cfg.kl_coef * kl - self.cfg.entropy_coef * entropy

        self.scaler.scale(loss / self.cfg.grad_accum_steps).backward()
        if (step_grad + 1) % self.cfg.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad()

        return {
            "loss_policy": float(loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
