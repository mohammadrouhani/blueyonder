import argparse, os, random, json
import torch, yaml
from src.models.moe_policy import MoEPolicy, PolicyConfig
from src.data.gsm8k_env import GSM8KEnvironment, format_prompt
from src.rl.ppo import PPOTrainer, PPOConfig
from src.utils.logger import CSVLogger
from src.utils.checkpoint import save_checkpoint

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=str, default="configs/default.yaml"); return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))

    pol_cfg = PolicyConfig(
        model_name=cfg["model_name"],
        trust_remote_code=cfg.get("trust_remote_code", True),
        load_in_4bit=cfg.get("load_in_4bit", True),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        use_lora=cfg.get("use_lora", True),
        lora_r=cfg.get("lora",{}).get("r", 8),
        lora_alpha=cfg.get("lora",{}).get("alpha", 16),
        lora_dropout=cfg.get("lora",{}).get("dropout", 0.05),
        lora_target_modules=tuple(cfg.get("lora",{}).get("target_modules", [])),
        max_new_tokens=cfg.get("max_new_tokens", 128),
        temperature=cfg.get("temperature", 0.0),
        top_p=cfg.get("top_p", 1.0),
        device=cfg.get("device", "auto"),
    )
    policy = MoEPolicy(pol_cfg)
    ref_cfg = PolicyConfig(**{**pol_cfg.__dict__, "use_lora": False})
    ref_policy = MoEPolicy(ref_cfg)
    for p in ref_policy.model.parameters(): p.requires_grad = False
    ref_policy.model.eval()

    ppo_cfg = PPOConfig(**cfg["ppo"], fp16=cfg.get("fp16", True))
    trainer = PPOTrainer(policy, ref_policy, ppo_cfg)

    env = GSM8KEnvironment(split="train", dataset_config=cfg["dataset_config"], n_samples=cfg["train_samples"], prompt_style=cfg["prompt_style"], seed=cfg["seed"])
    log = CSVLogger(cfg["log_dir"])
    global_step = 0

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    for batch in env.iter_batches(ppo_cfg.batch_size):
        prompts = [format_prompt(ex.question, cfg["prompt_style"]) for ex in batch]
        with torch.no_grad():
            gens = policy.generate(prompts)

        responses = []
        for p, g in zip(prompts, gens):
            responses.append(g[len(p):].strip() if g.startswith(p) else g.split("Answer:")[-1].strip())

        rewards = torch.tensor([env.reward(r, ex.answer) for r, ex in zip(responses, batch)], dtype=torch.float32)

        input_ids, attention_mask, labels, label_mask = trainer._prepare_inputs(policy.tokenizer, prompts, responses)
        input_ids = input_ids.to(policy.model.device); attention_mask = attention_mask.to(policy.model.device)
        with torch.no_grad():
            logprobs, _ = policy.forward_logprobs(input_ids, attention_mask)
            token_logps = logprobs.gather(-1, labels.to(policy.model.device).unsqueeze(-1).clamp_min(0)).squeeze(-1)
            mask = label_mask.to(policy.model.device).float()
            old_lp = (token_logps * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        for epoch in range(ppo_cfg.update_epochs):
            metrics = trainer.step(policy.tokenizer, prompts, responses, rewards, old_lp, global_step)

        global_step += 1

        if global_step % 10 == 0:
            r_mean = rewards.mean().item(); r_std = rewards.std().item() if len(rewards)>1 else 0.0
            log.log({"step": global_step, "split":"train","reward_mean": r_mean, "reward_std": r_std, **metrics})

        if global_step % cfg["save_steps"] == 0:
            save_checkpoint(policy.model, policy.tokenizer, trainer.optimizer, cfg["ckpt_dir"], global_step, is_peft=cfg.get("use_lora", True))

        if global_step >= cfg["ppo"]["max_steps"]:
            break

    save_checkpoint(policy.model, policy.tokenizer, trainer.optimizer, cfg["ckpt_dir"], global_step, is_peft=cfg.get("use_lora", True))

if __name__ == "__main__": main()
