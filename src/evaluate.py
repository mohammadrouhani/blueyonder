import argparse, json, random, yaml, torch
from src.models.moe_policy import MoEPolicy, PolicyConfig
from src.data.gsm8k_env import GSM8KEnvironment, format_prompt
from src.utils.checkpoint import latest_checkpoint

def set_seed(seed): random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))

    policy = MoEPolicy(PolicyConfig(
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
    ))

    ckpt = args.ckpt or latest_checkpoint(cfg["ckpt_dir"])
    if ckpt:
        print(f"Loading adapter/checkpoint from: {ckpt}")
        try:
            policy.model.load_adapter(ckpt)
        except Exception as e:
            print(f"Warning: couldn't load adapter automatically: {e}")

    env = GSM8KEnvironment(split="test", dataset_config=cfg["dataset_config"], n_samples=cfg["eval_samples"], prompt_style=cfg["prompt_style"], seed=cfg["seed"])

    correct, total = 0, 0
    for batch in env.iter_batches(batch_size=4):
        prompts = [format_prompt(ex.question, cfg["prompt_style"]) for ex in batch]
        gens = policy.generate(prompts)
        for p, g, ex in zip(prompts, gens, batch):
            resp = g[len(p):].strip() if g.startswith(p) else g
            r = env.reward(resp, ex.answer)
            correct += int(r > 0.5); total += 1

    acc = correct / max(1,total)
    print(json.dumps({"accuracy": acc, "total": total}, indent=2))

if __name__ == "__main__": main()
