import os, torch

def save_checkpoint(model, tokenizer, optimizer, out_dir: str, step: int, is_peft: bool=False):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"step-{step}"
    save_dir = os.path.join(out_dir, tag)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    torch.save({"optimizer": optimizer.state_dict()}, os.path.join(save_dir, "optim.pt"))
    return save_dir

def latest_checkpoint(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir): return None
    steps = []
    for n in os.listdir(ckpt_dir):
        if n.startswith("step-"):
            try: steps.append((int(n.split("-")[-1]), n))
            except: pass
    if not steps: return None
    steps.sort()
    return os.path.join(ckpt_dir, steps[-1][1])
