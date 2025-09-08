from dataclasses import dataclass
from typing import List, Iterable, Optional
import re, random
from datasets import load_dataset

NUM_RE = re.compile(r"-?\d+(\.\d+)?")

def extract_numeric(text: str) -> Optional[str]:
    """Extract final numeric answer. Prefer '#### 42', else last number."""
    if text is None: return None
    m = re.search(r"####\s*([\-]?\d+(?:\.\d+)?)", text)
    if m: return m.group(1)
    m3 = re.findall(r"[\-]?\d+(?:\.\d+)?", text)
    return m3[-1] if m3 else None

@dataclass
class GSM8KExample:
    question: str
    answer: str

def format_prompt(q: str, style: str="cot") -> str:
    if style == "short":
        return f"Question: {q}\nAnswer:"
    return (        "You are a careful math tutor. Solve the problem step by step and give the final numeric answer.\n"
        f"Problem: {q}\n"
        "Reasoning (concise):"
    )

class GSM8KEnvironment:
    def __init__(self, split: str="train", dataset_config="main", n_samples: int=1000, prompt_style: str="cot", seed: int=42):
        self.ds = load_dataset("gsm8k", dataset_config, split=split)
        if n_samples is not None and n_samples > 0:
            self.ds = self.ds.select(range(min(n_samples, len(self.ds))))
        self.prompt_style = prompt_style
        random.seed(seed)

    def iter_batches(self, batch_size: int) -> Iterable[list[GSM8KExample]]:
        batch = []
        for item in self.ds:
            batch.append(GSM8KExample(question=item["question"], answer=item["answer"]))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch: yield batch

    @staticmethod
    def reward(generated: str, ground_truth: str) -> float:
        pred = extract_numeric(generated); gold = extract_numeric(ground_truth)
        if pred is None or gold is None: return 0.0
        return 1.0 if str(pred) == str(gold) else 0.0
