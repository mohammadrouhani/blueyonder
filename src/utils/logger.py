import csv, os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class CSVLogger:
    out_dir: str
    filename: str = "metrics.csv"
    fieldnames: list = field(default_factory=lambda: [
        "step","split","reward_mean","reward_std","kl","entropy","loss_policy","lr","acc","notes"
    ])
    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, self.filename)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Dict[str, Any]):
        row = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
