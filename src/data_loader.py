import os
from datasets import load_dataset, DatasetDict

class PoquadDataLoader:
    def __init__(self, data_dir: str):
        self.train_path = os.path.join(data_dir, "train.json")
        self.dev_path = os.path.join(data_dir, "dev.json")
        self._validate_paths()

    def _validate_paths(self):
        for path in [self.train_path, self.dev_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")

    def load_raw_datasets(self) -> DatasetDict:
        data_files = {
            "train": self.train_path,
            "validation": self.dev_path
        }
        return load_dataset("json", data_files=data_files, field="data")