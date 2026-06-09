import os
import json
import random
import re
from datasets import Dataset, DatasetDict

from tokenizer import (
    NO_ANSWER,
    answer_supported_by_context,
    canonicalize_answer_text,
    is_noisy_answer_text,
    sanitize_text,
)


class PoquadDataLoader:
    def __init__(self, data_dir: str):
        self.train_path = os.path.join(data_dir, "train.json")
        self.dev_path = os.path.join(data_dir, "dev.json")
        self._validate_paths()

    def _validate_paths(self):
        for path in [self.train_path, self.dev_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")

    @staticmethod
    def _extract_answer(qa: dict) -> str:
        if qa.get("is_impossible", False):
            return NO_ANSWER
        for answer in qa.get("answers", []):
            for key in ("generative_answer", "text"):
                candidate = canonicalize_answer_text(answer.get(key, ""))
                if candidate == NO_ANSWER:
                    continue
                if is_noisy_answer_text(candidate):
                    continue
                return candidate
        return NO_ANSWER

    @staticmethod
    def _normalize_key_text(text: str) -> str:
        lowered = sanitize_text(text).lower()
        return re.sub(r"\s+", " ", lowered).strip()

    def _flatten_split(self, path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        rows: list[dict] = []
        for article in payload.get("data", []):
            for paragraph in article.get("paragraphs", []):
                context = sanitize_text(paragraph.get("context", ""))
                if not context:
                    continue
                for qa in paragraph.get("qas", []):
                    question = sanitize_text(qa.get("question", ""))
                    if not question:
                        continue
                    answer = self._extract_answer(qa)
                    rows.append(
                        {
                            "question": question,
                            "context": context,
                            "answer": answer,
                        }
                    )
        return rows

    @staticmethod
    def _filter_examples(
        examples: list[dict],
        *,
        drop_noisy_answers: bool,
        require_answer_in_context: bool,
    ) -> list[dict]:
        filtered: list[dict] = []
        for example in examples:
            answer = canonicalize_answer_text(example["answer"])
            if answer != NO_ANSWER and drop_noisy_answers and is_noisy_answer_text(answer):
                continue
            if (
                answer != NO_ANSWER
                and require_answer_in_context
                and not answer_supported_by_context(answer, example["context"])
            ):
                continue

            filtered.append(
                {
                    "question": sanitize_text(example["question"]),
                    "context": sanitize_text(example["context"]),
                    "answer": answer,
                }
            )
        return filtered

    def _deduplicate_examples(self, examples: list[dict]) -> list[dict]:
        unique_examples: list[dict] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for example in examples:
            key = (
                self._normalize_key_text(example["question"]),
                self._normalize_key_text(example["context"]),
                self._normalize_key_text(example["answer"]),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_examples.append(example)
        return unique_examples

    @staticmethod
    def _rebalance_unanswerable_examples(
        examples: list[dict],
        *,
        target_unanswerable_ratio: float | None,
        seed: int,
    ) -> list[dict]:
        if target_unanswerable_ratio is None:
            return examples
        if not 0.0 < target_unanswerable_ratio < 1.0:
            raise ValueError("target_unanswerable_ratio must be in the (0, 1) range.")

        answerable = [example for example in examples if example["answer"] != NO_ANSWER]
        unanswerable = [example for example in examples if example["answer"] == NO_ANSWER]
        if not answerable or not unanswerable:
            return examples

        target_unanswerable_count = round(
            (target_unanswerable_ratio / (1.0 - target_unanswerable_ratio)) * len(answerable)
        )
        if target_unanswerable_count >= len(unanswerable):
            return examples
        if target_unanswerable_count <= 0:
            return answerable

        rng = random.Random(seed)
        rng.shuffle(unanswerable)
        balanced = answerable + unanswerable[:target_unanswerable_count]
        rng.shuffle(balanced)
        return balanced

    @staticmethod
    def _to_dataset(examples: list[dict]) -> Dataset:
        return Dataset.from_list(examples)

    def load_raw_datasets(
        self,
        *,
        deduplicate_train: bool = True,
        drop_noisy_answers: bool = True,
        require_answer_in_context: bool = True,
        target_unanswerable_ratio: float | None = 0.1836,
        sampling_seed: int = 42,
    ) -> DatasetDict:
        train_examples = self._flatten_split(self.train_path)
        validation_examples = self._flatten_split(self.dev_path)

        train_examples = self._filter_examples(
            train_examples,
            drop_noisy_answers=drop_noisy_answers,
            require_answer_in_context=require_answer_in_context,
        )
        validation_examples = self._filter_examples(
            validation_examples,
            drop_noisy_answers=False,
            require_answer_in_context=False,
        )

        if deduplicate_train:
            train_examples = self._deduplicate_examples(train_examples)
        train_examples = self._rebalance_unanswerable_examples(
            train_examples,
            target_unanswerable_ratio=target_unanswerable_ratio,
            seed=sampling_seed,
        )

        return DatasetDict(
            {
                "train": self._to_dataset(train_examples),
                "validation": self._to_dataset(validation_examples),
            }
        )
