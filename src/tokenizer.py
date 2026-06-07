from transformers import AutoTokenizer
from datasets import DatasetDict


class PoquadTokenizer:
    def __init__(self, model_checkpoint: str = "allegro/plt5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_poquad_raw_json(self, examples):
        inputs = []
        targets = []

        for paragraphs_list in examples["paragraphs"]:
            for paragraph in paragraphs_list:
                context = paragraph["context"].strip()

                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    is_impossible = qa.get("is_impossible", False)

                    input_str = f"pytanie: {question} kontekst: {context}"
                    inputs.append(input_str)

                    if not is_impossible and "answers" in qa and len(qa["answers"]) > 0:
                        gen_ans = qa["answers"][0].get("generative_answer", "").strip()
                        targets.append(gen_ans if gen_ans != "" else "brak_odpowiedzi")
                    else:
                        targets.append("brak_odpowiedzi")

        model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)
        labels = self.tokenizer(text_target=targets, max_length=64, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def transform(self, raw_datasets: DatasetDict) -> DatasetDict:
        return raw_datasets.map(
            self.preprocess_poquad_raw_json,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )