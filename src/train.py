import os

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from tokenizer import PoquadTokenizer

MODEL_NAME = "allegro/plt5-base"


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": os.path.join(project_dir, "data", "train.json"),
            "validation": os.path.join(project_dir, "data", "dev.json"),
        },
        field="data",
    )

    poquad_tokenizer = PoquadTokenizer(MODEL_NAME)
    tokenized_datasets = poquad_tokenizer.transform(raw_datasets)
    tokenizer = poquad_tokenizer.tokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(project_dir, "outputs", "plt5-task1"),
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
