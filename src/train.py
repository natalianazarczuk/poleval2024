import argparse
import os
import json

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from data_loader import PoquadDataLoader
from tokenizer import PoquadTokenizer
from metrics import compute_poleval_metrics

MODEL_NAME = "allegro/plt5-small"
MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 64


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PoQuAD model")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a tiny fast training pass to smoke-test the pipeline",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit number of training samples",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit number of validation samples",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training length with a fixed number of optimizer steps",
    )
    return parser.parse_args()


def _load_seq2seq_model_with_sanitized_config(model_name: str):
    # Some hub configs store float-like values as ints (e.g. initializer_factor=1),
    # which strict validation in newer huggingface_hub rejects.
    config_path = hf_hub_download(repo_id=model_name, filename="config.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)

    if isinstance(config_dict.get("initializer_factor"), int):
        config_dict["initializer_factor"] = float(config_dict["initializer_factor"])

    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError(f"Missing 'model_type' in config for model: {model_name}")

    config = AutoConfig.for_model(
        model_type,
        **{key: value for key, value in config_dict.items() if key != "model_type"},
    )
    config.tie_word_embeddings = False
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)


def make_compute_metrics(tokenizer):
    def _prepare_token_ids(values):
        values = np.asarray(values)

        if values.ndim == 3:
            values = values.argmax(axis=-1)

        values = np.where(np.isfinite(values), values, tokenizer.pad_token_id)
        values = np.where(values < 0, tokenizer.pad_token_id, values)
        return values.astype(np.int64, copy=False)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = _prepare_token_ids(predictions)
        labels = _prepare_token_ids(labels)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        return compute_poleval_metrics(decoded_preds, decoded_labels)

    return compute_metrics


def _subset_dataset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def main():
    args = parse_args()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_loader = PoquadDataLoader(os.path.join(project_dir, "data"))
    raw_datasets = data_loader.load_raw_datasets()

    poquad_tokenizer = PoquadTokenizer(
        MODEL_NAME,
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
    )
    tokenized_datasets = poquad_tokenizer.transform(raw_datasets)
    tokenizer = poquad_tokenizer.tokenizer

    if args.debug:
        args.max_train_samples = args.max_train_samples or 256
        args.max_eval_samples = args.max_eval_samples or 64
        args.max_steps = args.max_steps or 5

    train_dataset = _subset_dataset(tokenized_datasets["train"], args.max_train_samples)
    eval_dataset = _subset_dataset(tokenized_datasets["validation"], args.max_eval_samples)

    model = _load_seq2seq_model_with_sanitized_config(MODEL_NAME)
    # Keep training/eval config separate from inference-time generation config.
    model.generation_config.max_new_tokens = None
    model.generation_config.min_new_tokens = None
    model.generation_config.max_length = None
    model.generation_config.min_length = None

    use_fixed_steps = args.max_steps is not None
    warmup_steps = 0 if use_fixed_steps else max(1, int(len(train_dataset) * 0.06 / 8))
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(
            project_dir,
            "outputs",
            "plt5-task1-debug" if use_fixed_steps else "plt5-task1",
        ),
        num_train_epochs=1,
        max_steps=args.max_steps if use_fixed_steps else -1,
        learning_rate=1e-5,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        eval_strategy="steps" if use_fixed_steps else "epoch",
        eval_steps=1 if use_fixed_steps else None,
        save_strategy="no" if use_fixed_steps else "epoch",
        save_total_limit=2,
        load_best_model_at_end=not use_fixed_steps,
        metric_for_best_model="eval_score",
        greater_is_better=True,
        report_to="none",
        fp16=False,
        dataloader_pin_memory=False,
        # Use a single worker on macOS to avoid torch shared-memory issues.
        dataloader_num_workers=0,
        predict_with_generate=True,
        generation_num_beams=1,
        logging_steps=1 if use_fixed_steps else 50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
