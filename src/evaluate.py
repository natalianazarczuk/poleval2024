import argparse
import json
from pathlib import Path

from inference import PolEvalInferencePipeline
from metrics import (
    binary_f1,
    compute_poleval_metrics,
    is_no_answer,
    normalize_answer,
    normalized_levenshtein_similarity,
)
from tokenizer import sanitize_text

NO_ANSWER = "brak_odpowiedzi"


def extract_reference_answer(qa: dict) -> str:
    if qa.get("is_impossible", False):
        return NO_ANSWER

    for answer in qa.get("answers", []):
        for key in ("generative_answer", "text"):
            candidate = sanitize_text(answer.get(key, ""))
            if candidate:
                return candidate
    return NO_ANSWER


def iter_dev_samples(dev_data: dict):
    for article in dev_data.get("data", []):
        article_id = article.get("id", "")
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = sanitize_text(paragraph.get("context", ""))
            for question_index, qa in enumerate(paragraph.get("qas", [])):
                question = sanitize_text(qa.get("question", ""))
                reference = extract_reference_answer(qa)
                sample_id = f"{article_id}_{paragraph_index}_{question_index}"
                yield sample_id, question, context, reference


def compute_breakdown(predictions: list[str], references: list[str]) -> dict[str, float | int]:
    expected_abstentions = [is_no_answer(reference) for reference in references]
    predicted_abstentions = [is_no_answer(prediction) for prediction in predictions]

    answerable_pairs = [
        (prediction, reference)
        for prediction, reference in zip(predictions, references)
        if not is_no_answer(reference)
    ]

    if answerable_pairs:
        answerable_levenshtein = sum(
            normalized_levenshtein_similarity(prediction, reference)
            for prediction, reference in answerable_pairs
        ) / len(answerable_pairs)
        answerable_exact_match = sum(
            normalize_answer(prediction) == normalize_answer(reference)
            for prediction, reference in answerable_pairs
        ) / len(answerable_pairs)
    else:
        answerable_levenshtein = 0.0
        answerable_exact_match = 0.0

    true_positives = sum(
        predicted and expected
        for predicted, expected in zip(predicted_abstentions, expected_abstentions)
    )
    false_positives = sum(
        predicted and not expected
        for predicted, expected in zip(predicted_abstentions, expected_abstentions)
    )
    false_negatives = sum(
        not predicted and expected
        for predicted, expected in zip(predicted_abstentions, expected_abstentions)
    )

    abstention_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    abstention_recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return {
        "samples_total": len(references),
        "samples_answerable": len(answerable_pairs),
        "samples_unanswerable": sum(expected_abstentions),
        "answerable_normalized_levenshtein": 100.0 * answerable_levenshtein,
        "answerable_exact_match": 100.0 * answerable_exact_match,
        "abstention_precision": 100.0 * abstention_precision,
        "abstention_recall": 100.0 * abstention_recall,
        "abstention_f1": 100.0 * binary_f1(predicted_abstentions, expected_abstentions),
    }


def run_evaluation(
    model_dir: str,
    dev_json: str,
    limit: int | None = None,
    predictions_path: str | None = None,
):
    pipeline = PolEvalInferencePipeline(model_dir)

    with open(dev_json, "r", encoding="utf-8") as handle:
        dev_data = json.load(handle)

    predictions: list[str] = []
    references: list[str] = []
    detailed_rows: list[dict[str, str]] = []

    for sample_index, (sample_id, question, context, reference) in enumerate(
        iter_dev_samples(dev_data),
        start=1,
    ):
        if limit is not None and sample_index > limit:
            break

        prediction = pipeline.predict(question, context)
        predictions.append(prediction)
        references.append(reference)

        if predictions_path:
            detailed_rows.append(
                {
                    "id": sample_id,
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                }
            )

    if not predictions:
        raise ValueError("No samples were evaluated. Check your dev.json structure.")

    overall_metrics = compute_poleval_metrics(predictions, references)
    breakdown_metrics = compute_breakdown(predictions, references)

    result = {
        "model_dir": model_dir,
        "dev_json": dev_json,
        "limit": limit,
        "overall": overall_metrics,
        "breakdown": breakdown_metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if predictions_path:
        output_path = Path(predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in detailed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained PoQuAD model on dev.json")
    parser.add_argument(
        "--model-dir",
        default="outputs/plt5-task1",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--dev-json",
        default="data/dev.json",
        help="Path to PoQuAD dev JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N samples (for quick checks)",
    )
    parser.add_argument(
        "--predictions-path",
        default=None,
        help="Optional JSONL output path with per-sample predictions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        dev_json=args.dev_json,
        limit=args.limit,
        predictions_path=args.predictions_path,
    )
