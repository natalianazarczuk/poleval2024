from collections.abc import Sequence
import re
import unicodedata

NO_ANSWER = "brak_odpowiedzi"


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def normalize_answer(text: str) -> str:
    text = _strip_accents(text)
    text = text.replace("_", " ")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


def is_no_answer(text: str) -> bool:
    normalized = normalize_answer(text)
    compact = normalized.replace(" ", "")
    return normalized in {"", "brak", "brak odpowiedzi"} or compact == "brakodpowiedzi"


def normalized_levenshtein_similarity(prediction: str, reference: str) -> float:
    prediction = normalize_answer(prediction)
    reference = normalize_answer(reference)
    longest = max(len(prediction), len(reference))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein_distance(prediction, reference) / longest


def levenshtein_distance(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first
    previous_row = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current_row = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            insertion = current_row[second_index - 1] + 1
            deletion = previous_row[second_index] + 1
            substitution = previous_row[second_index - 1] + (
                    first_character != second_character
            )
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def binary_f1(predictions: Sequence[bool], references: Sequence[bool]) -> float:
    pairs = list(zip(predictions, references))
    true_positives = sum(predicted and expected for predicted, expected in pairs)
    false_positives = sum(predicted and not expected for predicted, expected in pairs)
    false_negatives = sum(not predicted and expected for predicted, expected in pairs)

    denominator = 2 * true_positives + false_positives + false_negatives
    return 0.0 if denominator == 0 else 2 * true_positives / denominator


def compute_poleval_metrics(
        predictions: Sequence[str], references: Sequence[str]
) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not references:
        raise ValueError("At least one prediction is required.")

    answerable_similarities = [
        normalized_levenshtein_similarity(
            "" if is_no_answer(prediction) else prediction,
            reference,
        )
        for prediction, reference in zip(predictions, references)
        if not is_no_answer(reference)
    ]
    levenshtein = (
        sum(answerable_similarities) / len(answerable_similarities)
        if answerable_similarities
        else 0.0
    )

    predicted_abstentions = [is_no_answer(value) for value in predictions]
    expected_abstentions = [is_no_answer(value) for value in references]
    abstention_f1 = binary_f1(predicted_abstentions, expected_abstentions)

    exact_matches = [
        normalize_answer(prediction) == normalize_answer(reference)
        for prediction, reference in zip(predictions, references)
        if not is_no_answer(reference)
    ]
    exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0

    return {
        "normalized_levenshtein": 100.0 * levenshtein,
        "answerability_f1": 100.0 * abstention_f1,
        "score": 50.0 * (levenshtein + abstention_f1),
        "exact_match": 100.0 * exact_match,
    }
