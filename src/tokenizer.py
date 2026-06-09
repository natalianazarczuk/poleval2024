import re
import unicodedata
from html import unescape

from transformers import T5TokenizerFast
from datasets import DatasetDict

NO_ANSWER = "brak_odpowiedzi"
_NO_ANSWER_ALIASES = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "brak",
    "brak odpowiedzi",
    "brak_odpowiedzi",
    "nie wiem",
}
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^(?:odpowiedz|odpowiedź|answer|ans|odp|a)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
_LIST_MARKER_PATTERN = re.compile(r"^\d+[.)]\s*")
_QUOTE_TRANSLATION_TABLE = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "‚": "'",
        "—": "-",
        "–": "-",
        "−": "-",
        "…": "...",
    }
)


def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    text = unescape(str(text))
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_QUOTE_TRANSLATION_TABLE)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
    text = re.sub(r"\[\s*\d+\s*]", " ", text)
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)
    text = re.sub(r"\s+([,.:;!?%])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s*-\s*", " - ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'")
    return text


def build_prompt(question: str, context: str) -> str:
    return f"pytanie: {sanitize_text(question)} kontekst: {sanitize_text(context)}"


def _canonical_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", sanitize_text(text).replace("_", " ").strip().lower())


def is_no_answer_text(text: str) -> bool:
    canonical = _canonical_for_match(text)
    compact = canonical.replace(" ", "")
    return canonical in _NO_ANSWER_ALIASES or compact == "brakodpowiedzi"


def canonicalize_answer_text(text: str) -> str:
    cleaned = sanitize_text(text)
    if not cleaned:
        return NO_ANSWER
    cleaned = _ANSWER_PREFIX_PATTERN.sub("", cleaned)
    cleaned = _LIST_MARKER_PATTERN.sub("", cleaned)
    cleaned = sanitize_text(cleaned)
    if is_no_answer_text(cleaned):
        return NO_ANSWER
    return cleaned or NO_ANSWER


def is_noisy_answer_text(text: str) -> bool:
    cleaned = canonicalize_answer_text(text)
    if cleaned == NO_ANSWER:
        return False
    alpha_numeric = sum(character.isalnum() for character in cleaned)
    if alpha_numeric == 0:
        return True
    if len(cleaned) >= 6 and alpha_numeric / len(cleaned) < 0.35:
        return True
    if re.fullmatch(r"(?:\d+[.)]?\s*){2,}", cleaned):
        return True
    return False


def _normalize_for_search(text: str) -> str:
    normalized = unicodedata.normalize("NFD", sanitize_text(text).lower())
    normalized = "".join(
        character for character in normalized if unicodedata.category(character) != "Mn"
    )
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def answer_supported_by_context(answer: str, context: str) -> bool:
    canonical_answer = canonicalize_answer_text(answer)
    if canonical_answer == NO_ANSWER:
        return True

    answer_normalized = _normalize_for_search(canonical_answer)
    context_normalized = _normalize_for_search(context)
    if not answer_normalized or not context_normalized:
        return False
    if answer_normalized in context_normalized:
        return True

    tokens = [token for token in answer_normalized.split() if len(token) > 2]
    if len(tokens) < 2:
        return False
    token_hits = sum(token in context_normalized for token in tokens)
    return token_hits / len(tokens) >= 0.8


class PoquadTokenizer:
    def __init__(
            self,
            model_checkpoint: str = "allegro/plt5-base",
            max_input_length: int = 384,
            max_target_length: int = 64,
    ):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_checkpoint)
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    @staticmethod
    def _build_training_batch(examples: dict) -> tuple[list[str], list[str]]:
        if {"question", "context", "answer"}.issubset(examples.keys()):
            questions = [sanitize_text(question) for question in examples["question"]]
            contexts = [sanitize_text(context) for context in examples["context"]]
            answers = [canonicalize_answer_text(answer) for answer in examples["answer"]]
            inputs = [build_prompt(question, context) for question, context in zip(questions, contexts)]
            return inputs, answers

        raise ValueError(
            "Unsupported dataset schema. Expected flattened columns: question, context, answer."
        )

    def preprocess_poquad_raw_json(self, examples):
        inputs, targets = self._build_training_batch(examples)

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def transform(self, raw_datasets: DatasetDict) -> DatasetDict:
        return raw_datasets.map(
            self.preprocess_poquad_raw_json,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )
