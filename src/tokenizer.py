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
    def _extract_target_answer(qa: dict) -> str:
        if qa.get("is_impossible", False):
            return NO_ANSWER

        for answer in qa.get("answers", []):
            candidate = sanitize_text(answer.get("generative_answer", ""))
            if candidate and is_no_answer_text(candidate):
                return NO_ANSWER
            if candidate:
                return candidate
            candidate = sanitize_text(answer.get("text", ""))
            if candidate and is_no_answer_text(candidate):
                return NO_ANSWER
            if candidate:
                return candidate
        return NO_ANSWER

    def preprocess_poquad_raw_json(self, examples):
        inputs = []
        targets = []

        for paragraphs_list in examples["paragraphs"]:
            for paragraph in paragraphs_list:
                context = sanitize_text(paragraph.get("context", ""))
                if not context:
                    continue

                for qa in paragraph["qas"]:
                    question = sanitize_text(qa.get("question", ""))
                    if not question:
                        continue

                    inputs.append(build_prompt(question, context))
                    targets.append(self._extract_target_answer(qa))

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
