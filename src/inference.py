import json
import torch
from transformers import AutoConfig, T5TokenizerFast, AutoModelForSeq2SeqLM

try:
    from .tokenizer import NO_ANSWER, build_prompt, is_no_answer_text, sanitize_text
except ImportError:  # pragma: no cover - supports running as a script
    from tokenizer import NO_ANSWER, build_prompt, is_no_answer_text, sanitize_text


class PolEvalInferencePipeline:
    def __init__(self, model_dir: str):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_dir)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        config = AutoConfig.from_pretrained(model_dir)
        config.tie_word_embeddings = False
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, config=config).to(self.device)
        generation_config = self.model.generation_config
        generation_config.max_new_tokens = generation_config.max_new_tokens or 64
        generation_config.min_new_tokens = generation_config.min_new_tokens or 1
        generation_config.max_length = None
        generation_config.min_length = None
        generation_config.num_beams = 4
        generation_config.length_penalty = 0.8
        generation_config.early_stopping = True
        self.model.eval()

    def predict(self, question: str, context: str) -> str:
        prompt = build_prompt(question, context)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=384,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()
        if is_no_answer_text(decoded):
            return NO_ANSWER
        return sanitize_text(decoded) or NO_ANSWER

    def generate_submissions(self, test_in_tsv: str, test_context_json: str, output_tsv: str):
        with open(test_context_json, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)

        context_map = {}
        for article in ref_data.get('data', []):
            art_id = article['id']
            for p_idx, paragraph in enumerate(article.get('paragraphs', [])):
                context = paragraph.get('context', '').strip()
                for q_idx, qa in enumerate(paragraph.get('qas', [])):
                    lookup_key = f"{art_id}_{p_idx}_{q_idx}"
                    context_map[lookup_key] = {
                        "question": sanitize_text(qa.get("question", "")),
                        "context": sanitize_text(context),
                    }

        with open(test_in_tsv, 'r', encoding='utf-8') as infile, \
                open(output_tsv, 'w', encoding='utf-8') as outfile:

            for line in infile:
                id_key = line.strip()
                if not id_key:
                    continue

                data_slice = context_map.get(id_key)
                if not data_slice:
                    outfile.write("\n")
                    continue

                prediction = self.predict(
                    data_slice["question"],
                    data_slice["context"],
                )

                if is_no_answer_text(prediction):
                    outfile.write("\n")
                else:
                    outfile.write(f"{prediction}\n")
