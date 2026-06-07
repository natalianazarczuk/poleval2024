import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class PolEvalInferencePipeline:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, question: str, context: str) -> str:
        prompt = f"pytanie: {question} kontekst: {context}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=64)

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

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
                        "question": qa['question'].strip(),
                        "context": context
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

                if prediction.lower() == "brak_odpowiedzi":
                    outfile.write("\n")
                else:
                    outfile.write(f"{prediction}\n")
