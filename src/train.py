import os
import json
import pandas as pd
import numpy as np
import Levenshtein
from sklearn.metrics import f1_score
from datasets import load_dataset

# KLUCZOWY IMPORT, KTÓREGO BRAKOWAŁO:
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

print("--- Ładowanie lokalnych plików JSON po uprzednim pobraniu ---")

# Pobieramy ścieżkę do katalogu, w którym znajduje się plik train.py (czyli .../poleval2024/src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Przechodzimy poziom wyżej do głównego folderu projektu i wchodzimy do 'data'
base_project_dir = os.path.dirname(current_dir)
train_path = os.path.join(base_project_dir, "data", "train.json")
dev_path = os.path.join(base_project_dir, "data", "dev.json")

# Słownik z bezwzględnymi (pewnymi) ścieżkami systemowymi
data_files = {
    "train": train_path,
    "validation": dev_path
}

# Sprawdzenie dla pewności przed uruchomieniem load_dataset
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Błąd! Plik train.json powinien być w: {train_path}, ale go tam nie ma. Sprawdź strukturę folderów!")

raw_datasets = load_dataset("json", data_files=data_files, field="data")

print(raw_datasets)

# 2. TOKENIZACJA
print("--- Inicjalizacja tokenizatora ---")
model_checkpoint = "allegro/plt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download=True)

def preprocess_poquad_raw_json(examples):
    inputs = []
    targets = []

    # Przechodzimy przez strukturę: artykuły -> paragrafy -> pytania
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
                    if gen_ans == "":
                        targets.append("brak_odpowiedzi")
                    else:
                        targets.append(gen_ans)
                else:
                    targets.append("brak_odpowiedzi")

    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(text_target=targets, max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("--- Przetwarzanie i tokenizacja danych ---")
tokenized_datasets = raw_datasets.map(
    preprocess_poquad_raw_json,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# Przypisanie do zmiennych
tokenized_train = tokenized_datasets["train"]
tokenized_dev = tokenized_datasets["validation"]

# ===================================================
# DLA BEZPIECZEŃSTWA: Wybierz mały wycinek na Smoke Test (tylko 3 artykuły)
tokenized_train_sample = tokenized_train.select(range(3))
tokenized_dev_sample = tokenized_dev.select(range(2))
# ===================================================

print(tokenized_train_sample)