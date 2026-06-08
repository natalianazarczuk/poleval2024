# PolEval 2024 – Zadanie 1: PoQuAD (Generatywne Odpowiadanie na Pytania)

System NLP oparty na architekturze Seq2Seq (PLT5), który na podstawie pytania i kontekstu
(fragment artykułu z Wikipedii) generuje odpowiedź w języku polskim lub zwraca `brak_odpowiedzi`.

## Struktura projektu

```
poleval2024/
├── data/                    # Dane wejściowe (wykluczone z Git)
│   ├── train.json
│   └── dev.json
├── src/
│   ├── data_loader.py       # PoquadDataLoader – wczytuje JSON przez Apache Arrow
│   ├── tokenizer.py         # PoquadTokenizer – buduje prompt T5, tokenizuje
│   ├── metrics.py           # compute_poleval_metrics – nLev + F1 + final score
│   ├── train.py             # Główny skrypt treningowy (Seq2SeqTrainer + compute_metrics)
│   ├── inference.py         # PolEvalInferencePipeline – generowanie out.tsv
│   └── test_pipeline.py     # Smoke test: loader + tokenizer
├── outputs/                 # Artefakty treningu (wykluczone z Git)
├── requirements.txt
└── .gitignore
```

## Format promptu (T5 input)

```
pytanie: {question} kontekst: {context}
```

## Sanityzacja danych przed treningiem

Pipeline w `src/tokenizer.py` wykonuje automatycznie:
- normalizację Unicode (`NFC`) i dekodowanie encji HTML,
- usuwanie znaków kontrolnych/zero-width/NBSP,
- normalizację cudzysłowów i myślników,
- redukcję szumu formatowania (np. `"[1]"`, `"(2)"`, nadmiarowe spacje),
- filtrowanie pustych próbek (puste `question` lub `context`),
- normalizację odpowiedzi typu `brak_odpowiedzi` (`brak`, `n/a`, `none`, itp.).

Jeśli pytanie jest bez odpowiedzi (`is_impossible=True` lub pusta lista `answers`),
target modelu to: `brak_odpowiedzi`.

## Metryki

| Metryka | Opis |
|---|---|
| `normalized_levenshtein` | Podobieństwo tekstowe do gold answer (0–100) |
| `answerability_f1` | F1 dla decyzji „odpowiedz / abstain" (0–100) |
| `score` | Średnia powyższych – finalny wynik PolEval (0–100) |
| `exact_match` | Dokładne trafienia na pytaniach z odpowiedzią |

## Przygotowanie środowiska

```zsh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

W katalogu `data/` muszą znajdować się pliki `train.json` i `dev.json`.

## Uruchomienie

### Trening

```zsh
.venv/bin/python src/train.py
```

Model (`allegro/plt5-small`) zostanie pobrany automatycznie z Hugging Face Hub.
Artefakty są zapisywane do `outputs/plt5-task1/`.

> Aby użyć mocniejszego modelu (`allegro/plt5-base`) zmień `MODEL_NAME` w `src/train.py`.

### Szybki test treningu

```zsh
.venv/bin/python src/train.py --debug
```

To uruchamia krótki przebieg na małej próbce i robi ewaluację po 1 kroku, żeby szybko wykryć błędy w pipeline.

### Smoke test (loader + tokenizer)

```zsh
.venv/bin/python src/test_pipeline.py
```

### Inferencja (generowanie out.tsv do submisji)

```zsh
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from inference import PolEvalInferencePipeline
p = PolEvalInferencePipeline('outputs/plt5-task1')
p.generate_submissions('data/in.tsv', 'data/dev.json', 'outputs/out.tsv')
"
```

### Ewaluacja modelu na `dev.json`

```zsh
.venv/bin/python src/evaluate.py --model-dir outputs/plt5-task1 --dev-json data/dev.json
```

Szybka iteracja (np. 200 próbek):

```zsh
.venv/bin/python src/evaluate.py --model-dir outputs/plt5-task1 --dev-json data/dev.json --limit 200
```

Zapis predykcji per próbka (JSONL):

```zsh
.venv/bin/python src/evaluate.py --model-dir outputs/plt5-task1 --dev-json data/dev.json --predictions-path outputs/dev_predictions.jsonl
```

Skrypt wypisuje:
- `overall`: metryki konkursowe (`normalized_levenshtein`, `answerability_f1`, `score`, `exact_match`)
- `breakdown`: dodatkowy rozkład jakości (answerable vs unanswerable, precision/recall abstencji)

## Hiperparametry (domyślne)

| Parametr | Wartość |
|---|---|
| Model | `allegro/plt5-small` |
| Epoki | 1 |
| Learning rate | 1e-5 |
| Train batch size | 8 |
| Eval batch size | 8 |
| Gradient accumulation | 1 |
| Max input length | 384 tokenów |
| Max output length | 64 tokeny |
| Beam search podczas treningu/eval | 1 |
