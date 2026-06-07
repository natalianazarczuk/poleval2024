# PolEval 2024 - Task 1

Cały proces składa się z trzech kroków:

1. `train.py` uczy model na danych z `train.json`.
2. Wytrenowany model jest zapisywany w `outputs/plt5-task1`.
3. `test_pipeline.py` wczytuje zapisany model i sprawdza go na danych z
   `dev.json`.

Najważniejsze pliki:

- `src/train.py` - uruchamia trening i zapisuje model.
- `src/tokenizer.py` - przygotowuje pytania, teksty i odpowiedzi dla modelu.
- `src/inference.py` - używa wytrenowanego modelu do generowania odpowiedzi.
- `src/test_pipeline.py` - testuje model i wyświetla metryki.
- `src/metrics.py` - oblicza jakość odpowiedzi.

## Przygotowanie projektu

Utwórz środowisko Python:

```bash
python -m venv .venv
source .venv/bin/activate
```

Zainstaluj biblioteki:

```bash
pip install -r requirements.txt
```

W katalogu `data` muszą znajdować się:

```text
data/
├── train.json
└── dev.json
```

Przy pierwszym uruchomieniu zostanie pobrany model `allegro/plt5-base`.

## Kolejność uruchamiania

```bash
python src/train.py
python src/test_pipeline.py
```

Najpierw trzeba wytrenować i zapisać model. Dopiero potem można go testować.
