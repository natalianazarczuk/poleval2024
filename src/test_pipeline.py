import os
from data_loader import PoquadDataLoader
from tokenizer import PoquadTokenizer


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, "data")

    loader = PoquadDataLoader(data_dir=data_dir)
    raw_data = loader.load_raw_datasets()
    print(raw_data)

    tokenizer = PoquadTokenizer()
    final_data = tokenizer.transform(raw_data)
    print(final_data)

    first_row = final_data["train"][0]
    print(list(first_row.keys()))


if __name__ == "__main__":
    main()