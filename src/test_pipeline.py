import json
import os

from inference import PolEvalInferencePipeline
from metrics import NO_ANSWER, compute_poleval_metrics


MAX_EXAMPLES = 20


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_dir, "outputs", "plt5-task1")
    dev_path = os.path.join(project_dir, "data", "dev.json")

    pipeline = PolEvalInferencePipeline(model_dir)

    with open(dev_path, encoding="utf-8") as file:
        dev_data = json.load(file)["data"]

    predictions = []
    correct_answers = []

    for article in dev_data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                if qa.get("is_impossible") or not answers:
                    correct_answer = NO_ANSWER
                else:
                    correct_answer = (
                        answers[0].get("generative_answer", "").strip()
                        or NO_ANSWER
                    )

                prediction = pipeline.predict(qa["question"], context)
                predictions.append(prediction)
                correct_answers.append(correct_answer)

                print(f"Model: {prediction}")
                print(f"Poprawna odpowiedź: {correct_answer}\n")

                if len(predictions) == MAX_EXAMPLES:
                    metrics = compute_poleval_metrics(
                        predictions,
                        correct_answers,
                    )
                    print("Metryki:")
                    for name, value in metrics.items():
                        print(f"{name}: {value:.2f}")
                    return


if __name__ == "__main__":
    main()
