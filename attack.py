from llm import FinBERTSentimentAnalyzer, DistilrobertaSentimentAnalyzer
from tqdm import tqdm
import csv
import math
import pandas as pd

class LLMTester:
    def __init__(self, model_path: str, batch_size: int = 8):

        self.model_path = model_path
        self.batch_size = batch_size

        if model_path.startswith("finbert"):
            self.analyzer = FinBERTSentimentAnalyzer(model_path=model_path)
        elif model_path.startswith("distilroberta"):
            self.analyzer = DistilrobertaSentimentAnalyzer(model_path=model_path)
        else:
            raise ValueError("Invalid model path. Must start with 'finbert' or 'distilroberta'.")

        self.mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
        
        print(f"Model loaded from: {model_path}")

    def _process_single_text(self, article: str) -> dict:
        tokens = len(self.analyzer.tokenizer.tokenize(article))
        use_chunks = tokens >= 512
        
        result = self.analyzer.predict_sentiment(article, use_chunks=use_chunks)

        sentiment = result["sentiment"]
        probabilities = result["probabilities"]

        output = {
            "sentiment": sentiment,
            "prob_bearish": probabilities.get("Bearish", 0.0),
            "prob_neutral": probabilities.get("Neutral", 0.0),
            "prob_bullish": probabilities.get("Bullish", 0.0),
            "article": article,
        }
        return output

    def test_llm(self, texts: list, output_csv: str = "results.csv") -> None:
        if not texts:
            print("No texts provided.")
            return
        
        all_results = []

        total_texts = len(texts)
        num_batches = math.ceil(total_texts / self.batch_size)

        print(f"Total texts: {total_texts}, Batch size: {self.batch_size}, "
              f"Number of batches: {num_batches}")
        
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_texts = texts[start_idx:end_idx]

            for text in batch_texts:
                output_dict = self._process_single_text(text)
                all_results.append(output_dict)
        
        self._save_to_csv(all_results, output_csv)
        print(f"\nProcessing complete! Results saved to {output_csv}")

    def _save_to_csv(self, results: list, output_csv: str):
        fieldnames = ["sentiment", "prob_bearish", "prob_neutral", "prob_bullish", "article"]
        with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)


def main():
    model_path = "distilroberta_sentiment_model/checkpoint-618"
    df = pd.read_csv("./data/AAPL/AAPL_filtered.csv")
    texts = df["article"].tolist()

    tester = LLMTester(model_path=model_path, batch_size=8)
    tester.test_llm(texts=texts, output_csv="results.csv")

if __name__ == "__main__":
    main()