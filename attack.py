from llm import FinBERTSentimentAnalyzer, DistilrobertaSentimentAnalyzer
from tqdm import tqdm
import csv
import math
import pandas as pd
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import torch
import torch.nn.functional as F
from lstm import LSTMModel

class LLMTester:
    def __init__(self, model_path: str, batch_size: int = 8):
        self.model_path = model_path
        self.batch_size = batch_size

        if model_path.startswith("finbert"):
            self.analyzer = FinBERTSentimentAnalyzer()
        elif model_path.startswith("distilroberta"):
            self.analyzer = DistilrobertaSentimentAnalyzer()
        else:
            raise ValueError("Invalid model path. Must start with 'finbert' or 'distilroberta'.")

        self.mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
        self.attacker = AdverbAttacker(self.analyzer.model, self.analyzer.tokenizer)
        print(f"Model loaded from: {model_path}")

    def _process_single_text(self, article: str) -> dict:
        tokens = len(self.analyzer.tokenizer.tokenize(article))
        use_chunks = tokens >= 512

        result = self.analyzer.predict_sentiment(article, use_chunks=use_chunks)
        probabilities = result["probabilities"]

        output = {
            "sentiment": result["sentiment"],
            "prob_bearish": probabilities.get("Bearish", 0.0),
            "prob_neutral": probabilities.get("Neutral", 0.0),
            "prob_bullish": probabilities.get("Bullish", 0.0),
            "article": article,
        }
        return output

    def compare_attack(self, texts: list, num_replacements: int = 1):
        success_count = 0
        total_texts = len(texts)

        for text in tqdm(texts):
            original_result = self._process_single_text(text)
            original_probs = torch.tensor([
                original_result.get("prob_bearish", 0.0),
                original_result.get("prob_neutral", 0.0),
                original_result.get("prob_bullish", 0.0)
            ], device=self.analyzer.model.device)

            importance_scores = self.attacker.get_adverb_importance_scores(text, original_probs)
            adversarial_text = self.attacker.generate_adversarial_text(text, importance_scores, num_replacements)

            adversarial_result = self._process_single_text(adversarial_text)

            if original_result["sentiment"] != adversarial_result["sentiment"]:
                success_count += 1

        success_rate = success_count / total_texts
        print(f"Adversarial attack success rate: {success_rate:.2%}")

    def test_llm(self, texts: list, output_csv: str = "results.csv") -> None:
        if not texts:
            print("No texts provided.")
            return

        all_results = []

        total_texts = len(texts)
        num_batches = math.ceil(total_texts / self.batch_size)

        print(f"Total texts: {total_texts}, Batch size: {self.batch_size}, Number of batches: {num_batches}")

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_texts = texts[start_idx:end_idx]

            for text in batch_texts:
                output_dict = self._process_single_text(text)
                all_results.append(output_dict)

        self._save_to_csv(all_results, output_csv)
        print(f"\nProcessing complete! Results saved to {output_csv}")
    
    def test_from_csv(self, input_csv: str):
        articles = []
        true_sentiments = []

        with open(input_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                articles.append(row["article"])
                true_sentiments.append(row["sentiment"])

        total_texts = len(articles)
        num_batches = math.ceil(total_texts / self.batch_size)
        correct_predictions = 0

        print(f"Total texts: {total_texts}, Batch size: {self.batch_size}, Number of batches: {num_batches}")

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_articles = articles[start_idx:end_idx]
            batch_true_sentiments = true_sentiments[start_idx:end_idx]

            for article, true_sentiment in zip(batch_articles, batch_true_sentiments):
                prediction = self._process_single_text(article)

                if prediction["sentiment"] == true_sentiment:
                    correct_predictions += 1

        accuracy = correct_predictions / total_texts
        print(f"\nAccuracy: {accuracy:.2%}")

    def _save_to_csv(self, results: list, output_csv: str):
        fieldnames = ["sentiment", "prob_bearish", "prob_neutral", "prob_bullish", "article"]
        with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

class AdverbAttacker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_adverb_importance_scores(self, text, original_probs):
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        adverbs = [word for word, pos in pos_tags if pos in ['RB', 'RBR', 'RBS']]
        importance_scores = []

        for adverb in adverbs:
            masked_tokens = [token if token != adverb else self.tokenizer.mask_token for token in tokens]
            masked_text = " ".join(masked_tokens)
            masked_inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)

            with torch.no_grad():
                logits = self.model(**masked_inputs).logits
                masked_probs = F.softmax(logits, dim=-1)[0]

            importance = torch.abs(original_probs - masked_probs).sum().item()
            importance_scores.append((adverb, importance))

        return sorted(importance_scores, key=lambda x: x[1], reverse=True)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word, pos=wordnet.ADV):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def generate_adversarial_text(self, text, importance_scores, num_replacements=1):
        tokens = word_tokenize(text)
        for adverb, _ in importance_scores[:num_replacements]:
            synonyms = self.get_synonyms(adverb)
            if synonyms:
                replacement = synonyms[0]  # Use the first synonym
                tokens = [replacement if token == adverb else token for token in tokens]
        return " ".join(tokens)

def main():
    model_path = "distilroberta"
    df = pd.read_csv("data_test.csv")
    texts = df["article"].tolist()

    tester = LLMTester(model_path=model_path, batch_size=8)
    tester.test_from_csv("data_test.csv")

if __name__ == "__main__":
    main()