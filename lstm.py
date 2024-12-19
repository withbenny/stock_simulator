import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import csv

class LSTMDataset(Dataset):
    def __init__(self, articles: list, sentiments: list, vocab: dict = None) -> None:
        self.articles = articles
        self.sentiments = sentiments

        if vocab is None:
            all_words = [word for article in articles for word in word_tokenize(article.lower())]
            word_counts = Counter(all_words)
            self.vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.items())}
            self.vocab["<PAD>"] = 0
        else:
            self.vocab = vocab
    
    def __len__(self) -> int:
        return len(self.articles)
    
    def __getitem__(self, idx: int) -> tuple:
        article = self.articles[idx]
        tokens = word_tokenize(article.lower())
        encoded = [self.vocab.get(word, 0) for word in tokens]
        sentiment = self.sentiments[idx]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(sentiment, dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.5) -> None:
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(self.dropout(hidden[-1]))
        return output

class LSTMSentimentAnalyzer:
    def __init__(self, save_path: str = None, device: str = None, csv_path: str = None, batch_size: int = 32, epochs: int = 10, lr: float = 1e-5) -> None:
        self.device = device if device is not None else "cpu"
        self.csv_path = csv_path if csv_path is not None else "./DATA.csv"
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.vocab = None
        self.save_path = save_path if save_path is not None else "./lstm"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def preprocess_data(self) -> tuple:
        data = pd.read_csv(self.csv_path)
        articles = data["article"].values
        sentiments = LabelEncoder().fit_transform(data["sentiment"].values)
        
        X_train, X_temp, y_train, y_temp = train_test_split(articles, sentiments, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def collate_fn(self, batch: list) -> tuple:
        articles, sentiments = zip(*batch)
        articles_padded = pad_sequence(articles, batch_first=True, padding_value=0)
        sentiments = torch.stack(sentiments)

        return articles_padded, sentiments

    def load_dataset(self) -> tuple:
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data()
        
        train_dataset = LSTMDataset(X_train, y_train)
        val_dataset = LSTMDataset(X_val, y_val, vocab=train_dataset.vocab)
        test_dataset = LSTMDataset(X_test, y_test, vocab=train_dataset.vocab)
        torch.save(train_dataset.vocab, f"{self.save_path}/vocab.pth")
        vocab_size = len(train_dataset.vocab)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        return train_loader, val_loader, test_loader, vocab_size
    
    def train_model(self, evaluate: bool = True, embedding_dim: int = 100, hidden_dim: int = 128, output_dim: int = 3, num_layers: int = 1, dropout: float = 0.5) -> None:
        train_loader, val_loader, test_loader, vocab_size = self.load_dataset()

        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0

            for articles, sentiments in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                articles, sentiments = articles.to(self.device), sentiments.to(self.device)
                optimizer.zero_grad()
                output = model(articles)
                loss = criterion(output, sentiments)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0 
            
            with torch.no_grad():
                for articles, sentiments in val_loader:
                    articles, sentiments = articles.to(self.device), sentiments.to(self.device)
                    output = model(articles)
                    loss = criterion(output, sentiments)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")
        if evaluate:
            self.evaluate_model(model, test_loader, criterion)
        
        model_path = f"{self.save_path}/lstm_sentiment_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}!")
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> None:
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for articles, sentiments in test_loader:
                articles, sentiments = articles.to(self.device), sentiments.to(self.device)
                output = model(articles)
                loss = criterion(output, sentiments)
                test_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += sentiments.size(0)
                correct += (predicted == sentiments).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = correct / total
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {accuracy:.4f}")

def test_lstm(model_path: str = None, vocab: dict = None, texts: list = None) -> None:
    save_path = "./lstm"
    if vocab is None:
        vocab = torch.load(f"{save_path}/vocab.pth", weights_only=False)
    if model_path is None:
        model_path = f"{save_path}/lstm_sentiment_model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentiment_mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

    model = LSTMModel(len(vocab), 100, 128, 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    encoded_texts = [torch.tensor([vocab.get(word, 0) for word in word_tokenize(text.lower())], dtype=torch.long) for text in texts]
    padded_texts = pad_sequence(encoded_texts, batch_first=True, padding_value=0).to(device)

    with torch.no_grad():
        output = model(padded_texts)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        for text, sentiment, prob in zip(texts, predicted, probabilities):
            sentiment_label = sentiment_mapping[sentiment.item()]
            prob_distribution = {sentiment_mapping[i]: prob[i].item() for i in range(len(sentiment_mapping))}
            print(f"\nText: {text}\nSentiment: {sentiment_label}\nProbabilities:")
            for sentiment, prob in prob_distribution.items():
                print(f"{sentiment}: {prob:.4f}")

def process_batches(model, device, padded_texts, batch_size):
    predictions = []
    total_samples = len(padded_texts)
    num_batches = (total_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch = padded_texts[start_idx:end_idx]

        with torch.no_grad():
            output = model(batch)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)
            predictions.extend(predicted.cpu().tolist())

    return predictions

def test_from_csv_lstm(input_csv: str, model_path: str = None, vocab: dict = None, batch_size: int = 64):
    save_path = "./lstm"
    if vocab is None:
        vocab = torch.load(f"{save_path}/vocab.pth", weights_only=False)
    if model_path is None:
        model_path = f"{save_path}/lstm_sentiment_model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentiment_mapping = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

    model = LSTMModel(len(vocab), 100, 128, 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    articles = []
    true_sentiments = []
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles.append(row["article"])
            true_sentiments.append(row["sentiment"])

    encoded_texts = [
        torch.tensor([vocab.get(word, 0) for word in word_tokenize(article.lower())], dtype=torch.long)
        for article in articles
    ]
    padded_texts = pad_sequence(encoded_texts, batch_first=True, padding_value=0).to(device)
    predicted_labels = process_batches(model, device, padded_texts, batch_size)

    correct_predictions = 0
    for true_sentiment, predicted_label in zip(true_sentiments, predicted_labels):
        predicted_sentiment = sentiment_mapping[predicted_label]
        if predicted_sentiment == true_sentiment:
            correct_predictions += 1

    accuracy = correct_predictions / len(articles)
    print(f"\nAccuracy: {accuracy:.2%}")
    return accuracy

def main():
    test_from_csv_lstm("data_test.csv")

if __name__ == "__main__":
    main()