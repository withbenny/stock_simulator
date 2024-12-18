from llm import FinBERTSentimentAnalyzer, test_llm, DistilrobertaSentimentAnalyzer
from lstm import LSTMSentimentAnalyzer, test_lstm

llm = FinBERTSentimentAnalyzer(
    model_path = None, 
    model_name = "ProsusAI/finbert",
    max_length = 512,
    num_labels = 3,
    batch_size = 64,
    num_epochs = 10
)

dataset_path = "data_train.csv"
dataset = llm.load_and_preprocess_data(dataset_path)
tokenized_dataset = llm.prepare_datasets(dataset)
llm.train_model(
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["test"],
    output_dir = "finbert_sentiment_model",
)

# test_llm(model_path="finbert_sentiment_model",
#              texts=[
#                     "Apple's recent financial report shows that the company's profits have barely increase.",
#             ])

lstm = LSTMSentimentAnalyzer(save_path="./lstm", device="cuda", csv_path="DATA.csv", batch_size=64, epochs=10, lr=1e-5)
lstm.train_model()

# test_lstm(model_path="./lstm/lstm_sentiment_model.pth",
#           vocab="./lstm/vocab.pth",
#           texts=[
#                  "Apple's recent financial report shows that the company's profits have barely increase.",
#          ])

llm = DistilrobertaSentimentAnalyzer(
    model_path = None, 
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    max_length = 512,
    num_labels = 3,
    batch_size = 64,
    num_epochs = 10
)

dataset_path = "data_train.csv"
dataset = llm.load_and_preprocess_data(dataset_path)
tokenized_dataset = llm.prepare_datasets(dataset)
llm.train_model(
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["test"],
    output_dir = "distilroberta_sentiment_model",
)
