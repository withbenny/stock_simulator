from datasets import Dataset
from fastapi import FastAPI
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
import evaluate
import pandas as pd
import torch
import uvicorn

class CustomTrainer(Trainer):
    def compute_loss(self, model: torch.nn.Module, inputs: dict, return_outputs: bool = False, num_items_in_batch: int = None) -> torch.Tensor:
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        epsilon = 0.1
        n_classes = 3

        smooth_labels = torch.zeros_like(logits)
        smooth_labels.fill_(epsilon / (n_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - epsilon)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        
        return (loss, outputs) if return_outputs else loss
    
class FinBERTSentimentAnalyzer:
    def __init__(self, model_path: str = None, model_name: str = "ProsusAI/finbert", 
                 max_length: int = 512, batch_size: int = 32, num_epochs: int = 3, num_labels: int = 3) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._initialize_tokenizer()
        self.model = self._initialize_model(model_path)
        
    def _initialize_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.model_max_length = self.max_length
        return tokenizer
    
    def _initialize_model(self, model_path: str) -> AutoModelForSequenceClassification:
        if model_path is not None:
            print(f"Loading model from {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels
            ).to(self.device)
        else:
            print("Initializing new model")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            ).to(self.device)
        
        return model
    
    def _chunk_text(self, text: str, overlap_size: int = 50) -> list:
        encoding = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids']
        offset_mapping = encoding['offset_mapping']
        max_chunk_size = self.max_length - 2
        if len(input_ids) <= self.max_length:
            return [text]
            
        chunks = []
        start_idx = 0
        
        while start_idx < len(input_ids):
            end_idx = min(start_idx + max_chunk_size, len(input_ids))
            
            chunk_start = offset_mapping[start_idx][0]
            chunk_end = offset_mapping[end_idx - 1][1]
            chunk_text = text[chunk_start:chunk_end]
            chunks.append(chunk_text)

            start_idx = max(start_idx + max_chunk_size - overlap_size, end_idx)

            if start_idx >= len(input_ids):
                break
                
        return chunks
    
    def predict_sentiment(self, text: str, use_chunks: bool = True) -> dict:
        if not use_chunks:
            return self._predict_single(text)

        tokens = self.tokenizer(text, truncation=False, padding=False)['input_ids']

        if len(tokens) <= self.max_length:
            return self._predict_single(text)
        
        chunks = self._chunk_text(text)
        chunk_results = []
        
        for chunk in chunks:
            result = self._predict_single(chunk)
            chunk_results.append(result)
            
        return self._merge_chunk_results(chunk_results)
    
    
    def prepare_for_training(self) -> AutoModelForSequenceClassification:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value", "key"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        return self.model
    
    def load_and_preprocess_data(self, file_path: str, test_spilt: float = 0.2) -> Dataset:
        df = pd.read_csv(file_path)
        sentiment_map = {
            'Bullish': 2,
            'Neutral': 1,
            'Bearish': 0
        }
        df['label'] = df['sentiment'].map(sentiment_map).astype(int)
        dataset = Dataset.from_pandas(df[['article', 'label']]).train_test_split(test_size=test_spilt)
        return dataset
    
    def _tokenize_text(self, examples: dict) -> dict:
        results = self.tokenizer(
            examples['article'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        results['labels'] = examples['label']
        return results
    
    def prepare_datasets(self, dataset: Dataset) -> Dataset:
        return dataset.map(
            self._tokenize_text,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
    
    @staticmethod
    def compute_metrics(eval_pred: tuple) -> dict:
        accuracy_metric = evaluate.load('accuracy')
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return accuracy_metric.compute(predictions=predictions.numpy(), references=labels)
    
    def _predict_single(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            class_probabilities = probabilities[0].tolist()
        
        sentiment_mapping = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
        sentiment = sentiment_mapping[predicted_class]
        
        return {
            'sentiment': sentiment,
            'probabilities': {
                'Bearish': class_probabilities[0],
                'Neutral': class_probabilities[1],
                'Bullish': class_probabilities[2]
            }
        }
    
    def _merge_chunk_results(self, chunk_results: list) -> dict:
        avg_probs = {
            'Bearish': 0.0,
            'Neutral': 0.0,
            'Bullish': 0.0
        }
        
        for result in chunk_results:
            for sentiment, prob in result['probabilities'].items():
                avg_probs[sentiment] += prob
                
        num_chunks = len(chunk_results)
        avg_probs = {k: v/num_chunks for k, v in avg_probs.items()}

        class_probabilities = [
            avg_probs['Bearish'],
            avg_probs['Neutral'],
            avg_probs['Bullish']
        ]
        predicted_class = class_probabilities.index(max(class_probabilities))
        sentiment_mapping = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
        
        return {
            'sentiment': sentiment_mapping[predicted_class],
            'probabilities': avg_probs
        }
            
    def train_model(self, train_dataset: Dataset, eval_dataset: Dataset, output_dir: str = "./finbert-lora-sentiment-final") -> AutoModelForSequenceClassification:
        self.prepare_for_training()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-4,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            fp16=True,
            lr_scheduler_type='cosine',
            report_to='none',
            save_total_limit=5,
        )
        
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        print(f"Training complete! Saving model to {output_dir}")
        trainer.save_model(output_dir)
        return self.model

class DistilrobertaSentimentAnalyzer(FinBERTSentimentAnalyzer):
    def __init__(self, model_path: str = None, model_name: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", 
                 max_length: int = 512, batch_size: int = 32, num_epochs: int = 3, num_labels: int = 3) -> None:
        super().__init__(
            model_path=model_path, 
            model_name=model_name, 
            max_length=max_length, 
            batch_size=batch_size, 
            num_epochs=num_epochs, 
            num_labels=num_labels
        )

    def train_model(self, train_dataset: Dataset, eval_dataset: Dataset, output_dir: str = "./distilroberta-lora-sentiment-final") -> AutoModelForSequenceClassification:
        self.prepare_for_training()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-4,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            fp16=True,
            lr_scheduler_type='cosine',
            report_to='none',
            save_total_limit=5
        )
        
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        print(f"Training complete! Saving model to {output_dir}")
        trainer.save_model(output_dir)
        return self.model

def test_llm(model_path: str, texts: list) -> None:
    print(f"Loading model from {model_path}...")
    if model_path.startswith('finbert') or model_path.startswith('./finbert'):
        analyzer = FinBERTSentimentAnalyzer(model_path=model_path)
    elif model_path.startswith('distilroberta') or model_path.startswith('./distilroberta'):
        analyzer = DistilrobertaSentimentAnalyzer(model_path=model_path)
    else:
        raise ValueError("Invalid model path. Please specify and start with either 'finbert' or 'distilroberta'")
    
    print("\nStart testing...")
    for i, text in enumerate(texts, 1):
        tokens = len(analyzer.tokenizer.tokenize(text))
        print(f"\nText Length: {tokens} tokens")
        print("Text:", text[:100], "..." if len(text) > 100 else "")
        
        if tokens >= 512:
            use_chunks = True
            print("This content is too long to process in one go. Splitting into chunks...")
        else:
            use_chunks = False
        result = analyzer.predict_sentiment(text, use_chunks=use_chunks)
        
        print("Sentiment:", result['sentiment'])
        print("Probabilities:")
        for sentiment, prob in result['probabilities'].items():
            print(f"{sentiment}: {prob:.4f}")

class SentimentAnalyzerAPI:
    def __init__(self, model_path: str):
        self.model_path = model_path
        if model_path.startswith('finbert') or model_path.startswith('./finbert'):
            self.analyzer = FinBERTSentimentAnalyzer(model_path=model_path)
        elif model_path.startswith('distilroberta') or model_path.startswith('./distilroberta'):
            self.analyzer = DistilrobertaSentimentAnalyzer(model_path=model_path)
        else:
            raise ValueError("Invalid model path. Must start with 'finbert' or 'distilroberta'.")
        
        self.app = FastAPI(title='Sentiment Analyzer API', version='0.1.0')

        class TextPayload(BaseModel):
            text: str
            use_chunks: bool = True
        
        @self.app.get('/')
        def root():
            return {"message": "Welcome to the Sentiment Analyzer API!"}

        @self.app.post('/analyze')
        def analyze_sentiment(payload: TextPayload):
            result = self.analyzer.predict_sentiment(payload.text, use_chunks=payload.use_chunks)
            return result
    
    def run(self, host: str = '0.0.0.0', port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    api = SentimentAnalyzerAPI(model_path="./distilroberta_sentiment_model/checkpoint-618")
    api.run()