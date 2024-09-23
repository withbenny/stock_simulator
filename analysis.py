import json

class NewsAnalysis:
    def __init__(self, news:json) -> None:
        self.news = news
        self.total_news = 0
        self.sentiment_counter = {
            'Bearish': 0,
            'Somewhat-Bearish': 0,
            'Neutral': 0,
            'Somewhat-Bullish': 0,
            'Bullish': 0
        }
        self.sentiment_scores = []
        self.weighted_sentiment_scores = []
        self.process_news()

    def process_news(self) -> None:
        for idx in self.news['feed']:
            for ticker in idx['ticker_sentiment']:
                if ticker['ticker'] == 'AAPL':
                    self.total_news += 1
                    sentiment_score = float(ticker['ticker_sentiment_score'])
                    relevance_score = float(ticker['relevance_score'])
                    self.sentiment_scores.append(sentiment_score)
                    self.weighted_sentiment_scores.append(sentiment_score * relevance_score)

                    if sentiment_score <= -0.35:
                        self.sentiment_counter['Bearish'] += 1
                    elif -0.35 < sentiment_score <= -0.15:
                        self.sentiment_counter['Somewhat-Bearish'] += 1
                    elif -0.15 < sentiment_score < 0.15:
                        self.sentiment_counter['Neutral'] += 1
                    elif 0.15 <= sentiment_score < 0.35:
                        self.sentiment_counter['Somewhat-Bullish'] += 1
                    elif sentiment_score >= 0.35:
                        self.sentiment_counter['Bullish'] += 1

    def calculate_percentages(self) -> dict:
        percentages = {label: (self.sentiment_counter[label] / self.total_news * 100 if self.total_news > 0 else 0)
                       for label in ['Bearish', 'Somewhat-Bearish', 'Neutral', 'Somewhat-Bullish', 'Bullish']}
        return percentages
    
    def combined_percentages(self) -> tuple:
        pessimistic_percentage = (self.sentiment_counter['Bearish'] + self.sentiment_counter['Somewhat-Bearish']) / self.total_news * 100
        neutral_percentage = self.sentiment_counter['Neutral'] / self.total_news * 100
        optimistic_percentage = (self.sentiment_counter['Somewhat-Bullish'] + self.sentiment_counter['Bullish']) / self.total_news * 100
        
        return pessimistic_percentage, neutral_percentage, optimistic_percentage

    def average_sentiment(self) -> tuple:
        unweighted_avg_sentiment = sum(self.sentiment_scores) / self.total_news
        weighted_avg_sentiment = sum(self.weighted_sentiment_scores) / self.total_news
        
        return unweighted_avg_sentiment, weighted_avg_sentiment
    
    def sentiment_label(self, avg_score:float) -> str:
        if avg_score <= -0.35:
            return "Bearish"
        elif -0.35 < avg_score <= -0.15:
            return "Somewhat-Bearish"
        elif -0.15 < avg_score < 0.15:
            return "Neutral"
        elif 0.15 <= avg_score < 0.35:
            return "Somewhat-Bullish"
        else:
            return "Bullish"
    
    def get_summary(self) -> dict:
        percentages = self.calculate_percentages()
        pessimistic_percentage, neutral_percentage, optimistic_percentage = self.combined_percentages()
        unweighted_avg_sentiment, weighted_avg_sentiment = self.average_sentiment()
        overall_sentiment = self.sentiment_label(unweighted_avg_sentiment)
        return {
            'total_news': self.total_news,
            'sentiment_percentages': percentages,
            'pessimistic_percentage': pessimistic_percentage,
            'neutral_percentage': neutral_percentage,
            'optimistic_percentage': optimistic_percentage,
            'unweighted_avg_sentiment': unweighted_avg_sentiment,
            'weighted_avg_sentiment': weighted_avg_sentiment,
            'overall_sentiment': overall_sentiment
        }