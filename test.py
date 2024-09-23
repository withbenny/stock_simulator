import json
from collections import Counter

# Load the JSON file
with open('/mnt/data/AAPL_news.json', 'r') as file:
    data = json.load(file)

# Initialize variables to collect information
total_news = 0
sentiment_counter = Counter()
sentiment_scores = []
weighted_sentiment_scores = []

# Iterate through each news item
for news in data['feed']:
    for ticker in news['ticker_sentiment']:
        if ticker['ticker'] == 'AAPL':
            total_news += 1
            sentiment_score = float(ticker['ticker_sentiment_score'])
            relevance_score = float(ticker['relevance_score'])
            sentiment_scores.append(sentiment_score)
            weighted_sentiment_scores.append(sentiment_score * relevance_score)

            # Classify the sentiment labels based on the provided score ranges
            if sentiment_score <= -0.35:
                sentiment_counter['Bearish'] += 1
            elif -0.35 < sentiment_score <= -0.15:
                sentiment_counter['Somewhat-Bearish'] += 1
            elif -0.15 < sentiment_score < 0.15:
                sentiment_counter['Neutral'] += 1
            elif 0.15 <= sentiment_score < 0.35:
                sentiment_counter['Somewhat-Bullish'] += 1
            else:
                sentiment_counter['Bullish'] += 1

# Calculate percentages for each sentiment category
percentages = {label: count / total_news * 100 for label, count in sentiment_counter.items()}

# Calculate combined percentages for all pessimistic, neutral, and optimistic sentiments
pessimistic_percentage = (sentiment_counter['Bearish'] + sentiment_counter['Somewhat-Bearish']) / total_news * 100
neutral_percentage = sentiment_counter['Neutral'] / total_news * 100
optimistic_percentage = (sentiment_counter['Somewhat-Bullish'] + sentiment_counter['Bullish']) / total_news * 100

# Calculate unweighted and weighted average sentiment scores
unweighted_avg_sentiment = sum(sentiment_scores) / total_news
weighted_avg_sentiment = sum(weighted_sentiment_scores) / total_news

# Assign overall sentiment label based on average sentiment score
def sentiment_label(avg_score):
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

overall_unweighted_label = sentiment_label(unweighted_avg_sentiment)
overall_weighted_label = sentiment_label(weighted_avg_sentiment)

# Display the results
import ace_tools as tools; tools.display_dataframe_to_user(name="AAPL News Sentiment Analysis", dataframe=None)

{
    "total_news": total_news,
    "sentiment_percentages": percentages,
    "pessimistic_percentage": pessimistic_percentage,
    "neutral_percentage": neutral_percentage,
    "optimistic_percentage": optimistic_percentage,
    "unweighted_avg_sentiment": unweighted_avg_sentiment,
    "weighted_avg_sentiment": weighted_avg_sentiment,
    "overall_unweighted_label": overall_unweighted_label,
    "overall_weighted_label": overall_weighted_label
}

