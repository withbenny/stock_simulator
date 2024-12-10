from llm import FinBERTSentimentAnalyzer, test_finbert

llm = FinBERTSentimentAnalyzer(
    model_path = None, 
    model_name = "ProsusAI/finbert",
    max_length = 512,
    num_labels = 3,
    batch_size = 32,
    num_epochs = 3
)

dataset_path = "DATA.csv"
dataset = llm.load_and_preprocess_data(dataset_path)
tokenized_dataset = llm.prepare_datasets(dataset)
llm.train_model(
    train_dataset = tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["test"],
    output_dir = "finbert_sentiment_model",
)

test_finbert(model_path="finbert_sentiment_model",
             texts=[
                 "One of Berkshire Hathaway's most successful investments, from a purely dollar-gain perspective, has got to be Apple (AAPL 1.02%). The conglomerate first purchased shares in the first quarter of 2016, and since the start of that year through the end of 2023, the stock has skyrocketed 631%, representing nearly half of Berkshire's portfolio today at a market value of $169 billion. As of this writing, Apple carries a gargantuan market cap of $2.9 trillion, making it the world's most valuable company. Even so, some investors might have their eyes on the iPhone maker as a potential portfolio addition. But is it too late to buy this FAANG stock right now? Let's analyze Apple with a fresh perspective to find the answer to that pressing question. Apple might be hitting a plateau With Apple having generated a whopping $383 billion in fiscal 2023, investors are right to wonder how much growth it really has left in the tank. That trailing-12-month sales figure was down nearly 3% on a year-over-year basis. However, to its credit, the declines have improved with each passing quarter throughout the last fiscal year. Nonetheless, Apple can't overcome the fact that more than half of its revenue still comes from a single product, the iPhone. While it might have been the case a decade ago to pay up for each new model that comes out every year, nowadays, the updated features are likely only marginally better, which allows consumers to hold onto their current smartphones for longer. This isn't what Apple executives want to see. But the management team is still optimistic, unsurprisingly. ""Our installed base of over 2 billion active devices continues to grow at a nice pace and establishes a solid foundation for the future expansion of the ecosystem,"" CFO Luca Maestri said on the fourth-quarter 2023 earnings call. Ongoing macro uncertainty will make things difficult, though, at least in the near term. The company's services segment, which includes Apple TV+, Pay, iCloud, and Music, is still posting solid gains. Revenue for the last fiscal year increased by 9%. But by representing just 22% of overall sales, it has a lot of work to do to make a sizable impact. I think it's out of the question to expect consistent double-digit top-line growth from Apple going forward. There is a caveat, though. If the business can introduce a game-changing product that can move the needle from a financial perspective, then maybe Apple can get a boost. An autonomous car is the only thing that comes to mind, but there are no concrete plans for this product. Arrows-Out Expand NASDAQ: AAPL Apple Today's Change (1.02%) $2.40 Current Price $237.33 Arrow-Thin-Down YTD1w1m3m6m1y 5y +274.11% Price VS S&P AAPL Key Data Points Market Cap $3,587B Day's Range $233.97 - $237.81 52wk Range $164.07 - $237.81 Volume 28,481,377 Avg Vol 48,560,301 Gross Margin 46.21% Dividend Yield 0.42% Show me the math There's no question that Apple is one of the most dominant companies the world has ever seen. This simple fact might prompt some eager investors to rush and buy the stock. However, you have to question whether shares can outperform the S&P 500 in the next five years. To get straight to the point, I don't believe they will. Right now, Apple's stock trades at a price-to-earnings ratio of 31.4. That's significantly more expensive than the S&P 500. And it's about 50% higher than Apple's trailing-10-year average multiple. If we assume that in five years, Apple's P/E ratio drops to 25, which seems like a reasonable valuation, it means that the company's earnings per share would need to increase by about 19% per year through fiscal 2028 for the stock to double for investors. This looks like a lofty expectation. Moreover, it's not out of the question for the valuation to drop even lower than that as Apple becomes larger and more mature. As things stand today, I believe it's too late to buy Apple stock."
             ])
