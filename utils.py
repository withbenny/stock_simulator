from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import AutoTokenizer
from urllib.parse import urlparse
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

class GetDataset:
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.data = None

    def read_json(self) -> None:
        with open(self.input_path, 'r') as file:
            self.data = json.load(file)
    
    def get(self, symbol: str) -> None:
        self.read_json()
        data = []
        for content in self.data['feed']:
            for sentiment in content['ticker_sentiment']:
                if sentiment['ticker'] == symbol:
                    formmated_time = datetime.strptime(content['time_published'], '%Y%m%dT%H%M%S')
                    data.append({
                        'symbol': symbol,
                        'url': content['url'],
                        'article': None,
                        'sentiment': sentiment['ticker_sentiment_label'],
                        'time': formmated_time
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_path, index=False)

class CombineDataset:
    def __init__(self) -> None:
        self.files_path = None
        self.output_path = None
    
    def multi_combine(self, files_path: str, output_path: str) -> None:
        self.files_path = files_path
        self.output_path = output_path
        files_list = sorted(glob.glob(self.files_path))
        combined_df = pd.DataFrame()
        for file in files_list:
            df = pd.read_csv(file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        combined_df.to_csv(self.output_path, index=False)
    
    def combine2(self, file_path1: str, file_path2: str, output_path: str) -> None:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
        combined_df = pd.concat([df1, df2], ignore_index=True)
        combined_df.to_csv(output_path, index=False)

    def results_combine(self, results_path: str, output_path: str) -> None:
        new_order = ["symbol", "url", "sentiment", "time", "source", "article"]
        results_list = sorted(glob.glob(results_path))
        merged_df = pd.DataFrame(columns=new_order)
        for result in results_list:
            df = pd.read_csv(result)
            check_cols = [col for col in new_order if col not in df.columns]
            if check_cols:
                print(f"File: {result} is missing columns: {', '.join(check_cols)}")
            df = df[new_order]
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        
        merged_df = merged_df.drop_duplicates(subset=['url'], keep='first')
        merged_df.to_csv(output_path, index=False)
    
    def remove_duplicates(self, input_path: str, output_path: str) -> None:
        df = pd.read_csv(input_path)
        df.drop_duplicates(subset=['url'], inplace=True)
        df.to_csv(output_path, index=False)
    
    def split_data(self, input_path: str, train_output_path: str, test_output_path: str) -> None:
        df = pd.read_csv(input_path)
        
        if 'sentiment' not in df.columns:
            raise ValueError("Sentiment column not found in dataset")

        sentiment_counts = df['sentiment'].value_counts()
        print(f"Original sentiment counts: {sentiment_counts}")
        min_count = min(sentiment_counts)

        target_counts = {
            'Bearish': min_count,
            'Bullish': min_count * 3,
            'Neutral': min_count * 4
        }

        train_dfs = []

        for sentiment, count in target_counts.items():
            subset = df[df['sentiment'] == sentiment]
            if len(subset) < count:
                raise ValueError(f"Not enough samples for sentiment '{sentiment}' to create the target dataset.")

            train_subset = subset.sample(count, random_state=42)
            train_dfs.append(train_subset)

        train_df = pd.concat(train_dfs, ignore_index=True)

        # Split train_df into train and test datasets (90%/10%)
        train_final_df, test_df = train_test_split(train_df, test_size=0.1, random_state=42)

        # Verify the sentiment distribution in both datasets
        train_sentiments_counts = train_final_df['sentiment'].value_counts()
        test_sentiments_counts = test_df['sentiment'].value_counts()

        print(f"Final train sentiment counts: {train_sentiments_counts}")
        print(f"Test sentiment counts: {test_sentiments_counts}")

        # Save to CSV files
        train_final_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

class WebCrawler:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        self.data = None
    
    def read_csv(self) -> None:
        self.data = pd.read_csv(self.dataset_path)
    
    def compute_source(self) -> pd.DataFrame:
        self.read_csv()
        if 'url' in self.data.columns:
            self.data['source'] = self.data['url'].apply(lambda x: urlparse(x).netloc)
            self.data.to_csv(self.dataset_path, index=False)
            domain_count = self.data['source'].value_counts()
            domain_summary = domain_count.reset_index()
            domain_summary.columns = ['source', 'count']

            print(domain_summary)
            return domain_summary
        else:
            raise ValueError("URL column not found in dataset")
    
    def get_article(self) -> None:
        self.read_csv()
        self.data['article'] = self.data.apply(lambda row: self.fetch_article(row['url'], row['source']), axis=1)
        self.data.to_csv(self.dataset_path, index=False)
    
    def fetch_article(self, url: str, source_domain: str) -> str:
        ua = UserAgent()
        headers = {
            "User-Agent": ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        options = Options()
        options.add_argument("--headless")
        skip = False
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            content = driver.page_source
            # response = requests.get(url, headers=headers, timeout=10)
            # response.raise_for_status()
            soup = BeautifulSoup(content, 'html.parser')

            if source_domain in ["www.fool.com", "www.benzinga.com", "www.globenewswire.com", "www.foxbusiness.com"]:
                div = soup.find('div', class_='article-body')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.zacks.com":
                div = soup.find('div', class_='commentary_body')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.investors.com":
                div = soup.find('div', class_='single-post-content')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.business-standard.com":
                div = soup.find('div', id='parent_top_div')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.marketwatch.com":
                # Will be blcoked
                div = soup.find('div', class_='column-full')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.investorideas.com":
                div = soup.find('div', class_='col-md-9')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())    
            elif source_domain == "www.forbes.com":
                # Limited 4 free articles
                div = soup.find('div', class_="article-body")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "moneymorning.com":
                div = soup.find('div', class_='single-content')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "markets.businessinsider.com":
                div = soup.find('div', class_='news-content')
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.kiplinger.com":
                div = soup.find('div', id="article-body")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain in ["www.cnn.com", "edition.cnn.com", "amp.cnn.com", "us.cnn.com"]:
                div = soup.find('div', class_="article__content")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.prnewswire.com":
                div = soup.find('div', class_="col-lg-10")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "investingnews.com":
                div = soup.find('div', class_="body-description")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "apnews.com":
                div = soup.find('div', class_="RichTextStoryBody")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "cointelegraph.com":
                div = soup.find('div', class_="post-content")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.financialexpress.com":
                div = soup.find('div', id="pcl-full-content")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "stocknews.com":
                div = soup.find('div', class_="post_content")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "aap.thestreet.com":
                # Will be blocked
                div = soup.find('div', class_="m-detail--body")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.moneycontrol.com":
                div = soup.find('div', class_="content_wrapper")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.businessinsider.com":
                div = soup.find('div', class_="content-lock-content")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "decrypt.co":
                div = soup.find('div', class_="grid-cols-1")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "www.newswire.ca":
                div = soup.find('div', class_="col-lg-10")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "cfo.economictimes.indiatimes.com":
                div = soup.find('div', class_="article-section__body__news")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())   
            elif source_domain in ["theweek.com", "www.ft.com", "www.kiplinger.com"]:
                div = soup.find('div', id="article-body")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "africa.businessinsider.com":
                div = soup.find('div', class_="container-wrapper")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            elif source_domain == "":
                div = soup.find('div', class_="")
                if div:
                    article = div.text.strip()
                    article = " ".join(article.split())
            else:
                print(f"Source domain {source_domain} not supported. Skipping...")
                skip = True
                pass
            print(article)
            driver.quit()
            return article
        except Exception as e:
            if not skip:
                print(f"You might be blocked by {source_domain} or url: {url} is wrong. Error: {e}")
            else:
                print(f"Error: {e}")
            return None
    
    def remove_resources(self, output_path: str = None) -> None:
        self.read_csv()
        # Pay wall: www.cnbc.com, www.barrons.com, www.economist.com, www.wsj.com
        # Website down: stockmarket.com,
        # Hard to get article: www.benzinga.com
        # Error: consent.google.com
        sources_to_remove = ["www.cnbc.com", "stockmarket.com", "www.barrons.com", "www.benzinga.com", "www.economist.com", "www.wsj.com",
                             "consent.google.com"]
        if output_path is None:
            output_path = self.dataset_path.replace(".csv", "_cleaned.csv")
        if 'source' in self.data.columns:
            self.data = self.data[~self.data['source'].isin(sources_to_remove)]
        
        with open(output_path, 'w', newline='') as file:
            self.data.to_csv(file, index=False)
    
    def get_max_token_length(self) -> int:
        self.read_csv()
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        if 'article' in self.data.columns:
            token_lengths = self.data['article'].apply(lambda x: len(tokenizer.tokenize(x)))
            plt.figure(figsize=(20, 8))
            plt.hist(token_lengths, bins=50, alpha=0.75)
            plt.xlabel('Token Length')
            plt.ylabel('Frequency')
            plt.title('Token Length Distribution')
            plt.show()
            max_length = int(np.percentile(token_lengths, 90))
        else:
            max_length = 0
            
        print(f"Max token length: {max_length}")
        return max_length

    def dataset_filter(self, min_length: int = 20, max_length: int = None, output_path: str = None) -> None:
        self.read_csv()
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        if 'article' not in self.data.columns:
            raise ValueError("Article column not found in dataset")
        
        self.data['sentiment'] = self.data['sentiment'].replace({
            'Somewhat-Bearish': 'Bearish',
            'Somewhat-Bullish': 'Bullish'
        })

        if max_length is None:
            filtered_df = self.data[self.data['article'].notnull() & 
                                    (self.data['article'].apply(lambda x: len(tokenizer.tokenize(x)) if isinstance(x, str) else 0) > min_length)]
        else:
            filtered_df = self.data[self.data['article'].notnull() & 
                                    (self.data['article'].apply(lambda x: len(tokenizer.tokenize(x)) if isinstance(x, str) else 0) > min_length) & 
                                    (self.data['article'].apply(lambda x: len(tokenizer.tokenize(x)) if isinstance(x, str) else 0) < max_length)]
        
        if output_path is None:
            output_path = self.dataset_path.replace(".csv", "_filtered.csv")
        filtered_df.to_csv(output_path, index=False)

    def get_empty(self) -> None:
        self.read_csv()
        empty_articles = self.data[self.data['article'].isna() | (self.data['article'] == '')]
        empty_articles.to_csv(self.dataset_path.replace(".csv", "_empty.csv"), index=False)

    def count_sentiment(self) -> None:
        self.read_csv()
        sentiment_count = self.data['sentiment'].value_counts()
        bearish = sentiment_count.get('Bearish', 0)
        bullish = sentiment_count.get('Bullish', 0)
        neutral = sentiment_count.get('Neutral', 0)

        print(f"Bearish: {bearish}, \nBullish: {bullish}, \nNeutral: {neutral}")