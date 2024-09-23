import re
import csv
from datetime import datetime
import json
import math
import os
import re
import time
from collections import namedtuple
from typing import Tuple
from enum import Enum
import requests
import analysis
INIT_MONEY = 100_000

class ConfigLoader:
    def __init__(self, config_path:str, source:str) -> None:
        self.config_path = config_path
        self.source = source
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def get_api_key(self) -> str:
        source = self.source
        return self.config[source]['api_key']
    
    def get_api_url(self) -> str:
        source = self.source
        return self.config[source]['api_url']

class GetTicker:
    def __init__(self, symbol:str, config_loader:ConfigLoader) -> None:
        self.symbol = symbol
        self.config_loader = config_loader
        self.api_url = self.config_loader.get_api_url()
        self.api_key = self.config_loader.get_api_key()
        self.current_year = datetime.now().year
        self.current_month = datetime.now().month
        self.time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def latestData(self, date=None, interval:str='5min', time_series:str='TIME_SERIES_INTRADAY') -> None:
        if self.config_loader.source == 'alphavantage':
            if date is None:
                url = f"{self.api_url}query?function={time_series}&symbol={self.symbol}&interval={interval}&outputsize=full&apikey={self.api_key}&datatype=csv"
                response = requests.get(url)
                data = response.text
            
            else:
                if not re.match(r'\d{4}-\d{2}', date) :
                    raise ValueError("Invalid date format.")
                
                year, month = int(date.split('-')[0]), int(date.split('-')[1])
                if year < 2000 or month < 1 or month > 12:
                    raise ValueError("Invalid date. The API can only get data after 2000-01.")
                elif year > self.current_year or (year == self.current_year and month > self.current_month):
                    raise ValueError("You cannot get future data. Current date is " + str(self.current_year) + '-' + str(self.current_month) + '.')
                
                url = f"{self.api_url}query?function={time_series}&symbol={self.symbol}&interval={interval}&month={date}&outputsize=full&apikey={self.api_key}&datatype=csv"
                response = requests.get(url)
                data = response.text
                self.time_str = date

            file_path = f'{self.symbol}_{self.time_str}.csv'
            with open(file_path, 'w') as f:
                f.write(data)

            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)

            data[0].append('unix_timestamp')
            for row in data[1:]:
                if len(row) < 1:
                    continue
                dt = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                unix_timestamp = int(time.mktime(dt.timetuple()))
                row.append(unix_timestamp)

            data = [data[0]] + data[1:][::-1]
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data)

    def longData(self, start_date=None, end_date=None, interval:str='5min') -> None:
        if self.config_loader.source == 'alphavantage':
            if start_date is None:
                start_date = f'{self.current_year}-01'
            if end_date is None:
                end_date = f'{self.current_year}-{self.current_month}'

            if not re.match(r'\d{4}-\d{2}', start_date) or not re.match(r'\d{4}-\d{2}', end_date):
                raise ValueError("Invalid date format.")
            
            start_year, start_month = int(start_date.split('-')[0]), int(start_date.split('-')[1])
            end_year, end_month = int(end_date.split('-')[0]), int(end_date.split('-')[1])

            for year in range(start_year, end_year+1):
                for month in range(1, 13):
                    if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                        continue
                    elif year == self.current_year and month > self.current_month:
                        break

                    date = f'{year}-{month:02}'
                    self.latestData(date, interval)

    def getNews(self, topics:str=None, time_from:int=None, time_to:int=None, num:int=50, sort:str='LATEST') -> None:
        if time_from is not None:
            _time_from  = f"&from={time_from}"
        else:
            _time_from = ''       
        if time_to is not None:
            _time_to = f"&to={time_to}"
        else:
            _time_to = ''       
        if topics is not None:
            _topics = f"&topics={topics}"
        else:
            _topics = ''

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.symbol}{_topics}{_time_from}{_time_to}&limit={num}&sort={sort}&apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        
        with open(f'{self.symbol}_news{self.time_str}.json', 'w') as f:
            json.dump(data, f, indent=4)
    
    def newsAnalysis(self, news_path:json=None) -> dict:
        with open(news_path, 'r') as f:
            news = json.load(f)
        result = analysis.NewsAnalysis(news)
        
        return result.get_summary()

class Simulate:
    # Data is stored in username.csv
    def __init__(self, username: str, config_loader: ConfigLoader) -> None:
        self.username = username
        self.holdings = {}
        self.config_loader = config_loader

        # username can only be numbers[0-9] and letters[a-z, A-Z]
        # username cannot start with a number
        # username must be longer than 5
        if not self.isValidName(username):
            raise ValueError("Invalid username.")

        # If username is new, create a new file
        if not os.path.exists(username + '.csv'):
            with open(username + '.csv', 'w', newline='') as csvfile:
                headers = ['symbol', 'quantity']
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerow({'symbol': username + 'CASH', 'quantity': INIT_MONEY})

        # If username exists, load the data
        else:
            with open(username + '.csv', 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    symbol = row['symbol']
                    quantity = row['quantity']
                    self.holdings[symbol] = quantity

                if self.username + 'CASH' in self.holdings:
                    self.cash = self.holdings.get(self.username + 'CASH')
                else:
                    self.cash = INIT_MONEY
                    with open(username + '.csv', 'w', newline='') as csvfile:
                        headers = ['symbol', 'quantity']
                        writer = csv.DictWriter(csvfile, fieldnames=headers)
                        writer.writeheader()
                        writer.writerow({'symbol': username + 'CASH', 'quantity': self.cash})
        print(str(self.holdings))

    def isValidName(self, username: str) -> bool:
        pattern = r'^[a-zA-Z][a-zA-Z0-9]*$'
        if re.match(pattern, username):
            if len(username) >= 5:
                return True
            else:
                return False
        else:
            return False

    def currentCash(self) -> float:
        return self.cash

    def trade(self, symbol: str, mode: str, share: float) -> None:
        if share <= 0:
            raise ValueError("Invalid values.")

        self.cash = self.holdings.get(self.username + 'CASH')
        cash = self.cash
        cash = float(cash)

        # share keeps four decimal places to support fractional shares
        share = math.floor(share * 10000) / 10000

        # BUY
        # Ignore the case of 'buy' and 'sell'
        if mode.casefold() == 'buy':
            price, _, _ = GetTicker(symbol, self.config_loader).latestPrice(GetTicker(symbol, self.config_loader).latestData())
            cost = price * share
            if cash - cost >= 0:
                cash = cash - cost
                # Change the current cash
                self.holdings[self.username + 'CASH'] = cash

                if symbol in self.holdings:
                    self.holdings[symbol] = math.floor((float(self.holdings.get(symbol)) + share) * 10000) / 10000
                else:
                    self.holdings[symbol] = math.floor(share * 10000) / 10000
            else:
                raise ValueError("Invalid action.")

        # SELL    
        elif mode.casefold() == 'sell':
            price, _, _ = GetTicker(symbol, self.config_loader).latestPrice(GetTicker(symbol, self.config_loader).latestData())
            if symbol in self.holdings:
                before = self.holdings.get(symbol)
                if share <= before:
                    income = price * share
                    cash = cash + income
                    # Change the current cash
                    self.holdings[self.username + 'CASH'] = cash
                    if share == before:
                        del self.holdings[symbol]
                    else:
                        self.holdings[symbol] = math.floor((before - share) * 10000) / 10000
                else:
                    raise ValueError("Invalid action.")
            else:
                raise ValueError("Invalid action.")

        # Other MODE
        else:
            raise ValueError("Invalid trade mode.")

        print(str(self.holdings))
        self.saveData()

    def saveData(self) -> None:
        # Save the data
        with open(self.username + '.csv', 'w', newline='') as csvfile:
            headers = ['symbol', 'quantity']
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            # Save the cash
            writer.writerow({'symbol': self.username + 'CASH', 'quantity': self.holdings[self.username + 'CASH']})
            # Save the holdings
            for stock_symbol, quantity in self.holdings.items():
                if stock_symbol != self.username + 'CASH':
                    writer.writerow({'symbol': stock_symbol, 'quantity': quantity})