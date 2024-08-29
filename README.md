**This project is NOT finished yet!**

## How to USE
is NOT finished yet!
## Data Explanation
### 1. API

You can see the example of [api data](examples\api.example.json) in ```/examples``` folder.

The translation of its important part as follows:
| | 中文 |
| :--- | :--- |
| "region"| 發行國 |
| "quoteType" | 報價類型 |
| "currency" | 貨幣 |
| "bid" | 買價 |
| "ask" | 賣價 |
| "regularMarketOpen" | 開盤價 |
| "averageDailyVolume3Month" | 三個月平均日交易量 |
| "tradeable" | 交易狀態 |
| "regularMarketTime" | 市場時間 |
| "regularMarketDayHigh/Low" | 今日最高/低價 |
| "regularMarketVolume" | 今日交易量 |
| "regularMarketPreviousClose" | 前日收盤價 |
| "regularMarketPrice" | 現在價格 |
| "marketState" | 市場狀態 |
| "regularMarketChangePercent" | 市場變化 |
| "symbol" | 代碼 |


## Standard of I/O
![](images/flow.drawio_1.png)
### 1. Financial News Standard
| | type | 中文 |
| :--- | :--- | :--- |
| Title | / | 標題 |
| Content | / | 內文 |
| --- | --- | --- | 
| Date | date | 日期 |
| Source | str | 來源 |
| Symbol/Industry | str | 股票/行業 |
| Class | str | 類型 |
| Weight | .2f | 權重 |

### 2. Simulator Standard
For Files:
| | type |
| :--- | :--- |
| Stock Data from Stock API | json -> dict | 
| Trade Data from USER | csv |
| History Data^ | csv |

NOTE: ^ History Data can be download from [Yahoo Finanace HK](https://hk.finance.yahoo.com/quote/AAPL/history?p=AAPL), but only for DAILY DATA. 

---

For Stock API:

is NOT finished yet!


For USER Trade data:
| | type | 中文 |
| :--- | :--- | :--- |
| username^^ | str | 用戶名 |
| symbol | str | 股票代碼 |
| quantity | .1f | 持有量 |

NOTE: ^^ Username is filename, follow the rule as below:
1. Username starts with alphabetic character (either lowercase and upper case).
2. Username can ONLY uses alphabetic characters and digits.
3. In symbol, username + CASH is the user's CASH.

---

For History Data Infomation:
| | type | 中文 |
| :--- | :--- | :--- |
| Date | YYYY-MM-DD(date) | 年-月-日 |
| Time Range | Trading Days(int) | 總交易日天數 | 
| Symbol | str | 股票代碼 |
| Currency | str | 貨幣 |
| Rate | data per day(/hour/minute) | 每天(/時/分)資料 |
| Industry Type (GICS)^^^ | e.g. 15104050 (Steel) | 行業類型 |

NOTE: ^^^ Industry types use the Sub-Industry of [GICS](https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard).

For Yahoo History Data CSV:
| | type | 中文 |
| :--- | :--- | :--- |
| Date | date | 日期 |
| Open | .2f | 開盤價 |
| High | .2f | 最高價 |
| Low | .2f | 最低價 |
| Close | .2f | 收盤價 |
| AdjClose | .5f | 調整收盤價 | 
| Volume | int | 交易量 |
