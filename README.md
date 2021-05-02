# For the dataloading, there are two methods

## First method

1. Get the API from Polenix
```bash
# Use the public API (PREFERRED)
from urllib.request import urlopen
import requests
df = requests.get('https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1405699200&end=9999999999&period=14400')
df_json = df.json()
df2 = pd.DataFrame(df_json)
df2.head()
df2.to_csv('project_df.csv', encoding = 'utf-8')

# Get Private API
pip install Polenix
polo = Poloniex()
API_KEY = "[your api]"
API_SECRET = '[your api]'
api_key = os.environ.get(API_KEY)
api_secret = os.environ.get(API_SECRET)
polo = Poloniex(api_key, api_secret)
ticker = polo.returnTicker()['BTC_ETH']

````

## Second method
2. Download data folder above
I imported data from the Polenix and exported into CSV format. Save the dataset name as df2

# Fast start
After data import, recommend to run lstm_keras_Vfinal.py -> rnn_keras_VFinal.py -> gru_keras_VFinal.py.
