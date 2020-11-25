#!/usr/bin/env python
# coding: utf-8

# In[6]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_dataReader.web as web

style.use('ggplot')
start = dt.datetime(2020,1,1)
end = dt.datetime(2020,5,12)

df=web.dataReader('TSLA', 'yahoo', start, end)
print(df.head())


# In[9]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as pd

style.use('ggplot')
start = dt.datetime(2020,1,1)
end = dt.datetime(2020,5,12)

df=pd.datareader('TSLA', 'yahoo', start, end)
print(df.head())


# In[10]:


import pandas as pd


# In[13]:


import pandas_datareader as dr


# In[14]:


get_ipython().system('pip install pandas_datareader')


# In[16]:


import pandas_datareader as dr


# In[23]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as dr

style.use('ggplot')
start = dt.datetime(2020,1,1)
end = dt.datetime(2020,5,12)

tsla=dr.get_data('TSLA', 'yahoo', start, end)
print(tsla.head())


# In[25]:


from pandas_datareader import data
import datetime as dt
ticker = 'GLD'
begdate = '2014-11-11'
enddate = '2016-11-11'
data1 = data.DataReader(ticker,'yahoo',dt.datetime(2014,11,11),dt.datetime(2016,11,11))
print(data1)


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')

my_year_month_fmt = mdates.DateFormatter('%m/%y')

data = pd.read_pickle('./data.pkl')
data.head(10)


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
modes = ['full', 'same', 'valid']
for m in modes:
    plt.plot(np.convolve(np.ones((200,)), np.ones((50,))/50, mode=m));
plt.axis([-10, 251, -.1, 1.1]);
plt.legend(modes, loc='lower center');
plt.show()


# In[29]:


import talib as ta
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import time as t

PAIR = info.primary_pair
PERIOD = 30

def initialize():
    storage.reset()
    storage.elapsed = storage.get('elapsed', [0,0,0,0,0,0])

def cumsum_sma(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

def pandas_sma(array, period):
    return pd.rolling_mean(array, period)

def api_sma(array, period):
    # this method is native to Tradewave and does NOT return an array
    return (data[PAIR].ma(PERIOD))

def talib_sma(array, period):
    return ta.MA(array, period)
def convolve_sma(array, period):
    return np.convolve(array, np.ones((period,))/period, mode='valid')

def fftconvolve_sma(array, period):    
    return scipy.signal.fftconvolve(
        array, np.ones((period,))/period, mode='valid')    

def tick():

    close = data[PAIR].warmup_period('close')

    t1 = t.time()
    sma_api = api_sma(close, PERIOD)
    t2 = t.time()
    sma_cumsum = cumsum_sma(close, PERIOD)
    t3 = t.time()
    sma_pandas = pandas_sma(close, PERIOD)
    t4 = t.time()
    sma_talib = talib_sma(close, PERIOD)
    t5 = t.time()
    sma_convolve = convolve_sma(close, PERIOD)
    t6 = t.time()
    sma_fftconvolve = fftconvolve_sma(close, PERIOD)
    t7 = t.time()

    storage.elapsed[-1] = storage.elapsed[-1] + t2-t1
    storage.elapsed[-2] = storage.elapsed[-2] + t3-t2
    storage.elapsed[-3] = storage.elapsed[-3] + t4-t3
    storage.elapsed[-4] = storage.elapsed[-4] + t5-t4
    storage.elapsed[-5] = storage.elapsed[-5] + t6-t5    
    storage.elapsed[-6] = storage.elapsed[-6] + t7-t6        

    plot('sma_api', sma_api)  
    plot('sma_cumsum', sma_cumsum[-5])
    plot('sma_pandas', sma_pandas[-10])
    plot('sma_talib', sma_talib[-15])
    plot('sma_convolve', sma_convolve[-20])    
    plot('sma_fftconvolve', sma_fftconvolve[-25])

def stop():

    log('ticks....: %s' % info.max_ticks)

    log('api......: %.5f' % storage.elapsed[-1])
    log('cumsum...: %.5f' % storage.elapsed[-2])
    log('pandas...: %.5f' % storage.elapsed[-3])
    log('talib....: %.5f' % storage.elapsed[-4])
    log('convolve.: %.5f' % storage.elapsed[-5])    
    log('fft......: %.5f' % storage.elapsed[-6])


# In[31]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as pd
from pandas_datareader import data

start_date = dt.datetime(2020,1,1)
end_date = dt.datetime(2020,5,12)
ticker='AMZN'

data = data.get_data_yahoo(ticker, start_date, end_date)
data.head()


# In[32]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data['Adj Close'].plot()
plt.show()


# In[33]:


# Plot the adjusted close price
data['Adj Close'].plot(figsize=(10, 7))
# Define the label for the title of the figure
plt.title("Adjusted Close Price of %s" % ticker, fontsize=16)
# Define the labels for x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)
# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
# Show the plot
plt.show()


# In[35]:


# Define the figure size for the plot
plt.figure(figsize=(10, 7))
# Plot the adjusted close price
data['Adj.Close'].plot()
# Define the label for the title of the figure
plt.title("Adjusted Close Price of %s" % ticker, fontsize=16)
# Define the labels for x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)
# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()


# In[36]:


# Plot all the close prices
data.plot(figsize=(10, 7))
# Show the legend
plt.legend()
# Define the label for the title of the figure
plt.title("Adjusted Close Price", fontsize=16)
# Define the labels for x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)
# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()


# In[40]:


import pandas as pd
import technicalanalysis as ta
from ta.utils import dropna
from ta.volatility import BollingerBands


# Load datas
df = pd.read_csv('ta/tests/data/datas.csv', sep=',')

# Clean NaN values
df = dropna(df)

# Initialize Bollinger Bands Indicator
indicator_bb = BollingerBands(close=df["Close"], n=20, ndev=2)

# Add Bollinger Bands features
df['bb_bbm'] = indicator_bb.bollinger_mavg()
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()

# Add Bollinger Band high indicator
df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

# Add Bollinger Band low indicator
df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

# Add Width Size Bollinger Bands
df['bb_bbw'] = indicator_bb.bollinger_wband()

# Add Percentage Bollinger Bands
df['bb_bbp'] = indicator_bb.bollinger_pband()


# In[42]:


import pandas as pd
import numpy as np
from pandas_datareader import data as web
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def get_stock(stock,start,end):
 return web.DataReader(stock,'google',start,end)['Close']
def RSI(series, period):
 delta = series.diff().dropna()
 u = delta * 0
 d = u.copy()
 u[delta > 0] = delta[delta > 0]
 d[delta < 0] = -delta[delta < 0]
 u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
 u = u.drop(u.index[:(period-1)])
 d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
 d = d.drop(d.index[:(period-1)])
 rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) /  pd.stats.moments.ewma(d, com=period-1, adjust=False)
 return 100 - 100 / (1 + rs)
df = pd.DataFrame(get_stock('FB', '1/1/2016', '12/31/2016'))
df['RSI'] = RSI(df['Close'], 14)
df.tail()
df.plot(y=['Close'])
df.plot(y=['RSI'])


# In[44]:


import pandas as pd
import numpy as np
from pandas_datareader import data as web
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def get_stock(stock,start,end):
 return web.DataReader(stock,'yahoo',start,end)['Close']
 
def RSI(series, period):
 delta = series.diff().dropna()
 u = delta * 0
 d = u.copy()
 u[delta > 0] = delta[delta > 0]
 d[delta < 0] = -delta[delta < 0]
 u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
 u = u.drop(u.index[:(period-1)])
 d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
 d = d.drop(d.index[:(period-1)])
 rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) /  pd.stats.moments.ewma(d, com=period-1, adjust=False)
 return 100 - 100 / (1 + rs)
 
df = pd.DataFrame(get_stock('FB', '1/1/2016', '12/31/2016'))
df['RSI'] = RSI(df['Close'], 14)
df.tail()


# In[47]:


import numpy as np
import numba as nb

nb.jit(fastmath=True, nopython=True)   
def calc_rsi( array, deltas, avg_gain, avg_loss, n ):

    # Use Wilder smoothing method
    up   = lambda x:  x if x > 0 else 0
    down = lambda x: -x if x < 0 else 0
    i = n+1
    for d in deltas[n+1:]:
        avg_gain = ((avg_gain * (n-1)) + up(d)) / n
        avg_loss = ((avg_loss * (n-1)) + down(d)) / n
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            array[i] = 100 - (100 / (1 + rs))
        else:
            array[i] = 100
        i += 1

    return array

def get_rsi( array, n = 14 ):   

    deltas = np.append([0],np.diff(array))

    avg_gain =  np.sum(deltas[1:n+1].clip(min=0)) / n
    avg_loss = -np.sum(deltas[1:n+1].clip(max=0)) / n

    array = np.empty(deltas.shape[0])
    array.fill(np.nan)

    array = calc_rsi( array, deltas, avg_gain, avg_loss, n )
    return array
rsi = get_rsi(array or series, 14 )


# In[48]:


import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests

style.use('ggplot')


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# save_sp500_tickers()
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'morningstar', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')


def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


visualize_data()


# In[ ]:




