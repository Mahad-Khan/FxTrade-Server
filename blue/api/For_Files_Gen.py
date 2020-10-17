import os
import pprint
import talib
import sys
import numpy as np
from scipy import stats
import plotly.express as px
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import cufflinks as cf
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso
from itertools import cycle, islice
import statsmodels.api as sm
#import plotly.offline as py
import chart_studio.plotly as py
cf.go_offline()
init_notebook_mode(connected=True)
import plotly.graph_objects as go
from matplotlib import pyplot
import matplotlib.ticker as ticker
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pyplot import figure
import matplotlib
import matplotlib.pyplot as plt
import ppscore as pps
MEDIUM_SIZE = 10
plt.rc('axes', labelsize=MEDIUM_SIZE) 
from matplotlib.pyplot import figure
import datetime
from datetime import datetime
import glob
import csv
from pyexcel.cookbook import merge_all_to_a_book
import pandas as pd
global gpath
from indicators import (adx, macd, macd_percentile, moving_average,
                        plot_agg_divergence, plot_agg_momentum, plot_data, pnl,
                        pnl_n, pnl_percentile, rsi, rsi_percentile, williams,cci_percentile)
from oanda import get_candles
specific_insts=None
ggpath=None
plt.style.use('ggplot')
from server_function import date_to_list

PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data/daily/v8'
)

DATA_FILES = {
    'Open': 'open.xls',
    'High': 'high.xls',
    'Low': 'low.xls',
    'Close': 'close.xls',
    'Volume': 'volume.xls',
    'Date': 'date.xls',
}

COLUMNS = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
]

SORT_COLUMN = 'adx'

# start and end period in days
START_PERIOD = -100
END_PERIOD = -1
# last day
SELECTED_PERIOD = -1
SELECTED_INSTRUMENT = 'AUD_CAD'


# OANDA API variables
INSTRUMENTS = [
    'AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD', 'AUD_SGD',
    'AUD_USD', 'BCO_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CAD_SGD',
    'CHF_HKD', 'CHF_JPY', 'CHF_ZAR', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF',
    'EUR_CZK', 'EUR_DKK', 'EUR_GBP', 'EUR_HKD', 'EUR_HUF', 'EUR_JPY',
    'EUR_NOK', 'EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_SGD', 'EUR_TRY',
    'EUR_USD', 'EUR_ZAR', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_HKD',
    'GBP_JPY', 'GBP_NZD', 'GBP_PLN', 'GBP_SGD', 'GBP_USD', 'GBP_ZAR',
    'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD', 'NZD_JPY', 'NZD_SGD',
    'NZD_USD', 'SGD_CHF', 'SGD_HKD', 'SGD_JPY', 'TRY_JPY', 'USB02Y_USD',
    'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'USD_CAD', 'USD_CHF',
    'USD_CNH', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR',
    'USD_JPY', 'USD_MXN', 'USD_NOK', 'USD_PLN', 'USD_SAR', 'USD_SEK',
    'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'ZAR_JPY',
]
# https://developer.oanda.com/rest-live-v20/instrument-df/#CandlestickGranularity
GRANULARITY = [
    'S5', #5 second candlesticks, minute alignment
    'S10', #10 second candlesticks, minute alignment
    'S15', # 15 second candlesticks, minute alignment
    'S30', # 30 second candlesticks, minute alignment
    'M1', #  1 minute candlesticks, minute alignment
    'M2', #  2 minute candlesticks, hour alignment
    'M4', #  4 minute candlesticks, hour alignment
    'M5', #  5 minute candlesticks, hour alignment
    'M10', # 10 minute candlesticks, hour alignment
    'M15', # 15 minute candlesticks, hour alignment
    'M30', # 30 minute candlesticks, hour alignment
    'H1', #  1 hour candlesticks, hour alignment
    'H2', #  2 hour candlesticks, day alignment
    'H3', #  3 hour candlesticks, day alignment
    'H4', #  4 hour candlesticks, day alignment
    'H6', #  6 hour candlesticks, day alignment
    'H8', #  8 hour candlesticks, day alignment
    'H12', # 12 hour candlesticks, day alignment
    'D',# 1 day candlesticks, day alignment
    'W', #  1 week candlesticks, aligned to start of week
    'M', # 1 month candlesticks, aligned to first day of the month
]

PERIODS = 500

def get_file_data():
    '''
    Read the data from the csv or excel files
    Key is olhc
    '''
    values = {}
    for key, filename in DATA_FILES.items():
        path = os.path.join(PATH, filename)
        if not os.path.exists(path):
            print(f'file {path} not found')
            continue
        if key not in values:
            values[key] = None
        if filename.endswith('.csv'):
            values[key] = pd.read_csv(path)
        elif filename.endswith('.xls'):
            values[key] = pd.read_excel(path)
    return values


def get_oanda_api(
    instruments,
    granularity='D',
    prices=['B'],
    since=None,
    until=None,
    count=PERIODS,
    daily_alignment=17,
):
    # https://developer.oanda.com/rest-live-v20/instrument-ep
    params = {
        'granularity': granularity,
        'prices': prices,
        'since': since,
        'until': until,
        'count': count,
        'dailyAlignment': daily_alignment,
        'alignmentTimezone': 'Europe/London',
    }
    return get_candles(instruments, params)


def get_ig_api(
    instruments,
    granularity='D',
    prices=['B'],
    since=None,
    until=None,
    count=PERIODS
):
    params = {
        'granularity': granularity,
        'prices': prices,
        'since': since,
        'until': until,
        'count': count
    }
    return get_candles(instruments, params)


dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
gpath = dirname + "/api/"


def rank_formulation(array):
    values=[]
    for index in range(len(array)):
        value = array[index]
        compare_array = array[:index]
        rank = stats.percentileofscore(compare_array,value)
        rank = rank / 100
        values.append(rank)
    return values

def change_in_price(price):
    cprice=[]
    #cprice.append(price[0])
    #price=np.delete(price,0)
    for i in range(0,len(price)):
        if i==0:
            cprice.append(price[i])
        else:
            a=np.log(price[i]/price[i-1])
            cprice.append(a)
    return np.array(cprice)

def instrument_selection_rules(df):
    top5=df.head(5)
    last5=df.tail(5)
    top5_indicies = top5[(top5['RSI']<30)].index.tolist()
    last5_indicies=last5[(last5['RSI']>70)].index.tolist()
    return top5_indicies,last5_indicies


def Function_for_file_generation():
    global specific_insts
    global ggpath
    ggpath=gpath
    debug = False
    if not debug:
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.width', 1000)
    # user_input=input("Do You Want Select Instrument Manually:y or n")
    # man_inst_names=None
    # if user_input=="y":
    #     man_inst_names=input("Enter Instrument names Separated by Space:")
    #     man_inst_names = man_inst_names.split()

    

    
    instruments = INSTRUMENTS
    # # instruments = ['AUD_CAD',]
    # # instruments = ['AUD_CHF',]
    # instruments = ['AUD_CHF', 'AUD_CAD']
    # instruments = ['AUD_CAD',]
    #data = get_file_data()
    stock = get_oanda_api(instruments, granularity='D')
  

    # print (stock['AUD_CAD'])

    # stock = get_ig_api(instruments)
    # print (stock['AUD_CAD'])


    # instruments = data['Close'].columns.values
    # # Initialize all assign all instrument data to dataframes
    # stock = {}
    # for instrument in instruments:
    #     values = {}
    #     for key in COLUMNS:
    #         values[key] = data.get(key, {}).get(instrument)
    #     values['Date'] = data.get('Date').iloc[:len(values[key]), 0]
    #     stock[instrument] = pd.DataFrame(values, columns=COLUMNS)

    # print(stock[SELECTED_INSTRUMENT])
    # return
    # Calculate the MACD, RSI and Profit and Loss for all instrument paid
    # Also, Calculate the MACD, RSI and Profit and Loss percentile for all
    # instruments
    instruments_list=[]
    CCI_list=[]
    dic_for_all_cci={}

    for instrument in instruments:
        
    
        nsize = len(stock[instrument]['Close'])

        # Calculate MACD
        stock[instrument] = stock[instrument].join(
            macd(stock[instrument]['Close'])
        )


        # Calculate RSI for n = 14
        stock[instrument] = stock[instrument].join(
            rsi(stock[instrument]['Close'])
        )

        #changeInPrice
        stock[instrument]["Change In Price"] = change_in_price(stock[instrument]["Close"].values)

    

        # Calculate Profile and Loss
        stock[instrument] = stock[instrument].join(
            pnl(stock[instrument]['Close'])
        )
        # Calculate MACD Percentile
        stock[instrument] = stock[instrument].join(
            macd_percentile(stock[instrument]['MACD'])
        )
        # Calculate RSI Percentile
        stock[instrument] = stock[instrument].join(
            rsi_percentile(stock[instrument]['RSI'])
        )
        # Calculate  Profile and Loss Percentile
        stock[instrument] = stock[instrument].join(
            pnl_percentile(stock[instrument]['Profit/Loss'])
        )



        #Calculate CCI
        high = stock[instrument]["High"].values
        close = stock[instrument]["Close"].values
        low = stock[instrument]["Low"].values
        #create instrument dataframe
        ccis=talib.CCI(high,low,close,timeperiod=14)
        #ccis=list(ccis)
        instruments_list.append(instrument)
        CCI_list.append(ccis[-1])
        dic_for_all_cci[instrument]=ccis
        stock[instrument]["CCI"]=ccis
      

        # Calculate Divergence factor 1 and 2
        stock[instrument] = stock[instrument].join(
            pd.Series(
                (
                    stock[instrument]['MACD Percentile'] + 0.1 -
                    stock[instrument]['RSI Percentile']
                ) / 2.0,
                name='Divergence Factor 1'
            )
        )
        stock[instrument] = stock[instrument].join(
            pd.Series(
                stock[instrument]['Divergence Factor 1'] -
                stock[instrument]['PNL Percentile'],
                name='Divergence Factor 2'
            )
        )
        
        # Calculate Divergence factor 3
        n = 19
        for i in range(nsize):
            stock[instrument].loc[i: nsize, 'Macd_20'] = (
                stock[instrument]['MACD'].iloc[i] -
                stock[instrument]['MACD'].iloc[i - n]
            )
            stock[instrument].loc[i: nsize, 'Prc_20'] = (
                (stock[instrument]['Close'].iloc[i] -
                    stock[instrument]['Close'].iloc[i - n])
            ) / stock[instrument]['Close'].iloc[i - n]
            stock[instrument].loc[i: nsize, 'Divergence Factor 3'] = (
                stock[instrument]['Macd_20'].iloc[i] /
                stock[instrument]['Close'].iloc[i]
            ) - stock[instrument]['Prc_20'].iloc[i]

        stock[instrument] = stock[instrument].join(
            rsi(stock[instrument]['Close'], 20, name='RSI_20')
        )

        # Calculate the momentum factors
        stock[instrument] = stock[instrument].join(
            pnl_n(stock[instrument]['Close'], 10)
        )
        stock[instrument] = stock[instrument].join(
            pnl_n(stock[instrument]['Close'], 30)
        )

        stock[instrument]['Close_fwd'] = stock[instrument]['Close'].shift(-2)
        stock[instrument].loc[-1: nsize, 'Close_fwd'] = stock[instrument]['Close'].iloc[-1]
        stock[instrument].loc[-2: nsize, 'Close_fwd'] = stock[instrument]['Close'].iloc[-2]

        stock[instrument] = stock[instrument].join(
            macd(
                stock[instrument]['Close_fwd'],
                name='MACD_fwd'
            )
        )
        n = 19
        stock[instrument] = stock[instrument].join(
            pd.Series(
                stock[instrument]['MACD_fwd'].diff(n) - stock[instrument]['MACD'],
                name='M_MACD_CHANGE'
            )
        )

        stock[instrument] = stock[instrument].join(
            rsi(stock[instrument]['Close_fwd'], n=20, name='RSI_20_fwd')
        )
        stock[instrument] = stock[instrument].join(
            pd.Series(
                stock[instrument]['RSI_20_fwd'] - stock[instrument]['RSI_20'],
                name='M_RSI_CHANGE'
            )
        )

        # Calculate the ADX, PDI & MDI
        _adx, _pdi, _mdi = adx(stock[instrument])

        stock[instrument] = stock[instrument].join(_adx)
        stock[instrument] = stock[instrument].join(_pdi)
        stock[instrument] = stock[instrument].join(_mdi)

        # Calculate the Moving Averages: 5, 10, 20, 50, 100
        for period in [5, 10, 20, 50, 100]:
            stock[instrument] = stock[instrument].join(
                moving_average(
                    stock[instrument]['Close'],
                    period,
                    name=f'{period}MA'
                )
            )

        # Calculate the Williams PCTR
        stock[instrument] = stock[instrument].join(
            williams(stock[instrument])
        )

        # Calculate the Minmax Range
        n = 17
        for i in range(nsize):
            maxval = stock[instrument]['High'].iloc[i - n: i].max()
            minval = stock[instrument]['Low'].iloc[i - n: i].min()
            rng = abs(maxval) - abs(minval)
            # where is the last price in the range of minumimn to maximum
            pnow = stock[instrument]['Close'].iloc[i - n: i]
            if len(pnow.iloc[-1: i].values) > 0:
                whereinrng = (
                    (pnow.iloc[-1: i].values[0] - abs(minval)) / rng
                ) * 100.0
                stock[instrument].loc[i: nsize, 'MinMaxPosition'] = whereinrng
                stock[instrument].loc[i: nsize, 'High_Price(14)'] = maxval
                stock[instrument].loc[i: nsize, 'Low_Price(14)'] = minval

        stock[instrument]['Divergence factor Avg'] = (
            stock[instrument]['Divergence Factor 1'] +
            stock[instrument]['Divergence Factor 2'] +
            stock[instrument]['Divergence Factor 3']
        ) / 3.0

        stock[instrument]['Momentum Avg'] = (
            stock[instrument]['M_MACD_CHANGE'] +
            stock[instrument]['M_RSI_CHANGE'] +
            stock[instrument]['Profit/Loss_10'] +
            stock[instrument]['Profit/Loss_30']
        ) / 4.0

        df_instrument=pd.DataFrame()
        df_instrument["Open"]=stock[instrument]["Open"]
        df_instrument["High"]=stock[instrument]['High']
        df_instrument["Low"]=stock[instrument]['Low']
        df_instrument["Close"]=stock[instrument]['Close']
        df_instrument["Volume"]=stock[instrument]['Volume']
        df_instrument["Price"]=stock[instrument]['Close']
        df_instrument["Change In Price"]=change_in_price(stock[instrument]['Close'].values)
        df_instrument["CCI"]=stock[instrument]['CCI']
        df_instrument["PNL Percentile"]=stock[instrument]['PNL Percentile']
        df_instrument["Divergence Factor 1"]=stock[instrument]['Divergence Factor 1']
        df_instrument["Divergence Factor 2"]=stock[instrument]['Divergence Factor 2']
        df_instrument["Divergence Factor 3"]=stock[instrument]['Divergence Factor 3']

        df_instrument["Momentum Factor 1"]=stock[instrument]["M_MACD_CHANGE"]
        df_instrument["Momentum Factor 2"]=stock[instrument]['M_RSI_CHANGE']
        df_instrument["Momentum Factor 3"]=stock[instrument]['Profit/Loss_10']
        df_instrument["Momentum Factor 4"]=stock[instrument]['Profit/Loss_30']

        df_instrument["RSI"]=stock[instrument]["RSI"]
        df_instrument["MACD"]=stock[instrument]["MACD"]
        df_instrument["WPCTR"]=stock[instrument]["Williams PCTR"]
        df_instrument["pdi"]=stock[instrument]["pdi"]
        df_instrument["mdi"]=stock[instrument]["mdi"]
        df_instrument["adx"]=stock[instrument]["adx"]
        #df_instrument= df_instrument[pd.notnull(df_instrument['CCI'])]
        df_instrument=df_instrument.dropna(how="any")
        df_instrument["CCI Percentile"]=cci_percentile(df_instrument["CCI"])
        df_instrument["Divergence Factor 4"]=df_instrument["CCI Percentile"]-df_instrument["PNL Percentile"]
        df_instrument['Divergence Factor 1 Rank'] = rank_formulation(df_instrument['Divergence Factor 1'].values)
        df_instrument['Divergence Factor 2 Rank'] = rank_formulation(df_instrument['Divergence Factor 2'].values)
        df_instrument['Divergence Factor 3 Rank'] = rank_formulation(df_instrument['Divergence Factor 3'].values)
        df_instrument['Divergence Factor 4 Rank'] = rank_formulation(df_instrument['Divergence Factor 4'].values)
        df_instrument['DF Avg Rank'] = (
            df_instrument['Divergence Factor 1 Rank'] +
            df_instrument['Divergence Factor 2 Rank'] +
            df_instrument['Divergence Factor 3 Rank'] +
            df_instrument['Divergence Factor 4 Rank']
        ) / 4.0

        df_instrument['Momentum Factor 1 Rank'] = rank_formulation(df_instrument['Momentum Factor 1'].values)
        df_instrument['Momentum Factor 2 Rank'] = rank_formulation(df_instrument['Momentum Factor 2'].values)
        df_instrument['Momentum Factor 3 Rank'] = rank_formulation(df_instrument['Momentum Factor 3'].values)
        df_instrument['Momentum Factor 4 Rank'] = rank_formulation(df_instrument['Momentum Factor 4'].values)
        df_instrument['MF Avg Rank'] = (
            df_instrument['Momentum Factor 1 Rank'] +
            df_instrument['Momentum Factor 2 Rank'] +
            df_instrument['Momentum Factor 3 Rank'] +
            df_instrument['Momentum Factor 4 Rank']
        ) / 4.0



        df_instrument["% Rank of DF Avgs"] =rank_formulation(df_instrument['DF Avg Rank'].values)
        df_instrument["% Rank of MF Avgs"] =rank_formulation(df_instrument['MF Avg Rank'].values)
        df_instrument=df_instrument[['Open','High','Low','Close','Volume','Price','Change In Price',
           'Divergence Factor 1', 'Divergence Factor 2', 'Divergence Factor 3', 'Divergence Factor 4','DF Avg Rank', '% Rank of DF Avgs',
           'Divergence Factor 1 Rank', 'Divergence Factor 2 Rank', 'Divergence Factor 3 Rank','Divergence Factor 4 Rank',
           'Momentum Factor 1','Momentum Factor 2','Momentum Factor 3','Momentum Factor 4',
           'Momentum Factor 1 Rank','Momentum Factor 2 Rank','Momentum Factor 3 Rank','Momentum Factor 4 Rank','MF Avg Rank', '% Rank of MF Avgs',
           'RSI', 'MACD', 'WPCTR', 'CCI', 'CCI Percentile', 'PNL Percentile','pdi', 'mdi', 'adx',]]
        df_instrument.to_csv(gpath+"all_folders/"+instrument+".csv")
    
    ccis_df=pd.DataFrame(dic_for_all_cci)
    cci_percentile_list=[]
    dic={"Instrument":instruments_list,"CCI":CCI_list}
    new_df=pd.DataFrame(dic)
    cci_percentile_list=cci_percentile(new_df["CCI"]).to_list()
 
    #sys.exit()
    # calculate the aggregrate for each oeruod
    # calculate the Divergence_Macd_Prc_Rank

    for nrow in range(nsize):
        row = [
            stock[instrument]['Divergence Factor 3'].iloc[nrow] for
            instrument in instruments
        ]
        series = pd.Series(row).rank() / len(row)
        for i, instrument in enumerate(instruments):
            stock[instrument].loc[nrow: nsize, 'Divergence_Macd_Prc_Rank'] = series.iloc[i]

    # calculate the Divergence and Momentum average rank
    indices = [instrument for instrument in instruments]
    columns = [
        'Price',
        "Change In Price",
        'Divergence Factor 1',
        'Divergence Factor 2',
        'Divergence Factor 3',
        'Divergence Factor 1 Rank',
        'Divergence Factor 2 Rank',
        'Divergence Factor 3 Rank',
        'M_MACD_CHANGE',
        'M_RSI_CHANGE',
        'Profit/Loss_10',
        'Profit/Loss_30',
        'M_MACD_CHANGE Rank',
        'M_RSI_CHANGE Rank',
        'Profit/Loss_10 Rank',
        'Profit/Loss_30 Rank',
        'MF Avg Rank', 
        '% Rank of MF Avgs',
        'MinMaxPosition',
        'RSI',
        'WPCTR',
        'pdi', 'mdi', 'adx',
        'High_Price(14)',
        'Low_Price(14)',
        '5MA', '10MA', '20MA', '50MA', '100MA',
        "MACD",
        'PNL Percentile',
        "DF Avg Rank",
        "% Rank of DF Avgs",
    ]

    periods = []
    for i in range(nsize):

        period = []

        for instrument in instruments:

            period.append([
                stock[instrument]['Close'].iloc[i],
                stock[instrument]["Change In Price"].iloc[i],
                stock[instrument]['Divergence Factor 1'].iloc[i],
                stock[instrument]['Divergence Factor 2'].iloc[i],
                stock[instrument]['Divergence Factor 3'].iloc[i],
                None,
                None,
                None,
                stock[instrument]['M_MACD_CHANGE'].iloc[i],
                stock[instrument]['M_RSI_CHANGE'].iloc[i],
                stock[instrument]['Profit/Loss_10'].iloc[i],
                stock[instrument]['Profit/Loss_30'].iloc[i],
                None,
                None,
                None,
                None,
                None,
                None,
                stock[instrument]['MinMaxPosition'].iloc[i],
                stock[instrument]['RSI'].iloc[i],
                stock[instrument]['Williams PCTR'].iloc[i],
                stock[instrument]['pdi'].iloc[i],
                stock[instrument]['mdi'].iloc[i],
                stock[instrument]['adx'].iloc[i],
                stock[instrument]['High_Price(14)'].iloc[i],
                stock[instrument]['Low_Price(14)'].iloc[i],
                stock[instrument]['5MA'].iloc[i],
                stock[instrument]['10MA'].iloc[i],
                stock[instrument]['20MA'].iloc[i],
                stock[instrument]['50MA'].iloc[i],
                stock[instrument]['100MA'].iloc[i],
                stock[instrument]["MACD"].iloc[i],
                stock[instrument]['PNL Percentile'].iloc[i],
                None,
                None,
            ])
        df = pd.DataFrame(data=period, index=indices, columns=columns)
        df['Divergence Factor 1 Rank'] =rank_formulation(df["Divergence Factor 1"].values)
        df['Divergence Factor 2 Rank'] = rank_formulation(df["Divergence Factor 2"].values)
        df['Divergence Factor 3 Rank'] = rank_formulation(df["Divergence Factor 3"].values)

        df['Momentum Factor 1 Rank'] = rank_formulation(df['M_MACD_CHANGE'].values)
        df['Momentum Factor 2 Rank'] = rank_formulation(df['M_RSI_CHANGE'].values)
        df['Momentum Factor 3 Rank'] = rank_formulation(df['Profit/Loss_10'].values)
        df['Momentum Factor 4 Rank'] = rank_formulation(df['Profit/Loss_30'].values)
     
        df['MF Avg Rank'] = (
            df['Momentum Factor 1 Rank'] +
            df['Momentum Factor 1 Rank'] +
            df['Momentum Factor 1 Rank'] +
            df['Momentum Factor 1 Rank']
        ) / 4.0
        df['% Rank of MF Avgs'] = rank_formulation(df['MF Avg Rank'].values)

        #df.to_excel("target_data.xlsx")
        periods.append(df)
    pnl_percentile_nparaay=np.array(df["PNL Percentile"].values)
    cci_percentile_nparray=cci_percentile_list
    divergent_factor_4=cci_percentile_nparray-pnl_percentile_nparaay
    df["CCI"]=CCI_list
    df["CCI Percentile"]=cci_percentile_list
    df["Divergence Factor 4"]=divergent_factor_4
    df['Divergence Factor 4 Rank'] = rank_formulation(df['Divergence Factor 1'].values)
    df['DF Avg Rank'] = (
            df['Divergence Factor 1 Rank'] +
            df['Divergence Factor 2 Rank'] +
            df['Divergence Factor 3 Rank'] +
            df['Divergence Factor 4 Rank']
        ) / 4.0
    df["% Rank of DF Avgs"] = rank_formulation(df['DF Avg Rank'].values)
    df=df[['Price', 'Change In Price','Divergence Factor 1', 'Divergence Factor 2', 'Divergence Factor 3', 'Divergence Factor 4',
           'Divergence Factor 1 Rank', 'Divergence Factor 2 Rank', 'Divergence Factor 3 Rank','Divergence Factor 4 Rank',
           'DF Avg Rank', '% Rank of DF Avgs', #'Momentum Factor 1','Momentum Factor 2','Momentum Factor 3','Momentum Factor 4',
           #'Momentum Factor 1 Rank','Momentum Factor 2 Rank','Momentum Factor 3 Rank','Momentum Factor 4 Rank','MF Avg Rank', '% Rank of MF Avgs',
           'M_MACD_CHANGE', 'M_RSI_CHANGE', 'Profit/Loss_10', 'Profit/Loss_30','M_MACD_CHANGE Rank', 'M_RSI_CHANGE Rank', 'Profit/Loss_10 Rank', 'Profit/Loss_30 Rank',
           'MF Avg Rank', '% Rank of MF Avgs', 'MinMaxPosition', 'RSI', 'MACD', 'WPCTR', 'CCI', 'CCI Percentile', 'PNL Percentile',
           'pdi', 'mdi', 'adx', 'High_Price(14)', 'Low_Price(14)', '5MA', '10MA', '20MA', '50MA', '100MA']]
    df.to_excel(gpath+"all_folders/"+"target_data.xlsx")
    df.sort_values(by="% Rank of DF Avgs",inplace=True)
    df.to_excel(gpath+"all_folders/"+"ordered_target_data.xlsx")

    top5,last5=instrument_selection_rules(df)
    specific_insts=None
    specific_insts=top5+last5
    dflist1=[]
    if specific_insts is not None:
        '''dflist1=Graph_Plots_For_Individual_Instrument(specific_insts,False)
        dflist1.to_csv(gpath+"all_folders"+"/"+"ins_ind_flag.csv")
        dfDiverge=dflist.copy()  
        dfDiverge=dfDiverge.loc[dfDiverge['imp_var'].isin(['Divergence Factor 1','Divergence Factor 2','Divergence Factor 3','Divergence Factor 4'])]
        dfDiverge.reset_index(drop=True,inplace=True)
        dfDiverge.to_csv(gpath+"all_folders"+"/"+"selected_ins_ind_flag.csv")'''
        for rule_instrument in specific_insts:
            data=pd.read_csv(gpath+"all_folders/"+rule_instrument+".csv",index_col="Date")
            data.to_csv(gpath+"rule_select_inst/"+rule_instrument+".csv")



'''
    calculation(dfDiverge) #perform rolling,MinMax,Diff calculation

    if specific_insts is not None:
        dflist1=make_impVar_for_momentum(specific_insts,False)
    if man_inst_names is not None:
        dflist1=make_impVar_for_momentum(man_inst_names,True)
    dflist=dflist1+dflist2
    dflist=pd.concat(dflist)
    dflist.to_csv("all_folders"+"/"+"ImpVar_forMomentum.csv")
    dfMomentum=dflist.copy()  
    dfMomentum=dfMomentum.loc[dfMomentum['imp_var'].isin(['Momentum Factor 1','Momentun Factor 2','Momentum Factor 3','Momentum Factor 4'])]
    dfMomentum.reset_index(drop=True,inplace=True)
    calculation_forMomentum(dfMomentum) #perform rolling,MinMax,Diff calculation
    
    '''
    