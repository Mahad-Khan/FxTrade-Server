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
#from indicators import (adx, macd, macd_percentile, moving_average,
#                        plot_agg_divergence, plot_agg_momentum, plot_data, pnl,
#                        pnl_n, pnl_percentile, rsi, rsi_percentile, williams,cci_percentile)
#from oanda import get_candles
specific_insts=None

plt.style.use('ggplot')
from server_function import date_to_list

PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data/daily/v8'
)

dirname = os.path.dirname(os.path.abspath(__file__))
dirname_list = dirname.split("/")[:-1]
dirname = "/".join(dirname_list)
ggpath = dirname + "/api"

def heatmap1(x, y, **kwargs):
    inst_name=kwargs["instrument_name"]
    del kwargs["instrument_name"]
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order', 'xlabel', 'ylabel'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=30, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')
    ax.set_title("Correaltion of "+inst_name)
    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 


def corrplot(data, size_scale=500, marker='s',instrument_name="instrument"):
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    heatmap1(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale,
        instrument_name=instrument_name
    )

##PPS
def heatmap(df,instrument_name):
    df = df[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    ax = sns.heatmap(df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    ax.set_title("PPS matrix of "+instrument_name)
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
    return ax



def Plot_Multi_Axes(instrument_name,flag):
    #instrument_name = "AUD_USD"
    data = pd.read_csv(ggpath+"/all_folders/"+instrument_name + ".csv")
    data = data.tail(90)
    
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)

    N = 100
    x = data.index
    
    x2=pd.to_datetime(data.index)
    x2=date_to_list(x2)
    print("my type is ",type(x2))
    y = data["Change In Price"]
    y_send=y.to_list()
    y2 = data["DF Avg Rank"]
    y2_send=y2.to_list()
    #y3 = data["% Rank of DF Avgs"]
    
    #data_plot = data[["Price","DF Avg Rank","% Rank of DF Avgs"]]

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(14, 16))
    # make a plot
    ax.plot(x, y, color="orange", marker="o")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=10)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=70)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    # set y-axis label
    ax.set_ylabel("Change In Price",color="orange",fontsize=16)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(x, y2,color="blue",marker="o")
    ax2.set_ylabel("Average Divergence Percentile Rank",color="blue",fontsize=16)

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax2.xaxis.get_minorticklabels(), rotation=70)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=70)

    title = "Price change and Average Divergence Percentile Rank over the time for "+ instrument_name +"."
    plt.title(title,fontname="Times New Roman",fontweight="bold", fontsize=20)

    #plt.show()
    # save the plot as a file
  
    # fig.savefig("api/"+"man_select_inst"+"/"+instrument_name +'_two_different_y_axis.jpg',
    #         format='png',
    #         dpi=100,
    #         bbox_inches='tight')
        
    
    name="Multiple axes plot"
  
    return name,x2,y_send,y2_send


def Divergence_Plots_For_Single_Instrument(instrument_name,flag):
    
    data = pd.read_csv(ggpath+"/all_folders/"+instrument_name + ".csv")
    
    # Making a copy of data frame and dropping all the null values
    df_copy = data.copy()
    df_copy = df_copy.dropna(axis = 1)
    df_copy = df_copy.dropna()

   
    
    X = df_copy[["CCI","RSI","MACD","WPCTR","pdi","mdi","adx","Divergence Factor 1","Divergence Factor 2","Divergence Factor 3","Divergence Factor 4"]]
    y = df_copy["DF Avg Rank"]

    #print(len(X),len(y))
    # Embedded method
    reg = LassoCV()
    reg.fit(X, y)
  #  print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
   # print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)

    #print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    dic={"instrument_name":instrument_name,"imp_var":X.columns,"score":reg.coef_,"flag":flag}
    rdf=pd.DataFrame(dic)
    rdf= rdf[rdf.score>0]
    rdf=rdf.loc[rdf['imp_var'].isin(['Divergence Factor 1','Divergence Factor 2','Divergence Factor 3','Divergence Factor 4'])]
    rdf.reset_index(drop=True,inplace=True)
    rdf.sort_values(by="score",ascending=True,inplace=True)

    rdf.to_csv(ggpath+"/man_select_inst/"+"Imp_Indicator.csv")

    fig,ax = plt.subplots(1)
    sns.set()
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (10, 14)
    matplotlib.rcParams['ytick.labelsize'] = 3
    matplotlib.rcParams['axes.labelsize']  = 3
    matplotlib.rcParams['legend.fontsize']  = 1

    plt.rc('axes', titlesize=12)
    plt.yticks(fontsize=9.5)
    my_colors = list(islice(cycle(['orange','b', 'r', 'g', 'y', 'k','m']), None, len(df_copy)))
    ax = imp_coef.plot(kind = "barh", stacked=True, color=my_colors, width=0.91,align='edge')
    ax.yaxis.label.set_size(3)
    list_index=[]
    list_row=[]
########################################################
    for index, row in imp_coef.items():
     print("index: ",index);print("row",row)
     list_index.append(index)
     list_row.append(row)
###########################################



    title = f"Feature importance of {instrument_name} using the Lasso Model"
    plt.title(title)

    
    name="Feature Importance"
    return name,list_row,list_index


def decompose_plot(instrument_name,flag):
    data = pd.read_csv(ggpath+"/all_folders/"+instrument_name + ".csv")
    data = data.tail(90)
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)
    series = data["DF Avg Rank"]
    
    result = seasonal_decompose(series, model='additive',period=12)
    
    
    observed = result.observed

       
    list_rowO=[]
    list_rowt=[]
    list_rows=[]
    list_rowr=[]
########################################################
    for index, row in observed.items():
        print("index: ",index);print("row",row)
        
        list_rowO.append(row)

    trend = result.trend
    for index, row in trend.items():
        print("index: ",index);print("row",row)
        
        list_rowt.append(row)

    seasonal = result.seasonal
    for index, row in seasonal.items():
        print("index: ",index);print("row",row)
        
        list_rows.append(row)

    residual = result.resid
    for index, row in residual.items():
        print("index: ",index);print("row",row)
        
        list_rowr.append(row)
    df = pd.DataFrame({"observed":observed,"trend":trend, "seasonal":seasonal,"residual":residual})
 
   
    
    x=pd.to_datetime(df.index)
    x=date_to_list(x)
  


    name="Decompose_Time_series"
    return name,x, list_rowO,list_rowt,list_rows,list_rowr

def Plot_Items(instrument_name,flag):
    data = pd.read_csv(ggpath+"/all_folders/"+instrument_name+".csv")
    data = data.tail(90)
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)
    N = 100
    x = data.index
    y = data["Change In Price"]
    y2 = data["DF Avg Rank"]
    y3 = data["% Rank of DF Avgs"]
    
    data_plot = data[["Change In Price","DF Avg Rank","% Rank of DF Avgs"]]
    title = instrument_name + "'s Change In Price graph"
    # create figure and axis objects with subplots()
    fig0,ax = plt.subplots(figsize=(15, 17))
    # make a plot
    ax.plot(x, y, color="orange", marker="o")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=10)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=40)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40)
    # set y-axis label
    ax.set_ylabel("Change In Price",color="orange",fontsize=26)

    
    title = "Price change over time for " + instrument_name + "."
    plt.title(title,fontname="Times New Roman",fontweight="bold", fontsize=22)

    # save the plot as a file
    if flag==True:
        fig0.savefig("/home/mahad/Freelance-Work/FxTrade/Divergence-Server/Divergence_server1/blue/api/man_select_inst"+"/"+instrument_name +'_price.png',
                format='png')

def Rolling_20_Period(instrument_name):
    data = pd.read_csv(ggpath+"/man_select_inst/"+instrument_name+"_rolling.csv")
    data1 = pd.read_csv(ggpath+"/man_select_inst/"+"Imp_Indicator.csv")
    imp_var=data1["imp_var"][0]
    print(imp_var)
    x=data["Date"].to_list()
    y=data["Rolling correlation "+imp_var+" % Rank"].to_list()
    name=instrument_name+" "+imp_var+" 20 Period Rolling Correlation"
    return name,x,y


    



def MinMax_Diff(lst,period):
    anslst=[]
    j=0
    for i in range(0,len(lst)):
        if i>(period-2):
            sublst=lst[j:i+1]
            max1=np.nanmax(sublst)
            min1=np.nanmin(sublst)
            ans=max1-min1
            anslst.append(ans)
            j=j+1
        else:    
            anslst.append(np.nan)
    return anslst

def percent_change(col,period):
    l=[]
    j=0
    for i in range(0,len(col)):
        if i>(period-2):
            a=col[i]-col[j]
            b=a/col[j]
            l.append(b)
            j=j+1
        else:
            l.append(np.nan)
    return l


def calculation():
    df = pd.read_csv(ggpath+"/man_select_inst/"+"Imp_Indicator.csv")
    for index, row in df.iterrows():
        inst_name=row["instrument_name"]
        imp_var=row["imp_var"]
        flag=row["flag"]
        readDf=pd.read_csv(ggpath+"/all_folders/"+inst_name+".csv")
        rollingCorrdf=readDf[imp_var].rolling(20).corr(readDf['Price']) #cal. rolling imp var
        rollingCorrAvg=readDf['% Rank of DF Avgs'].rolling(20).corr(readDf['Price']) #fixed
        nperiod_price_change=percent_change(readDf['Price'].values,20) #fixed
        nperiod_df_change=percent_change(readDf[imp_var].values,20) #cal. div.Factor %age change
        nperiod_MinMax_df=MinMax_Diff(nperiod_df_change,20) #cal MinMax Diverg. difference
        nperiod_MinMax_price=MinMax_Diff(readDf["Price"].values,20) #fixed ,cal MinMax Price difference
        df_duration=np.array(nperiod_MinMax_df)/np.array(nperiod_MinMax_price)
    
        
        readDf["Rolling correlation "+imp_var+" % Rank"]=rollingCorrdf
        readDf["Rolling correlation Avg DF % Rank"]=rollingCorrAvg
        readDf["20 period percentage price change"]=nperiod_price_change
        readDf["20 period percentage " +imp_var+" change"]=nperiod_df_change
        readDf["20 PERIOD MIN MAX DIVERGENCE DIFF"]=nperiod_MinMax_df
        readDf["20 PERIOD MIN MAX PRICE DIFF"]=nperiod_MinMax_price
        readDf["DIVERGENCE DURATION"]=df_duration
        readDf.to_csv(ggpath+"/man_select_inst/"+inst_name+"_rolling"+".csv",na_rep="NaN")
    
    

    


def calculation_forMomentum(df):
    for index, row in df.iterrows():
        inst_name=row["instrument_name"]
        imp_var=row["imp_var"]
        flag=row["flag"]
        if flag==True:
            readDf=pd.read_csv("man_select_inst/"+inst_name+"_impvar"+".csv")
        else:
            readDf=pd.read_csv("rule_select_inst/"+inst_name+"_impvar"+".csv")
        rollingCorrdf=readDf[imp_var].rolling(20).corr(readDf['Price']) #cal. rolling imp var
        rollingCorrAvg=readDf['% Rank of MF Avgs'].rolling(20).corr(readDf['Price']) #fixed
        nperiod_price_change=percent_change(readDf['Price'].values,20) #fixed
        nperiod_df_change=percent_change(readDf[imp_var].values,20) #cal. div.Factor %age change
        nperiod_MinMax_df=MinMax_Diff(nperiod_df_change,20) #cal MinMax Diverg. difference
        nperiod_MinMax_price=MinMax_Diff(readDf["Price"].values,20) #fixed ,cal MinMax Price difference
        df_duration=np.array(nperiod_MinMax_df)/np.array(nperiod_MinMax_price)
        
        readDf["Rolling correlation "+imp_var+" % Rank"]=rollingCorrdf
        readDf["Rolling correlation Avg MF % Rank"]=rollingCorrAvg
        readDf["20 period percentage price change"]=nperiod_price_change
        readDf["20 period percentage " +imp_var+" change"]=nperiod_df_change
        readDf["20 PERIOD MIN MAX DIVERGENCE DIFF"]=nperiod_MinMax_df
        readDf["20 PERIOD MIN MAX PRICE DIFF"]=nperiod_MinMax_price
        readDf["Momentum DURATION"]=df_duration
        if flag==True:
            readDf.to_csv("man_select_inst"+"/"+inst_name+"_impvar"+".csv",na_rep="NaN")
        else:
            readDf.to_csv("rule_select_inst"+"/"+inst_name+"_impvar"+".csv",na_rep="NaN")


def  Avg_Diverg_PercRank(instrument_name):
    data = pd.read_csv(ggpath+"/all_folders/"+instrument_name + ".csv")
    dates = data["Date"].tolist()
    values = data["DF Avg Rank"].tolist()
    name=instrument_name+" Average Divergence Percentile Rank"
    return name,dates,values


 

def main_function(instrument_name):
    flag =True
    graph_name,X_1,Y_1=Divergence_Plots_For_Single_Instrument(instrument_name,flag)
    graph_name2,X_2,Y1_2,Y2_2=Plot_Multi_Axes(instrument_name,flag)
    graph_name3,X_3, observed,trend,seasonal,residual=decompose_plot(instrument_name,flag)

    graph_name4,X_4,Y_4=Avg_Diverg_PercRank(instrument_name)
    calculation()
    graph_name5,X_5,Y_5=Rolling_20_Period(instrument_name)
    #graph_name6,X_6,Y_6=Rolling_20_ChangePrice(instrument_name)
    
    main_dic={graph_name:{"X":X_1,"Y":Y_1},graph_name2:{"X":X_2,"Y1":Y1_2,"Y2":Y2_2},graph_name4:{"X":X_4,"Y":Y_4},graph_name5:{"X":X_5,"Y":Y_5},graph_name3:{"X":X_3,"Observed":observed,"trend":trend,"seasonal":seasonal,"residual":residual}}
    Plot_Items(instrument_name,flag)
    return main_dic

