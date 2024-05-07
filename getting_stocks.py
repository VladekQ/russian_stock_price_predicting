#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def get_all_stocks_list():
    start = 0
    dataframe = pd.DataFrame()
    response = requests.get('https://iss.moex.com/iss/securities.json?engine=stock&market=shares&start={}'.format(start)).json()
    dataframe_to_append = pd.DataFrame(data=response['securities']['data'],
                                       columns=response['securities']['columns'])[['secid', 'name', 'is_traded',
                                                                                   'emitent_title', 'emitent_inn',
                                                                                   'type']]
    dataframe = pd.concat([dataframe, dataframe_to_append])
    start += 100
    while len(dataframe_to_append) > 0:
        response = requests.get('https://iss.moex.com/iss/securities.json?engine=stock&market=shares&start={}'.format(start)).json()
        dataframe_to_append = pd.DataFrame(data=response['securities']['data'],
                                           columns=response['securities']['columns'])[['secid', 'name', 'is_traded',
                                                                                       'emitent_title', 'emitent_inn',
                                                                                       'type']]
        dataframe = pd.concat([dataframe, dataframe_to_append])
        start += 100
        
    return dataframe

def get_stocks_history(start_date=None, end_date=None,
                       symbol='GAZP'):
    # If start_date or end_date is null
    # Get date interval
    if not start_date or not end_date:
        response = requests.get('https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{}/dates.html'.format(symbol))
        soup = BeautifulSoup(response.text, 'html.parser')
        interval = [td.text for td in soup.find_all('td')]
        start_date_hist, end_date_hist = interval[0], interval[1]
        del interval
        
    if not start_date:
        start_date = start_date_hist
    if not end_date:
        end_date = end_date_hist
    
    start = 0
    dataframe = pd.DataFrame()
    response = requests.get('https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{}.json?from={}&till={}&start={}'.format(symbol, start_date, end_date, start)).json()
    dataframe_to_append = pd.DataFrame(data=response['history']['data'],
                                       columns=response['history']['columns'])[['TRADEDATE', 'SHORTNAME', 'SECID',
                                                                                'NUMTRADES', 'VALUE', 'OPEN',
                                                                                'LOW', 'HIGH', 'CLOSE', 'TRENDCLSPR']]
    dataframe = pd.concat([dataframe, dataframe_to_append])
    start += 100
    while len(dataframe_to_append) > 0:
        response = requests.get('https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{}.json?from={}&till={}&start={}'.format(symbol, start_date, end_date, start)).json()
        dataframe_to_append = pd.DataFrame(data=response['history']['data'],
                                           columns=response['history']['columns'])[['TRADEDATE', 'SHORTNAME', 'SECID',
                                                                                    'NUMTRADES', 'VALUE', 'OPEN',
                                                                                    'LOW', 'HIGH', 'CLOSE', 'TRENDCLSPR']]
        dataframe = pd.concat([dataframe, dataframe_to_append])
        start += 100
        
    dataframe = dataframe.sort_values(by=['TRADEDATE', 'NUMTRADES'], ascending=[True, False])
    dataframe = dataframe.drop_duplicates(subset=['TRADEDATE'], keep='first')
    dataframe = dataframe.set_index('TRADEDATE')
    dataframe.index = [pd.to_datetime(x) for x in dataframe.index]
    return dataframe


def get_stocks_list_history(start_date=None, end_date=None,
                            symbols=['GAZP']):
    all_data = {}
    for symbol in symbols:
        all_data[symbol] = get_stocks_history(start_date, end_date, symbol)
        
    return all_data

