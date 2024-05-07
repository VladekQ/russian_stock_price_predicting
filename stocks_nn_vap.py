#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime
import pandas as pd
# import pandas_datareader
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sklearn
import seaborn as sns
sns.set_theme(style="whitegrid")
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import timedelta
# import lightgbm as lgb
# import optuna
import os
import time
import warnings as w
import getting_stocks as gs
w.filterwarnings('ignore')

# def get_stocks_data(start_time=datetime.datetime(2016, 1, 10),
#                     end_time=datetime.datetime.today(),
#                     symbols=['GAZP']):
#     format='%Y-%m-%d'
#     if type(start_time) == str:
#         sdate = datetime.datetime.strptime(start_time, format)
#     else:
#         sdate = datetime.datetime(2016, 1, 10)
#     if type(end_time) == str:
#         edate = datetime.datetime.strptime(end_time, format)
#     else:
#         edate = datetime.datetime.today()
#     datasets = {}

#     for symbol in symbols:
#         code = symbol
#         t1 = time.perf_counter()
#         full_data = pandas_datareader.DataReader(code, 'moex', start=sdate, end=edate)
#         t2 = time.perf_counter()
#         data = full_data[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
#         data = data[['CLOSE']].copy()
#         data.columns = ['close']
#         datasets[code] = data.copy()
#         print('{} data is ready. Time takes: {:.0f} seconds'.format(code, t2-t1))
#     print('Getting data is done!')
#     return datasets

def get_stocks_data(start_date=None, end_date=None, symbols=['GAZP']):
    datasets = gs.get_stocks_list_history(start_date, end_date, symbols=symbols)
    datasets = {key: value[['CLOSE']] for key, value in datasets.items()}
    lower_columns = [column.lower() for column in datasets[list(datasets.keys())[0]].columns]
    for symbol in datasets.keys():
        datasets[symbol].columns = lower_columns
    return datasets

def preprocess_stocks_data(datasets):
    def make_features(data, max_lag, rolling_mean_size):
        df = data.copy()
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['hour'] = df.index.hour
        
        for lag in range(1, max_lag + 1):
            df['lag_{}'.format(lag)] = df['close'].shift(lag)
    
        df['rolling_mean'] = df['close'].rolling(rolling_mean_size, closed='left').mean()
        return df
    symbols = list(datasets.keys())
    preprocessed_datasets = {}
    for symbol in symbols:
        t1 = time.perf_counter()
        preprocessed_datasets[symbol] = make_features(datasets[symbol].dropna(), 10, 10)
        t2 = time.perf_counter()
        print('{} data is ready. Time takes: {:.0f} seconds'.format(symbol, t2-t1))
        
    return preprocessed_datasets

def build_fit_stocks_model(preprocessed_datasets, symbols=['GAZP']):
    def fit_neural_network_large(preprocessed_datasets, symbols=['GAZP']):
        global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
        scalers_final = {}
        full_data_tf_scaled_final = {}
        full_data_target_tf = {}
        fill_data_target_indexes = {}
        
        
        for symbol in symbols:
            scalers_final[symbol] = MinMaxScaler()
            X = preprocessed_datasets[symbol].drop('close', axis=1).copy()
            y = preprocessed_datasets[symbol]['close'].copy()
            
            X_tf = tf.constant(tf.cast(X.dropna(), dtype=tf.float32))
            y_tf = tf.constant(tf.cast(y[X.dropna().index], dtype=tf.float32))
            
            # Turn our numpy arrays into tensors
            X_tf_scaled = scalers_final[symbol].fit_transform(X_tf)
            
            full_data_tf_scaled_final[symbol] = X_tf_scaled
            full_data_target_tf[symbol] = y_tf
            fill_data_target_indexes[symbol] = y.index.copy()
        
        # Build a neural network
        tf.random.set_seed(42)
        
        # Set EarlyStopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
        
        # Create the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ],
        name='main_model')
        
        # Compile the model
        model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        # Fitting model on all datasets
        for symbol in symbols:
            t1 = time.perf_counter()
            history_time_model = model.fit(
                full_data_tf_scaled_final[symbol],
                full_data_target_tf[symbol],
                epochs=1000,
                # validation_data=(testing_data_tf_scaled[symbol], testing_target_data_tf[symbol]),
                batch_size=256,
                callbacks=[callback],
                verbose=0
            );
            mae_test_score = model.evaluate(full_data_tf_scaled_final[symbol], full_data_target_tf[symbol])
            preds_tmp = model.predict(full_data_tf_scaled_final[symbol])
            r2_test_score = r2_score(full_data_target_tf[symbol], preds_tmp)
            t2 = time.perf_counter()
            print('{} data is ready. Time takes: {:.0f} seconds. MAE: {:.2f} rub. R2: {:.2f}'.format(symbol, t2-t1, mae_test_score, r2_test_score))
            
        path = "models__"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created!")
            
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
        for symbol in symbols:
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            cloned_model.set_weights(model.get_weights())
            t1 = time.perf_counter()
            history_time_model = cloned_model.fit(
                full_data_tf_scaled_final[symbol],
                full_data_target_tf[symbol],
                epochs=1000,
                # validation_data=(testing_data_tf_scaled[symbol], testing_target_data_tf[symbol]),
                batch_size=256,
                callbacks=[callback],
                verbose=0
            );
            mae_test_score = cloned_model.evaluate(full_data_tf_scaled_final[symbol], full_data_target_tf[symbol])
            preds_tmp = cloned_model.predict(full_data_tf_scaled_final[symbol])
            r2_test_score = r2_score(full_data_target_tf[symbol], preds_tmp)
            cloned_model.save('models__/{}_model_large.h5'.format(symbol))
            t2 = time.perf_counter()
            print('{} data is ready. Time takes: {:.0f} seconds. MAE: {:.2f} rub. R2: {:.2f}'.format(symbol, t2-t1, mae_test_score, r2_test_score))
    
    def fit_neural_network_medium(preprocessed_datasets, symbols=['GAZP']):
        global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
        scalers_final = {}
        full_data_tf_scaled_final = {}
        full_data_target_tf = {}
        fill_data_target_indexes = {}
        
        
        for symbol in symbols:
            scalers_final[symbol] = MinMaxScaler()
            X = preprocessed_datasets[symbol].drop('close', axis=1).copy()
            y = preprocessed_datasets[symbol]['close'].copy()
            X_tf = tf.constant(tf.cast(X.dropna(), dtype=tf.float32))
            y_tf = tf.constant(tf.cast(y[X.dropna().index], dtype=tf.float32))
            # Turn our numpy arrays into tensors
            X_tf_scaled = scalers_final[symbol].fit_transform(X_tf)
            
            full_data_tf_scaled_final[symbol] = X_tf_scaled
            full_data_target_tf[symbol] = y_tf
            fill_data_target_indexes[symbol] = y.index.copy()
        
        # Build a neural network
        tf.random.set_seed(42)
        
        # Set EarlyStopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
        
        # Create the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation=None),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ],
        name='main_model')
        
        # Compile the model
        model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        # Fitting model on all datasets
        for symbol in symbols:
            t1 = time.perf_counter()
            history_time_model = model.fit(
                full_data_tf_scaled_final[symbol],
                full_data_target_tf[symbol],
                epochs=1000,
                # validation_data=(testing_data_tf_scaled[symbol], testing_target_data_tf[symbol]),
                batch_size=256,
                callbacks=[callback],
                verbose=0
            );
            mae_test_score = model.evaluate(full_data_tf_scaled_final[symbol], full_data_target_tf[symbol])
            preds_tmp = model.predict(full_data_tf_scaled_final[symbol])
            r2_test_score = r2_score(full_data_target_tf[symbol], preds_tmp)
            t2 = time.perf_counter()
            print('{} data is ready. Time takes: {:.0f} seconds. MAE: {:.2f} rub. R2: {:.2f}'.format(symbol, t2-t1, mae_test_score, r2_test_score))
            
        path = "models__"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created!")
            
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
        for symbol in symbols:
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            cloned_model.set_weights(model.get_weights())
            t1 = time.perf_counter()
            history_time_model = cloned_model.fit(
                full_data_tf_scaled_final[symbol],
                full_data_target_tf[symbol],
                epochs=1000,
                # validation_data=(testing_data_tf_scaled[symbol], testing_target_data_tf[symbol]),
                batch_size=256,
                callbacks=[callback],
                verbose=0
            );
            mae_test_score = cloned_model.evaluate(full_data_tf_scaled_final[symbol], full_data_target_tf[symbol])
            preds_tmp = cloned_model.predict(full_data_tf_scaled_final[symbol])
            r2_test_score = r2_score(full_data_target_tf[symbol], preds_tmp)
            cloned_model.save('models__/{}_model.h5'.format(symbol))
            t2 = time.perf_counter()
            print('{} data is ready. Time takes: {:.0f} seconds. MAE: {:.2f} rub. R2: {:.2f}'.format(symbol, t2-t1, mae_test_score, r2_test_score))
            
    def fit_neural_network_small(preprocessed_datasets, symbols=['GAZP']):
        global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
        scalers_final = {}
        full_data_tf_scaled_final = {}
        full_data_target_tf = {}
        fill_data_target_indexes = {}
        
        
        for symbol in symbols:
            scalers_final[symbol] = MinMaxScaler()
            X = preprocessed_datasets[symbol].drop('close', axis=1).copy()
            y = preprocessed_datasets[symbol]['close'].copy()
            
            X_tf = tf.constant(tf.cast(X.dropna(), dtype=tf.float32))
            y_tf = tf.constant(tf.cast(y[X.dropna().index], dtype=tf.float32))
            
            # Turn our numpy arrays into tensors
            X_tf_scaled = scalers_final[symbol].fit_transform(X_tf)
            
            full_data_tf_scaled_final[symbol] = X_tf_scaled
            full_data_target_tf[symbol] = y_tf
            fill_data_target_indexes[symbol] = y.index.copy()
        
        # Build a neural network
        tf.random.set_seed(42)
        
        # Set EarlyStopping
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
        
        # Create the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=None),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ],
        name='main_model')
        
        # Compile the model
        model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        # Fitting model on all datasets
        for symbol in symbols:
            t1 = time.perf_counter()
            history_time_model = model.fit(
                full_data_tf_scaled_final[symbol],
                full_data_target_tf[symbol],
                epochs=1000,
                # validation_data=(testing_data_tf_scaled[symbol], testing_target_data_tf[symbol]),
                batch_size=256,
                callbacks=[callback],
                verbose=0
            );
            mae_test_score = model.evaluate(full_data_tf_scaled_final[symbol], full_data_target_tf[symbol])
            preds_tmp = model.predict(full_data_tf_scaled_final[symbol])
            r2_test_score = r2_score(full_data_target_tf[symbol], preds_tmp)
            t2 = time.perf_counter()
            print('{} data is ready. Time takes: {:.0f} seconds. MAE: {:.2f} rub. R2: {:.2f}'.format(symbol, t2-t1, mae_test_score, r2_test_score))
            
        path = "models__"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created!")
            
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)
        for symbol in symbols:
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            cloned_model.set_weights(model.get_weights())
            t1 = time.perf_counter()
            history_time_model = cloned_model.fit(
                full_data_tf_scaled_final[symbol],
                full_data_target_tf[symbol],
                epochs=1000,
                # validation_data=(testing_data_tf_scaled[symbol], testing_target_data_tf[symbol]),
                batch_size=256,
                callbacks=[callback],
                verbose=0
            );
            mae_test_score = cloned_model.evaluate(full_data_tf_scaled_final[symbol], full_data_target_tf[symbol])
            preds_tmp = cloned_model.predict(full_data_tf_scaled_final[symbol])
            r2_test_score = r2_score(full_data_target_tf[symbol], preds_tmp)
            cloned_model.save('models__/{}_model_small.h5'.format(symbol))
            t2 = time.perf_counter()
            print('{} data is ready. Time takes: {:.0f} seconds. MAE: {:.2f} rub. R2: {:.2f}'.format(symbol, t2-t1, mae_test_score, r2_test_score))
        
    fit_neural_network_medium(preprocessed_datasets, symbols)
    fit_neural_network_small(preprocessed_datasets, symbols)
    fit_neural_network_large(preprocessed_datasets, symbols)
    print('All models have been saved in "models_" folder')
        
def predict_future_stock_price(preprocessed_datasets, future_days=3, symbol='SBER', plot_=True):
    def prediction_by_neual_network_medium(preprocessed_datasets, future_days=3, symbol='SBER'):
        global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
        
        scalers_final = {}
        full_data_tf_scaled_final = {}
        full_data_target_tf = {}
        fill_data_target_indexes = {}
        
        scalers_final[symbol] = MinMaxScaler()
        X = preprocessed_datasets[symbol].drop('close', axis=1).copy()
        y = preprocessed_datasets[symbol]['close'].copy()
        
        X_tf = tf.constant(tf.cast(X.dropna(), dtype=tf.float32))
        y_tf = tf.constant(tf.cast(y[X.dropna().index], dtype=tf.float32))
        
        # Turn our numpy arrays into tensors
        X_tf_scaled = scalers_final[symbol].fit_transform(X_tf)
        
        full_data_tf_scaled_final[symbol] = X_tf_scaled
        full_data_target_tf[symbol] = y_tf
        fill_data_target_indexes[symbol] = y.index.copy()
        # print('{} data is ready. Time takes: {:.0f} seconds'.format(symbol, t2-t1))
        
        def create_future(days, symbol):
            global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
            tf.random.set_seed(42)
            symbol_model = tf.keras.models.load_model('models__/{}_model.h5'.format(symbol))
            # symbol_model.compile(loss=tf.keras.losses.mae,
            #           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
            
            X_ = full_data_tf_scaled_final[symbol]
            y_ = full_data_target_tf[symbol]
            
            # symbol_model_history = symbol_model.fit(
            #     X_,
            #     y_,
            #     epochs=1000,
            #     batch_size=256,
            #     callbacks=[callback],
            #     verbose=0
            # );
            
            answers = pd.Series(dtype='float64')
            days_list = pd.bdate_range(start=fill_data_target_indexes[symbol][-1] + timedelta(days=1), end=fill_data_target_indexes[symbol][-1] + timedelta(days=days))
            for day in days_list:
                lags = pd.DataFrame(pd.Series(y_).tail(10)).T
                lags.columns = ['lag_{}'.format(i) for i in range(len(lags.columns), 0, -1)]
                lags = lags[[lags.columns[col] for col in range(len(lags.columns)-1, -1, -1)]]
                lags.index = [day]
                lags.index = pd.to_datetime(lags.index)
                
                future_day = pd.DataFrame({'year': lags.index.year[0],
                                                'month': lags.index.month[0],
                                                'day': lags.index.day[0],
                                                'dayofweek': lags.index.dayofweek[0],
                                                'hour': lags.index.hour[0]}, index=[lags.index[0]])
                future_day = future_day.join(lags)
                future_day['rolling_mean'] = pd.Series(y_).tail(10).mean()
                X_tf = tf.constant(tf.cast(future_day, dtype=tf.float32))
                X_tf = scalers_final[symbol].transform(X_tf)
                X_ = np.append(X_, X_tf.reshape(1, -1), axis=0)
                day_prediction_to_append = symbol_model.predict(X_tf, verbose=0)
                # print(day_prediction_to_append)
                day_prediction = pd.Series(day_prediction_to_append[0], index=future_day.index)
                y_ = tf.concat([y_, day_prediction_to_append[0]], 0)
    #             answers = answers.append(day_prediction)
                answers = pd.concat([answers, day_prediction])
                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                symbol_model_history = symbol_model.fit(
                    X_,
                    y_,
                    epochs=50,
                    batch_size=256,
                    callbacks=[callback],
                    verbose=0
                );
    #             print('day #{}'.format(day))
            return answers
    
#     predictions_symbols = {}

#     predictions_symbols[symbol] = create_future(future_days, symbol)
        predictions_symbols = create_future(future_days, symbol)
        if plot_:
            max_tail = 60 if len(preprocessed_datasets[symbol]) > 60 else len(preprocessed_datasets[symbol])
            prediction_final = pd.DataFrame({
                'True Value (Full Data)': preprocessed_datasets[symbol]['close'].tail(60),
                'Prediction (Future)': predictions_symbols
            })
            
            fig, axs = plt.subplots(1, 1, figsize=(8, 3))
    #         prediction_final.plot(ax=axs, color=['blue', 'red']);
            ax = sns.lineplot(data=prediction_final, ax=axs, palette=['b', 'r'])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.xticks(rotation=75)
            plt.title('Predictions by Neural Network Medium')
    #         axs.tick_params(axis='x', rotation=90)
            plt.show()
        
        return predictions_symbols
        
    def prediction_by_neual_network_small(preprocessed_datasets, future_days=3, symbol='SBER'):
        global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
        
        scalers_final = {}
        full_data_tf_scaled_final = {}
        full_data_target_tf = {}
        fill_data_target_indexes = {}
        
        scalers_final[symbol] = MinMaxScaler()
        X = preprocessed_datasets[symbol].drop('close', axis=1).copy()
        y = preprocessed_datasets[symbol]['close'].copy()
        
        X_tf = tf.constant(tf.cast(X.dropna(), dtype=tf.float32))
        y_tf = tf.constant(tf.cast(y[X.dropna().index], dtype=tf.float32))
        
        # Turn our numpy arrays into tensors
        X_tf_scaled = scalers_final[symbol].fit_transform(X_tf)
        
        full_data_tf_scaled_final[symbol] = X_tf_scaled
        full_data_target_tf[symbol] = y_tf
        fill_data_target_indexes[symbol] = y.index.copy()
        # print('{} data is ready. Time takes: {:.0f} seconds'.format(symbol, t2-t1))
        
        def create_future(days, symbol):
            global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
            tf.random.set_seed(42)
            symbol_model = tf.keras.models.load_model('models__/{}_model_small.h5'.format(symbol))
            # symbol_model.compile(loss=tf.keras.losses.mae,
            #           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
            
            X_ = full_data_tf_scaled_final[symbol]
            y_ = full_data_target_tf[symbol]
            
            # symbol_model_history = symbol_model.fit(
            #     X_,
            #     y_,
            #     epochs=1000,
            #     batch_size=256,
            #     callbacks=[callback],
            #     verbose=0
            # );
            
            answers = pd.Series(dtype='float64')
            days_list = pd.bdate_range(start=fill_data_target_indexes[symbol][-1] + timedelta(days=1), end=fill_data_target_indexes[symbol][-1] + timedelta(days=days))
            for day in days_list:
                lags = pd.DataFrame(pd.Series(y_).tail(10)).T
                lags.columns = ['lag_{}'.format(i) for i in range(len(lags.columns), 0, -1)]
                lags = lags[[lags.columns[col] for col in range(len(lags.columns)-1, -1, -1)]]
                lags.index = [day]
                lags.index = pd.to_datetime(lags.index)
                
                future_day = pd.DataFrame({'year': lags.index.year[0],
                                                'month': lags.index.month[0],
                                                'day': lags.index.day[0],
                                                'dayofweek': lags.index.dayofweek[0],
                                                'hour': lags.index.hour[0]}, index=[lags.index[0]])
                future_day = future_day.join(lags)
                future_day['rolling_mean'] = pd.Series(y_).tail(10).mean()
                X_tf = tf.constant(tf.cast(future_day, dtype=tf.float32))
                X_tf = scalers_final[symbol].transform(X_tf)
                X_ = np.append(X_, X_tf.reshape(1, -1), axis=0)
                day_prediction_to_append = symbol_model.predict(X_tf, verbose=0)
                # print(day_prediction_to_append)
                day_prediction = pd.Series(day_prediction_to_append[0], index=future_day.index)
                y_ = tf.concat([y_, day_prediction_to_append[0]], 0)
    #             answers = answers.append(day_prediction)
                answers = pd.concat([answers, day_prediction])
                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                symbol_model_history = symbol_model.fit(
                    X_,
                    y_,
                    epochs=50,
                    batch_size=256,
                    callbacks=[callback],
                    verbose=0
                );
    #             print('day #{}'.format(day))
            return answers
    
#     predictions_symbols = {}

#     predictions_symbols[symbol] = create_future(future_days, symbol)
        predictions_symbols = create_future(future_days, symbol)
        if plot_:
            max_tail = 60 if len(preprocessed_datasets[symbol]) > 60 else len(preprocessed_datasets[symbol])
            prediction_final = pd.DataFrame({
                'True Value (Full Data)': preprocessed_datasets[symbol]['close'].tail(60),
                'Prediction (Future)': predictions_symbols
            })
            
            fig, axs = plt.subplots(1, 1, figsize=(8, 3))
    #         prediction_final.plot(ax=axs, color=['blue', 'red']);
            ax = sns.lineplot(data=prediction_final, ax=axs, palette=['b', 'r'])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.xticks(rotation=75)
            plt.title('Predictions by Neural Network Small')
    #         axs.tick_params(axis='x', rotation=90)
            plt.show()
        
        return predictions_symbols
    
    def prediction_by_neual_network_large(preprocessed_datasets, future_days=3, symbol='SBER'):
        global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
        
        scalers_final = {}
        full_data_tf_scaled_final = {}
        full_data_target_tf = {}
        fill_data_target_indexes = {}
        
        scalers_final[symbol] = MinMaxScaler()
        X = preprocessed_datasets[symbol].drop('close', axis=1).copy()
        y = preprocessed_datasets[symbol]['close'].copy()
        
        X_tf = tf.constant(tf.cast(X.dropna(), dtype=tf.float32))
        y_tf = tf.constant(tf.cast(y[X.dropna().index], dtype=tf.float32))
        
        # Turn our numpy arrays into tensors
        X_tf_scaled = scalers_final[symbol].fit_transform(X_tf)
        
        full_data_tf_scaled_final[symbol] = X_tf_scaled
        full_data_target_tf[symbol] = y_tf
        fill_data_target_indexes[symbol] = y.index.copy()
        # print('{} data is ready. Time takes: {:.0f} seconds'.format(symbol, t2-t1))
        
        def create_future(days, symbol):
            global full_data_tf_scaled_final, full_data_target_tf, fill_data_target_indexes, scalers_final
            tf.random.set_seed(42)
            symbol_model = tf.keras.models.load_model('models__/{}_model_large.h5'.format(symbol))
            # symbol_model.compile(loss=tf.keras.losses.mae,
            #           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
            
            X_ = full_data_tf_scaled_final[symbol]
            y_ = full_data_target_tf[symbol]
            
            # symbol_model_history = symbol_model.fit(
            #     X_,
            #     y_,
            #     epochs=1000,
            #     batch_size=256,
            #     callbacks=[callback],
            #     verbose=0
            # );
            
            answers = pd.Series(dtype='float64')
            days_list = pd.bdate_range(start=fill_data_target_indexes[symbol][-1] + timedelta(days=1), end=fill_data_target_indexes[symbol][-1] + timedelta(days=days))
            for day in days_list:
                lags = pd.DataFrame(pd.Series(y_).tail(10)).T
                lags.columns = ['lag_{}'.format(i) for i in range(len(lags.columns), 0, -1)]
                lags = lags[[lags.columns[col] for col in range(len(lags.columns)-1, -1, -1)]]
                lags.index = [day]
                lags.index = pd.to_datetime(lags.index)
                
                future_day = pd.DataFrame({'year': lags.index.year[0],
                                                'month': lags.index.month[0],
                                                'day': lags.index.day[0],
                                                'dayofweek': lags.index.dayofweek[0],
                                                'hour': lags.index.hour[0]}, index=[lags.index[0]])
                future_day = future_day.join(lags)
                future_day['rolling_mean'] = pd.Series(y_).tail(10).mean()
                X_tf = tf.constant(tf.cast(future_day, dtype=tf.float32))
                X_tf = scalers_final[symbol].transform(X_tf)
                X_ = np.append(X_, X_tf.reshape(1, -1), axis=0)
                day_prediction_to_append = symbol_model.predict(X_tf, verbose=0)
                # print(day_prediction_to_append)
                day_prediction = pd.Series(day_prediction_to_append[0], index=future_day.index)
                y_ = tf.concat([y_, day_prediction_to_append[0]], 0)
    #             answers = answers.append(day_prediction)
                answers = pd.concat([answers, day_prediction])
                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                symbol_model_history = symbol_model.fit(
                    X_,
                    y_,
                    epochs=50,
                    batch_size=256,
                    callbacks=[callback],
                    verbose=0
                );
    #             print('day #{}'.format(day))
            return answers
    
#     predictions_symbols = {}

#     predictions_symbols[symbol] = create_future(future_days, symbol)
        predictions_symbols = create_future(future_days, symbol)
        if plot_:
            max_tail = 60 if len(preprocessed_datasets[symbol]) > 60 else len(preprocessed_datasets[symbol])
            prediction_final = pd.DataFrame({
                'True Value (Full Data)': preprocessed_datasets[symbol]['close'].tail(60),
                'Prediction (Future)': predictions_symbols
            })
            
            fig, axs = plt.subplots(1, 1, figsize=(8, 3))
    #         prediction_final.plot(ax=axs, color=['blue', 'red']);
            ax = sns.lineplot(data=prediction_final, ax=axs, palette=['b', 'r'])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.xticks(rotation=75)
            plt.title('Predictions by Neural Network Large')
    #         axs.tick_params(axis='x', rotation=90)
            plt.show()
        
        return predictions_symbols
        
    preds_by_nn_1 = prediction_by_neual_network_medium(preprocessed_datasets, future_days, symbol)
    preds_by_nn_2 = prediction_by_neual_network_small(preprocessed_datasets, future_days, symbol)
    preds_by_nn_3 = prediction_by_neual_network_large(preprocessed_datasets, future_days, symbol)
    preds_full = pd.concat([preds_by_nn_1, preds_by_nn_2, preds_by_nn_3], axis=1)
    preds_full.columns = ['medium', 'small', 'large']
    preds_full['mean_prediction'] = preds_full.mean(axis=1)
    
    max_tail = 60 if len(preprocessed_datasets[symbol]) > 60 else len(preprocessed_datasets[symbol])
    prediction_final = pd.DataFrame({
        'True Value (Full Data)': preprocessed_datasets[symbol]['close'].tail(60),
        'Prediction (Future)': preds_full['mean_prediction']
    })
    fig, axs = plt.subplots(1, 1, figsize=(8, 3))
    ax = sns.lineplot(prediction_final, ax=axs, palette=['b', 'g'])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xticks(rotation=75)
    plt.title('Mean Predictions of all models')
    
    return preds_full