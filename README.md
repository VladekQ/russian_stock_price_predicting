# Russian Stock Price Prediction with Neural Networks
The "Russian Stock Price Prediction with Neural Networks" project is a Python package designed to provide investors and financial analysts with valuable insights into the future performance of Russian stocks. By leveraging historical stock price data and advanced machine learning algorithms, this package offers a solution to the complex problem of stock price prediction.

With this package, users can easily obtain historical stock price data for any Russian company of their choice, and use this data to train a neural network to predict future stock prices. The package includes a range of powerful functions and tools to help users create and optimize their neural networks, ensuring that they achieve the most accurate and reliable predictions possible.

One of the key benefits of this project is its ability to help investors make more informed decisions about their investments. By providing accurate predictions about future stock prices, the package can help investors identify potential opportunities for profit, as well as potential risks. For example, an investor who is considering buying shares in a particular company can use this package to predict how the price of those shares is likely to change over the next few days or weeks. This information can help the investor decide whether to buy the shares now, or wait for a better opportunity.

Another important benefit of this project is its ability to help financial analysts and traders develop more effective trading strategies. By using the package to predict future stock prices, analysts can identify trends and patterns in the market, and use this information to develop more sophisticated trading strategies. For example, an analyst may use the package to predict how the price of a particular stock is likely to change in response to a specific news event or economic indicator. This information can then be used to develop a trading strategy that takes advantage of these changes.

Overall, the "Russian Stock Price Prediction with Neural Networks" project is an innovative and powerful tool that can help investors, analysts, and traders make more informed decisions and develop more effective trading strategies. By providing accurate predictions about future stock prices, the package offers a valuable solution to the complex problem of stock price prediction, and has the potential to revolutionize the way that investors and analysts approach the Russian stock market.

# How it works
You can use the code from `Example.ipynb` file to try it with yourself

## Import Package
```python
import stocks_nn_vap as snv
```

## Getting Data
Function `get_stocks_data` takes a few arguments:
  - start_time - stating data in string format
  - end_time - ending data in string format
  - symbols - list of companies whose stocks you are interested in. You can see some symbols here: https://www.tradingview.com/markets/stocks-russia/market-movers-large-cap/

**Example:**
```python
symbols = ['ROSN', 'GAZP']
datasets = snv.get_stocks_data(start_date='2016-01-01', end_date='2024-04-02', symbols=['GAZP'])
```

**Output:** 
The function returns a dictionary with company keys and data on stock prices for the specified period

![image](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/454024f3-6d7a-46b8-a753-01cdcb3e09b5)

## Preprocessing Data
Function `preprocess_stocks_data` takes one argument:
  - datasets - dictionary with data sets obtained in the previous step

```python
preprocessed_datasets = snv.preprocess_stocks_data(datasets)
```

**Output:** 
The function returns a dictionary with processed data ready to enter the model

![image](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/6ee08a6f-4e85-41fb-ad2e-c59eeb401a7b)


## Creating and Fitting models
Function `build_fit_stocks_model` takes a few arguments:
  - preprocessed_datasets - preprocessed datasets obtained in the previous step
  - symbols - list of companies whose stocks you are interested in

```python
metrics = snv.build_fit_stocks_model(preprocessed_datasets, symbols=symbols)
```

**Output:** 
The function creates and trains models, and also returns a dataframe with model metrics

![image](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/d03bcc67-1596-46ea-88dd-9d368f3fc0b4)

After executing the function, a “models__” folder will be created in the project folder, in which the trained models for each company will be saved

## Predicting
Function `predict_future_stock_price` takes a few arguments:
  - preprocessed_datasets - preprocessed datasets obtained in the "Preprocessing Data" step
  - future_days - number of days for which you need to predict the price
  - symbol - name of the company for which the stock price is predicted
  - plot_ - argument responsible for displaying the graph after prediction, True by default

```python
preds = snv.predict_future_stock_price(preprocessed_datasets, future_days=10, symbol='ROSN')
```

The function return `pd.DataFrame` with perdictions of every model and mean predictions like:

![image](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/7ff91d26-2271-46a3-8cd4-bee9e5359234)


And a plots like:

Predictions of Medium Model:

![Plot_1](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/9a9d020a-d79c-40e8-b509-cd32cb12aa12)

Predictions of Small Model:

![Plot_2](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/0c643afd-200e-47d8-869a-9e39cfa44e32)

Predictions of Large Model:

![Plot_3](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/dcfecbfa-d7d7-4dd7-8946-d2f040414b41)

And Mean Predictions of all models:

![Plot_4](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/5480a1c3-605b-4ca9-bfa6-c43f66f499ef)
