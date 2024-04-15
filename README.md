# Russian Stock Price Prediction with Neural Networks
The "Russian Stock Price Prediction with Neural Networks" project is a Python package designed to provide investors and financial analysts with valuable insights into the future performance of Russian stocks. By leveraging historical stock price data and advanced machine learning algorithms, this package offers a solution to the complex problem of stock price prediction.

With this package, users can easily obtain historical stock price data for any Russian company of their choice, and use this data to train a neural network to predict future stock prices. The package includes a range of powerful functions and tools to help users create and optimize their neural networks, ensuring that they achieve the most accurate and reliable predictions possible.

One of the key benefits of this project is its ability to help investors make more informed decisions about their investments. By providing accurate predictions about future stock prices, the package can help investors identify potential opportunities for profit, as well as potential risks. For example, an investor who is considering buying shares in a particular company can use this package to predict how the price of those shares is likely to change over the next few days or weeks. This information can help the investor decide whether to buy the shares now, or wait for a better opportunity.

Another important benefit of this project is its ability to help financial analysts and traders develop more effective trading strategies. By using the package to predict future stock prices, analysts can identify trends and patterns in the market, and use this information to develop more sophisticated trading strategies. For example, an analyst may use the package to predict how the price of a particular stock is likely to change in response to a specific news event or economic indicator. This information can then be used to develop a trading strategy that takes advantage of these changes.

Overall, the "Russian Stock Price Prediction with Neural Networks" project is an innovative and powerful tool that can help investors, analysts, and traders make more informed decisions and develop more effective trading strategies. By providing accurate predictions about future stock prices, the package offers a valuable solution to the complex problem of stock price prediction, and has the potential to revolutionize the way that investors and analysts approach the Russian stock market.

# How it works
You can use the code from `Example.ipynb` file to try it with yourself

## Getting Data
Function `get_stocks_data` takes a few arguments:
  - start_time - stating data in string format
  - end_time - ending data in string format
  - symbols - list of companies whose stocks you are interested in. You can see some symbols here: https://www.tradingview.com/markets/stocks-russia/market-movers-large-cap/

**Example:**
```python
import stocks_nn_vap as snv
datasets = snv.get_stocks_data(start_time='2016-01-01', end_time='2024-04-02', symbols=['GAZP'])
```

## Preprocessing Data
Function `preprocess_stocks_data` takes one argument:
  - datasets - вictionary with data sets obtained in the previous step

```python
preprocessed_datasets = snv.preprocess_stocks_data(datasets)
```

## Creating and Fitting models
Function `build_fit_stocks_model` takes a few arguments:
  - preprocessed_datasets - preprocessed datasets obtained in the previous step
  - symbols - list of companies whose stocks you are interested in

```python
snv.build_fit_stocks_model(preprocessed_datasets, symbols=['GAZP'])
```

After executing the function, a “models__” folder will be created in the project folder, in which the trained models for each company will be saved

## Predicting
Function `predict_future_stock_price` takes a few arguments:
  - preprocessed_datasets - preprocessed datasets obtained in the "Preprocessing Data" step
  - future_days - number of days for which you need to predict the price
  - symbol - name of the company for which the stock price is predicted
  - plot_ - argument responsible for displaying the graph after prediction, True by default

```python
preds = snv.predict_future_stock_price(preprocessed_datasets, future_days=10, symbol='SBER')
```

The function return `pd.Series` like:

![image](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/da9ac18c-16a3-4e0b-87e8-6b1b6194137f)

And a plot like:

![image](https://github.com/VladekQ/russian_stock_price_predicting/assets/72941961/3c4da955-a2c1-4794-86fb-eea292555d44)


