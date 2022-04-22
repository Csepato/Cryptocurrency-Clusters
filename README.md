# Cryptocurrency Clusters

![Hero-image](resources/images/Cryptocurrencies.jpeg)

## Background

* You are on the Advisory Services Team of a financial consultancy. One of your clients, a prominent investment bank, is interested in offering a new cryptocurrency investment portfolio for its customers. The company, however, is lost in the vast universe of cryptocurrencies. They’ve asked you to create a report that includes what cryptocurrencies are on the trading market and determine whether they can be grouped to create a classification system for this new investment.

* You have been handed raw data, so you will first need to process it to fit the machine learning models. Since there is no known classification system, you will need to use unsupervised learning. You will use several clustering algorithms to explore whether the cryptocurrencies can be grouped together with other similar cryptocurrencies. You will use data visualization to share your findings with the investment bank.

## DataSource

The dataset was obtained from [CryptoCompare](https://min-api.cryptocompare.com/data/all/coinlist).

## Data Cleaning

Firstly we start by discarding all cryptocurrencies that are not being traded. In others words, filter all currencies that are currently being traded. 
Once this done, we drop the IsTrading column from the dataframe.

```python
#Discard all cryptocurrencies that are not being traded
crypto_df = crypto_df[crypto_df['IsTrading'] == True]
crypto_df.head(10)
```

```python
#Drop IsTrading column from the dataframe
crypto_df.drop(columns=['IsTrading'], inplace=True)
crypto_df
```

```python
```

```python

```
```python
```

```python
```
## Recomendations
Based on my findings, there is not enough features in the dataset to extact a meaningful grouping. Our elbow chart trends downwards with no elbow point,  there is not enough of an 'elbow' in our K-Means plot to signify a meaningful cluster in this dataset. This clustering did not provide much insight into the cryptocurrency trends. More features should be added.

## References

Crypto Coin Comparison Ltd. (2020) Coin market capitalization lists of crypto currencies and prices. Retrieved from [https://www.cryptocompare.com/coins/list/all/USD/1](https://www.cryptocompare.com/coins/list/all/USD/1)
