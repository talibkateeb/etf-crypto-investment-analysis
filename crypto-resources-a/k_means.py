"""
    Title: Sample K-means Clustering Strategy
    Description: This is a long short strategy based on the K-means clustering
                 and simple moving average.
    Style tags: Systematic
    Asset class: Cryptocurrencies
    Data requirement: Data subscription is required for crytp data feed.

    ############################# DISCLAIMER #############################
    This is a strategy template only and should not be
    used for live trading without appropriate backtesting and tweaking of
    the strategy parameters.
    ######################################################################
"""
# Import numpy
import numpy as np

# Import KMeans from sklearn
from sklearn.cluster import KMeans

def initialize(context):

    # Define symbol
    context.security = symbol('ETH')
    
    # Schedule the strategy logic every day at market close
    schedule_function(
        rebalance,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_close()
    )


def rebalance(context, data):

    # Fetch price data
    price = data.history(
        context.security, ['open', 'high', 'low', 'close'], 60, '1m')

    # Calculate SMA
    price['SMA'] = price.close.shift(1).rolling(30).mean()

    # Calculate the absolute value of percentage change in prices
    price['Movement'] = price.close.shift(1).pct_change().abs()

    # Calculate the market return values
    price['Return'] = price.close.pct_change()

    # Drop all the null values
    price = price.dropna()

    # Instantiate a K-Means model
    clus = KMeans(2)

    # Fit the K-Means model to the train data to train the model.
    # Here we need to reshape the data as the input contains a single column
    clus.fit(np.reshape(price.Movement.values, (-1, 1)))

    # Assign the predicted cluster values to the cluster column.
    # We use the predict function to predict the cluster values
    price['Cluster'] = clus.predict(np.reshape(price.Movement.values, (-1, 1)))

    long_entry = (price.SMA[-1] < price.close[-2]
                  and price.Cluster[-1] == 0)

    short_entry = (price.SMA[-1] > price.close[-2]
                   and price.Cluster[-1] == 0)

    # Place the orders 
    if long_entry:
        order_target_percent(context.security, 1)

    elif short_entry:
        order_target_percent(context.security, -1)

    else:
        order_target_percent(context.security, 0)