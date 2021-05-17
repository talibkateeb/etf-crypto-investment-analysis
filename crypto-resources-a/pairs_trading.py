"""
    Title: Sample Pairs Trading Strategy
    Description: Pairs trading is a market neutral strategy where we go long 
    on one crypto pair and short other crypto pair, betting that the 
    spread between the two would eventually converge to their mean. 
    Style tags: Mean Reversion
    Asset class: Cryptocurrencies
    Data requirement: Data subscription is required for crytp data feed.

    ############################# DISCLAIMER #############################
    This is a strategy template only and should not be
    used for live trading without appropriate backtesting and tweaking of
    the strategy parameters.
    ######################################################################
"""

# Import pandas
import pandas as pd

# Import statsmodel
import statsmodels.api as sm

def initialize(context):

    # Define symbols
    context.security_1 = symbol('BTC')
    context.security_2 = symbol('XMR')

    # The lookback for calculating the hedge ratio
    context.lookback = 90

    # The lookback for calculating the bollinger bands
    context.spread_lookback = 20

    # The position on the spread
    context.position = 0

    # The standard deviation multiplier
    context.mult = 0.5

    # The quantity to be purchased of the dependant asset
    context.quantity = 100

    # Schedule the rebalance function
    schedule_function(
                        rebalance,
                        date_rule=date_rules.every_day(),
                        time_rule=time_rules.market_close(minutes=5)
                     )


def rebalance(context, data):
    """
        A function to rebalance the portfolio. This function is called by the
        schedule_function above
    """

    # Fetch lookback no. days data for the first security
    prices_1 = data.history(
        context.security_1,
        ['close'],
        context.lookback,
        '1d')

    # Fetch lookback no. days data for the second security
    prices_2 = data.history(
        context.security_2,
        ['close'],
        context.lookback,
        '1d')

    # Fit the OLS model and calculate the hedge ratio
    model = sm.OLS(prices_1['close'], prices_2['close'])
    model = model.fit()

    # Dataframe to store the spread and bollinger band information
    df = pd.DataFrame()
    df['spread'] = prices_1['close'] - model.params[0] * prices_2['close']

    # Moving Average and Moving Standard Deviation
    df['moving_average'] = df.spread.rolling(context.spread_lookback).mean()
    df['moving_std_dev'] = df.spread.rolling(context.spread_lookback).std()

    # Upper band and lower band
    df['upper_band'] = df.moving_average + context.mult * df.moving_std_dev
    df['lower_band'] = df.moving_average - context.mult * df.moving_std_dev

    # Get the signal
    long_entry = (df.spread < df.lower_band)[-1]
    long_exit = (df.spread >= df.moving_average)[-1]
    short_entry = (df.spread > df.upper_band)[-1]
    short_exit = (df.spread >= df.moving_average)[-1]

    # Place the order
    if long_entry and context.position == 0:
        # Take long position in the spread
        order_target(context.security_1, context.quantity)
        order_target(context.security_2, -model.params[0]*context.quantity)
        context.position = 1

    elif short_entry and context.position == 0:
        # Take short position in the spread
        order_target(context.security_1, -context.quantity)
        order_target(context.security_2, model.params[0]*context.quantity)
        context.position = -1

    elif long_exit and context.position == 1:
        # Exit the positions
        order_target(context.security_1, 0)
        order_target(context.security_2, 0)
        context.position = 0

    elif short_exit and context.position == -1:
        # Exit the positions
        order_target(context.security_1, 0)
        order_target(context.security_2, 0)
        context.position = 0