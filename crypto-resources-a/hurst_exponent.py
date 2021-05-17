"""
    Title: Sample Hurst Exponent Strategy
    Description: This is a long short strategy based on the Hurst exponent and
                 RSI indicator. The strategy rebalances everday at the market open.             
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

# Import talib
import talib as ta

# Import hurst
from hurst import compute_Hc

def initialize(context):

    context.security = symbol('ETH')

    schedule_function(
            rebalance,
            date_rule=date_rules.every_day(),
            time_rule=time_rules.market_open(minutes=1)
        )

def rebalance(context, data):

    price = data.history(
        context.security, ['close'], 60, '1m')

    # Calculate Hurst exponenet
    H = compute_Hc(series, kind='price')[0]

    # Calculate RSI
    rsi = ta.RSI(price.Close.shift(1).values, 14).iloc[-1]

    # Buy signal
    condition_1 = rsi > 75 and H > 0.65
    condition_2 = rsi < 25 and H < 0.35
    buy_signal = condition_1 and condition_2

    # Sell signal
    condition_3 = rsi > 75 and H < 0.35
    condition_4 = rsi < 35 and H > 0.75
    sell_signal = condition_3 and condition_4

    # Place the order
    if buy_signal:
        order_target_percent(security, 1)
        
    elif sell_signal:
        order_target_percent(security, -1)
            
    else:
        order_target_percent(security, 0)