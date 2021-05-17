#%%

# Import the required libraries and dependencies

import os

import requests

import json

import pandas as pd

from dotenv import load_dotenv

import alpaca_trade_api as tradeapi

#from MCForecastTools import MCSimulation

import datetime


# Load the environment variables from the .env file
#by calling the load_dotenv function

load_dotenv()

#%%

# The Free Crypto API Call endpoint URLs for the held cryptocurrency assets

btc_url = "https://api.alternative.me/v2/ticker/Bitcoin/?convert=USD"

eth_url = "https://api.alternative.me/v2/ticker/Ethereum/?convert=USD"

link_url = "https://api.alternative.me/v2/ticker/Chainlink/?convert=USD"

zec_url = "https://api.alternative.me/v2/ticker/Zcash/?convert=USD"

atom_url = "https://api.alternative.me/v2/ticker/Cosmos/?convert=USD"

algo_url = "https://api.alternative.me/v2/ticker/Algorand/?convert=USD"

#%%

# Using the Python requests library, make an API call to access the current price of BTC

btc_response = requests.get(btc_url).json()

# Use the json.dumps function to review the response data from the API call
# Use the indent and sort_keys parameters to make the response object readable
# YOUR CODE HERE

print(json.dumps(btc_response, indent=8, sort_keys=True))

#%%

# Using the Python requests library, make an API call to access the current price ETH

eth_response = requests.get(eth_url).json()

# YOUR CODE HERE

# Use the json.dumps function to review the response data from the API call
# Use the indent and sort_keys parameters to make the response object readable
# YOUR CODE HERE
print(json.dumps(eth_response, indent=8, sort_keys=True))

#%%

link_responce = requests.get(link_url).json()


print(json.dumps(link_responce, indent=8, sort_keys=True))

#%%

zec_responce = requests.get(zec_url).json()


print(json.dumps(zec_responce, indent=8, sort_keys=True))

#%%

atom_res = requests.get(atom_url).json()


print(json.dumps(atom_res, indent=8, sort_keys=True))

#%%

algo_responce = requests.get(algo_url).json()


print(json.dumps(algo_responce, indent=8, sort_keys=True))

#%%

btc_price = btc_response['data']['1']["quotes"]["USD"]["price"]

btc_price

#%%

btc_percentage_change_7d = btc_response['data']['1']["quotes"]["USD"]["percent_change_7d"]
btc_percentage_change_7d

#%%

eth_price = eth_response['data']['1027']["quotes"]["USD"]["price"]

eth_price

#%%

eth_percentage_change_7d = eth_response['data']['1027']["quotes"]["USD"]["percent_change_7d"]
eth_percentage_change_7d

#%%

link_price = link_responce['data']['1975']["quotes"]["USD"]["price"]

link_price

#%%

link_percentage_change_7d = link_responce['data']['1975']["quotes"]["USD"]["percent_change_7d"]

link_percentage_change_7d

#%%

zcash_percentage_change_7d =  zec_responce['data']['1437']["quotes"]["USD"]["percent_change_7d"]

#%%

zcash_percentage_change_7d

#%%

atom_prices = atom_res['data']['1420']["quotes"]["USD"]["price"]
atom_prices

#%%

atom_change = atom_res['data']['1420']["quotes"]["USD"]["percent_change_7d"]
atom_change

#%%

algo_prices = algo_responce['data']['0']["quotes"]["USD"]["price"]
algo_prices

#%%

algo_pct = algo_responce['data']['0']["quotes"]["USD"]["percent_change_7d"]
algo_pct
