# These functions take inputs up factor, down factor, underlying price, strike price, RFIR, Time to expiration
#Jaydon Thinh-To Oct 29 2025

import numpy as np


up_factor = 1.1
down_factor = 0.9
underlying_price = 100
strike_price = 105
risk_free_rate = 0.389
timeframe = 3

def underlying_value_tree(up_factor, down_factor, underlying_price, strike_price, risk_free_rate, t): #t = time

    option_price_tree = np.zeros((t + 1, t + 1)) #Create an empty array size t + 1
    option_price_tree[0,0]= underlying_price

    for i in range(1, t + 1):
        option_price_tree[0, i] = option_price_tree[0, i-1] * up_factor
        for j in range(1, i + 1):
            option_price_tree[j, i] = option_price_tree[j-1, i-1] * down_factor

    return option_price_tree

print(underlying_value_tree(up_factor,down_factor,underlying_price,strike_price,risk_free_rate,timeframe))

def option_value_tree