# These functions take inputs up factor, down factor, underlying price, strike price, RFR, Time to expiration
# CHECKLIST:
#-Underlying Value Tree (✅)
#-Option Value Tree (X)
#       -Risk Neutral Prob (✅)
#       -Option Value Node Function (✅)

#Jaydon Thinh-To Oct 29 2025

import numpy as np
from math import e

#test variables
up_factor = 1.1
down_factor = 0.9
underlying_price = 100
strike_price = 105
risk_free_rate = 0.389
timeframe = 3

#NOTE rfr is risk free rate and t = time in most cases

def underlying_value_tree(up_factor, down_factor, underlying_price, t):
    #OUTPUTS a single list form t+1 by t+1. Can iterate through it by calling list[row, colum]

    price_tree = np.zeros((t + 1, t + 1)) #Create an empty array size t + 1
    price_tree[0,0]= underlying_price

    #Recursive loop to create the tree (multiplying top row by up factor, then multiplying all the rows to the bottom
    for i in range(1, t + 1):
        price_tree[0, i] = price_tree[0, i-1] * up_factor
        for j in range(1, i + 1):
            price_tree[j, i] = price_tree[j-1, i-1] * down_factor

    return price_tree



def option_value_tree(up_factor, down_factor, underlying_price, strike_price, rfr, t, underlying_value_tree_matrix):
    pass


def risk_neutral_probability(node_price_up, node_price_down, rfr, t):
    #calculate risk neutral probability (p)
    p = ((e** (rfr * t)) * underlying_price - node_price_down) / (node_price_up - node_price_down)
    return p

def option_value_node(node_price_up, node_price_down, rfr, t, risk_neutral_prob):
    #calculates option value for each node
    #To be used for the option_value_tree creation recursive function
    option_value = e**(-rfr * t) * (risk_neutral_prob * node_price_up + (1 - risk_neutral_prob) * node_price_down)
    return option_value

#test underlying_tree
underlying_tree = underlying_value_tree(up_factor,down_factor,underlying_price,timeframe)
print(underlying_tree)
