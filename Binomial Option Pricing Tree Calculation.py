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
up_factor = 1.2
down_factor = 0.9
underlying_price = 100
strike_price = 105
risk_free_rate = 0.0389
timeframe = 3
option = 'call'



def underlying_value_tree(up_factor, down_factor, k, t):
    #OUTPUTS a single list form t+1 by t+1. Can iterate through it by calling list[row, colum]
    underlying_price_tree = np.zeros((t + 1, t + 1)) #Create an empty array size t + 1
    underlying_price_tree[0,0]= k

    #Recursive loop to create the tree (multiplying top row by up factor, then multiplying all the rows to the bottom
    for i in range(1, t + 1):
        underlying_price_tree[0, i] = underlying_price_tree[0, i-1] * up_factor
        for j in range(1, i + 1):
            underlying_price_tree[j, i] = underlying_price_tree[j-1, i-1] * down_factor

    return underlying_price_tree


def calculate_risk_neutral_probability(up_f, down_f, rfr, t):
    #calculate risk neutral probability (p)
    p = ((np.exp(rfr) - down_f) / (up_f - down_f))
    return p

def calculate_option_value_node(node_price_up, node_price_down, rfr, t, risk_neutral_prob):
    #calculates option value for each node
    #To be used for the option_value_tree creation recursive function
    option_value = np.exp(-rfr) * (risk_neutral_prob * node_price_up + (1 - risk_neutral_prob) * node_price_down)

    return option_value


def create_payoff_tree(strike_k, t, underlying_value_tree_matrix, option_type):
    #Returns the payoff tree (underlying value matrix with each node substracted by the strike price)

    payoff = underlying_value_tree_matrix
    payoff[0,0] = 0

    for a in range(1, t + 1):
        for b in range(0, a + 1):

            if  underlying_value_tree_matrix[b,a] - strike_k >= 0 and (option_type == 'call'): #checks for non-negativity, passes if option type is a call
                payoff[b,a] =  underlying_value_tree_matrix[b,a] - strike_k #inserts payoff into payoff table

            elif strike_k - underlying_value_tree_matrix[b,a] >= 0 and (option_type == 'put'): #hecks for non-negativity, passes if option type is a put
                payoff[b,a] = strike_k - underlying_value_tree_matrix[b,a]

            else:
                payoff[b,a] = 0

    return payoff

def option_value_tree(up_f, down_f, k, rfr, t, payoff_tree_matrix):


underlying_tree = underlying_value_tree(up_factor,down_factor,underlying_price,timeframe)
print(underlying_tree)
payoff_tree = create_payoff_tree(strike_price,timeframe,underlying_tree,option)
print(payoff_tree)
print(option_value_tree(up_factor, down_factor, underlying_price, risk_free_rate, timeframe, payoff_tree))
