# These functions take inputs up factor, down factor, underlying price, strike price, RFR, Time to expiration and option type to-
#create a binomial option pricing tree for european style options.
# CHECKLIST:
#-American Style-Option Pricing
#-Arbitrage Detector
#-Calculating Delta-Neutral Portfolio

#Jaydon Thinh-To Oct 29 2025

#Updated: Oct 30 2025

import numpy as np
import matplotlib.pyplot as plt



#VARIABLES AND REFFERAL NAMES
up_factor = 1.1 #up_f
down_factor = 0.9 #down_f
underlying_price = 100 #k
strike_price = 90 #strike_k
risk_free_rate = 0.0389 #rfr
timeframe = 3 #t
option = 'put'


def underlying_value_tree(up_f, down_f, k, t):
    #OUTPUTS a single list form t+1 by t+1. Can iterate through it by calling list[row, colum]
    underlying_price_tree = np.zeros((t + 1, t + 1)) #Create an empty array size t + 1
    underlying_price_tree[0,0]= k

    #Recursive loop to create the tree (multiplying top row by up factor, then multiplying all the rows to the bottom
    for i in range(1, t + 1):
        underlying_price_tree[0, i] = underlying_price_tree[0, i-1] * up_f
        for j in range(1, i + 1):
            underlying_price_tree[j, i] = underlying_price_tree[j-1, i-1] * down_f

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
    #Returns the payoff tree (underlying value matrix with each node subtracted by the strike price)

    payoff = underlying_value_tree_matrix
    payoff[0,0] = 0

    for a in range(1, t + 1):
        for b in range(0, a + 1):

            if  underlying_value_tree_matrix[b,a] - strike_k >= 0 and (option_type == 'call'): #checks for non-negativity, passes if option type is a call
                payoff[b,a] =  underlying_value_tree_matrix[b,a] - strike_k #inserts payoff into payoff table

            elif strike_k - underlying_value_tree_matrix[b,a] >= 0 and (option_type == 'put'): #checks for non-negativity, passes if option type is a put
                payoff[b,a] = strike_k - underlying_value_tree_matrix[b,a]

            else: #Else condition if negative, do not sell option thus keep's payoff at 0
                payoff[b,a] = 0

    return payoff


def option_value_tree(up_f, down_f, k, rfr, t, payoff_tree_matrix):
    #Returns the option value tree by iterating backwards through the payoff tree

    risk_neutral_prob = calculate_risk_neutral_probability(up_f, down_f, rfr, t)
    option_value_tree = payoff_tree_matrix.copy()

    for q in range(t-1, -1, -1): #Iterates backwards through the matrix starting at T - 1 to 0
        for x in range(q , -1, -1):
            option_value_node = calculate_option_value_node(
            option_value_tree[x,q + 1], #Sends the value of the node up
            option_value_tree[x + 1,q + 1], #Sends the value of the node down
            rfr,
            t,
            risk_neutral_prob
            )

            option_value_tree[x,q] = round(option_value_node,2) #Inserts the calculated option value into the option value tree


    return option_value_tree, risk_neutral_prob


def print_tree(tree, tree_title):
    steps = int(tree.shape[0]) #number of steps in the tree
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title(tree_title)

    mang0 = '#e3994f'

    for six in range(steps): #67
        for seven in range(six+1):

            y = seven - six /2
            value = tree[seven, six]

            if six < steps - 1:
                ax.plot([six, six + 1], [y, (y + 0.5)], color="grey", lw=1)       # up move
                ax.plot([six, six + 1], [y, (y - 0.5)], color="gray", lw=1)   # down move
            ax.scatter(six, seven - six / 2, color=mang0, s=800)
            ax.text(six, seven - six / 2, f'{value:.2f}$', color='black', ha='center', va='center', fontsize=8)

    plt.gca().invert_yaxis()
    ax.set_xticks(range(steps))
    ax.set_yticks([])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Nodes Steps")
    ax.set_aspect('equal')
    ax.grid(False)
    plt.tight_layout(h_pad=2, w_pad=2)
    plt.show()



underlying_tree = underlying_value_tree(
    up_factor,
    down_factor,
    underlying_price,
    timeframe
)

copy_under_tree = underlying_tree.copy()

payoff_tree = create_payoff_tree(
    strike_price,
    timeframe,
    copy_under_tree,
    option
)


option_tree, risk_n_prob = option_value_tree(
    up_factor,
    down_factor,
    underlying_price,
    risk_free_rate,
    timeframe,
    payoff_tree
)


print_tree(underlying_tree, f"Underlying Price Tree, {option}")
print_tree(option_tree, f"Option Price Tree, {option}")

