#binomial model
# -American/Ueorpean Style-Option Pricing
# -Calculating Delta-Neutral Portfolio
#
# Roan McReynolds Oct 29 2025
# Updated: Jan 29 2026

import numpy as np
import matplotlib.pyplot as plt
import math
import yfinance as yf

ticker = str(input("Input Option Ticker"))

option_type: str = input("Please enter option type (call or put): ").strip().lower() 
while option_type not in ("call", "put"):
    option_type = input("Invalid input. Please enter 'call' or 'put': ").strip().lower()
style = input("Please enter option style (american or european): ").strip().lower()
while style not in ("american", "european"):
    style = input("Invalid input. Please enter 'american' or 'european': ").strip().lower()
lookback: str = "1y",               # used for volatility estimation
interval_b: str = "1d",
vol_method: str = "log",          
timeSteps: int = 200,
pick: str = "atm"    

### OLD METHOD OF IMPUTS                        ###
# annVol = float(input("Please enter the annualized volatility: "))
# timeToExpiry = float(input("Please enter the time to expiry (in years):"))
# underlying_price = float(input("Please enter the underlying price S₀ (e.g., 100): "))
# strike_price = float(input("Please enter the strike price K (e.g., 90): "))
# risk_free_rate = float(input("Please enter the risk-free rate r (decimal, e.g., 0.0389): "))
# timeSteps = int(input("Please enter the number of desired time steps: "))
###                                             ###

tk = yf.Ticker(ticker)

#Historical price of each day for the last 5 days
hist = tk.history(period="5d",interval=interval_b)

if hist.empty:
    raise ValueError("Not enough historical data")
underlying_price = float(hist["Close"].iloc[-1])

px = tk.history(period=lookback, interval=interval_b)["Close"].dropna()
if len(px) < 30:
    raise ValueError("Not enough historical data to estimate volatility. Try a longer lookback.")

if vol_method == "log":
    rets = np.log(px / px.shift(1)).dropna()
else:
    rets = px.pct_change().dropna()
    annVol = float(rets.std(ddof=1) * np.sqrt(252)) #252 -> num trading days 

expiries = tk.options
if not expiries:
    raise ValueError("No option expiries returned. This ticker may not have listed options on yfinance.")


# calculate up and down factors (CRR)
up_factor = math.exp(annVol * math.sqrt(timeToExpiry / timeSteps))
down_factor = math.exp(-annVol * math.sqrt(timeToExpiry / timeSteps))





def underlying_value_tree(up_f, down_f, k, t):
    underlying_price_tree = np.zeros((t + 1, t + 1))  
    underlying_price_tree[0, 0] = k

    # Recursive loop to create the tree
    for i in range(1, t + 1):
        underlying_price_tree[0, i] = underlying_price_tree[0, i - 1] * up_f
        for j in range(1, i + 1):
            underlying_price_tree[j, i] = underlying_price_tree[j - 1, i - 1] * down_f

    return underlying_price_tree


def calculate_risk_neutral_probability(up_f, down_f, rfr, ts, tte):
    dt = tte / ts
    p = (np.exp(rfr * dt) - down_f) / (up_f - down_f)
    return p


def calculate_option_value_node(node_price_up, node_price_down, rfr, dt, risk_neutral_prob):
    # calculates option value for each node
    option_value = np.exp(-rfr * dt) * (risk_neutral_prob * node_price_up + (1 - risk_neutral_prob) * node_price_down)
    return option_value


def create_payoff_tree(strike_k, t, underlying_value_tree_matrix, option_type):
    # Returns the payoff tree
    payoff = underlying_value_tree_matrix
    payoff[0, 0] = 0

    for a in range(1, t + 1):
        for b in range(0, a + 1):
            if underlying_value_tree_matrix[b, a] - strike_k >= 0 and (option_type == 'call'):
                payoff[b, a] = underlying_value_tree_matrix[b, a] - strike_k
            elif strike_k - underlying_value_tree_matrix[b, a] >= 0 and (option_type == 'put'):
                payoff[b, a] = strike_k - underlying_value_tree_matrix[b, a]
            else:
                payoff[b, a] = 0

    return payoff


def option_value_tree(up_f, down_f, k, rfr, timeSteps, timeToExpiry, payoff_tree_matrix):
    # Returns the option value tree by iterating backwards through the payoff tree

    dt = timeToExpiry / timeSteps
    risk_neutral_prob = calculate_risk_neutral_probability(up_f, down_f, rfr, timeSteps, timeToExpiry)

    option_value_tree = payoff_tree_matrix.copy()
    delta_tree = np.zeros_like(option_value_tree)

    for q in range(timeSteps - 1, -1, -1):
        for x in range(q, -1, -1):

            # Child node option values and underlying prices
            V_u = option_value_tree[x, q + 1]
            V_d = option_value_tree[x + 1, q + 1]

           
            S_u = underlying_tree[x, q + 1]
            S_d = underlying_tree[x + 1, q + 1]

            # Binomial delta at node (x, q)
            delta_tree[x, q] = (V_u - V_d) / (S_u - S_d)

            hold = calculate_option_value_node(V_u, V_d, rfr, dt, risk_neutral_prob)

            if style == "american":
                # Immediate exercise 
                S = underlying_tree[x, q]
                if option == "call":
                    exercise = max(S - k, 0.0)
                else:
                    exercise = max(k - S, 0.0)

                option_value_tree[x, q] = max(hold, exercise)
            else:
                option_value_tree[x, q] = hold

    return option_value_tree, risk_neutral_prob, delta_tree


def print_tree(tree, tree_title):
    steps = int(tree.shape[0])  # number of steps in the tree
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(tree_title)

    mang0 = '#e3994f'

    for six in range(steps):
        for seven in range(six + 1):

            y = seven - six / 2
            value = tree[seven, six]

            if six < steps - 1:
                ax.plot([six, six + 1], [y, (y + 0.5)], color="grey", lw=1)  # up move
                ax.plot([six, six + 1], [y, (y - 0.5)], color="gray", lw=1)  # down move
            ax.scatter(six, seven - six / 2, color=mang0, s=800)
            ax.text(six, seven - six / 2, f'{value:.2f}', color='black', ha='center', va='center', fontsize=8)

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
    timeSteps
)

copy_under_tree = underlying_tree.copy()

payoff_tree = create_payoff_tree(
    strike_price,
    timeSteps,
    copy_under_tree,
    option
)


option_tree, risk_n_prob, delta_tree = option_value_tree(
    up_factor,
    down_factor,
    strike_price,     
    risk_free_rate,
    timeSteps,
    timeToExpiry,     
    payoff_tree
)

shares_tree = -delta_tree * 100  # 1 contract = 100 shares

print("")
print_tree(underlying_tree, f"Underlying Price Tree, {option}")
print_tree(option_tree, f"Option Price Tree, {option}")
print_tree(delta_tree, f"Delta Tree, {option}")
print_tree(shares_tree, f"Hedge Shares Tree (Buy + / Sell -), {option}")

