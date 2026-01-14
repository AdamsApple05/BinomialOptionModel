#COX ROSS RUBENSTEIN STOCK PRICE CALCULATOR
#This script is to calculate the stock price tree given parameters to be used in the calculations of portfolio delta
#Jaydon Thinh-To Jan 12 2026


import numpy as np
import math as math


Time = 5 #Time | T
Strike = 10 #Strike Price | K
interest_rate = 0.2 #Interest Rate | r
volatility = 0.5 #Sigma (Volatility) |  vol
div_yield = 2 #Dividend Yield | q
height = 2 #Height of Tree | n

def crr_calculation(T, K, r, vol, q, n):
    if n == 0:
        return None

    su = math.exp(vol* math.sqrt(T/n))
    sd = 1/su
    p = math.exp((r*T/n-sd)/(su-sd))

    print(su, sd, p)




crr_calculation(Time, Strike, interest_rate, volatility, div_yield, height)