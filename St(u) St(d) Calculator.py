#COX ROSS RUBENSTEIN STOCK PRICE CALCULATOR
#This script is to calculate the stock price tree given parameters to be used in the calculations of portfolio delta
#Jaydon Thinh-To Jan 12 2026


import numpy as np
import math as math


Time = 0 #Time | T
Strike = 0 #Strike Price | K
interest_rate = 0.0 #Interest Rate | r
volatility = 0 #Sigma (Volatility) |  vol
div_yield = 0 #Dividend Yield | q
height = 0 #Height of Tree | n

def crr_calculation(T, K, r, vol, q, n):
    Su = math.exp(vol* math.sqrt(T/n))
    Sd = 1/Su
    p = math.exp((r*T/n-Sd)/(Su-Sd))
    print(Su, Sd)




crr_calculation(Time, Strike, interest_rate, volatility, div_yield, height)