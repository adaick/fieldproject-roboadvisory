# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:34:26 2022

@author: marcus.becker
"""

# =============================================================================
# Import Libraries
# =============================================================================

#import numpy as np
#For optimization! 
# Note: max f(x)  = - min(-f(x))
#from scipy.optimize import minimize 
# Approximate Jacobian and Hessian
#pip install numdifftools
#from numdifftools import Jacobian, Hessian
import pandas as pd
#import scipy.linalg as linalg
import matplotlib.pyplot as plt
#from heatmap import corrplot #pip install heatmapz
#import seaborn as sns #for pairwise scatter plots
#pip install seaborn==0.10.0.rc0
#Importing Random Numbers
#import random
#Importing Colormaps
#from matplotlib import cm
#Get more colors
#from matplotlib import colors
# For Timestamps
#import datetime
#Load the shares price from Yahoo Finance using Pandas Datareader
#from pandas_datareader import data
#conda install pandas-datareader
#or  pip install pandas-datareader
#from datetime import datetime
# pip install yfinance
import yfinance as yfin
#yfin.pdr_override() # overrides bug in pandas datareader

# Import the Robo Class from External .py file
# Note a .py file named "Robo_Advisor_Class.py" must be stored in the current
# working directory
from Green_Robo_Advisor_Class import RoboAdvisor

def getData(ticker, start_date, end_date):
    
    
    ticker_list = list(ticker.Ticker)
    
    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    panel_data =  yfin.download(ticker_list,
                                     start = start_date,
                                     end = end_date)


    # Create a Panda DatFrame from multicolumn data
    df = pd.DataFrame()

    for t in ticker.Ticker:
        #print(t)
        df[t] = pd.Series(panel_data[('Close',  t)].iloc[:].values, 
                       index = panel_data.index)

    # Use Ticker Labels as Column Name for easier reckognition
    df.columns = ticker.Label
    
    return df


def cleanData(data, method = 'forward'):
    
    
    if method == 'forward':
        
        # Use forward fill to replace missing data (Default)
        # If present ETF price is unknown, use last known price
        # Fundamental Theorem  S_t = E_Q[S_t+1/1+rf] => Martingale Property
        # If rf approx 0 exp. asset prices of tomorrow are similiar to todays prices!    
        cleanedData = data.ffill()
        
    elif method == 'backward':
        
        #Use future known price
        cleanedData = data.bfill()
        
    elif method == 'interpolate':
        
        # Interpolate between last known prices (before and after missing values)
        cleanedData = data.interpolate()
        
    elif method == 'dropna':
        
        # Drop missing price rows (reduces the dataset if nan values are prevelant)
        cleanedData = data.dropna()     
    
    else:
        
        print("Unknown cleaning method. Use forward, backward, interpolate or dropna as option.")
        
    return cleanedData
    




# =============================================================================
#  Import Data
# =============================================================================
    
# Get the ticker list including labels of the Tickers
ticker = pd.read_excel("Green_ETF_Selection.xlsx",sheet_name = "ETF_Universe")

### Select certain ETFs only
#ticker = ticker.iloc[[0, 1, 2, 3, 5, 6]] # with healthcasre and real estate excluded

start_date = '2021-04-01'
end_date = '2025-04-01'

# Ge the pricing Data (needs internet connection)
df = getData(ticker, start_date, end_date)


# =============================================================================
# Clean Data
# =============================================================================

# Resampling dataset with mean intraday prices
df_cleaned = df.resample('D').mean()

# use forward/backward propagation for missing prices 
df_cleaned = cleanData(df_cleaned, method = 'backward') # if  very first entry is missing
df_cleaned = cleanData(df_cleaned, method = 'forward')  # if  very last entry is missing


# =============================================================================
#  Descriptives   
# =============================================================================

# Have a look at the first 10 price rows
print(df_cleaned.head(10))

# Describe Data (count, mean, std, min, max, median etc)
descriptives = df_cleaned.describe()
print(descriptives)

# Get the data Types
print(df_cleaned.dtypes)


# =============================================================================
# Create a Robo Advisor Object    
# =============================================================================

# Creates a Robo Advisor Object
RA = RoboAdvisor(df_cleaned, rf = 'Green Bonds', benchmark = 'MSCI World SRI')

# Creates Log Returns (cleaned)
logR = RA.logR

# =============================================================================
# Plot Data
# =============================================================================

# Choose a colormap
CMAP = "viridis"

# Plot pricing Data
plt.figure()
df_cleaned.plot( figsize=(12,7) , cmap = CMAP )   
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

 # Plot the Daily log-Returns
plt.figure()
logR.plot( figsize=(12,7), title = "Daily Log Returns", cmap = CMAP )
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

# Plot the Cumulative Log-Returns
plt.figure()
logR.cumsum().plot( figsize=(12,7), title = "Cumulative Log-Returns", cmap = CMAP)
plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.show()

# Histogram Plots of Log Returns
RA.plotHistograms()

# Get a Heatmap of Correlations
RA.plotCorr()

# Alternative: Get Pairwise Scatter Plot (also including Histograms)
#sns.pairplot(logR)

# =============================================================================
# Optimize Weights (static 'Buy & Hold' over the investment horizon)
# =============================================================================


print("Number of Assets:", RA.n)

# Create an equaly weight portfolio as starting pont for optimization
w0 = RA.getEqualWeight()

# Get Mean Log Returns
mu = RA.mu

# Get Log Returns of Equal Weights
mu0 = w0 @ mu.T
print("Portfolio Return of an Equal Weight Portfoli", mu0)

# Create a portfolio just in the global equity ETF (like MSCI benchmark)
w_bench = RA.getBenchmark()

# Get return of the Benchmark Portfolio suhc as MSCI
mu_bench = w_bench @ mu.T
print("Benchmark Portfolio Return (MSCI)",mu_bench)

# Descriptives of the Log Returns
descriptives_logR = logR.describe()

# Create a Solution dictionary
solDict = {}

### Min. Variance ###
sol = RA.optimizeWeights(mup = mu_bench,
                     strategy = 'min-var')

print('Min. Variance Solution Weights:', sol)

solDict['min-var'] = sol

### Max. Expected Return ###
Cov = RA.cov
sigma_bench = w_bench @ Cov @ w_bench.T
print("Benchmark Vola (MSCI): sigma_p = ", sigma_bench)

sol1 = RA.optimizeWeights(sigmap = sigma_bench, 
                      strategy = 'max-exp',
                      optimizer = 'trust-constr')

print('Max. Expected Return Solution Weights:', sol1)

solDict['max-exp'] = sol1

### Max. Sharpe Ratio ### 
sol2 = RA.optimizeWeights(strategy = 'max-sharpe-ratio' )

print('Max. Sharpe Ratio Solution Weights:', sol2)

solDict['max-sharpe-ratio'] = sol2


# details = {
#     'strategy' : ['min-var', 'max-exp', 'max-sharpe-ratio'],
#     'starting values' : [w0] * 3,
#     'solver' : ['SLSQP', 'trust-constr', 'SLSQP'],
#     'mup' : [w_bench @ mu.T] * 3,
#     'sigmap' : [ w_bench @ Cov @ w_bench.T] * 3,
#     }

# dftest = pd.DataFrame(details, index = details['strategy'])
# dftest = dftest.drop(columns=['strategy'])

# dftest['solutions'] = solDict.values()
# dftest.dtypes

# test = dftest.loc[ 'min-var' , 'solver']

#Get all optimal strategies straight away
df_solutions = RA.get_all_optimal_solutions()


# =============================================================================
# Plot mu-sigma-diagram with optimal solutions (Monte Carlo Simulation)
# =============================================================================

RA.plot_mu_sigma_diagram(maxSim = 5000, solDict = dict(df_solutions['solutions']), save = True)

# =============================================================================
# Dynamic Portfolio Weights
# =============================================================================

start_time = 180
period_width = 90

# Optimization Period
optimizer_period  = "customized"  # use all datapoints from "start" or from last "period" width or "customized" (define customize period)
custom_period =  3 * period_width #270
# Optimization Strategy
strategy = "min-var"  #"max-exp" or "min-var" or "max-sharpe-ratio"
#Optimization algorithm
optimizer = "SLSQP" # use "SLSQP" for min Var and max-sharpe-ratio use "trust-constr" for max-exp


time, W = RA.get_optimal_allocation_over_time(start_time = start_time,
                                              period_width = period_width,
                                              optimizer_period = optimizer_period,
                                              custom_period = custom_period,
                                              strategy = strategy,
                                              optimizer = optimizer,
                                              bnds = [0.1, 0.9]
                                              )

# Alternative (Default)
#time, W = RA.get_optimal_allocation_over_time()

print("Time:", time)
print("Dynamic Weights", W)

#Associate the time index
time_idx = [df_cleaned.index[t-1] for t in time]
time_idx.append(df.index[-1])

# Plot optimal Weights over time
RA.weightPlot(time = time_idx, 
              W = W, 
              title = "Optimal Portfolio Allocation (without exp. smoothing) \n Strategy: " + strategy + ' Solver: ' + optimizer, 
              save = "False", 
              name = "weight_plot_without_smoothing")

# Alternative
#RA.weightPlot()

# =============================================================================
# Exponential Smoothing
# =============================================================================

# Smoothing parameter (higher alpha giving more weigth to the optimized weights in the current period
# Formula exp. smoothing: y_t^=  alpha*y_t + (1-alpha)* y_t-1
alpha = 0.3

W_smoothing = RA.exp_smoothing(time = time, W = W, alpha = alpha)

# Alternative
# W_smoothing = RA.exp_smoothing()

print('Smoothing weights', W_smoothing)
 
# Plot smoothed optimal Weights over time
RA.weightPlot(time = time_idx, 
              W = W_smoothing, 
              title = "Optimal Portfolio Allocation (with exp. smoothing) \n Strategy: " + strategy + ' Solver: ' + optimizer, 
              save = "True", 
              name = "weight_plot_with_smoothing")

    
# =============================================================================
# Backtesting Results
# (Single ETFs, Benchmarks, static Buy & Hold vs. Dynamic Optimization Results)
# =============================================================================
    
W_dynamic, W_dynamic_smooth, logR_opt = RA.getBacktestingResults(df = df_cleaned, df_solutions = df_solutions,  #for static Buy & Hold Strategies
                                                                 dynamic_strategy = strategy, #For dynamic Strategies
                                                                 start_time = start_time, 
                                                                 time = time, 
                                                                 W = W, 
                                                                 W_smoothing = W_smoothing,                     
                                                                 place_legend = True, #Place legends behind the asset prices
                                                                 save = True
                                                                 )


# Alternative
# RA.getBacktestingResults()

# =============================================================================
# Simulation of different Backtesting Strategies
# =============================================================================

TIME = False #set to False to suppress intensive grid search simulation else True

if TIME:

    df_benchmark_results = pd.DataFrame(logR_opt.cumsum().iloc[-1][:-2])
    df_benchmark_results.rename({logR_opt.cumsum().iloc[-1].name: 'Cum. Return'}, axis=1, inplace=True)
    df_benchmark_results['Min'] =  logR_opt.cumsum().min().iloc[:-2]
    df_benchmark_results['Max'] =  logR_opt.cumsum().max().iloc[:-2]
    df_benchmark_results['Max. Drawdown'] =  logR_opt.min().iloc[:-2]
    df_benchmark_results['Alpha'] =  logR_opt.cumsum().iloc[-1,:-2] - logR_opt.cumsum().iloc[-1].loc['MSCI World SRI']
    
    
    
    testing_dict = {"Strategy" : ["min-var", "max-sharpe-ratio"], #, "max-exp"],
                    "Optimizer" : ["SLSQP"], #, "trust-constr"],
                    "Optimizer Period" : ["customized"], #"start", 
                    "Start Time" : [30, 90, 180],
                    "Period Width" : [30, 90, 180],
                    "Custom Period" : [30, 90, 180, 270],
                    "Smoothing" : [0.25, 0.5, 0.75],
                    "b_min" :[0.05, 0.1],
                    "b_max" :[0.9, 0.95],
                    }
    
    
    
    # Grid Search Optimization
    from sklearn.model_selection import ParameterGrid
    
    for count, params in enumerate(ParameterGrid(testing_dict)):
        
        # Dynamic Portfolio Weights
        time, W = RA.get_optimal_allocation_over_time(start_time = params["Start Time"],
                                                      period_width = params["Period Width"],
                                                      optimizer_period = params["Optimizer Period"],
                                                      custom_period = params["Custom Period"],
                                                      strategy = params["Strategy"],
                                                      optimizer = params["Optimizer"],
                                                      bnds = [params["b_min"], params["b_max"]]
                                                      )
        
        #Associate the time index
        time_idx = [df_cleaned.index[t-1] for t in time]
        time_idx.append(df.index[-1])
        
        
        #Exponential Smoothing Weihts
        W_smoothing = RA.exp_smoothing(time = time, W = W, alpha = params["Smoothing"])
        
        
        W_dynamic, W_dynamic_smooth, logR_opt = RA.getBacktestingResults(df = df_cleaned, df_solutions = df_solutions,  #for static Buy & Hold Strategies
                                                                         dynamic_strategy = params["Strategy"], #For dynamic Strategies
                                                                         start_time = params["Start Time"], 
                                                                         time = time, 
                                                                         W = W, 
                                                                         W_smoothing = W_smoothing, 
                                                                         show = False, #Show daily simple returns and cum returns
                                                                         place_legend = False, #Place legends behind the asset prices
                                                                         save = False
                                                                         )
    
        
        params_ex_alpha = params.copy()
        params_ex_alpha['Smoothing'] = 0
        
        new_strategy = pd.Series({'Cum. Return' : logR_opt.cumsum().iloc[-1,-2], 
                                 'Min' : logR_opt.cumsum().min().iloc[-2], 
                                 'Max' : logR_opt.cumsum().max().iloc[-2], 
                                 'Max. Drawdown' :  logR_opt.min().iloc[-2],
                                 'Alpha' : logR_opt.cumsum().iloc[-1,-2] - logR_opt.cumsum().iloc[-1].loc['MSCI World SRI'],
                                 'Parameter' : params_ex_alpha
                                },
                                name = params["Strategy"] + " " + str(count)
                                )
        
        df_benchmark_results = df_benchmark_results.append(new_strategy)
        
        
        new_strategy_smoothing = pd.Series({'Cum. Return' : logR_opt.cumsum().iloc[-1,-1], 
                                         'Min' : logR_opt.cumsum().min().iloc[-1], 
                                         'Max' : logR_opt.cumsum().max().iloc[-1],
                                         'Max. Drawdown' :  logR_opt.min().iloc[-1],
                                         'Alpha' : logR_opt.cumsum().iloc[-1,-1] - logR_opt.cumsum().iloc[-1].loc['MSCI World SRI'],
                                         'Parameter' : params
                                        },
                                        name = params["Strategy"] + ' (Smoothing)' + " " + str(count)
                                        )
        
        df_benchmark_results = df_benchmark_results.append(new_strategy_smoothing)
        
     
    df_benchmark_results = df_benchmark_results.sort_values( by = ['Alpha', 'Max. Drawdown', 'Min'], 
                                            ascending = [False, False, False])
