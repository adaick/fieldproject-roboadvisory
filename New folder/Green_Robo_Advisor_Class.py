# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 19:44:09 2022
Prototype of a Robo Advisor for the Fields Project (SoSe 2022)
@author: marcus.becker
"""
# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
#For optimization! 
# Note: max f(x)  = - min(-f(x))
from scipy.optimize import minimize 
# Approximate Jacobian and Hessian
#pip install numdifftools
#from numdifftools import Jacobian, Hessian
import pandas as pd
#import scipy.linalg as linalg
import matplotlib.pyplot as plt
#from heatmap import corrplot #pip install heatmapz
import seaborn as sns #for pairwise scatter plots
#pip install seaborn==0.10.0.rc0
#Importing Random Numbers
import random
#Importing Colormaps
from matplotlib import cm
#Get more colors
#from matplotlib import colors
# For Timestamps
import datetime
#Load the shares price from Yahoo Finance using Pandas Datareader
#from pandas_datareader import data
#conda install pandas-datareader
#or  pip install pandas-datareader
# We would like all available data from 01/01/2021 until 12/31/2021.
#from datetime import datetime


# =============================================================================
# Create a Robo Advisor Class
# =============================================================================
class RoboAdvisor:
    
    ### Initializer
    def __init__(self, df, rf, benchmark): # needs cleaned ETF data as input 
        self.data = df #ETF data
        self.n = len(self.data.columns) # number of assets
        self.logR = pd.DataFrame(np.log(self.data).diff()).dropna() # Log Returns
        self.mu = self.logR.mean()  # Mean log Returns
        self.sigma = self.logR.std() 
        self.cov = self.logR.cov()
        self.corr = self.logR.corr()
        self.rf = self.mu[rf] #Proxy for the riskfree interest rate
        self.benchmark = benchmark #Proxy for the benchmark portfolio such as MSCI
        self.w0 = np.ones(self.n) / self.n 
        self.w_bench = np.where(self.data.columns == self.benchmark, 1, 0)

    
    #### Methods
            
    # Create an equal weights portfolio
    def getEqualWeight(self):
        return np.ones(self.n) / self.n 
    
    #Get a benchmark Portolio like the MSCI
    def getBenchmark(self): 
        return np.where(self.data.columns == self.benchmark, 1, 0)
    
    # Calculate Log Returns for a given DataFrame
    def getLogReturn(self, df, plot = False):
        logR = pd.DataFrame(np.log(df).diff())
        logR = logR.dropna()
        if plot:
            # Plot the log-Returns
            plt.figure()
            logR.plot( figsize=(12,7) )
        return logR
    
    # Slice arbitrary DataFrames
    def sliceDataframe(self, df = None, start = 0,  end = None ):
        
        if df is None:
            df = self.data
        
        if end is None:
            end = len(df)
        return df[start:end]
    
    
    # Plot Histograms of Log Returns on Single ETFs
    def plotHistograms(self):
        # For all ETFs
        plt.figure()
        for etf in self.data.columns:
            plt.hist(self.logR[etf])
            plt.title(etf)
            plt.xlabel("log r_i")
            plt.ylabel("Frequency")
            plt.show()
    
    
    def plotCorr(self):
        # Plot heatmap of Correlations
        plt.figure()
        #Source: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
        ax = sns.heatmap(
            self.corr, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n = 200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
        
        #Alternative Correlations plot
        #corrplot(Corr, size_scale=300);

    # For min. Variance only and max-sharpe-ratio
    def optimizeWeights(self,
                        logR = None, # for given log Returns
                        rf = None, # Give a riskfree interest rate
                        mup = 0.0005, # desired portfolion returns
                        sigmap =  0.0001, # maximum volatility threshold  
                        strategy = "min-var",  # or "max-sharpe-ratio", or "max-exp"
                        optimizer = "SLSQP" , # or "SLSQP" or "trust-constr"
                        bnds = [0,1], #minimal and optimal bound per asset
                        w0 = None,  #make a starting guess (uses equal weights if missing)
                        printSol = False):
        
        #Seting Random Seed (for reproducibility)
        random.seed(0)
        
        # If no log Returns are provided use inital ETF time series
        if logR is None:
            logR = self.logR
            
        # If no riskfree interest rate provided use rf chosen from self.data
        if rf is None:
            rf = self.rf
            
        # Number of assets:
        n = len(logR.columns)
        
        if w0 is None :
            w0 = np.ones(n) / n 
            # Test
            #print("Starting Point w0 =", w0)
        
        # Calculate the Variance-Covariance Matrix
        Cov = logR.cov() 
        
        # Calculate the Mean Log Returns
        mu = logR.mean()  
        
        # Limit portfolio holding
        b_min = bnds[0]
        b_max = bnds[1]
        
        if strategy == "min-var":
            #Objective function (Min. portfolio Standard Deviation)
            obj = lambda weights: np.sqrt(weights @ Cov @ weights.T)
            
            #weights constraints
            b=(b_min,b_max) 
            bnds = (b,)
            for i in range(self.n-1):
                bnds = bnds + (b,)
                
            #Define non-linear equality constraints (functions defined above)
            con1={'type':'eq','fun': lambda weights:  weights @ mu.T - mup}
            con2={'type':'eq','fun': lambda weights: weights @ np.ones(len(weights)) - 1.0}
            cons=[con1,con2]
            #Optimal portfolio allocation
            sol = minimize(obj,w0 , method = optimizer, 
                           bounds=bnds, 
                           constraints = cons)
            if printSol:
                print (sol)
        
        elif strategy == "max-exp":
            #Objective function (Min. portfolio variance)
            obj = lambda weights: - weights @ mu.T #max exp. Return
            
            #weights constraints
            b=(b_min,b_max) 
            bnds = (b,)
            for i in range(n-1):
                bnds = bnds + (b,)
                
            #Define non-linear equality constraints (functions defined above)
            con1={'type':'eq','fun': lambda weights: np.sqrt(weights @ Cov @ weights.T) - sigmap}
            con2={'type':'eq','fun': lambda weights: weights @ np.ones(len(weights)) - 1.0}
            cons=[con1,con2]
            #Optimal portfolio allocation
            sol = minimize(obj,w0 , method = optimizer, 
                           bounds=bnds, 
                           constraints = cons)
            if printSol:
                print (sol)
                
        elif strategy == "max-sharpe-ratio":
            
            #Objective function (max-sharpe-ratio)
            obj = lambda weights: - (weights @ mu.T - rf) / np.sqrt(weights @ Cov @ weights.T)        
            
            #weights constraints
            b=(b_min,b_max) 
            bnds = (b,)
            for i in range(n-1):
                bnds = bnds + (b,)
                
            #Define non-linear equality constraints (functions defined above)
            con={'type':'eq','fun': lambda weights: weights @ np.ones(len(weights)) - 1.0}
            #Optimal portfolio allocation
            sol = minimize(obj,w0 , method = optimizer, 
                           bounds=bnds, 
                           constraints = con)
            if printSol:
                print (sol)
            
        else:
            print("Unknown strategy: use min-var, max-exp or max-sharpe-ratio as input-strategy")
            
        return sol.x
    
    def get_all_optimal_solutions(self, details = None, bnds =[0,1] ):
        
        # Optimization Details
        if details is None:
           
            details = {
                'strategy' : ['min-var', 'max-exp', 'max-sharpe-ratio'],
                'starting values' : [self.w0] * 3,
                'solver' : ['SLSQP', 'trust-constr', 'SLSQP'],
                'b_min' : [bnds[0]] * 3,
                'b_max' : [bnds[1]] * 3,
                'mup' : [self.w_bench @ self.mu.T, None,None] ,
                'sigmap' : [ None, np.sqrt(self.w_bench @ self.cov @ self.w_bench.T), None],
                }
  
        # Create a DataFrame from details dictionary
        df = pd.DataFrame(details, index = details['strategy'])
        df = df.drop(columns=['strategy'])
        
        # Solution Dictionary
        solDict = {}
        
        for s in df.index:

            sol = self.optimizeWeights(mup = df.loc[s,'mup'],
                                        sigmap = df.loc[s,'sigmap'],
                                        strategy = s,
                                        optimizer = df.loc[s,'solver'],
                                        w0 = df.loc[s,'starting values'],
                                        bnds = bnds,
                                        )
            
            solDict[s] = sol

        df['solutions'] = solDict.values()          

        return df
    
    def get_optimal_allocation_over_time(self,
                                         df = None, # Provide a pricing Dataframe over time
                                         start_time = 30, # you need some data to do the 1st estimate
                                         period_width = 30, # when does re-allocation take place
                                         strategy = "min-var", #"max-exp" or "min-var" or "max-sharpe-ratio"
                                         optimizer = 'SLSQP', # use "SLSQP" for min Var ans max-sharpe-ratio use "trust-constr" for max-exp
                                         optimizer_period = 'customized', # use all datapoints from "start" or from last "period" width ür "customized"
                                         custom_period = None,  # Used for "customized" lookback periods only 
                                         bnds =[0,1] #bounds consraints
                                         ):
        
        if df is None:
            df = self.data
        
        # Number of assets
        n = len(df.columns)
        
        if custom_period is None:
            custom_period = 3 * period_width
        
        time = [t for t in range(start_time, len(df), period_width) ]
        #time.append(len(df))
        
        #Initialize weighting function
        W = np.zeros((len(time) + 1,n))
        W[0] = np.ones(n) / n 
        i = 1
        
        for t in time:
            
            if optimizer_period  == "start":
                df_backtest = self.sliceDataframe(df, end = t) #estimation from t = 0
            elif optimizer_period  == "period":
                df_backtest = self.sliceDataframe(df, start = t - period_width , end = t) # too much deviation in opt. solution
            elif optimizer_period  == "customized":
                
                if custom_period > t:
                    df_backtest = self.sliceDataframe(df, start = 0 , end = t) # if lookback period goes too far into the past
                else:    
                    df_backtest = self.sliceDataframe(df, start = t - custom_period , end = t) # too much deviation in opt. solution
            else:
                print("Unknown optimization period! Set optimizer_period either to start or period or customized.")
            logR_backtest =  self.getLogReturn(df_backtest)
            # Solve the optimization problem dynamically
            w_backtest = self.optimizeWeights(logR_backtest, 
                                              strategy = strategy, 
                                              optimizer = optimizer,
                                              bnds = bnds
                                              )
            # Update optimal portfolio weights over time
            W[i] = w_backtest 
            i += 1
          
        # Returns the optimal portfolio weights over time   
        W = pd.DataFrame(W, columns = df.columns)
        
        return time, W
    
    def exp_smoothing(self,time = None, W = None, alpha = 0.5, bnds = [0,1]):
        
        if W is None:
                
            print("Note: Method uses Deafault values!")
            start_time = 30
            period_width = 30
            df = self.data
            custom_period = 3 * period_width
            
            time, W = self.get_optimal_allocation_over_time(
                                                 df, # Provide a pricing Dataframe over time
                                                 start_time = start_time, # you need some data to do the 1st estimate
                                                 period_width = period_width, # when does re-allocation take place
                                                 strategy = "min-var", #"max-exp" or "min-var" or "max-sharpe-ratio"
                                                 optimizer = 'SLSQP', # use "SLSQP" for min Var ans max-sharpe-ratio use "trust-constr" for max-exp
                                                 optimizer_period = 'customized', # use all datapoints from "start" or from last "period" width ür "customized"
                                                 custom_period = custom_period,  # Used for "customized" lookback periods only       
                                                 bnds = bnds #bound constrains  
                                                 )
        
        
        # Number of assets
        n = len(W.columns)
        # Get column Names
        columnNames = W.columns
        

        # Initialize the smoothing Weight Matrix over time
        W_smoothing_arr = np.zeros((len(time) + 1, n))
        
        # Transform W to np.array
        W_arr = np.array(W)
        W_smoothing_arr[0] = W_arr[0] # starting with the 1st opt. solution

        #test
        #print(alpha * W[1]  +  (1-alpha) * W[0])

        for i in range(1, len(time) + 1):
            W_smoothing_arr[i] = alpha * W_arr[i]  +  (1-alpha) * W_smoothing_arr[i-1]
        
        
        W_smoothing = pd.DataFrame( W_smoothing_arr, columns = columnNames) 
        
        return W_smoothing
    
    
    def plotSingleETFs(self, name):
        w = np.where(self.data.columns == name, 1, 0)
        mu = w.T @ self.mu
        sigma = np.sqrt(w.T @ self.cov @ w)
        plt.plot(sigma, mu, "o", label = name)
    
    
    def plot_mu_sigma_diagram(self, maxSim = 5000,  #number of portfolios simulated
                             showETFs = True, #Show single ETFS
                             showEqualWeight = True, # Show benchmark portfolios (MSCI and Equal Weight)
                             showEfficientFrontier = True, # Show the efficient frontier 
                             solDict = None,  #Display Optimal solution  
                             save = False # Save Figure Mode 
                             ): 

        plt.figure(figsize=(12,8),)
    
        for i in range(maxSim):
    
            w = np.random.rand(self.n, 1)
            #print(sum(w))
            
            # weights summing up to 100%
            w = 1/sum(w) * w
            #print(sum(w))
            
            mu_pf = w.T @ self.mu
            sigma_pf = np.sqrt(w.T @ self.cov @ w)
            sigma_pf = sigma_pf[0].values # reformat to floating number
            
            plt.plot(sigma_pf, mu_pf, c = 'lightgray', alpha = 0.7, marker='o')
    
    
        if showETFs:
            for etf in self.data.columns:
                self.plotSingleETFs(name = etf)
    
        
        if showEqualWeight:    
            # Add plot for equal variance portfolio
            mu_0 = self.w0.T @ self.mu
            sigma_0 = np.sqrt(self.w0.T @ self.cov @ self.w0)
            plt.plot(sigma_0, mu_0, "om", label = "Equal Weights")
        
        
        if solDict is not None:
            
            #Plot optimal solution for each dictionary entry
            for s in solDict:
                
                # Get the optimal weights
                w = solDict[s]
                mu_opt = w @ self.mu
                sigma_opt = np.sqrt(w @ self.cov @ w.T)
                plt.plot(sigma_opt, mu_opt, '*', markersize = 10, label = s)
        
        if showEfficientFrontier:
            
            if 'max-sharpe-ratio' in solDict.keys():
                
                # Get optimal weights of max-sharpe-ration
                w = solDict['max-sharpe-ratio']
                # Get maximum sigma
                buffer  = 1.0 # set to one if no buffer else >1 e.g. 1.3-1.5 (might deliver unstable optiization results)
                sigma_max = np.sqrt(w @ self.cov @ w.T) * buffer

            else: 
                sigma_max = 0.0065 #on a daily basis
            
            # Add plot for Efficient frontier (until optimal sharpe ratio sigma if possible else o.0065)
            sigma_x = np.linspace(start = 0, stop = sigma_max, num = 100)
            
            # Objective Lambda Function
            obj = lambda weights: - weights @ self.mu.T #max exp. Return

            #Weight constraints (do not allow single asset solutions)
            b=(0,1) 
            bnds = (b,)
            for i in range(self.n-1):
                bnds = bnds + (b,)
                
            w_start = self.w0

            # Holding no cash constraint (indepnedent of sigma_x)
            con2={'type':'eq','fun': lambda weights: weights @ np.ones(len(weights)) - 1.0}    

            for sx in sigma_x:
                #print('sigma_x =', sx)
                #Define non-linear equality constraints (dependix on sigma_x)
                con1={'type':'eq','fun': lambda weights: np.sqrt(weights @ self.cov @ weights.T) - sx}
                
                #List constraints    
                cons=[con1,con2]
             
                #Optimal portfolio allocation
                sol_temp = minimize(obj,w_start , method = "SLSQP", tol = 1e-9,
                                    bounds=bnds, 
                                    constraints = cons)
                #print('sol_temp =', sol_temp.x)
                mu_opt_temp = sol_temp.x @ self.mu
                #print('mu_opt_temp =', mu_opt_temp)
                sigma_opt_temp = np.sqrt(sol_temp.x @ self.cov @ sol_temp.x.T)
                #print('sigma_opt_temp =', sigma_opt_temp)
                plt.plot(sigma_opt_temp, mu_opt_temp,  ".r")
                #Update Starting value in the area of the recent optimal solution
                #w_start = sol_temp.x
                      

        plt.title("mu-sigma-diagram")
        plt.xlabel("sigma")
        plt.ylabel("mu")
        plt.legend( loc = 'lower right')
        #plt.legend(bbox_to_anchor=(1.0, 1.0))
        if save:
            plt.savefig("mu_sigma_diagram.jpg", bbox_inches = 'tight')
        plt.show()
        
    
    def weightPlot(self, 
                   time = None, 
                   W = None, 
                   bnds =[0,1], #bounds consraints
                   title = "Optimal Portfolio Allocation over time", 
                   save = "False", 
                   name = "weight_plot"):
       
        
        
        if W is None:
                
            print("Note: Method uses Deafault values!")
            start_time = 30
            period_width = 30
            df = self.data
            custom_period = 3 * period_width
            
            time, W = self.get_optimal_allocation_over_time(
                                                 df, # Provide a pricing Dataframe over time
                                                 start_time = start_time, # you need some data to do the 1st estimate
                                                 period_width = period_width, # when does re-allocation take place
                                                 strategy = "min-var", #"max-exp" or "min-var" or "max-sharpe-ratio"
                                                 optimizer = 'SLSQP', # use "SLSQP" for min Var ans max-sharpe-ratio use "trust-constr" for max-exp
                                                 optimizer_period = 'customized', # use all datapoints from "start" or from last "period" width ür "customized"
                                                 custom_period = custom_period,  # Used for "customized" lookback periods only  
                                                 bnds = bnds #bounds constraints
                                                 )
    
        # Transofrm to Numpy array for plotting purposes
        W_arr = np.array(W)
        # Getting Column Names
        columnNames = W.columns
        
        #Set period width
        period_width = time[1] - time[0] # Assumes a constant period width
        n_classes = len(W_arr.T)
        cmp = cm.get_cmap("viridis", n_classes)  
        
        plt.figure(figsize=(15,7))
        color = cmp.colors[0]
        plt.bar(time, W_arr[:,0], width = period_width, color = color) 
        w_sum = np.zeros(len(W_arr))
        
        for i in range(1, n_classes):
            color = cmp.colors[i]
            w_sum +=  W_arr[:,i-1]
            plt.bar(time, W_arr[:, i], width = period_width, bottom = w_sum, color = color)
        
        plt.title(title)    
        plt.legend(columnNames, loc= 'center left', bbox_to_anchor=(1, 0.5))
        # Saving the image
        if save:
            plt.savefig(name + ".jpg",  bbox_inches = 'tight')
        plt.show()
    
    def getBacktestingResults(self, df = None, df_solutions = None,  #for static Buy & Hold Strategies
                              dynamic_strategy = 'min-var', start_time = 30, #For dynamic Strategies
                              bnds = [0,1], #minimum and maximum bounds per weight
                              time = None, W = None, W_smoothing = None, #For dynamic Strategies
                              show = True, #Plot daily simple returns and cum rezurns
                              place_legend = False, #Place legends behind the asset prices
                              save = False # Saving images
                              ):
 
        
        if df is None:
            print('Note: Method uses default values for df!')
            df = self.data
        
        if df_solutions is None:
            print('Note: Method uses default values for df_solutions!')
            df_solutions = self.get_all_optimal_solutions( bnds = bnds )
        
        if (time is None) or (W is None):
            print('Note: Method uses default values for time and W!')
            time, W = self.get_optimal_allocation_over_time( bnds = bnds )
        
        if W_smoothing is None:
            print('Note: Method uses default values for time and W_smoothing!')
            W_smoothing = self.exp_smoothing(time = time, W = W, alpha = 0.5, bnds = bnds)
        
        n = len(df.columns)
        
        #### Comparison with static Buy & Hold Strategies
        
        # Equal Weights
        w0 = np.ones(n)/ n 
        optimum0 = df @ w0.T
        # Min-Var
        w = df_solutions.loc['min-var', 'solutions']
        optimum = df @ w.T
        # Max-Exp
        w1 = df_solutions.loc['max-exp', 'solutions']
        optimum1 = df @ w1.T
        # Max Sharpe Ratio
        w2 = df_solutions.loc['max-sharpe-ratio', 'solutions']
        optimum2 = df @ w2.T
        
        # Collect optimal strategies in an overall dataframe including single ETFS
        df_opt = df.copy()
        df_opt["Equal Weights"] = optimum0 #comparison with equal weights portfolio
        df_opt["Min Variance Portfolio (buy & hold)"] = optimum  #comparison with min. Variance portfolio
        df_opt["Max Expected Return (buy & hold)"] = optimum1  #comparison with min. Variance portfolio
        df_opt["Max Sharpe Ratio (buy & hold)"] = optimum2 #Comparison with max. Sharpe Ratiow

        #Plot optimal portfolios comparing to basis titles
        cmap = "viridis"
        
        if show:
            df_opt.plot( figsize=(12,7) , cmap = cmap )
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            if save:
                plt.savefig("asset_prices_static.jpg", bbox_inches = 'tight')
            plt.show()

        logR_opt = pd.DataFrame(np.log(df_opt).diff())
        logR_opt = logR_opt.dropna()
        
        if show:
            logR_opt.plot( figsize=(12,7), title = "Daily Log Returns", cmap = cmap)
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            if save:
                plt.savefig("daily_log_returns_static.jpg", bbox_inches = 'tight')
            plt.show()
        
        # Plot the Cumulative Log-Returns
        if show:
            logR_opt.cumsum().plot( figsize=(12,7), title = "Cumulative Log-Returns" , cmap = cmap )
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            if save:
                plt.savefig("cum_log_returns_static.jpg", bbox_inches = 'tight')
            plt.show()
        
        #### Comparison with Dynamic Trading Strategy (starting from Equal Weights)
        
        # Transforming dynamic strategies into Numpy Arrays,
        W = np.array(W)
        W_smoothing = np.array(W_smoothing)
        
        # Assuming constant weights over one investmend period
        W_dynamic = np.vstack( [ W[0] ] * start_time ) 
        W_dynamic_smooth = np.vstack( [ W_smoothing[0] ] * start_time ) 
        
        period_width = time[1] - time[0] # assuming constant period widths

        for i in range(1, len(W)-1):
            w_block = np.vstack( [ W[i] ] * period_width ) 
            W_dynamic = np.vstack( (W_dynamic, w_block) )
            
            # Smoothing Weights
            w_block_smooth = np.vstack( [ W_smoothing[i] ] * period_width ) 
            W_dynamic_smooth = np.vstack( (W_dynamic_smooth, w_block_smooth) )

        W_dynamic = np.vstack( (W_dynamic, [ W[-1] ] * ( len(df) - len(W_dynamic) ) ) )
        W_dynamic_smooth = np.vstack( (W_dynamic_smooth, [ W_smoothing[-1] ] * ( len(df) - len(W_dynamic_smooth) ) ) )
        
        
        df_backtest_all = df.copy() #self.sliceDataframe(df, end = time[-1]) #estimation from t = 0
          
        """  
        Pre-Check and Demonstration: 

        A = np.arange(1,13).reshape(3,4)
        B = np.array( [1/i for i in range(1,13)] ).reshape(3,4)

        A * B #okay
        np.sum(A * B, axis = 1) #okay
        """
          
        dynamic_opt = np.sum(df_backtest_all * W_dynamic, axis = 1) # elementwise multiplication summed up across each row 
        dynamic_opt_smooth = np.sum(df_backtest_all * W_dynamic_smooth, axis = 1) # elementwise multiplication summed up across each row 


        df_opt["Dynamic " + dynamic_strategy] = dynamic_opt #Comparison with dynamic Min. Var Portfolio
        df_opt["Dynamic " + dynamic_strategy + " Smooth"] = dynamic_opt_smooth #Comparison with dynamic Min. Var Portfolio


        #Coosing a Colormap
        cmap = "viridis"

        #Plots optimal portfolios comparing to basis titles
        if show:
            df_opt.plot( figsize=(12,7), cmap = cmap )
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            if save:
                plt.savefig("asset_prices_all.jpg", bbox_inches = 'tight')
            plt.show()

        logR_opt = pd.DataFrame(np.log(df_opt).diff())
        logR_opt = logR_opt.dropna()
        
        if show:
            logR_opt.plot( figsize=(12,7), title = "Daily Log Returns", cmap = cmap )
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            if save:
                plt.savefig("daily_log_returns_all.jpg", bbox_inches = 'tight')
            plt.show()
        # Plot the Cumulative Log-Returns
        if show:
            logR_opt.cumsum().plot( figsize=(12,7), title = "Cumulative Log-Returns" , cmap = cmap )
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            if save:
                plt.savefig("cum_log_returns_all.jpg", bbox_inches = 'tight')
            plt.show()
        
        if place_legend:
            
            sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.2})
            #sns.set_style("whitegrid")
            custom_style = {
                        'grid.color': '0.8',
                        'grid.linestyle': '--',
                        'grid.linewidth': 0.05,
            }
            sns.set_style(custom_style)  

            dataset = logR_opt.cumsum()
            
            #print(datetime.timedelta(days=1))

            positions = self.legend_positions(df = dataset, y = dataset.columns, rel_Values = True)

            f, ax = plt.subplots(figsize=(20,15))        
            cmap = plt.cm.get_cmap('viridis', len(dataset.columns))

            for i, (column, position) in enumerate(positions.items()):

                # Get a color
                color = cmap(float(i)/len(positions))
                # Plot each line separatly so we can be explicit about color
                ax = dataset.plot(y=column, legend=False, ax=ax, color=color)

                # Add the text to the right
                ax.text(
                    dataset[column].last_valid_index() + datetime.timedelta(days = 1),
                    position, column, fontsize=12,
                    color=color # Same color as line
                )
            ax.set_ylabel('Cumulative Returns')
            # Add percent signs
            ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in ax.get_yticks()])
            sns.despine()
            if save:
                plt.savefig("cum_log_returns_all_place_legends.jpg", bbox_inches = 'tight')
            
        #Return Weightallocations and returns for texting purposes
        return W_dynamic, W_dynamic_smooth, logR_opt
        
    # Helper Function to get the last positions of the label
    # For documentation see http://maxberggren.se/2016/11/21/right-labels/
    # Update: Replace iteritems by items allowing for varying length of columns  
    def legend_positions(self, df, y, rel_Values = False):
        """ Calculate position of labels to the right in plot... """
        positions = {}
        #Adding divisor for relative numbers visualization
        divisor = 1
        if rel_Values:
            divisor = 100
        for column in y:
                positions[column] = df[column].values[-1] - 0.5 / divisor 

        def push():
            """
            ...by puting them to the last y value and
            pushing until no overlap
            """
            collisions = 0
            for column1, value1 in positions.items():
                for column2, value2 in positions.items():
                    if column1 != column2:
                        dist = abs(value1-value2)                        
                        if dist < 2.5 / divisor :
                            collisions += 1
                            if value1 < value2:
                                positions[column1] -= .1 / divisor 
                                positions[column2] += .1 / divisor 
                            else:
                                positions[column1] += .1 / divisor 
                                positions[column2] -= .1 / divisor 
                            return True
        while True:
            pushed = push()
            if not pushed:
                break

        return positions


      
