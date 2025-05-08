# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 13:19:04 2025

@author: adars
"""

from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime
from Green_Robo_Advisor_Class import RoboAdvisor
import yfinance as yfin
import os

app = Flask(__name__)

# Globals for simplicity (can be modularized)
ticker_data = pd.read_excel("Green_ETF_Selection.xlsx", sheet_name="ETF_Universe")
df_cleaned = None
RA = None

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Input Form Page
@app.route('/form')
def form():
    return render_template('form.html')

# Handle Form Submission
@app.route('/results', methods=['POST'])
def results():
    global df_cleaned, RA

    name = request.form.get('name')
    risk_level = request.form.get('risk')
    horizon = int(request.form.get('horizon'))
    start_date = request.form.get('start_date') or '2021-04-01'
    end_date = request.form.get('end_date') or '2025-04-01'

    # Get ETF prices
    tickers = list(ticker_data.Ticker)
    panel_data = yfin.download(tickers, start=start_date, end=end_date)
    df = pd.DataFrame({label: panel_data[('Close', t)] for t, label in zip(ticker_data.Ticker, ticker_data.Label)})
    df = df.resample('D').mean().bfill().ffill()
    df_cleaned = df

    # Run Robo Advisor
    RA = RoboAdvisor(df, rf='Green Bonds', benchmark='MSCI World SRI')

    # Strategy Mapping
    strategy_map = {
        'Low': 'min-var',
        'Medium': 'max-sharpe-ratio',
        'High': 'max-exp'
    }
    strategy = strategy_map.get(risk_level, 'min-var')
    sol = RA.optimizeWeights(strategy=strategy)
    mu = round(sol @ RA.mu.T * 100, 2)
    sigma = round((sol @ RA.cov @ sol.T)**0.5 * 100, 2)
    sharpe = round((mu/100 - RA.rf) / (sigma/100), 2) if sigma != 0 else 0

    weights = {col: round(w*100, 2) for col, w in zip(df.columns, sol)}

    return render_template(
        'results.html',
        name=name,
        strategy=strategy,
        mu=mu,
        sigma=sigma,
        sharpe=sharpe,
        weights=weights
    )

if __name__ == '__main__':
    app.run(debug=True)
