#strategies.py

import json
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import os
from config import BACKTEST_CONFIG
from indicators import TechnicalIndicators

class TradingStrategies:
    @staticmethod
    def ema_strategy(df, custom_params=None):
        """
        EMA strategy with optional custom parameters
        """
        signals = pd.Series(index=df.index, data=0)
        
        if custom_params is not None:
            short = custom_params['short']
            medium = custom_params['medium']
        else:
            short = BACKTEST_CONFIG['ema']['short']
            medium = BACKTEST_CONFIG['ema']['medium']
        
        # Generate buy signal when short EMA crosses above medium EMA
        signals[df[f'ema_{short}'] > df[f'ema_{medium}']] = 1
        # Generate sell signal when short EMA crosses below medium EMA
        signals[df[f'ema_{short}'] < df[f'ema_{medium}']] = -1
        
        return signals

    @staticmethod
    def macd_strategy(df, custom_params=None):
        """
        MACD strategy with optional custom parameters
        """
        signals = pd.Series(index=df.index, data=0)
        
        # Buy when MACD line crosses above signal line
        signals[df['macd_line'] > df['signal_line']] = 1
        # Sell when MACD line crosses below signal line
        signals[df['macd_line'] < df['signal_line']] = -1
        
        return signals

    @staticmethod
    def rsi_strategy(df, custom_params=None):
        """
        RSI strategy with optional custom parameters
        """
        signals = pd.Series(index=df.index, data=0)
        
        if custom_params is not None:
            overbought = custom_params['overbought']
            oversold = custom_params['oversold']
        else:
            overbought = BACKTEST_CONFIG['rsi']['overbought']
            oversold = BACKTEST_CONFIG['rsi']['oversold']
        
        # Buy when RSI crosses below oversold level
        signals[df['rsi'] < oversold] = 1
        # Sell when RSI crosses above overbought level
        signals[df['rsi'] > overbought] = -1
        
        return signals

    @staticmethod
    def stochastic_strategy(df, custom_params=None):
        """
        Stochastic strategy with optional custom parameters
        """
        signals = pd.Series(index=df.index, data=0)
        
        if custom_params is not None:
            overbought = custom_params['overbought']
            oversold = custom_params['oversold']
        else:
            overbought = BACKTEST_CONFIG['stochastic']['overbought']
            oversold = BACKTEST_CONFIG['stochastic']['oversold']
        
        # Buy when both %K and %D are below oversold level
        signals[(df['stoch_k'] < oversold) & 
               (df['stoch_d'] < oversold)] = 1
        
        # Sell when both %K and %D are above overbought level
        signals[(df['stoch_k'] > overbought) & 
               (df['stoch_d'] > overbought)] = -1
        
        return signals

    @staticmethod
    def vwap_strategy(df, custom_params=None):
        """
        VWAP strategy with optional custom parameters
        """
        signals = pd.Series(index=df.index, data=0)
        
        if custom_params is not None:
            overbought = custom_params['overbought']
            oversold = custom_params['oversold']
        else:
            overbought = BACKTEST_CONFIG['vwap']['overbought']
            oversold = BACKTEST_CONFIG['vwap']['oversold']
        
        # Calculate price to VWAP ratio
        price_to_vwap = df['close_price'] / df['vwap']
        
        # Buy when price is below VWAP by oversold threshold
        signals[price_to_vwap < oversold] = 1
        # Sell when price is above VWAP by overbought threshold
        signals[price_to_vwap > overbought] = -1
        
        return signals

    @staticmethod
    def volume_rsi_strategy(df, custom_params=None):
        """
        Volume RSI strategy with optional custom parameters
        """
        signals = pd.Series(index=df.index, data=0)
        
        if custom_params is not None:
            overbought = custom_params['overbought']
            oversold = custom_params['oversold']
        else:
            overbought = BACKTEST_CONFIG['volume_rsi']['overbought']
            oversold = BACKTEST_CONFIG['volume_rsi']['oversold']
        
        # Buy when Volume RSI is below oversold level
        signals[df['volume_rsi'] < oversold] = 1
        # Sell when Volume RSI is above overbought level
        signals[df['volume_rsi'] > overbought] = -1
        
        return signals

    @classmethod
    def get_all_strategies(cls):
        """
        Returns all base strategies (excluding optimized strategy which requires additional parameters)
        """
        return {
            'EMA': cls.ema_strategy,
            'MACD': cls.macd_strategy,
            'RSI': cls.rsi_strategy,
            'Stochastic': cls.stochastic_strategy,
            'Volume RSI': cls.volume_rsi_strategy,
            'VWAP': cls.vwap_strategy
        }