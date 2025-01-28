# indicators.py

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG

class TechnicalIndicators:
    @staticmethod
    def add_ema(df, custom_params=None):
        """Add EMA indicators with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['ema']
        
        df[f'ema_{config["short"]}'] = df['close_price'].ewm(span=config['short'], adjust=False).mean()
        df[f'ema_{config["medium"]}'] = df['close_price'].ewm(span=config['medium'], adjust=False).mean()
        df[f'ema_{config["long"]}'] = df['close_price'].ewm(span=config['long'], adjust=False).mean()
        
        return df

    @staticmethod
    def add_macd(df, custom_params=None):
        """Add MACD indicators with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['macd']
        
        # Calculate MACD line
        exp1 = df['close_price'].ewm(span=config['fast'], adjust=False).mean()
        exp2 = df['close_price'].ewm(span=config['slow'], adjust=False).mean()
        df['macd_line'] = exp1 - exp2
        
        # Calculate Signal line
        df['signal_line'] = df['macd_line'].ewm(span=config['signal'], adjust=False).mean()
        
        # Calculate MACD histogram
        df['macd_histogram'] = df['macd_line'] - df['signal_line']
        
        return df

    @staticmethod
    def add_rsi(df, custom_params=None):
        """Add RSI indicator with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['rsi']
        
        # Calculate price changes
        delta = df['close_price'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=config['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config['period']).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def add_stochastic(df, custom_params=None):
        """Add Stochastic indicators with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['stochastic']
        
        # Calculate %K
        lowest_low = df['low_price'].rolling(window=config['k_period']).min()
        highest_high = df['high_price'].rolling(window=config['k_period']).max()
        df['stoch_k'] = 100 * (df['close_price'] - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=config['d_period']).mean()
        
        return df

    @staticmethod
    def add_volume_rsi(df, custom_params=None):
        """Add Volume RSI with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['volume_rsi']
        
        # Calculate volume changes
        delta = df['volume_crypto'].diff()
        
        # Separate increases and decreases
        gain = (delta.where(delta > 0, 0)).rolling(window=config['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config['period']).mean()
        
        # Calculate RS and Volume RSI
        rs = gain / loss
        df['volume_rsi'] = 100 - (100 / (1 + rs))
        
        return df

    @staticmethod
    def add_vwap(df, custom_params=None):
        """Add VWAP with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['vwap']
        
        # Start fresh each day
        df['date'] = df.index.date
        
        # Group by date and calculate VWAP
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        df['pv'] = df['typical_price'] * df['volume_crypto']
        
        # Calculate running sum for each day
        df['cum_pv'] = df.groupby('date')['pv'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume_crypto'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cum_pv'] / df['cum_volume']
        
        # Clean up temporary columns
        df.drop(['typical_price', 'pv', 'cum_pv', 'cum_volume', 'date'], axis=1, inplace=True)
        
        return df

    @staticmethod
    def add_market_characteristics(df):
        """Add various market characteristics for correlation analysis"""
        df = df.copy()
        
        # Volatility (20-period standard deviation of returns)
        df['volatility'] = df['close_price'].pct_change().rolling(window=20).std()
        
        # Trend Direction and Strength
        if 'ema_9' not in df.columns or 'ema_50' not in df.columns:
            df = TechnicalIndicators.add_ema(df)
        df['trend_direction'] = np.sign(df['ema_9'] - df['ema_50'])
        df['trend_strength'] = ((df['ema_9'] - df['ema_50']).abs() / df['ema_50']).fillna(0)
        
        # Volume characteristics
        df['volume_ma'] = df['volume_crypto'].rolling(window=20).mean()
        df['relative_volume'] = df['volume_crypto'] / df['volume_ma']
        
        # Price momentum (5-period returns)
        df['momentum'] = df['close_price'].pct_change(periods=5)
        
        # True Range and ATR
        high_low = df['high_price'] - df['low_price']
        high_close = abs(df['high_price'] - df['close_price'].shift())
        low_close = abs(df['low_price'] - df['close_price'].shift())
        df['true_range'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        return df

    @classmethod
    def add_all_indicators(cls, df, custom_params=None):
        """
        Add all technical indicators to the DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame
            custom_params (dict, optional): Custom parameters for indicators
        """
        print("\nAdding all indicators...")
        df = df.copy()
        
        # Add market characteristics first
        df = cls.add_market_characteristics(df)
        
        # Add each indicator with either custom or default parameters
        indicators = ['ema', 'macd', 'rsi', 'stochastic', 'volume_rsi', 'vwap']
        for indicator in indicators:
            try:
                params = custom_params.get(indicator) if custom_params else None
                df = getattr(cls, f'add_{indicator}')(df, params)
            except Exception as e:
                print(f"Error adding {indicator}: {str(e)}")
                raise
        
        # Drop NaN values created by indicators
        df.dropna(inplace=True)
        
        return df