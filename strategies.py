#strategies.py

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

    @staticmethod
    def optimize_strategy_parameters(df, symbol, start_date, end_date):
        """
        Optimizes strategy parameters for different market conditions using training data.
        """
        print(f"\nOptimizing strategy parameters for {symbol}")
        print(f"Training period: {start_date} to {end_date}")
        
        # Filter data for training period
        mask = (df.index >= start_date) & (df.index <= end_date)
        train_df = df[mask].copy()
        
        # Initialize results storage
        best_params = {}
        strategy_weights = {}
        market_condition_params = {
            'high_volatility': {},
            'low_volatility': {},
            'strong_trend': {},
            'weak_trend': {},
            'high_volume': {},
            'low_volume': {}
        }
        
        # Calculate market conditions
        volatility = train_df['close_price'].pct_change().rolling(20).std()
        trend_strength = ((train_df['ema_9'] - train_df['ema_50']).abs() / train_df['ema_50']).fillna(0)
        volume_ma = train_df['volume_crypto'].rolling(20).mean()
        relative_volume = train_df['volume_crypto'] / volume_ma
        
        # Define market condition masks
        conditions = {
            'high_volatility': volatility > volatility.quantile(0.75),
            'low_volatility': volatility < volatility.quantile(0.25),
            'strong_trend': trend_strength > trend_strength.quantile(0.75),
            'weak_trend': trend_strength < trend_strength.quantile(0.25),
            'high_volume': relative_volume > relative_volume.quantile(0.75),
            'low_volume': relative_volume < relative_volume.quantile(0.25)
        }
        
        # Get parameter ranges from config
        param_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges']
        
        # Process each strategy
        for strategy_name, param_range in param_ranges.items():
            print(f"\nOptimizing {strategy_name} strategy...")
            best_sharpe = -np.inf
            best_params_overall = None
            
            # Generate parameter combinations
            param_combinations = list(itertools.product(*[
                param_range[param] if isinstance(param_range[param], (list, range)) 
                else [param_range[param]] 
                for param in param_range.keys()
            ]))
            
            total_combinations = len(param_combinations)
            print(f"Testing {total_combinations} parameter combinations...")
            
            for i, params in enumerate(param_combinations):
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{total_combinations} combinations tested")
                    
                param_dict = dict(zip(param_range.keys(), params))
                try:
                    # Recalculate indicators with current parameters
                    df_with_indicators = TechnicalIndicators.add_all_indicators(
                        train_df, 
                        custom_params={strategy_name: param_dict}
                    )
                    
                    # Get strategy signals
                    strategy_func = getattr(TradingStrategies, f"{strategy_name}_strategy")
                    signals = strategy_func(df_with_indicators, custom_params=param_dict)
                    
                    # Calculate returns and Sharpe ratio
                    returns = df_with_indicators['close_price'].pct_change() * signals.shift(1)
                    returns = returns.fillna(0)
                    
                    if len(returns) > 0 and returns.std() != 0:
                        sharpe = np.sqrt(365 * 24) * returns.mean() / returns.std()
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params_overall = param_dict
                            
                            # Evaluate for each market condition
                            for condition_name, mask in conditions.items():
                                condition_returns = returns[mask]
                                if len(condition_returns) > 0 and condition_returns.std() != 0:
                                    condition_sharpe = np.sqrt(365 * 24) * condition_returns.mean() / condition_returns.std()
                                    
                                    # Update if first time seeing this condition or better than previous best
                                    if (strategy_name not in market_condition_params[condition_name] or 
                                        condition_sharpe > market_condition_params[condition_name][strategy_name]['sharpe']):
                                        market_condition_params[condition_name][strategy_name] = {
                                            'params': param_dict.copy(),
                                            'sharpe': condition_sharpe
                                        }
                
                except Exception as e:
                    print(f"Error testing parameters {param_dict}: {str(e)}")
                    continue
            
            best_params[strategy_name] = best_params_overall
            strategy_weights[strategy_name] = max(0, best_sharpe)
        
        # Normalize strategy weights
        total_weight = sum(strategy_weights.values())
        if total_weight > 0:
            strategy_weights = {k: v/total_weight for k, v in strategy_weights.items()}
        
        # Save results to file
        results_dir = BACKTEST_CONFIG['results_dir']
        symbol_dir = os.path.join(results_dir, symbol.replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(symbol_dir, f'optimization_results_{timestamp}.txt')
        
        with open(results_file, 'w') as f:
            f.write(f"Strategy Optimization Results for {symbol}\n")
            f.write(f"Training period: {start_date} to {end_date}\n\n")
            
            f.write("Best Parameters:\n")
            for strategy, params in best_params.items():
                f.write(f"{strategy}: {params}\n")
            
            f.write("\nStrategy Weights:\n")
            for strategy, weight in strategy_weights.items():
                f.write(f"{strategy}: {weight:.4f}\n")
            
            f.write("\nMarket Condition-Specific Parameters:\n")
            for condition, strategies in market_condition_params.items():
                f.write(f"\n{condition}:\n")
                for strategy, result in strategies.items():
                    f.write(f"  {strategy}:\n")
                    f.write(f"    Parameters: {result['params']}\n")
                    f.write(f"    Sharpe Ratio: {result['sharpe']:.4f}\n")
        
        return {
            'best_params': best_params,
            'strategy_weights': strategy_weights,
            'market_condition_params': market_condition_params,
            'results_file': results_file
        }

    @staticmethod
    def optimized_adaptive_strategy(df, optimization_results):
        """
        Uses optimized parameters and weights to generate trading signals.
        Calculates indicators once for each parameter set.
        """
        signals = pd.Series(index=df.index, data=0.0)
        best_params = optimization_results['best_params']
        strategy_weights = optimization_results['strategy_weights']
        market_condition_params = optimization_results.get('market_condition_params', {})
        
        # Calculate market conditions once for the entire dataset
        print("\nCalculating market conditions...")
        volatility = df['close_price'].pct_change().rolling(20).std()
        trend_strength = ((df['ema_9'] - df['ema_50']).abs() / df['ema_50']).fillna(0)
        volume_ma = df['volume_crypto'].rolling(20).mean()
        relative_volume = df['volume_crypto'] / volume_ma
        
        # Pre-calculate indicator DataFrames for each parameter set
        print("\nPre-calculating indicators for all parameter sets...")
        indicator_dfs = {}
        
        # Add base parameters
        for strategy_name, params in best_params.items():
            key = (strategy_name, 'base')
            indicator_dfs[key] = TechnicalIndicators.add_all_indicators(
                df.copy(),
                custom_params={strategy_name: params}
            )
        
        # Add market condition specific parameters
        for condition, strategies in market_condition_params.items():
            for strategy_name, strategy_info in strategies.items():
                key = (strategy_name, condition)
                if 'params' in strategy_info:
                    indicator_dfs[key] = TechnicalIndicators.add_all_indicators(
                        df.copy(),
                        custom_params={strategy_name: strategy_info['params']}
                    )
        
        print("\nGenerating signals...")
        # Generate signals for each bar
        for i in range(len(df)):
            if i < 20:  # Skip first 20 bars to allow for indicator calculation
                continue
                
            # Determine current market condition
            vol_quantile = volatility.iloc[:i+1].rank(pct=True).iloc[-1]
            trend_quantile = trend_strength.iloc[:i+1].rank(pct=True).iloc[-1]
            volume_quantile = relative_volume.iloc[:i+1].rank(pct=True).iloc[-1]
            
            if vol_quantile > 0.75:
                current_condition = 'high_volatility'
            elif vol_quantile < 0.25:
                current_condition = 'low_volatility'
            elif trend_quantile > 0.75:
                current_condition = 'strong_trend'
            elif trend_quantile < 0.25:
                current_condition = 'weak_trend'
            elif volume_quantile > 0.75:
                current_condition = 'high_volume'
            elif volume_quantile < 0.25:
                current_condition = 'low_volume'
            else:
                current_condition = None
            
            weighted_signal = 0
            for strategy_name, weight in strategy_weights.items():
                try:
                    # Select appropriate pre-calculated indicators
                    if (current_condition and 
                        (strategy_name, current_condition) in indicator_dfs):
                        key = (strategy_name, current_condition)
                    else:
                        key = (strategy_name, 'base')
                    
                    data_with_indicators = indicator_dfs[key].iloc[:i+1]
                    
                    # Get strategy signal
                    strategy_func = getattr(TradingStrategies, f"{strategy_name}_strategy")
                    if (current_condition and 
                        strategy_name in market_condition_params.get(current_condition, {})):
                        params = market_condition_params[current_condition][strategy_name]['params']
                    else:
                        params = best_params[strategy_name]
                    
                    strategy_signal = strategy_func(
                        data_with_indicators,
                        custom_params=params
                    ).iloc[-1]
                    
                    weighted_signal += strategy_signal * weight
                    
                except Exception as e:
                    print(f"Error calculating signal for {strategy_name}: {str(e)}")
                    continue
            
            # Convert weighted signal to final signal
            if weighted_signal > 0.3:
                signals.iloc[i] = 1
            elif weighted_signal < -0.3:
                signals.iloc[i] = -1
            
            # Risk management based on market conditions
            if current_condition in ['high_volatility', 'low_volume']:
                signals.iloc[i] *= 0.5  # Reduce position size in high-risk conditions
        
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