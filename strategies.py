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

    @staticmethod
    def optimize_strategy_parameters(df, symbol, start_date, end_date):
        """
        Optimizes strategy parameters with improved validation and selection criteria.
        Fixes the issue where combined score was incorrectly used to compute final
        strategy weights. Now we separate the 'combined score' for selecting the best
        parameters from the actual Sharpe ratio used for weighting.

        Args:
            df (pd.DataFrame): Full dataset with at least 'close_price' and 'volume_crypto'.
            symbol (str): Symbol being optimized, e.g. "LTC/USD"
            start_date (str): Training period start date in "YYYY-MM-DD"
            end_date (str): Training period end date in "YYYY-MM-DD"

        Returns:
            dict: A dictionary containing the best parameters and performance for each
                strategy, along with final strategy weights and market condition data.
        """
        print(f"\nOptimizing strategy parameters for {symbol}")
        print(f"Training period: {start_date} to {end_date}")
        
        # 1. Filter data to the training period
        mask = (df.index >= start_date) & (df.index <= end_date)
        train_df = df[mask].copy()
        if len(train_df) < BACKTEST_CONFIG['optimization']['min_training_days']:
            raise ValueError(f"Insufficient training data for {symbol}")

        # 2. Calculate asset volatility (annualized, assuming ~365*24 hours/year for crypto)
        returns_series = train_df['close_price'].pct_change()
        hours_per_year = 365 * 24
        avg_volatility = returns_series.std() * np.sqrt(hours_per_year)
        print(f"Annual Volatility for {symbol}: {avg_volatility:.2f}")

        # Prepare the overall optimization results dictionary
        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
        optimization_results = {
            'optimization_id': f"opt_{safe_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'training_period': {'start': start_date, 'end': end_date},
            'volatility': float(avg_volatility),
            'strategies': {},
            'market_conditions': {},
            'timestamp': datetime.now().isoformat()
        }

        # 3. Market conditions calculation (identical to your original approach)
        volatility_20 = returns_series.rolling(20).std() * np.sqrt(252)
        trend_strength_20 = abs(train_df['close_price'].pct_change(20))
        volume_ma = train_df['volume_crypto'].rolling(20).mean()
        relative_volume = train_df['volume_crypto'] / volume_ma

        conditions = {
            'high_volatility': volatility_20 > volatility_20.quantile(0.75),
            'low_volatility': volatility_20 < volatility_20.quantile(0.25),
            'strong_trend': trend_strength_20 > trend_strength_20.quantile(0.75),
            'weak_trend': trend_strength_20 < trend_strength_20.quantile(0.25),
            'high_volume': relative_volume > relative_volume.quantile(0.75),
            'low_volume': relative_volume < relative_volume.quantile(0.25)
        }

        # 4. Parameter ranges from BACKTEST_CONFIG
        param_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges'].copy()
        total_sharpe = 0.0
        min_required_trades = 10  # Skip parameter sets with fewer trades

        # 5. Optimize each strategy
        for strategy_name, strategy_params in param_ranges.items():
            print(f"\nOptimizing {strategy_name} strategy...")

            strategy_results = {
                'best_parameters': None,
                'performance': {
                    'sharpe_ratio': None,
                    'returns': None,
                    'volatility': None,
                    'max_drawdown': None,
                    'win_rate': None,
                    'num_trades': None
                },
                'weight': 0.0
            }

            # We separate the combined score from the actual Sharpe
            best_score_for_strategy = -np.inf
            best_sharpe_for_strategy = -np.inf

            # Generate all parameter combinations
            param_combinations = list(itertools.product(*[
                strategy_params[p] if isinstance(strategy_params[p], (list, range))
                else [strategy_params[p]]
                for p in strategy_params.keys()
            ]))
            print(f"Testing {len(param_combinations)} parameter combinations...")

            # 6. Evaluate each parameter combination
            for params in param_combinations:
                param_dict = dict(zip(strategy_params.keys(), params))

                try:
                    # Calculate indicators with these custom params
                    df_with_inds = TechnicalIndicators.add_all_indicators(
                        train_df.copy(),
                        custom_params={strategy_name: param_dict}
                    )
                    strategy_func = getattr(TradingStrategies, f"{strategy_name}_strategy")
                    signals = strategy_func(df_with_inds, custom_params=param_dict)

                    # Count trades
                    num_trades = ((signals.shift(1) != signals) & (signals != 0)).sum()
                    if num_trades < min_required_trades:
                        continue

                    # Calculate returns from signals
                    strategy_returns = df_with_inds['close_price'].pct_change() * signals.shift(1)
                    strategy_returns = strategy_returns.fillna(0)

                    # Skip if no variance in returns
                    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                        continue

                    # Compute metrics (annualizing for hourly data)
                    ann_factor = np.sqrt(365 * 24)
                    sharpe_val = ann_factor * (strategy_returns.mean() / strategy_returns.std())
                    ann_return = strategy_returns.mean() * (365 * 24)
                    max_dd = (1 - (1 + strategy_returns).cumprod() /
                            (1 + strategy_returns).cumprod().cummax()).max()
                    win_rate = (strategy_returns[strategy_returns != 0] > 0).mean()

                    # Our combined "score" for choosing the best parameter set
                    score = sharpe_val * (1 - max_dd) * win_rate

                    if score > best_score_for_strategy:
                        best_score_for_strategy = score
                        best_sharpe_for_strategy = sharpe_val  # store the real Sharpe
                        strategy_results['best_parameters'] = param_dict
                        strategy_results['performance'].update({
                            'sharpe_ratio': float(sharpe_val),
                            'returns': float(ann_return),
                            'volatility': float(strategy_returns.std() * ann_factor),
                            'max_drawdown': float(max_dd),
                            'win_rate': float(win_rate),
                            'num_trades': int(num_trades)
                        })

                        # Calculate market-condition-specific performance
                        for condition_name, condition_mask in conditions.items():
                            cond_returns = strategy_returns[condition_mask]
                            if len(cond_returns) > 0 and cond_returns.std() != 0:
                                cond_sharpe = ann_factor * (cond_returns.mean() / cond_returns.std())
                                if condition_name not in optimization_results['market_conditions']:
                                    optimization_results['market_conditions'][condition_name] = {}
                                optimization_results['market_conditions'][condition_name][strategy_name] = {
                                    'parameters': param_dict.copy(),
                                    'sharpe_ratio': float(cond_sharpe)
                                }

                except Exception as e:
                    print(f"Error testing parameters {param_dict}: {str(e)}")
                    continue

            # 7. If we found valid parameters, add them to overall results
            if strategy_results['best_parameters'] is not None:
                optimization_results['strategies'][strategy_name] = strategy_results
                # Add the actual Sharpe ratio for weighting (not the combined score)
                if best_sharpe_for_strategy > 0:
                    total_sharpe += best_sharpe_for_strategy

        # 8. Weight each strategy by its actual Sharpe ratio
        if total_sharpe > 0:
            for strat_name, strat_data in optimization_results['strategies'].items():
                actual_sharpe = strat_data['performance']['sharpe_ratio']
                strat_data['weight'] = max(0, actual_sharpe) / total_sharpe
        else:
            # fallback if everything <= 0 Sharpe
            for strat_name in optimization_results['strategies']:
                optimization_results['strategies'][strat_name]['weight'] = 0.0

        # 9. Save results to JSON file
        results_dir = BACKTEST_CONFIG['results_dir']
        symbol_dir = os.path.join(results_dir, safe_symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        results_file = os.path.join(
            symbol_dir,
            f"optimization_{optimization_results['optimization_id']}.json"
        )

        print("\nOptimization Results Summary:")
        for strat_name, strat_data in optimization_results['strategies'].items():
            print(f"\n{strat_name}:")
            print(f"  Best Parameters: {strat_data['best_parameters']}")
            print(f"  Weight: {strat_data['weight']:.4f}")
            perf = strat_data['performance']
            print("  Performance:")
            print(f"    Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
            print(f"    Annual Return: {perf['returns']*100:.2f}%")
            print(f"    Max Drawdown: {perf['max_drawdown']*100:.2f}%")
            print(f"    Win Rate: {perf['win_rate']*100:.2f}%")
            print(f"    Number of Trades: {perf['num_trades']}")

        with open(results_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)

        return optimization_results

    @staticmethod
    def optimized_adaptive_strategy(df, optimization_results):
        """
        Uses optimized parameters and weights to generate trading signals.
        """
        print("\nApplying optimized adaptive strategy...")
        signals = pd.Series(index=df.index, data=0.0)
        best_params = optimization_results['best_params']
        strategy_weights = optimization_results['strategy_weights']
        market_condition_params = optimization_results.get('market_condition_params', {})
        
        print("Strategy weights being used:", strategy_weights)
        
        # Calculate market conditions for the entire dataset
        volatility = df['close_price'].pct_change().rolling(20).std()
        trend_strength = ((df['ema_9'] - df['ema_50']).abs() / df['ema_50']).fillna(0)
        volume_ma = df['volume_crypto'].rolling(20).mean()
        relative_volume = df['volume_crypto'] / volume_ma
        
        # Pre-calculate indicator DataFrames for each parameter set
        print("\nPre-calculating indicators for all parameter sets...")
        indicator_dfs = {}
        
        # Add base parameters
        for strategy_name, params in best_params.items():
            if params:  # Only add if parameters exist
                key = (strategy_name, 'base')
                try:
                    indicator_dfs[key] = TechnicalIndicators.add_all_indicators(
                        df.copy(),
                        custom_params={strategy_name: params}
                    )
                except Exception as e:
                    print(f"Error calculating indicators for {strategy_name}: {str(e)}")
                    continue
        
        # Add market condition specific parameters
        for condition, strategies in market_condition_params.items():
            for strategy_name, strategy_info in strategies.items():
                if 'parameters' in strategy_info:
                    key = (strategy_name, condition)
                    try:
                        indicator_dfs[key] = TechnicalIndicators.add_all_indicators(
                            df.copy(),
                            custom_params={strategy_name: strategy_info['parameters']}
                        )
                    except Exception as e:
                        print(f"Error calculating indicators for {strategy_name} in {condition}: {str(e)}")
                        continue
        
        print("\nGenerating signals...")
        last_condition = None
        last_params = {}
        
        for i in range(len(df)):
            if i < 20:  # Skip first 20 bars to allow for indicator calculation
                continue
            
            # Determine current market condition
            vol_rank = volatility.iloc[:i+1].rank(pct=True).iloc[-1]
            trend_rank = trend_strength.iloc[:i+1].rank(pct=True).iloc[-1]
            volume_rank = relative_volume.iloc[:i+1].rank(pct=True).iloc[-1]
            
            current_condition = None
            if vol_rank > 0.75:
                current_condition = 'high_volatility'
            elif vol_rank < 0.25:
                current_condition = 'low_volatility'
            elif trend_rank > 0.75:
                current_condition = 'strong_trend'
            elif trend_rank < 0.25:
                current_condition = 'weak_trend'
            elif volume_rank > 0.75:
                current_condition = 'high_volume'
            elif volume_rank < 0.25:
                current_condition = 'low_volume'
            
            # Print when market condition changes
            if current_condition != last_condition:
                print(f"\nBar {i}: Market condition changed to {current_condition}")
                last_condition = current_condition
            
            weighted_signal = 0
            for strategy_name, weight in strategy_weights.items():
                if weight <= 0 or strategy_name not in best_params:
                    continue
                    
                try:
                    # Select appropriate pre-calculated indicators
                    if (current_condition and 
                        (strategy_name, current_condition) in indicator_dfs and
                        strategy_name in market_condition_params.get(current_condition, {})):
                        key = (strategy_name, current_condition)
                        params = market_condition_params[current_condition][strategy_name]['parameters']
                    else:
                        key = (strategy_name, 'base')
                        params = best_params[strategy_name]
                    
                    if key not in indicator_dfs:
                        continue
                        
                    # Only print when parameters change
                    if params != last_params.get(strategy_name):
                        print(f"Strategy {strategy_name} using parameters: {params}")
                        last_params[strategy_name] = params
                    
                    data_with_indicators = indicator_dfs[key].iloc[:i+1]
                    
                    strategy_func = getattr(TradingStrategies, f"{strategy_name}_strategy")
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