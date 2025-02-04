# optimized_strategy.py

import os
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

from config import BACKTEST_CONFIG
from indicators import TechnicalIndicators
from strategies import TradingStrategies  # for the base signals (EMA, MACD, RSI, etc.)


def optimize_strategy_parameters(df, symbol, start_date, end_date):
    """
    1) Filters df to [start_date, end_date].
    2) For each base strategy, runs a parameter grid search to find the best param set.
    3) Assigns weights to each strategy based on its Sharpe ratio.
    4) Returns a dictionary with the best params, weights, and condition-specific data.
    """
    print(f"\nOptimizing strategy parameters for {symbol}")
    print(f"Training period: {start_date} to {end_date}")

    # 1) Filter data to training period
    mask = (df.index >= start_date) & (df.index <= end_date)
    train_df = df[mask].copy()
    if len(train_df) < BACKTEST_CONFIG['optimization']['min_training_days']:
        raise ValueError(f"Insufficient training data for {symbol}")

    # 2) Estimate volatility (annualized)
    returns_series = train_df['close_price'].pct_change()
    hours_per_year = 365 * 24
    avg_volatility = returns_series.std() * np.sqrt(hours_per_year)
    print(f"Annual Volatility for {symbol}: {avg_volatility:.2f}")

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

    # Market condition placeholders (optional)
    volatility_20 = returns_series.rolling(20).std() * np.sqrt(252)
    trend_strength_20 = abs(train_df['close_price'].pct_change(20))
    volume_ma = train_df['volume_crypto'].rolling(20).mean()
    relative_volume = train_df['volume_crypto'] / volume_ma

    conditions = {
        'high_volatility': (volatility_20 > volatility_20.quantile(0.75)),
        'low_volatility':  (volatility_20 < volatility_20.quantile(0.25)),
        'strong_trend':    (trend_strength_20 > trend_strength_20.quantile(0.75)),
        'weak_trend':      (trend_strength_20 < trend_strength_20.quantile(0.25)),
        'high_volume':     (relative_volume > relative_volume.quantile(0.75)),
        'low_volume':      (relative_volume < relative_volume.quantile(0.25))
    }

    param_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges'].copy()
    total_sharpe = 0.0
    min_required_trades = 10

    # Grid search each base strategy
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

        best_score_for_strategy = -np.inf
        best_sharpe_for_strategy = -np.inf

        # All param combos
        combos = list(itertools.product(*[
            strategy_params[p] if isinstance(strategy_params[p], (list, range))
            else [strategy_params[p]]
            for p in strategy_params.keys()
        ]))
        print(f"Testing {len(combos)} parameter combinations...")

        for params in combos:
            param_dict = dict(zip(strategy_params.keys(), params))
            try:
                # Recalc indicators with these custom params
                df_with_inds = TechnicalIndicators.add_all_indicators(
                    train_df.copy(),
                    custom_params={strategy_name: param_dict}
                )
                # Strategy function
                strat_func = getattr(TradingStrategies, f"{strategy_name}_strategy")
                signals = strat_func(df_with_inds, custom_params=param_dict)

                # Count trades
                num_trades = ((signals.shift(1) != signals) & (signals != 0)).sum()
                if num_trades < min_required_trades:
                    continue

                # Returns from signals
                strat_returns = df_with_inds['close_price'].pct_change() * signals.shift(1)
                strat_returns.fillna(0, inplace=True)
                if strat_returns.std() == 0:
                    continue

                ann_factor = np.sqrt(365 * 24)
                sharpe_val = ann_factor * (strat_returns.mean() / strat_returns.std())
                ann_return = strat_returns.mean() * (365 * 24)
                max_dd = (1 - (1 + strat_returns).cumprod() /
                          (1 + strat_returns).cumprod().cummax()).max()
                win_rate = (strat_returns[strat_returns != 0] > 0).mean()

                # Score to pick best params
                score = sharpe_val * (1 - max_dd) * win_rate
                if score > best_score_for_strategy:
                    best_score_for_strategy = score
                    best_sharpe_for_strategy = sharpe_val
                    strategy_results['best_parameters'] = param_dict
                    strategy_results['performance'].update({
                        'sharpe_ratio': float(sharpe_val),
                        'returns': float(ann_return),
                        'volatility': float(strat_returns.std() * ann_factor),
                        'max_drawdown': float(max_dd),
                        'win_rate': float(win_rate),
                        'num_trades': int(num_trades)
                    })

                    # (Optional) condition-specific performance
                    for cond_name, cond_mask in conditions.items():
                        cond_returns = strat_returns[cond_mask]
                        if len(cond_returns) > 0 and cond_returns.std() != 0:
                            cond_sharpe = ann_factor * (cond_returns.mean() / cond_returns.std())
                            if cond_name not in optimization_results['market_conditions']:
                                optimization_results['market_conditions'][cond_name] = {}
                            optimization_results['market_conditions'][cond_name][strategy_name] = {
                                'parameters': param_dict.copy(),
                                'sharpe_ratio': float(cond_sharpe)
                            }

            except Exception as e:
                print(f"Error testing parameters {param_dict}: {str(e)}")
                continue

        if strategy_results['best_parameters'] is not None:
            optimization_results['strategies'][strategy_name] = strategy_results
            if best_sharpe_for_strategy > 0:
                total_sharpe += best_sharpe_for_strategy

    # Assign weights from Sharpe ratio
    if total_sharpe > 0:
        for strat_name, strat_data in optimization_results['strategies'].items():
            s = strat_data['performance']['sharpe_ratio']
            strat_data['weight'] = max(0, s)/total_sharpe
    else:
        # fallback if all negative
        for strat_name in optimization_results['strategies']:
            optimization_results['strategies'][strat_name]['weight'] = 0.0

    # Save to JSON
    results_dir = BACKTEST_CONFIG['results_dir']
    symbol_dir = os.path.join(results_dir, safe_symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    results_file = os.path.join(symbol_dir, f"optimization_{optimization_results['optimization_id']}.json")
    optimization_results['results_file'] = results_file

    print("\nOptimization Results Summary:")
    for strat_name, strat_data in optimization_results['strategies'].items():
        print(f"\n{strat_name}:")
        print(f"  Best Params: {strat_data['best_parameters']}")
        print(f"  Weight: {strat_data['weight']:.4f}")
        perf = strat_data['performance']
        print("  Performance:")
        print(f"    Sharpe: {perf['sharpe_ratio']:.4f}")
        print(f"    Annual Return: {perf['returns']*100:.2f}%")
        print(f"    Max Drawdown: {perf['max_drawdown']*100:.2f}%")
        print(f"    Win Rate: {perf['win_rate']*100:.2f}%")
        print(f"    Trades: {perf['num_trades']}")

    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)

    return optimization_results


class OptimizedStrategy:
    """
    Your single advanced (adaptive) strategy. 
    It uses the results of optimize_strategy_parameters(...) 
    to generate signals on new data (the testing set).
    """

    def __init__(
        self,
        optimization_results,
        window_size=50,
        correlation_lookback=50,
        correlation_threshold=0.8,
        min_weight_for_inclusion=0.05
    ):
        self.optimization_results = optimization_results
        self.window_size = window_size
        self.correlation_lookback = correlation_lookback
        self.correlation_threshold = correlation_threshold
        self.min_weight_for_inclusion = min_weight_for_inclusion

        # Extract best_params & base_weights from the optimization dict
        self.best_params = {}
        self.base_weights = {}
        self.market_condition_params = {}

        # If 'strategies' is present, build from that
        if 'strategies' in optimization_results:
            self.best_params = {
                s: v['best_parameters'] for s,v in optimization_results['strategies'].items()
            }
            self.base_weights = {
                s: v['weight'] for s,v in optimization_results['strategies'].items()
            }
        # If 'market_conditions' is present
        if 'market_conditions' in optimization_results:
            self.market_condition_params = optimization_results['market_conditions']

        # For rolling performance
        self.recent_performance = {s: [] for s in self.base_weights}
        self.signal_history = {s: [] for s in self.base_weights}

        self.last_condition = None

    def detect_market_condition(self, df, i):
        """
        Example condition detection with short vs. long vol & trend.
        Customize as you wish.
        """
        if i < 60:
            return None

        short_returns = df['close_price'].pct_change().iloc[i-10:i]
        long_returns  = df['close_price'].pct_change().iloc[i-50:i]
        if short_returns.std() is None or long_returns.std() is None:
            return None

        short_vol = short_returns.std()*np.sqrt(365*24)
        long_vol  = long_returns.std()*np.sqrt(365*24)

        short_ema_diff = (df['ema_9'].iloc[i] - df['ema_50'].iloc[i])
        long_ema_diff  = (df['ema_9'].iloc[i-30] - df['ema_50'].iloc[i-30])

        if short_vol>long_vol*1.2 and short_ema_diff>0 and long_ema_diff>0:
            return 'high_vol_uptrend'
        elif short_vol>long_vol*1.2 and (short_ema_diff<0 or long_ema_diff<0):
            return 'high_vol_downtrend'
        elif short_vol<long_vol*0.8 and short_ema_diff<0 and long_ema_diff<0:
            return 'low_vol_downtrend'
        else:
            return 'normal'

    def adapt_parameters(self, strategy_name, condition):
        """
        If we have condition-specific params, use them; else fallback to best_params.
        """
        if not condition:
            return self.best_params.get(strategy_name, None)

        if (condition in self.market_condition_params and 
            strategy_name in self.market_condition_params[condition]):
            cdata = self.market_condition_params[condition][strategy_name]
            return cdata.get('parameters', self.best_params.get(strategy_name))
        return self.best_params.get(strategy_name)

    def update_strategy_weights(self, df, i):
        """
        Dynamically re-weight each strategy based on recent performance & correlation.
        Called every N bars (e.g. every 10 bars).
        """
        if i < self.window_size:
            return self.base_weights

        # 1) Compute recent Sharpe
        sharpe_dict = {}
        for s in self.base_weights:
            wret = self.recent_performance[s][-self.window_size:]
            if len(wret) < 2:
                sharpe_dict[s] = 0
                continue
            series = pd.Series(wret)
            if series.std()==0:
                sharpe_dict[s] = 0
                continue
            ann_factor=np.sqrt(365*24)
            shrp=ann_factor*(series.mean()/series.std())
            sharpe_dict[s]=shrp
        
        # 2) correlation among last correlation_lookback signals
        signal_dict = {}
        for s in self.base_weights:
            # last N signals
            sigs = self.signal_history[s][-self.correlation_lookback:]
            if len(sigs)<2:
                signal_dict[s]=pd.Series([0])
            else:
                signal_dict[s]=pd.Series(sigs).fillna(0)
        signals_df=pd.DataFrame(signal_dict)
        if len(signals_df)>1:
            corr_matrix=signals_df.corr()
        else:
            corr_matrix=pd.DataFrame(np.eye(len(self.base_weights)), 
                                     index=self.base_weights.keys(),
                                     columns=self.base_weights.keys())

        # 3) Convert Sharpe => raw weights
        raw_weights={}
        total=0
        for s in self.base_weights:
            val=sharpe_dict[s]
            if val>0:
                raw_weights[s]=val
                total+=val
            else:
                raw_weights[s]=0
        
        if total<=0:
            # fallback
            n=len(raw_weights)
            for k in raw_weights:
                raw_weights[k]=1/n
        else:
            for k in raw_weights:
                raw_weights[k]/=total
        
        # 4) correlation penalty
        for s1 in raw_weights:
            for s2 in raw_weights:
                if s1!=s2:
                    cval=corr_matrix.loc[s1,s2]
                    if cval>self.correlation_threshold:
                        # penalize the lower Sharpe
                        if sharpe_dict[s1]<sharpe_dict[s2]:
                            raw_weights[s1]*=0.5
                        else:
                            raw_weights[s2]*=0.5

        # re-normalize
        s=sum(raw_weights.values())
        if s>0:
            for k in raw_weights:
                raw_weights[k]/=s

        # 5) enforce min_weight
        changed=True
        while changed:
            changed=False
            for k in raw_weights:
                if raw_weights[k]<self.min_weight_for_inclusion:
                    raw_weights[k]=0
                    changed=True
            s2=sum(raw_weights.values())
            if s2>0 and changed:
                for kk in raw_weights:
                    raw_weights[kk]/=s2

        return raw_weights

def run(self, df):
    """Generate signals using weighted combination of optimized strategies"""
    print("\nRunning OptimizedStrategy with weights:")
    for s, w in self.base_weights.items():
        print(f"  {s}: {w:.4f}")
    
    signals = pd.Series(data=0.0, index=df.index)
    
    try:
        # Pre-compute indicators once
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Get signals from each strategy
        strategy_signals = {}
        for strat_name, weight in self.base_weights.items():
            if weight > 0:  # Only run strategies with non-zero weights
                try:
                    # Get strategy parameters and function
                    params = self.best_params[strat_name]
                    strat_func = getattr(TradingStrategies, f"{strat_name}_strategy")
                    
                    # Get signals and multiply by weight
                    strat_signals = strat_func(df, custom_params=params)
                    strategy_signals[strat_name] = strat_signals * weight
                    
                    print(f"  {strat_name} generated {len(strat_signals[strat_signals != 0])} signals")
                except Exception as e:
                    print(f"Error in strategy {strat_name}: {str(e)}")
                    continue
        
        # Combine weighted signals
        if strategy_signals:
            combined_signal = pd.concat(strategy_signals.values(), axis=1).sum(axis=1)
            
            # Convert to discrete signals based on thresholds
            signals[combined_signal > 0.3] = 1  # Strong buy signal when weighted sum > 0.3
            signals[combined_signal < -0.3] = -1  # Strong sell signal when weighted sum < -0.3
            
            # Apply market condition adjustments
            for i in range(len(signals)):
                if i >= 20:  # Skip first few bars
                    condition = self.detect_market_condition(df, i)
                    
                    # Reduce exposure in high volatility
                    if condition and 'high_vol' in condition:
                        signals.iloc[i] *= 0.5
                    # Increase exposure in strong trends that match signal direction
                    elif condition == 'strong_trend':
                        if signals.iloc[i] != 0 and i > 0:
                            price_trend = df['close_price'].iloc[i] / df['close_price'].iloc[i-20] - 1
                            if (signals.iloc[i] > 0 and price_trend > 0) or (signals.iloc[i] < 0 and price_trend < 0):
                                signals.iloc[i] *= 1.2
            
            print(f"\nOptimized strategy generated {len(signals[signals != 0])} total signals")
            print(f"  Buy signals: {len(signals[signals == 1])}")
            print(f"  Sell signals: {len(signals[signals == -1])}")
            
        else:
            print("No valid strategy signals generated")
            
    except Exception as e:
        print(f"Error in OptimizedStrategy.run: {str(e)}")
        return signals
    
    return signals
