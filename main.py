#main.py

import pandas as pd
import numpy as np
from database import DatabaseHandler
from indicators import TechnicalIndicators
from strategies import TradingStrategies
from backtester import Backtester
from config import BACKTEST_CONFIG
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from datetime import datetime

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_strategy_metrics(portfolio_df, trades_df, period_start, period_end, initial_capital):
    """Calculate strategy metrics for a specific period"""
    
    # Filter data for the period
    mask = (portfolio_df.index >= period_start) & (portfolio_df.index <= period_end)
    period_data = portfolio_df[mask]
    
    if len(period_data) == 0:
        return None
        
    # Calculate period metrics
    period_return = (period_data['total_value'].iloc[-1] - period_data['total_value'].iloc[0]) / period_data['total_value'].iloc[0]
    period_length = (period_data.index[-1] - period_data.index[0]).days / 365.0
    annual_return = period_return / period_length if period_length > 0 else 0
    
    # Calculate Sharpe Ratio
    daily_returns = period_data['total_value'].pct_change()
    sharpe_ratio = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
    
    # Calculate max drawdown
    peak = period_data['total_value'].expanding(min_periods=1).max()
    drawdown = (period_data['total_value'] - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate trade metrics for the period
    if trades_df is not None and not trades_df.empty:
        period_trades = trades_df[trades_df['date'].between(period_start, period_end)]
        num_trades = len(period_trades)
        win_rate = len(period_trades[period_trades['value'] > period_trades['value'].shift(1)]) / (num_trades // 2) if num_trades > 0 else 0
        total_fees = period_trades['fee'].sum() if 'fee' in period_trades.columns else 0
    else:
        num_trades = 0
        win_rate = 0
        total_fees = 0
    
    return {
        'Total Return': f"{period_return * 100:.2f}%",
        'Annual Return': f"{annual_return * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown * 100:.2f}%",
        'Number of Trades': num_trades,
        'Win Rate': f"{win_rate * 100:.2f}%",
        'Trading Fees': f"${total_fees:.2f}"
    }

def save_results_to_file(results_dict, symbol, output_dir, combined_results, optimization_results=None, train_end=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    summary_file = os.path.join(symbol_dir, f'summary_{timestamp}.txt')
    
    with open(summary_file, 'w') as f:
        f.write("=== Backtest Configuration ===\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Full Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}\n")
        f.write(f"Training Period: {BACKTEST_CONFIG['start_date']} to {train_end}\n")
        f.write(f"Testing Period: {train_end} to {BACKTEST_CONFIG['end_date']}\n")
        f.write(f"Initial Capital: ${BACKTEST_CONFIG['initial_capital']}\n\n")
        
        # Write optimization results if available
        if optimization_results and 'results_file' in optimization_results:
            f.write("=== Optimization Results (Training Period) ===\n")
            try:
                with open(optimization_results['results_file'], 'r') as opt_f:
                    f.write(opt_f.read())
                f.write("\n")
            except Exception as e:
                f.write(f"Error reading optimization results: {str(e)}\n\n")
        
        f.write("\n=== Testing Period Performance ===\n")
        f.write(f"Period: {train_end} to {BACKTEST_CONFIG['end_date']}\n\n")
        
        # Calculate testing period metrics for each strategy
        testing_metrics = []
        
        # Calculate buy and hold metrics for testing period
        first_result = next(iter(results_dict.values()))
        df = first_result['portfolio']
        test_mask = (df.index >= train_end) & (df.index <= BACKTEST_CONFIG['end_date'])
        test_df = df[test_mask]
        
        if len(test_df) > 0:
            # Buy and Hold metrics
            initial_price = float(test_df['close'].iloc[0])
            final_price = float(test_df['close'].iloc[-1])
            test_return = (final_price - initial_price) / initial_price
            test_period = (test_df.index[-1] - test_df.index[0]).days / 365.0
            test_annual_return = test_return / test_period if test_period > 0 else 0
            
            buy_hold_metrics = {
                'Strategy': 'Buy and Hold',
                'Total Return': f"{test_return * 100:.2f}%",
                'Annual Return': f"{test_annual_return * 100:.2f}%",
                'Number of Trades': 1,
                'Trading Fees': f"${BACKTEST_CONFIG['initial_capital'] * BACKTEST_CONFIG['trading_fee']:.2f}"
            }
            
            # Calculate metrics for each strategy
            for strategy_name, result in results_dict.items():
                metrics = calculate_strategy_metrics(
                    result['portfolio'],
                    result['trades'],
                    train_end,
                    BACKTEST_CONFIG['end_date'],
                    BACKTEST_CONFIG['initial_capital']
                )
                if metrics:
                    metrics['Strategy'] = strategy_name
                    testing_metrics.append(metrics)
            
            # Add buy and hold metrics
            testing_metrics.append(buy_hold_metrics)
            
            # Write testing period metrics table
            metrics_df = pd.DataFrame(testing_metrics)
            f.write(tabulate(metrics_df, headers="keys", tablefmt="grid", numalign="right"))
            
            # Add detailed statistics for optimized strategy in testing period
            if 'Optimized' in results_dict:
                f.write("\n\n=== Optimized Strategy Testing Period Details ===\n")
                opt_result = results_dict['Optimized']
                
                # Monthly returns during testing period
                test_portfolio = opt_result['portfolio'][test_mask]
                monthly_returns = test_portfolio['total_value'].resample('M').last().pct_change()
                
                f.write("\nMonthly Returns:\n")
                f.write(monthly_returns.to_string())
                
                # Trade analysis during testing period
                if not opt_result['trades'].empty:
                    test_trades = opt_result['trades'][
                        opt_result['trades']['date'].between(train_end, BACKTEST_CONFIG['end_date'])
                    ]
                    
                    f.write("\n\nTrade Analysis:\n")
                    trade_stats = {
                        'Total Trades': len(test_trades),
                        'Average Trade Duration': f"{test_trades['date'].diff().mean().total_seconds() / 3600:.1f} hours",
                        'Average Trade Size': f"${test_trades['value'].mean():.2f}",
                        'Largest Winning Trade': f"${test_trades['value'].max():.2f}",
                        'Largest Losing Trade': f"${test_trades['value'].min():.2f}",
                    }
                    
                    for stat, value in trade_stats.items():
                        f.write(f"{stat}: {value}\n")
        
        else:
            f.write("No data available for testing period\n")
    
    return summary_file
def save_test_period_results(results_dict, symbol, output_dir, train_end):
    """
    Creates a focused report on test period performance for all strategies.
    
    Args:
        results_dict (dict): Dictionary containing results for each strategy
        symbol (str): Trading symbol
        output_dir (str): Output directory path
        train_end (str): End date of training period
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    test_results_file = os.path.join(symbol_dir, f'test_period_results_{timestamp}.txt')
    
    # Calculate metrics for each strategy during test period
    test_metrics = []
    
    # Get first portfolio for buy and hold calculation
    first_result = next(iter(results_dict.values()))
    df = first_result['portfolio']
    
    # Calculate buy and hold for test period
    test_mask = (df.index >= train_end) & (df.index <= BACKTEST_CONFIG['end_date'])
    test_df = df[test_mask]
    
    if len(test_df) > 0:
        # Buy and Hold calculation
        initial_price = float(test_df['close'].iloc[0])
        final_price = float(test_df['close'].iloc[-1])
        bh_return = (final_price - initial_price) / initial_price
        test_days = (test_df.index[-1] - test_df.index[0]).days
        bh_annual_return = bh_return * (365 / test_days) if test_days > 0 else 0
        
        # Calculate strategy metrics
        for strategy_name, result in results_dict.items():
            portfolio = result['portfolio'][test_mask]
            trades = result['trades'][result['trades']['date'] >= train_end] if not result['trades'].empty else pd.DataFrame()
            
            # Calculate strategy returns
            strategy_return = (portfolio['total_value'].iloc[-1] - portfolio['total_value'].iloc[0]) / portfolio['total_value'].iloc[0]
            strategy_annual_return = strategy_return * (365 / test_days) if test_days > 0 else 0
            
            # Calculate other metrics
            daily_returns = portfolio['total_value'].pct_change()
            sharpe = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
            max_drawdown = ((portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()).max()
            
            test_metrics.append({
                'Strategy': strategy_name,
                'Total Return (%)': f"{strategy_return * 100:.2f}%",
                'Annual Return (%)': f"{strategy_annual_return * 100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Max Drawdown (%)': f"{max_drawdown * 100:.2f}%",
                'Number of Trades': len(trades),
                'Win Rate (%)': f"{(len(trades[trades['value'] > trades['value'].shift(1)]) / (len(trades) // 2) * 100):.2f}%" if len(trades) > 0 else "N/A"
            })
        
        # Add buy and hold metrics
        test_metrics.append({
            'Strategy': 'Buy and Hold',
            'Total Return (%)': f"{bh_return * 100:.2f}%",
            'Annual Return (%)': f"{bh_annual_return * 100:.2f}%",
            'Sharpe Ratio': "N/A",
            'Max Drawdown (%)': "N/A",
            'Number of Trades': 1,
            'Win Rate (%)': "N/A"
        })
        
        # Write results to file
        with open(test_results_file, 'w') as f:
            f.write(f"=== Test Period Performance for {symbol} ===\n")
            f.write(f"Test Period: {train_end} to {BACKTEST_CONFIG['end_date']}\n")
            f.write(f"Total Test Days: {test_days}\n\n")
            
            metrics_df = pd.DataFrame(test_metrics)
            f.write(tabulate(metrics_df, headers='keys', tablefmt='grid', numalign='right'))
            
            # Add percent improvement over buy and hold for each strategy
            f.write("\n\nStrategy Performance vs Buy and Hold:\n")
            bh_return_val = float(test_metrics[-1]['Total Return (%)'].rstrip('%'))
            for metric in test_metrics[:-1]:  # Exclude buy and hold
                strategy_return_val = float(metric['Total Return (%)'].rstrip('%'))
                improvement = strategy_return_val - bh_return_val
                f.write(f"{metric['Strategy']}: {improvement:+.2f}% vs Buy and Hold\n")
    
    else:
        with open(test_results_file, 'w') as f:
            f.write("No data available for test period\n")
    
    return test_results_file
def analyze_strategy_correlations(results_dict, df, symbol, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    # Define analysis_file path before using it
    analysis_file = os.path.join(symbol_dir, f'correlation_analysis_{timestamp}.txt')
    
    # Calculate daily returns for each strategy
    strategy_returns = {}
    for strategy_name, result in results_dict.items():
        strategy_returns[strategy_name] = result['portfolio']['total_value'].pct_change()
    
    returns_df = pd.DataFrame(strategy_returns)
    
    # Select the market characteristics we want to analyze
    market_chars = df[['volatility', 'trend_strength', 'relative_volume', 
                      'momentum', 'atr']].copy()
    
    # Calculate correlations
    correlations = pd.DataFrame()
    
    # Helper function for regime creation
    def create_regime_labels(series, n_bins=4):
        try:
            return pd.qcut(series, n_bins, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        except ValueError:
            try:
                return pd.qcut(series, 3, labels=['Low', 'Medium', 'High'])
            except ValueError:
                return pd.cut(series, 2, labels=['Low', 'High'])
    
    # Helper function to calculate regime statistics
    def calculate_regime_stats(returns_df, mask):
        stats = {}
        for strategy in returns_df.columns:
            strategy_returns = returns_df[strategy][mask]
            annual_return = strategy_returns.mean() * 252
            std = strategy_returns.std()
            sharpe = (strategy_returns.mean() / std * np.sqrt(252)) if std != 0 else 0
            win_rate = (strategy_returns > 0).mean()
            
            stats[strategy] = {
                'Annual Return': f"{annual_return*100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Win Rate': f"{win_rate*100:.2f}%"
            }
        return pd.DataFrame(stats).T
    
    with open(analysis_file, 'w') as f:
        f.write(f"=== Strategy Performance Correlation Analysis for {symbol} ===\n\n")
        
        # 1. Overall correlations with market characteristics
        f.write("1. Overall Correlations with Market Characteristics:\n")
        for char in market_chars.columns:
            for strategy in returns_df.columns:
                corr = returns_df[strategy].corr(market_chars[char])
                correlations.loc[strategy, char] = corr
        f.write(tabulate(correlations.round(3), headers='keys', tablefmt='grid'))
        f.write("\n\n")
        
        # 2. Performance in different market conditions
        f.write("2. Strategy Performance in Different Market Conditions:\n\n")
        
        # Analyze each market characteristic
        for char in ['volatility', 'trend_strength', 'relative_volume']:
            quartiles = create_regime_labels(market_chars[char])
            f.write(f"\na) Performance in Different {char.replace('_', ' ').title()} Regimes:\n")
            
            for regime in quartiles.unique():
                mask = quartiles == regime
                regime_stats = calculate_regime_stats(returns_df, mask)
                
                f.write(f"\n{char.replace('_', ' ').title()} Regime: {regime}\n")
                f.write(tabulate(regime_stats, headers='keys', tablefmt='grid'))
                f.write("\n")
        
        # 3. Key findings
        f.write("\n3. Key Strategy Recommendations:\n")
        f.write("\nBased on the analysis above:\n")
        
        # Find best strategy for each regime
        for char in ['volatility', 'trend_strength', 'relative_volume']:
            quartiles = create_regime_labels(market_chars[char])
            f.write(f"\n{char.replace('_', ' ').title()} conditions:\n")
            
            for regime in quartiles.unique():
                mask = quartiles == regime
                returns = returns_df[mask].mean() * 252  # Annualized returns
                best_strategy = returns.idxmax()
                best_return = returns.max()
                f.write(f"- {regime}: {best_strategy} (Return: {best_return*100:.2f}%)\n")
    
    return analysis_file

def plot_results(results_dict, symbol, output_dir):
    plt.style.use('default')
    
    # Create custom color map for strategies
    strategy_colors = {
        'Buy and Hold': 'black',
        'Adaptive': 'yellow',
        'EMA': 'blue',
        'MACD': 'green',
        'RSI': 'red',
        'Stochastic': 'purple',
        'Volume RSI': 'cyan',
        'VWAP': 'magenta'
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Get first result for buy and hold calculation
    first_result = next(iter(results_dict.values()))
    df = first_result['portfolio']
    initial_capital = float(BACKTEST_CONFIG['initial_capital'])
    initial_price = float(df['close'].iloc[0])
    buy_hold_units = initial_capital / initial_price
    buy_hold_values = df['close'] * buy_hold_units
    
    # Plot portfolio values
    ax1.plot(df.index, buy_hold_values, 
             label='Buy and Hold', linewidth=2, color=strategy_colors['Buy and Hold'], 
             linestyle='--')
    
    for strategy_name, result in results_dict.items():
        portfolio = result['portfolio']
        color = strategy_colors.get(strategy_name, 'gray')  # Default to gray if strategy not in color map
        ax1.plot(portfolio.index, portfolio['total_value'], 
                label=strategy_name, linewidth=1.5, color=color)
    
    ax1.set_title(f'Portfolio Value Over Time - {symbol}', fontsize=12, pad=20)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot drawdowns
    buy_hold_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    ax2.plot(df.index, buy_hold_dd, 
             label='Buy and Hold', linewidth=2, color=strategy_colors['Buy and Hold'], 
             linestyle='--')
    
    for strategy_name, result in results_dict.items():
        portfolio = result['portfolio']
        drawdown = (portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()
        color = strategy_colors.get(strategy_name, 'gray')  # Default to gray if strategy not in color map
        ax2.plot(portfolio.index, drawdown, label=strategy_name, linewidth=1.5, color=color)
    
    ax2.set_title('Strategy Drawdowns', fontsize=12, pad=20)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    plot_file = os.path.join(symbol_dir, f'performance_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_combined_results(combined_results, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = os.path.join(output_dir, f'combined_results_{timestamp}.txt')
    
    with open(combined_file, 'w') as f:
        f.write("=== Combined Backtest Results ===\n")
        f.write(f"Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}\n")
        f.write(f"Initial Capital per Symbol: ${BACKTEST_CONFIG['initial_capital']}\n\n")
        
        for symbol, results in combined_results.items():
            f.write(f"\n=== {symbol} Results ===\n")
            all_results = results['strategies'] + [results['buy_hold']]
            f.write(tabulate(all_results, headers="keys", tablefmt="grid", numalign="right"))
            f.write("\n" + "="*80 + "\n")
    
    return combined_file

def main():
    try:
        print("Starting backtesting process...")
        
        # Initialize database connection
        db = DatabaseHandler()
        
        # Create results directory
        output_dir = BACKTEST_CONFIG['results_dir']
        ensure_directory(output_dir)
        print(f"\nResults will be saved to: {output_dir}")
        
        # Store results for all symbols
        combined_results = {}
        optimization_results = {}
        
        # Process each symbol
        for symbol in BACKTEST_CONFIG['symbols']:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}...")
            print(f"{'='*50}")
            
            # Get historical data
            print("\nFetching historical data...")
            data = db.get_historical_data(
                symbol,
                BACKTEST_CONFIG['start_date'],
                BACKTEST_CONFIG['end_date']
            )
            
            if len(data) == 0:
                print(f"No data found for {symbol}")
                continue
                
            print(f"Loaded {len(data)} records")
            print("Initial columns:", data.columns.tolist())
            
            # Get training and testing dates from config
            train_start = BACKTEST_CONFIG['optimization']['training_start']
            train_end = BACKTEST_CONFIG['optimization']['training_end']
            test_start = BACKTEST_CONFIG['optimization']['testing_start']
            test_end = BACKTEST_CONFIG['optimization']['testing_end']
            
            print(f"\nTraining period: {train_start} to {train_end}")
            print(f"Testing period: {test_start} to {test_end}")
            
            # Add technical indicators for optimization
            print("\nCalculating technical indicators...")
            data_with_indicators = TechnicalIndicators.add_all_indicators(data.copy())
            print("Added indicators. Final columns:", data_with_indicators.columns.tolist())
            
            # Optimize strategy parameters using training data
            print("\nOptimizing strategy parameters...")
            optimization_results[symbol] = TradingStrategies.optimize_strategy_parameters(
                data_with_indicators,
                symbol,
                train_start,
                train_end
            )
            
            # Run backtests on testing period
            results = {}
            print("\nRunning backtests for each strategy:")
            
            # Get all base strategies
            base_strategies = TradingStrategies.get_all_strategies()
            
            # Remove the original adaptive and optimized strategies if they exist
                        # Remove the original adaptive and optimized strategies if they exist
            base_strategies.pop('Adaptive', None)
            base_strategies.pop('Optimized', None)
            
            # Create a closure for the optimized strategy that includes the optimization results
            opt_results = optimization_results[symbol]
            optimized_strategy = lambda df: TradingStrategies.optimized_adaptive_strategy(
                df, 
                {
                    'best_params': {k: v['best_parameters'] for k, v in opt_results['strategies'].items()},
                    'strategy_weights': {k: v['weight'] for k, v in opt_results['strategies'].items()},
                    'market_condition_params': opt_results['market_conditions']
                }
            )
            
            # Add optimized strategy to the base strategies
            base_strategies['Optimized'] = optimized_strategy
            
            # Run each strategy
            for strategy_name, strategy_func in base_strategies.items():
                print(f"\nBacktesting {strategy_name} strategy...")
                try:
                    backtester = Backtester(data_with_indicators, strategy_name, strategy_func)
                    results[strategy_name] = backtester.run()
                    print(f"✓ {strategy_name} strategy completed successfully")
                except Exception as e:
                    print(f"✗ Error in {strategy_name} strategy: {str(e)}")
                    continue  # Skip failed strategy but continue with others
            
            # Save results
            print("\nSaving results...")
            summary_file = save_results_to_file(
                results, symbol, output_dir, combined_results,
                optimization_results=optimization_results[symbol],
                train_end=train_end
            )
            print(f"✓ Results saved to {summary_file}")
            
            test_period_file = save_test_period_results(results, symbol, output_dir, train_end)
            print(f"✓ Test period results saved to {test_period_file}")
            
            # Generate correlation analysis
            print("\nGenerating correlation analysis...")
            analysis_file = analyze_strategy_correlations(
                results, data_with_indicators, symbol, output_dir
            )
            print(f"✓ Correlation analysis saved to {analysis_file}")
            
            # Create plots
            if BACKTEST_CONFIG['save_plots']:
                print("\nGenerating performance plots...")
                plot_results(results, symbol, output_dir)
                print("✓ Performance plots saved")
        
        # Save combined results
        if combined_results:
            print("\nSaving combined results...")
            combined_file = save_combined_results(combined_results, output_dir)
            print(f"✓ Combined results saved to {combined_file}")
        
        print("\nBacktesting process completed successfully!")
        
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        raise
    finally:
        print("\nClosing database connection...")
        db.close()

if __name__ == "__main__":
    main()