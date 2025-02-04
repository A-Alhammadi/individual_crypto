# main.py

import pandas as pd
import numpy as np
from database import DatabaseHandler
from indicators import TechnicalIndicators
from strategies import TradingStrategies
from backtester import Backtester
from config import BACKTEST_CONFIG

# Import the advanced optimization logic
from optimized_strategy import optimize_strategy_parameters, OptimizedStrategy

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from datetime import datetime

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_strategy_metrics(portfolio_df, trades_df, period_start, period_end, initial_capital):
    """Calculate strategy metrics for a specific period."""
    mask = (portfolio_df.index >= period_start) & (portfolio_df.index <= period_end)
    period_data = portfolio_df[mask]
    
    if len(period_data) == 0:
        return None

    period_return = (
        period_data['total_value'].iloc[-1] - period_data['total_value'].iloc[0]
    ) / period_data['total_value'].iloc[0]
    period_length = (period_data.index[-1] - period_data.index[0]).days / 365.0
    annual_return = period_return / period_length if period_length > 0 else 0

    daily_returns = period_data['total_value'].pct_change()
    sharpe_ratio = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0

    peak = period_data['total_value'].expanding(min_periods=1).max()
    drawdown = (period_data['total_value'] - peak) / peak
    max_drawdown = drawdown.min()

    if trades_df is not None and not trades_df.empty:
        period_trades = trades_df[trades_df['date'].between(period_start, period_end)]
        num_trades = len(period_trades)
        win_rate = (period_trades['value'] > period_trades['value'].shift(1)).sum() / (num_trades // 2) if num_trades > 0 else 0
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
        f.write(f"Training Period: {BACKTEST_CONFIG['optimization']['training_start']} to {BACKTEST_CONFIG['optimization']['training_end']}\n")
        f.write(f"Testing Period: {BACKTEST_CONFIG['optimization']['testing_start']} to {BACKTEST_CONFIG['optimization']['testing_end']}\n")
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
        f.write(f"Period: {BACKTEST_CONFIG['optimization']['testing_start']} to {BACKTEST_CONFIG['optimization']['testing_end']}\n\n")
        
        # Calculate testing period metrics for each strategy
        testing_metrics = []

        first_result = next(iter(results_dict.values()))
        df = first_result['portfolio']
        test_mask = (df.index >= BACKTEST_CONFIG['optimization']['testing_start']) & (df.index <= BACKTEST_CONFIG['optimization']['testing_end'])
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
            
            # Metrics for each strategy
            for strategy_name, result in results_dict.items():
                metrics = calculate_strategy_metrics(
                    result['portfolio'],
                    result['trades'],
                    BACKTEST_CONFIG['optimization']['testing_start'],
                    BACKTEST_CONFIG['optimization']['testing_end'],
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
            
            # Additional details if "Optimized" is present
            if 'Optimized' in results_dict:
                f.write("\n\n=== Optimized Strategy Testing Period Details ===\n")
                opt_result = results_dict['Optimized']
                
                # Monthly returns
                test_portfolio = opt_result['portfolio'][test_mask]
                monthly_returns = test_portfolio['total_value'].resample('M').last().pct_change()
                
                f.write("\nMonthly Returns:\n")
                f.write(monthly_returns.to_string())
                
                # Trade analysis
                if not opt_result['trades'].empty:
                    test_trades = opt_result['trades'][
                        opt_result['trades']['date'].between(
                            BACKTEST_CONFIG['optimization']['testing_start'], 
                            BACKTEST_CONFIG['optimization']['testing_end']
                        )
                    ]
                    f.write("\n\nTrade Analysis:\n")
                    trade_stats = {
                        'Total Trades': len(test_trades),
                        'Average Trade Duration': (
                            f"{test_trades['date'].diff().mean().total_seconds() / 3600:.1f} hours"
                            if len(test_trades) > 1 else "N/A"
                        ),
                        'Average Trade Size': (
                            f"${test_trades['value'].mean():.2f}" if not test_trades.empty else "N/A"
                        ),
                        'Largest Winning Trade': (
                            f"${test_trades['value'].max():.2f}" if not test_trades.empty else "N/A"
                        ),
                        'Largest Losing Trade': (
                            f"${test_trades['value'].min():.2f}" if not test_trades.empty else "N/A"
                        ),
                    }
                    
                    for stat, value in trade_stats.items():
                        f.write(f"{stat}: {value}\n")
        else:
            f.write("No data available for testing period\n")
    
    return summary_file

def save_test_period_results(results_dict, symbol, output_dir, train_end):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    test_results_file = os.path.join(symbol_dir, f'test_period_results_{timestamp}.txt')
    
    test_metrics = []
    first_result = next(iter(results_dict.values()))
    df = first_result['portfolio']
    test_mask = (df.index >= BACKTEST_CONFIG['optimization']['testing_start']) & (df.index <= BACKTEST_CONFIG['optimization']['testing_end'])
    test_df = df[test_mask]
    
    if len(test_df) > 0:
        # Buy and Hold
        initial_price = float(test_df['close'].iloc[0])
        final_price = float(test_df['close'].iloc[-1])
        bh_return = (final_price - initial_price) / initial_price
        test_days = (test_df.index[-1] - test_df.index[0]).days
        bh_annual_return = bh_return * (365 / test_days) if test_days > 0 else 0

        for strategy_name, result in results_dict.items():
            portfolio = result['portfolio'][test_mask]
            trades = result['trades'][result['trades']['date'] >= train_end] if not result['trades'].empty else pd.DataFrame()

            strategy_return = (
                portfolio['total_value'].iloc[-1] - portfolio['total_value'].iloc[0]
            ) / portfolio['total_value'].iloc[0]
            strategy_annual_return = strategy_return * (365 / test_days) if test_days > 0 else 0
            
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
                'Win Rate (%)': (
                    f"{(len(trades[trades['value'] > trades['value'].shift(1)]) / (len(trades)//2) * 100):.2f}%"
                    if len(trades) > 0 else "N/A"
                )
            })

        # Add buy & hold
        test_metrics.append({
            'Strategy': 'Buy and Hold',
            'Total Return (%)': f"{bh_return * 100:.2f}%",
            'Annual Return (%)': f"{bh_annual_return * 100:.2f}%",
            'Sharpe Ratio': "N/A",
            'Max Drawdown (%)': "N/A",
            'Number of Trades': 1,
            'Win Rate (%)': "N/A"
        })

        with open(test_results_file, 'w') as f:
            f.write(f"=== Test Period Performance for {symbol} ===\n")
            f.write(f"Test Period: {BACKTEST_CONFIG['optimization']['testing_start']} to {BACKTEST_CONFIG['optimization']['testing_end']}\n")
            f.write(f"Total Test Days: {test_days}\n\n")
            
            metrics_df = pd.DataFrame(test_metrics)
            f.write(tabulate(metrics_df, headers='keys', tablefmt='grid', numalign='right'))
            
            f.write("\n\nStrategy Performance vs Buy and Hold:\n")
            bh_return_val = float(test_metrics[-1]['Total Return (%)'].rstrip('%'))
            for metric in test_metrics[:-1]:
                strategy_return_val = float(metric['Total Return (%)'].rstrip('%'))
                improvement = strategy_return_val - bh_return_val
                f.write(f"{metric['Strategy']}: {improvement:+.2f}% vs Buy and Hold\n")
    else:
        with open(test_results_file, 'w') as f:
            f.write("No data available for test period\n")

    # >>> ADDITIONAL DETAILS FOR "Optimized" STRATEGY <<<
    if 'Optimized' in results_dict:
        with open(test_results_file, 'a') as f:
            f.write("\n=== Detailed Analysis of Optimized Strategy ===\n")
            opt_trades = results_dict['Optimized']['trades']
            # Filter just testing period
            test_trades = opt_trades[
                opt_trades['date'].between(
                    BACKTEST_CONFIG['optimization']['testing_start'],
                    BACKTEST_CONFIG['optimization']['testing_end']
                )
            ]
            f.write(f"Total Trades (Optimized): {len(test_trades)}\n")
            if not test_trades.empty:
                largest_win = test_trades['value'].max()
                largest_loss = test_trades['value'].min()
                f.write(f"Largest Win: ${largest_win:.2f}\n")
                f.write(f"Largest Loss: ${largest_loss:.2f}\n")
    
    return test_results_file


def analyze_strategy_correlations(results_dict, df, symbol, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    analysis_file = os.path.join(symbol_dir, f'correlation_analysis_{timestamp}.txt')
    
    strategy_returns = {}
    for strategy_name, result in results_dict.items():
        strategy_returns[strategy_name] = result['portfolio']['total_value'].pct_change()
    
    returns_df = pd.DataFrame(strategy_returns)

    # Select the market characteristics we want to analyze
    market_chars = df[['volatility', 'trend_strength', 'relative_volume', 'momentum', 'atr']].copy()

    correlations = pd.DataFrame()

    def create_regime_labels(series, n_bins=4):
        try:
            return pd.qcut(series, n_bins, labels=['Low','Med-Low','Med-High','High'])
        except ValueError:
            try:
                return pd.qcut(series, 3, labels=['Low','Medium','High'])
            except ValueError:
                return pd.cut(series, 2, labels=['Low','High'])
    
    def calculate_regime_stats(returns_df, mask):
        stats = {}
        for strategy in returns_df.columns:
            sr = returns_df[strategy][mask]
            annual_return = sr.mean()*252
            std_dev = sr.std()
            sharpe = (sr.mean()/std_dev*np.sqrt(252)) if std_dev != 0 else 0
            win_rate = (sr>0).mean()
            stats[strategy] = {
                'Annual Return': f"{annual_return*100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.2f}",
                'Win Rate': f"{win_rate*100:.2f}%"
            }
        return pd.DataFrame(stats).T

    with open(analysis_file, 'w') as f:
        f.write(f"=== Strategy Performance Correlation Analysis for {symbol} ===\n\n")
        f.write("1. Overall Correlations with Market Characteristics:\n")
        for char in market_chars.columns:
            for strategy in returns_df.columns:
                corr = returns_df[strategy].corr(market_chars[char])
                correlations.loc[strategy, char] = corr
        f.write(tabulate(correlations.round(3), headers='keys', tablefmt='grid'))
        f.write("\n\n")

        f.write("2. Strategy Performance in Different Market Conditions:\n\n")
        for char in ['volatility','trend_strength','relative_volume']:
            quartiles = create_regime_labels(market_chars[char])
            f.write(f"\na) Performance in Different {char.title()} Regimes:\n")
            for regime in quartiles.unique():
                mask = quartiles == regime
                regime_stats = calculate_regime_stats(returns_df, mask)
                f.write(f"\n{char.title()} Regime: {regime}\n")
                f.write(tabulate(regime_stats, headers='keys', tablefmt='grid'))
                f.write("\n")
        
        f.write("\n3. Key Strategy Recommendations:\n")
        f.write("\nBased on the analysis above:\n")

        for char in ['volatility','trend_strength','relative_volume']:
            quartiles = create_regime_labels(market_chars[char])
            f.write(f"\n{char.title()} conditions:\n")
            for regime in quartiles.unique():
                mask = quartiles == regime
                sr = returns_df[mask].mean()*252
                best_strat = sr.idxmax()
                best_ret = sr.max()
                f.write(f"- {regime}: {best_strat} (Return: {best_ret*100:.2f}%)\n")

    return analysis_file

def plot_results(results_dict, symbol, output_dir):
    """Plot full period results with enhanced Optimized strategy visibility"""
    print(f"\nPlotting results for {symbol}")
    print(f"Strategies to plot: {list(results_dict.keys())}")
    
    plt.style.use('default')
    strategy_colors = {
        'Optimized': '#FF4500',  # Bright orange-red
        'Buy and Hold': '#000000',
        'EMA': '#1E90FF',
        'MACD': '#32CD32',
        'RSI': '#DC143C',
        'Stochastic': '#9370DB',
        'Volume RSI': '#00CED1',
        'VWAP': '#FF69B4'
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    first_result = next(iter(results_dict.values()))
    df = first_result['portfolio']
    initial_capital = float(BACKTEST_CONFIG['initial_capital'])
    initial_price = float(df['close'].iloc[0])
    buy_hold_units = initial_capital / initial_price
    buy_hold_values = df['close'] * buy_hold_units
    
    # Plot buy and hold baseline
    ax1.plot(df.index, buy_hold_values, label='Buy and Hold', 
             linewidth=1.5, color=strategy_colors['Buy and Hold'], 
             linestyle='--', alpha=0.7)
    
    # Plot Optimized strategy first and prominently
    if 'Optimized' in results_dict:
        print("Plotting Optimized strategy...")
        portfolio = results_dict['Optimized']['portfolio']
        final_value = portfolio['total_value'].iloc[-1]
        print(f"Optimized strategy final value: ${final_value:.2f}")
        
        ax1.plot(portfolio.index, portfolio['total_value'], 
                label=f'Optimized (${final_value:.0f})', 
                linewidth=2.5, color=strategy_colors['Optimized'])
    
    # Plot other strategies
    for strategy_name, result in results_dict.items():
        if strategy_name not in ['Buy and Hold', 'Optimized']:
            portfolio = result['portfolio']
            final_value = portfolio['total_value'].iloc[-1]
            color = strategy_colors.get(strategy_name, 'gray')
            ax1.plot(portfolio.index, portfolio['total_value'], 
                    label=f'{strategy_name} (${final_value:.0f})', 
                    linewidth=1.5, color=color, alpha=0.6)
    
    ax1.set_title(f'Portfolio Value Over Time - {symbol}', fontsize=12, pad=20)
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Drawdown plot
    buy_hold_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    max_bh_dd = buy_hold_dd.min()
    ax2.plot(df.index, buy_hold_dd, 
             label=f'Buy and Hold (Max DD: {max_bh_dd:.1%})', 
             linewidth=1.5, color=strategy_colors['Buy and Hold'], 
             linestyle='--', alpha=0.7)
    
    # Plot Optimized strategy drawdown first
    if 'Optimized' in results_dict:
        portfolio = results_dict['Optimized']['portfolio']
        drawdown = (portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()
        max_dd = drawdown.min()
        ax2.plot(portfolio.index, drawdown, 
                label=f'Optimized (Max DD: {max_dd:.1%})', 
                linewidth=2.5, color=strategy_colors['Optimized'])
    
    # Plot other strategies' drawdowns
    for strategy_name, result in results_dict.items():
        if strategy_name not in ['Buy and Hold', 'Optimized']:
            portfolio = result['portfolio']
            drawdown = (portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()
            max_dd = drawdown.min()
            color = strategy_colors.get(strategy_name, 'gray')
            ax2.plot(portfolio.index, drawdown, 
                    label=f'{strategy_name} (Max DD: {max_dd:.1%})', 
                    linewidth=1.5, color=color, alpha=0.6)
    
    ax2.set_title('Strategy Drawdowns', fontsize=12, pad=20)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    plot_file = os.path.join(symbol_dir, f'performance_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Full period plot saved to {plot_file}")

def plot_test_period_results(results_dict, symbol, output_dir, test_start, test_end):
    """Plot test period results with enhanced Optimized strategy visibility"""
    print(f"\nPlotting test period results for {symbol}")
    print(f"Test period: {test_start} to {test_end}")
    
    plt.style.use('default')
    strategy_colors = {
        'Optimized': '#FF4500',  # Bright orange-red
        'Buy and Hold': '#000000',
        'EMA': '#1E90FF',
        'MACD': '#32CD32',
        'RSI': '#DC143C',
        'Stochastic': '#9370DB',
        'Volume RSI': '#00CED1',
        'VWAP': '#FF69B4'
    }

    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    first_result = next(iter(results_dict.values()))
    df = first_result['portfolio']
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    test_df = df.loc[test_mask]

    if test_df.empty:
        print(f"No data available for test period {test_start} to {test_end} for {symbol}")
        return

    initial_capital = float(BACKTEST_CONFIG['initial_capital'])
    initial_price = float(test_df['close'].iloc[0])
    buy_hold_units = initial_capital / initial_price
    buy_hold_values = test_df['close'] * buy_hold_units

    # Plot Buy and Hold baseline
    final_bh_value = buy_hold_values.iloc[-1]
    ax1.plot(test_df.index, buy_hold_values,
             label=f'Buy and Hold (${final_bh_value:.0f})', 
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)

    # Plot Optimized strategy first
    if 'Optimized' in results_dict:
        test_portfolio = results_dict['Optimized']['portfolio'].loc[test_mask]
        if not test_portfolio.empty:
            final_value = test_portfolio['total_value'].iloc[-1]
            print(f"Optimized strategy test period final value: ${final_value:.2f}")
            ax1.plot(test_portfolio.index, test_portfolio['total_value'],
                    label=f'Optimized (${final_value:.0f})',
                    linewidth=2.5, color=strategy_colors['Optimized'])

    # Plot other strategies
    for strategy_name, result in results_dict.items():
        if strategy_name not in ['Buy and Hold', 'Optimized']:
            test_portfolio = result['portfolio'].loc[test_mask]
            if test_portfolio.empty:
                continue
            final_value = test_portfolio['total_value'].iloc[-1]
            color = strategy_colors.get(strategy_name, 'gray')
            ax1.plot(test_portfolio.index, test_portfolio['total_value'],
                    label=f'{strategy_name} (${final_value:.0f})',
                    linewidth=1.5, color=color, alpha=0.6)

    ax1.set_title(f'{symbol} - Portfolio Value (Testing Period)', fontsize=12, pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot drawdowns
    bh_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    max_bh_dd = bh_dd.min()
    ax2.plot(test_df.index, bh_dd, 
             label=f'Buy and Hold (Max DD: {max_bh_dd:.1%})',
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)

    # Plot Optimized strategy drawdown first
    if 'Optimized' in results_dict:
        test_portfolio = results_dict['Optimized']['portfolio'].loc[test_mask]
        if not test_portfolio.empty:
            drawdown = (test_portfolio['total_value'].cummax() - test_portfolio['total_value']) / test_portfolio['total_value'].cummax()
            max_dd = drawdown.min()
            ax2.plot(test_portfolio.index, drawdown,
                    label=f'Optimized (Max DD: {max_dd:.1%})',
                    linewidth=2.5, color=strategy_colors['Optimized'])

    # Plot other strategies' drawdowns
    for strategy_name, result in results_dict.items():
        if strategy_name not in ['Buy and Hold', 'Optimized']:
            test_portfolio = result['portfolio'].loc[test_mask]
            if test_portfolio.empty:
                continue
            drawdown = (test_portfolio['total_value'].cummax() - test_portfolio['total_value']) / test_portfolio['total_value'].cummax()
            max_dd = drawdown.min()
            color = strategy_colors.get(strategy_name, 'gray')
            ax2.plot(test_portfolio.index, drawdown,
                    label=f'{strategy_name} (Max DD: {max_dd:.1%})',
                    linewidth=1.5, color=color, alpha=0.6)

    ax2.set_title('Drawdowns (Testing Period)', fontsize=12, pad=20)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(symbol_dir, f'testing_period_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Test period plot saved to {plot_file}")
    
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

        combined_results = {}
        optimization_results = {}
        
        for symbol in BACKTEST_CONFIG['symbols']:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}...")
            print(f"{'='*50}")

            # Fetch historical data
            print("\nFetching historical data...")
            data = db.get_historical_data(symbol, BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'])
            if len(data) == 0:
                print(f"No data found for {symbol}")
                continue
            print(f"Loaded {len(data)} records")
            print("Initial columns:", data.columns.tolist())

            # Get training and testing periods
            train_start = BACKTEST_CONFIG['optimization']['training_start']
            train_end   = BACKTEST_CONFIG['optimization']['training_end']
            test_start  = BACKTEST_CONFIG['optimization']['testing_start']
            test_end    = BACKTEST_CONFIG['optimization']['testing_end']

            print(f"\nTraining period: {train_start} to {train_end}")
            print(f"Testing period: {test_start} to {test_end}")

            # Add technical indicators
            print("\nCalculating technical indicators...")
            data_with_indicators = TechnicalIndicators.add_all_indicators(data.copy())
            print("Added indicators. Final columns:", data_with_indicators.columns.tolist())

            # Optimize strategy parameters (training period) using optimized_strategy
            print("\nOptimizing strategy parameters...")
            optimization_results[symbol] = optimize_strategy_parameters(
                data_with_indicators, symbol, train_start, train_end
            )

            print("\nSetting up strategies...")
            base_strategies = TradingStrategies.get_all_strategies()
            print(f"Base strategies: {list(base_strategies.keys())}")

            # Create the optimized strategy with better logging
            try:
                print("\nInitializing Optimized strategy...")
                opt_data = optimization_results[symbol]
                adv_strat = OptimizedStrategy(opt_data)
                
                # Print weights to verify optimization is loaded
                print("\nOptimized strategy weights:")
                for strat_name, weight in adv_strat.base_weights.items():
                    print(f"  {strat_name}: {weight:.4f}")
                
                base_strategies["Optimized"] = lambda df: adv_strat.run(df)
                print("✓ Added Optimized strategy to base_strategies")
                print(f"Final strategies: {list(base_strategies.keys())}")
            except Exception as e:
                print(f"Error setting up Optimized strategy: {str(e)}")
                raise

            # Backtest each strategy
            results = {}
            for strategy_name, strategy_func in base_strategies.items():
                print(f"\nBacktesting {strategy_name} strategy...")
                try:
                    backtester = Backtester(data_with_indicators, strategy_name, strategy_func)
                    results[strategy_name] = backtester.run()
                    print(f"✓ {strategy_name} strategy completed successfully")
                except Exception as e:
                    print(f"✗ Error in {strategy_name} strategy: {str(e)}")
                    continue
            
            # Save results
            print("\nSaving results...")
            summary_file = save_results_to_file(
                results, 
                symbol, 
                output_dir,
                combined_results,
                optimization_results=optimization_results[symbol],
                train_end=train_end
            )
            print(f"✓ Results saved to {summary_file}")

            test_period_file = save_test_period_results(results, symbol, output_dir, train_end)
            print(f"✓ Test period results saved to {test_period_file}")

            # Correlation analysis
            print("\nGenerating correlation analysis...")
            analysis_file = analyze_strategy_correlations(results, data_with_indicators, symbol, output_dir)
            print(f"✓ Correlation analysis saved to {analysis_file}")

            # Create plots
            if BACKTEST_CONFIG['save_plots']:
                print("\nGenerating performance plots...")
                # Full-range plot
                plot_results(results, symbol, output_dir)
                # NEW: test-only plot
                plot_test_period_results(results, symbol, output_dir, test_start, test_end)
                print("✓ Performance plots saved (including testing-only)")

        # Optionally, save combined results across symbols
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
