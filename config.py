#config.py

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "cryptocurrencies",
    "user": "myuser",
    "password": "mypassword"
}

BACKTEST_CONFIG = {
    # Date range
    "start_date": "2022-11-01",
    "end_date": "2024-12-31",
    
    # Trading pairs to test
    "symbols": ["LTC/USD", "BTC/USD"],
    
    # Initial capital for each currency
    "initial_capital": 10000,
    
    # Position size (percentage of capital)
    "position_size": 0.95,  # 95% of capital
    
    # Optimization configuration
    "optimization": {
        "training_start": "2022-01-01",  # Start date for training period
        "training_end": "2023-12-31",    # End date for training period
        "testing_start": "2024-01-01",   # Start date for testing period
        "testing_end": "2024-06-01",     # End date for testing period
        "min_training_days": 30,         # Minimum days required for training
        
        # Parameter ranges for grid search
        "parameter_ranges": {
            "ema": {
                "short": range(5, 15, 2),    # [5,7,9,11,13]
                "medium": range(15, 30, 3),   # [15,18,21,24,27]
                "long": range(30, 60, 5)      # [30,35,40,45,50,55]
            },
            "macd": {
                "fast": range(8, 16, 2),     # [8,10,12,14]
                "slow": range(20, 32, 3),    # [20,23,26,29]
                "signal": range(7, 12, 1)    # [7,8,9,10,11]
            },
            "rsi": {
                "period": range(10, 20, 2),  # [10,12,14,16,18]
                "overbought": range(65, 80, 5),  # [65,70,75]
                "oversold": range(20, 35, 5)     # [20,25,30]
            },
            "stochastic": {
                "k_period": range(10, 20, 2),    # [10,12,14,16,18]
                "d_period": range(3, 7, 1),      # [3,4,5,6]
                "overbought": range(75, 85, 5),  # [75,80]
                "oversold": range(15, 25, 5)     # [15,20]
            },
            "vwap": {
                "period": [12, 24, 36],
                "overbought": [1.01, 1.015, 1.02, 1.025, 1.03],
                "oversold": [0.97, 0.975, 0.98, 0.985, 0.99]
            },
            "volume_rsi": {
                "period": range(10, 20, 2),      # [10,12,14,16,18]
                "overbought": range(65, 80, 5),  # [65,70,75]
                "oversold": range(20, 35, 5)     # [20,25,30]
            }
        }
    },
    
    # Technical indicators baseline parameters
    "vwap": {
        "period": 24,
        "overbought": 1.02,
        "oversold": 0.98
    },
    "ema": {
        "short": 9,
        "medium": 21,
        "long": 50
    },
    "macd": {
        "fast": 12,
        "slow": 26,
        "signal": 9
    },
    "rsi": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "overbought": 80,
        "oversold": 20
    },
    "volume_rsi": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    },
    
    # Trading fees
    "trading_fee": 0,  # 0.001 = 0.1%
    
    # Output configuration
    "results_dir": "backtest_results",  # Directory to save results
    "save_trades": True,  # Whether to save detailed trade information
    "save_plots": True   # Whether to save plots as PNG files
}