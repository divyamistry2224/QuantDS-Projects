import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_cumulative_returns(daily_returns, output_path=None):
    """Plot cumulative returns for multiple tickers."""
    cum_returns = (1 + daily_returns).cumprod()

    plt.figure(figsize=(10, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col)

    plt.title("Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1 Investment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved plot to {output_path}")
    else:
        plt.show()


def plot_sharpe_ratios(sharpe_dict, output_path=None):
    """Bar plot of Sharpe ratios."""
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(sharpe_dict.keys()), y=list(sharpe_dict.values()), palette="viridis")
    plt.title("Sharpe Ratios by Stock")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Stock")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved plot to {output_path}")
    else:
        plt.show()
