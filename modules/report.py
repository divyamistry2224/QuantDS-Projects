# =========================================
# modules/report.py
# Generate Quant Performance Report (PDF)
# =========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# -----------------------------
# Helper: Save plots
# -----------------------------
def plot_returns(returns_series, output_dir):
    """Plot asset returns over time and save cumulative return chart."""
    plt.figure(figsize=(8, 4))
    (1 + returns_series).cumprod().plot(title="Cumulative Returns", linewidth=2, color="teal")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, "portfolio_returns.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_volatility(returns_series, output_dir):
    """Plot rolling volatility (standard deviation)."""
    plt.figure(figsize=(8, 4))
    returns_series.rolling(window=21).std().plot(
        title="21-Day Rolling Volatility", linewidth=2, color="orange"
    )
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, "volatility.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_weights(weights_path, output_dir):
    """Plot portfolio weights as bar chart."""
    weights_df = pd.read_csv(weights_path)
    plt.figure(figsize=(6, 4))
    plt.bar(weights_df["Ticker"], weights_df["Weight"], color="steelblue", edgecolor="black")
    plt.title("Optimized Portfolio Weights")
    plt.xlabel("Ticker")
    plt.ylabel("Weight")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, "weights.png")
    plt.savefig(path)
    plt.close()
    return path


# -----------------------------
# Main: Generate Report
# -----------------------------
def generate_report(weights_path, returns_series):
    report_path = os.path.join("data/processed", "quant_report.csv")
    weights = pd.read_csv(weights_path)
    report_df = weights.copy()
    report_df['AvgReturn'] = returns_series.mean()
    report_df.to_csv(report_path, index=False)
    return report_path

    # ---- Generate plots ----
    print("📊 Generating visualizations...")
    returns_plot = plot_returns(returns_series, output_dir)
    vol_plot = plot_volatility(returns_series, output_dir)
    weights_plot = plot_weights(weights_path, output_dir)

    # ---- Build PDF ----
    styles = getSampleStyleSheet()
    report = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []

    # Title and date
    story.append(Paragraph("📘 Quantitative Research Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
    )
    story.append(Spacer(1, 24))

    # ---- Summary statistics ----
    story.append(Paragraph("Summary Statistics", styles["Heading2"]))
    story.append(Spacer(1, 6))

    mean_return = returns_series.mean()
    volatility = returns_series.std()
    sharpe = mean_return / volatility if volatility != 0 else 0

    story.append(Paragraph(f"Average Daily Return: {mean_return:.4f}", styles["Normal"]))
    story.append(Paragraph(f"Daily Volatility: {volatility:.4f}", styles["Normal"]))
    story.append(Paragraph(f"Approx. Daily Sharpe Ratio: {sharpe:.4f}", styles["Normal"]))
    story.append(Spacer(1, 18))

    # ---- Charts ----
    story.append(Paragraph("Cumulative Returns", styles["Heading2"]))
    story.append(Image(returns_plot, width=400, height=250))
    story.append(Spacer(1, 12))

    story.append(Paragraph("21-Day Rolling Volatility", styles["Heading2"]))
    story.append(Image(vol_plot, width=400, height=250))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Portfolio Weights", styles["Heading2"]))
    story.append(Image(weights_plot, width=400, height=250))
    story.append(Spacer(1, 12))

    # ---- Build and Save ----
    report.build(story)
    print(f"✅ Report successfully saved to: {pdf_path}")
    return pdf_path


# -----------------------------
# Quick Local Test (Optional)
# -----------------------------
if __name__ == "__main__":
    # Run test only if needed
    sample_weights = "data/optimized_weights.csv"
    returns = pd.read_csv("data/processed/daily_returns.csv", index_col=0)["AAPL"]
    generate_report(sample_weights, returns)
