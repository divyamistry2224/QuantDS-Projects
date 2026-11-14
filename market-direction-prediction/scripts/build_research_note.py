"""Build a shareable finance research note from committed measured outputs."""
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "reports/metrics.json").read_text())
E = json.loads((ROOT / "experiments/v2/reports/metrics.json").read_text())
P = json.loads((ROOT / "reports/profitability.json").read_text())
GREEN, INK, GRAY = "#17695e", "#17313a", "#65777d"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "font.size": 10})


def page(title, number):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(.08, .95, "DIVYA MISTRY  /  FINANCIAL MACHINE LEARNING", color=GREEN, fontsize=8, weight="bold")
    fig.text(.08, .90, title, fontsize=23, weight="bold")
    fig.text(.08, .04, "Market Direction Prediction · Historical research", color=GRAY, fontsize=8)
    fig.text(.90, .04, f"0{number}", color=GRAY, fontsize=8)
    return fig


def paragraph(fig, y, text, width=93, size=10, color=GRAY):
    lines = textwrap.fill(text, width=width)
    fig.text(.08, y, lines, fontsize=size, color=color, va="top", linespacing=1.7)
    return y - (lines.count("\n") + 1) * .020 - .022


def build():
    selected = next(m for m in R["models"] if m["id"] == R["selected_model"])
    path = ROOT / "reports/market_research_note.pdf"
    with PdfPages(path) as pdf:
        fig = page("Market direction research", 1)
        y = paragraph(fig, .84, "Research question: can public economic-news sentiment and macroeconomic conditions add predictive information beyond market behavior, and does a cost-aware implementation generate positive financial outcomes?")
        fig.text(.08, y, "MEASURED RESULTS", fontsize=9, weight="bold", color=GREEN); y -= .04
        for title, text in [
            ("+12.15% net historical return", "The original validation-selected Logistic/full model, 2022–2025. Annualized return 2.92%, net Sharpe 0.40, maximum drawdown -6.98%."),
            ("+37.27% CatBoost comparison return", "CatBoost/market + macro over 2022–2025: annualized 8.28%, net Sharpe 0.78. This was not the validation-selected winner."),
            ("53.53% fresh balanced accuracy", "Enhanced CatBoost over January–October 2026: 53.11% accuracy versus 50.72% always-up. The 95% block-bootstrap balanced-accuracy interval is 48.24%–58.62%."),
            ("-0.44% fresh net trading return", "The fixed trading overlay does not make money after costs in that fresh period. Directional discrimination and net profitability are separate outcomes.")]:
            fig.text(.08, y, title, fontsize=13, weight="bold"); y -= .03
            y = paragraph(fig, y, text)
        y = paragraph(fig, y-.015, "Benchmark context: original SPY buy-and-hold returned +51.72% with -24.50% drawdown. It includes overnight exposure; the model holds only next-session open to close. Positive profit does not imply benchmark outperformance.")
        paragraph(fig, y, "This is a reproducible historical research project, not proven alpha or live execution. Latest-vintage data, assumed publication lags, a single ETF and a short new evaluation sample materially limit inference.", size=9)
        pdf.savefig(fig);plt.close(fig)

        fig = page("From data to financial P&L", 2)
        y = .84
        for heading, text in [
            ("1 / Public data, with provenance", "Yahoo Finance SPY OHLCV and adjusted prices; FRED DFF, T10Y2Y, VIXCLS, UNRATE and CPIAUCSL; FRBSF Daily News Sentiment Index. The extension adds QQQ, IWM, TLT, HYG, GLD and UUP. Committed normalized snapshots include URLs, UTC retrieval times and SHA-256 hashes."),
            ("2 / Causal features and explicit availability", "Original features combine momentum, volatility, volume, moving averages, rates, the term spread, unemployment, CPI inflation and sentiment trends. The extension adds reversal, RSI, close location, calendar effects and cross-asset moves. Backward joins never borrow future data. Daily macro uses a two-business-day lag; monthly data use month-end +45 days; sentiment uses +7 days. These assumptions do not eliminate revised-data bias."),
            ("3 / Chronological model evaluation", "CatBoost, Random Forest and Logistic are compared on 2019–2021 annual validation, with five purged sessions. The final original fit evaluates 2022–2025. After that result is observed, the extension explicitly reuses 2019–2025 as development and reserves January–October 2026. Thirty-two development configurations and sixteen thresholds select one fresh-test winner."),
            ("4 / Execution and costs", "Signal after close t; enter at open t+1, exit that close. Long or cash, no leverage or shorts. Every active day pays 2 bps entry plus 2 bps exit, including consecutive signals. Cash earns zero. Sharpe uses zero reference rate. Taxes, capacity, market impact and financing are not modeled."),
            ("5 / Research integrity", "Tests check checksum tampering, invalid OHLC, backward joins, future-input mutation invariance, next-session labels, purged splits, transaction-cost accounting, drawdown and agreement between prediction files, metrics and dashboard artifacts.")]:
            fig.text(.08, y, heading, fontsize=12, weight="bold");y-=.03
            y=paragraph(fig,y,text)
        pdf.savefig(fig);plt.close(fig)

        fig=page("Model comparison & equity",3)
        ax=fig.add_axes([.08,.62,.84,.21]);ax.axis('off')
        rows=[]
        for m in sorted(R['models'],key=lambda m:m['cv_balanced_accuracy'],reverse=True):
            rows.append([m['family'],m['features'],f"{m['cv_balanced_accuracy']:.1%}",f"{m['holdout']['balanced_accuracy']:.1%}",f"{m['strategy']['total_return']:+.1%}",f"{m['strategy']['sharpe']:.2f}"])
        table=ax.table(cellText=rows,colLabels=['Model','Features','Val BA','Test BA','Net return','Sharpe'],loc='center',cellLoc='left',colWidths=[.17,.25,.14,.14,.12,.10]);table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.4)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('#dce3e2');cell.set_linewidth(.4)
            if row==0:cell.set_facecolor('#edf3ef');cell.set_text_props(weight='bold')
        ax=fig.add_axes([.05,.21,.91,.37]);ax.imshow(plt.imread(ROOT/'reports/figures/holdout_equity.png'));ax.axis('off')
        paragraph(fig,.15,"Original selection uses validation balanced accuracy, not the largest retrospective profit. The selected model has a smaller drawdown and much less exposure than buy-and-hold, but lower cumulative return. Slow-moving macro/news features did not consistently improve one-session direction.",size=9)
        pdf.savefig(fig);plt.close(fig)

        fig=page("Interpretation & next steps",4)
        y=.84
        for heading,text in [
            ("What can be claimed", "The repository implements a connected financial research pipeline with real public-source data, reproducible model comparisons and cost-aware financial outputs. Some strategies generate positive historical net returns. The enhanced CatBoost records higher fresh-sample direction accuracy than always-up, with uncertainty spanning chance."),
            ("What cannot be claimed", "The study does not prove durable alpha, reliable future profits, statistical superiority, real-time data availability or cross-asset generalization. No brokerage integration or production forecasting service is implemented. The highest retrospective return is not automatically the best deployable strategy."),
            ("Testing history", "Extra Trees and histogram gradient boosting were added after the original test was inspected. Their historical backtests reuse that period and are not independent confirmation. Dashboard threshold and cost changes are scenario analysis. All studies and the extension protocol remain in version control."),
            ("Next finance research steps", "Use licensed point-in-time vintages and release calendars. Freeze a forward paper-trading protocol and collect timestamped signals. Test longer horizons suitable for economic data, independent universes and volatility regimes. Measure spread/slippage, cash financing and capacity. Separate calibration, selection and independent evaluation."),
            ("Reproduce and inspect", "github.com/divyamistry2224/QuantDS-Projects/tree/main/market-direction-prediction\nLive dashboard: divyamistry2224.github.io/QuantDS-Projects/market-direction/\nInstall the package, verify snapshots, run pytest, then reproduce the original experiment, extension and profitability comparisons. See the notebook, model card and data dictionary.")]:
            fig.text(.08,y,heading,fontsize=12,weight='bold');y-=.03;y=paragraph(fig,y,text)
        pdf.savefig(fig);plt.close(fig)
    shutil.copyfile(path,ROOT/'docs/research/market_research_note.pdf')
    return path


if __name__=='__main__': print(build())
