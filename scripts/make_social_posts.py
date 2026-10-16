"""
make_social_posts.py
--------------------
Builds a 7-slide carousel (1080 x 1350 px, Instagram/LinkedIn portrait size)
from the project's real outputs, plus a PDF version for LinkedIn
"document" posts.

    python main.py                      # generate results first
    python scripts/make_social_posts.py # -> images/social/
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

OUT = config.IMAGES_DIR / "social"
OUT.mkdir(parents=True, exist_ok=True)

BG, INK, MUTED = "#0f1b24", "#ffffff", "#9fb3c8"
TEAL, ORANGE, RED, YELLOW = "#2ec4b6", "#ff9f1c", "#e63946", "#ffd166"
TOTAL = 7

metrics = pd.read_csv(config.OUTPUTS_DIR / "model_metrics.csv", index_col=0)
policy = pd.read_csv(config.OUTPUTS_DIR / "policy_comparison.csv", index_col=0)
eda = pd.read_csv(config.OUTPUTS_DIR / "eda_summary.csv", index_col=0)["value"]
rf = metrics.loc["Random Forest"]
base = metrics.loc["Seasonal Naive (baseline)"]
old, new = policy.iloc[0], policy.iloc[1]
err_cut = (1 - rf["WAPE_%"] / base["WAPE_%"]) * 100
cost_cut = (1 - new["Total cost (Rs)"] / old["Total cost (Rs)"]) * 100
so_cut = (1 - new["Stockout days %"] / old["Stockout days %"]) * 100


def new_slide(n):
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 125)
    ax.axis("off")
    ax.text(6, 4, "Retail Forecasting & Inventory Optimization", color=MUTED,
            fontsize=13)
    ax.text(94, 4, f"{n}/{TOTAL}", color=MUTED, fontsize=13, ha="right")
    if n < TOTAL:
        ax.text(94, 119, "swipe →", color=TEAL, fontsize=15, ha="right")
    return fig, ax


def card(ax, x, y, w, h, color="#1b2b38"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                fc=color, ec="none"))


def add_image(fig, path, rect):
    img_ax = fig.add_axes(rect)
    img_ax.imshow(mpimg.imread(path))
    img_ax.axis("off")


def slide_cover():
    fig, ax = new_slide(1)
    ax.text(6, 100, "Empty shelves\nlose sales.", color=INK, fontsize=54,
            weight="bold", va="top", linespacing=1.1)
    ax.text(6, 74, "Full storerooms\nlock up cash.", color=ORANGE, fontsize=54,
            weight="bold", va="top", linespacing=1.1)
    ax.text(6, 46, "I built an ML system that predicts demand\nand tells a store "
            "what to order, and when.", color=MUTED, fontsize=22, va="top",
            linespacing=1.4)
    stats = [(f"−{so_cut:.0f}%", "stockout days"), (f"−{cost_cut:.0f}%", "inventory cost"),
             (f"−{err_cut:.0f}%", "forecast error")]
    for i, (big, small) in enumerate(stats):
        x = 6 + i * 30
        card(ax, x, 12, 26, 16)
        ax.text(x + 13, 22, big, color=TEAL, fontsize=36, weight="bold", ha="center")
        ax.text(x + 13, 15, small, color=INK, fontsize=15, ha="center")
    return fig


def slide_problem():
    fig, ax = new_slide(2)
    ax.text(6, 108, "The problem", color=TEAL, fontsize=22, weight="bold")
    ax.text(6, 101, "A 3-store supermarket reorders with\na rule of thumb:",
            color=INK, fontsize=32, weight="bold", va="top", linespacing=1.2)
    card(ax, 6, 70, 88, 13, "#22313f")
    ax.text(50, 76.5, "“When stock looks low,\norder 3 weeks of average sales.”",
            color=YELLOW, fontsize=24, ha="center", va="center", style="italic")
    ax.text(6, 62, "It ignores growth, seasons and Diwali. Over 2 years:",
            color=MUTED, fontsize=20)
    rows = [(f"{float(eda['stockout_rate_pct']):.1f}%", "of store-product days were out of stock"),
            (f"Rs. {float(eda['lost_sales_value']) / 1e7:.2f} Cr", "of revenue walked out of the door"),
            ("26%", "stockout days for Electronics")]
    for i, (big, text) in enumerate(rows):
        y = 46 - i * 14
        ax.text(6, y, big, color=RED, fontsize=38, weight="bold", va="center")
        ax.text(46, y, text, color=INK, fontsize=17, va="center")
    return fig


def slide_pipeline():
    fig, ax = new_slide(3)
    ax.text(6, 108, "How it works", color=TEAL, fontsize=22, weight="bold")
    ax.text(6, 101, "One command. Seven steps.", color=INK, fontsize=34,
            weight="bold", va="top")
    steps = [("Simulate", "2 yrs × 3 stores × 20 products: trend, seasons,\nfestivals, promos, supplier delays"),
             ("Clean", "duplicates, outliers, missing values → 0 missing"),
             ("Explore", "8 charts: weekday, seasonality, promo uplift"),
             ("Engineer", "17 leak-free features + festival-free baseline"),
             ("Forecast", "Baseline vs Linear Regression vs Random Forest"),
             ("Optimize", "Safety stock · Reorder point · EOQ"),
             ("Simulate impact", "old rule vs new policy on real demand")]
    for i, (title, desc) in enumerate(steps):
        y = 82 - i * 10.5
        ax.add_patch(plt.Circle((10, y), 3.2, color=TEAL))
        ax.text(10, y, str(i + 1), color=BG, fontsize=18, weight="bold",
                ha="center", va="center")
        ax.text(17, y + 1.6, title, color=INK, fontsize=21, weight="bold", va="center")
        ax.text(17, y - 2.6, desc, color=MUTED, fontsize=14, va="center")
    return fig


def slide_forecast():
    fig, ax = new_slide(4)
    ax.text(6, 108, "Forecasting", color=TEAL, fontsize=22, weight="bold")
    ax.text(6, 101, f"Random Forest beats the\nbaseline by {err_cut:.0f}%",
            color=INK, fontsize=34, weight="bold", va="top", linespacing=1.15)
    add_image(fig, config.IMAGES_DIR / "10_actual_vs_predicted.png",
              [0.04, 0.38, 0.92, 0.32])
    for i, (name, row) in enumerate(metrics.iterrows()):
        y = 40 - i * 7
        color = TEAL if name == "Random Forest" else MUTED
        ax.text(6, y, name, color=color, fontsize=18, weight="bold" if color == TEAL else None)
        ax.text(62, y, f"{row['WAPE_%']:.1f}%", color=color, fontsize=18, ha="right")
        ax.text(80, y, f"{row['Weekly_WAPE_%']:.1f}%", color=color, fontsize=18, ha="right")
        ax.text(94, y, f"{row['Bias_%']:+.0f}%", color=color, fontsize=18, ha="right")
    ax.text(62, 47, "daily WAPE", color=MUTED, fontsize=13, ha="right")
    ax.text(80, 47, "weekly", color=MUTED, fontsize=13, ha="right")
    ax.text(94, 47, "bias", color=MUTED, fontsize=13, ha="right")
    ax.text(6, 14, "Tested honestly: 56 days the model never saw,\nfeeding its own "
            "predictions forward like real life.", color=YELLOW, fontsize=16,
            linespacing=1.4)
    return fig


def slide_decisions():
    fig, ax = new_slide(5)
    ax.text(6, 108, "From forecast to decision", color=TEAL, fontsize=22, weight="bold")
    formulas = [("Safety stock", "= 1.645 × σ × √lead time"),
                ("Reorder point", "= lead-time demand + safety stock"),
                ("Order quantity", "= max(EOQ, target − stock)")]
    for i, (name, f) in enumerate(formulas):
        y = 96 - i * 9
        card(ax, 6, y - 3.5, 88, 7)
        ax.text(9, y, name, color=ORANGE, fontsize=19, weight="bold", va="center")
        ax.text(38, y, f, color=INK, fontsize=18, va="center", family="monospace")
    ax.text(6, 66, "Every morning: a list of what to order, and how much",
            color=INK, fontsize=21, weight="bold")
    add_image(fig, config.IMAGES_DIR / "14_reorder_alerts_table.png",
              [0.03, 0.08, 0.94, 0.44])
    return fig


def slide_impact():
    fig, ax = new_slide(6)
    ax.text(6, 108, "The impact", color=TEAL, fontsize=22, weight="bold")
    ax.text(6, 101, "Same demand. Same starting stock.\nOld rule vs my policy:",
            color=INK, fontsize=30, weight="bold", va="top", linespacing=1.2)
    ax.text(58, 79, "OLD", color=MUTED, fontsize=16, ha="center", weight="bold")
    ax.text(82, 79, "NEW", color=TEAL, fontsize=16, ha="center", weight="bold")
    rows = [("Fill rate", f"{old['Fill rate %']:.1f}%", f"{new['Fill rate %']:.1f}%"),
            ("Stockout days", f"{old['Stockout days %']:.1f}%", f"{new['Stockout days %']:.1f}%"),
            ("Lost revenue", f"₹{old['Lost revenue (Rs)'] / 1e5:.1f}L",
             f"₹{new['Lost revenue (Rs)'] / 1e5:.1f}L"),
            ("Purchase orders", f"{old['Purchase orders']:.0f}", f"{new['Purchase orders']:.0f}"),
            ("Total cost", f"₹{old['Total cost (Rs)'] / 1e5:.2f}L",
             f"₹{new['Total cost (Rs)'] / 1e5:.2f}L")]
    for i, (label, o, n) in enumerate(rows):
        y = 70 - i * 10
        card(ax, 6, y - 4, 88, 8)
        ax.text(9, y, label, color=INK, fontsize=20, va="center")
        ax.text(58, y, o, color=MUTED, fontsize=22, va="center", ha="center")
        ax.text(82, y, n, color=TEAL, fontsize=24, va="center", ha="center", weight="bold")
    inv = (new["Avg inventory value (Rs)"] / old["Avg inventory value (Rs)"] - 1) * 100
    ax.text(6, 14, f"Trade-off: {inv:.0f}% more average stock, held where demand "
            "needs it.\n(Simulated data, real methods.)", color=YELLOW,
            fontsize=16, linespacing=1.4)
    return fig


def slide_cta():
    fig, ax = new_slide(7)
    ax.text(6, 108, "Built with", color=TEAL, fontsize=22, weight="bold")
    stack = ["Python", "Pandas", "NumPy", "scikit-learn", "Matplotlib",
             "Seaborn", "Streamlit", "pytest"]
    for i, s in enumerate(stack):
        x, y = 6 + (i % 2) * 45, 94 - (i // 2) * 9
        card(ax, x, y - 3.2, 42, 6.4)
        ax.text(x + 21, y, s, color=INK, fontsize=20, ha="center", va="center")
    ax.text(6, 50, "What I learned", color=TEAL, fontsize=22, weight="bold")
    ax.text(6, 45, "• Time-based validation & data leakage\n"
            "• Why bias matters as much as accuracy\n"
            "• Safety stock, reorder point & EOQ\n"
            "• Turning model metrics into ₹ impact", color=INK, fontsize=19,
            va="top", linespacing=1.6)
    card(ax, 6, 8, 88, 11, TEAL)
    ax.text(50, 13.5, "Full code + dashboard on GitHub · link in bio",
            color=BG, fontsize=21, weight="bold", ha="center", va="center")
    return fig


if __name__ == "__main__":
    builders = [slide_cover, slide_problem, slide_pipeline, slide_forecast,
                slide_decisions, slide_impact, slide_cta]
    with PdfPages(OUT / "linkedin_carousel.pdf") as pdf:
        for i, build in enumerate(builders, start=1):
            fig = build()
            fig.savefig(OUT / f"slide_{i:02d}.png", facecolor=BG)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f"Saved {len(builders)} slides + linkedin_carousel.pdf to {OUT}")
