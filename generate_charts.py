#!/usr/bin/env python3
"""Generate research charts from collected exchange data"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

plt.style.use('dark_background')
COLORS = ['#00d4aa', '#ff6b6b', '#ffd93d', '#6bcfff', '#c084fc', '#ff9f43']
os.makedirs("figures", exist_ok=True)

# ============================================================
# Load data
# ============================================================
with open("data/binance_klines.json") as f:
    klines = json.load(f)

with open("data/binance_tickers.json") as f:
    tickers = json.load(f)

with open("data/okx_funding.json") as f:
    okx_funding = json.load(f)

with open("data/okx_oi.json") as f:
    okx_oi = json.load(f)

hl_meta = None
if os.path.exists("data/hyperliquid_meta.json"):
    with open("data/hyperliquid_meta.json") as f:
        hl_meta = json.load(f)

hl_funding = {}
for coin in ["btc", "eth", "sol", "doge", "xrp"]:
    fp = f"data/hyperliquid_funding_{coin}.json"
    if os.path.exists(fp):
        with open(fp) as f:
            hl_funding[coin.upper()] = json.load(f)

# ============================================================
# Chart 1: Price Action — BTC, ETH, SOL (20 days hourly)
# ============================================================
print("Chart 1: Multi-asset price action...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Hourly Price Action — Last 20 Days', fontsize=16, fontweight='bold', color='white')

for i, (symbol, color) in enumerate([("BTCUSDT", COLORS[0]), ("ETHUSDT", COLORS[1]), ("SOLUSDT", COLORS[3])]):
    data = klines[symbol]
    times = [datetime.fromtimestamp(d[0]/1000) for d in data]
    closes = [float(d[4]) for d in data]
    volumes = [float(d[5]) for d in data]
    
    ax = axes[i]
    ax.plot(times, closes, color=color, linewidth=1.2, alpha=0.9)
    ax.fill_between(times, closes, min(closes), alpha=0.1, color=color)
    ax.set_ylabel(symbol.replace("USDT", ""), fontsize=12, color=color, fontweight='bold')
    ax.tick_params(colors='#888')
    ax.grid(alpha=0.15)
    
    # Add price annotation
    ax.annotate(f'${closes[-1]:,.2f}', xy=(times[-1], closes[-1]), fontsize=10,
                color=color, fontweight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=color, alpha=0.8))

axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.tight_layout()
plt.savefig('figures/01_price_action.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 2: Volatility Comparison (Rolling 24h realized vol)
# ============================================================
print("Chart 2: Realized volatility...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('24-Hour Rolling Realized Volatility (Annualized)', fontsize=16, fontweight='bold', color='white')

for i, (symbol, name) in enumerate([("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("DOGEUSDT", "DOGE"), ("XRPUSDT", "XRP")]):
    data = klines[symbol]
    times = [datetime.fromtimestamp(d[0]/1000) for d in data]
    closes = np.array([float(d[4]) for d in data])
    returns = np.diff(np.log(closes))
    # Rolling 24h (24 periods) realized vol, annualized
    window = 24
    rolling_vol = []
    for j in range(len(returns)):
        if j < window:
            rolling_vol.append(np.nan)
        else:
            rv = np.std(returns[j-window:j]) * np.sqrt(365 * 24) * 100
            rolling_vol.append(rv)
    ax.plot(times[1:], rolling_vol, color=COLORS[i], linewidth=1.2, alpha=0.8, label=name)

ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.tight_layout()
plt.savefig('figures/02_realized_volatility.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 3: Volume Profile — Top 20 by Volume
# ============================================================
print("Chart 3: Volume profile...")
top20 = sorted(tickers, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)[:20]
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Top 20 USDT Pairs by 24h Volume (Binance Spot)', fontsize=16, fontweight='bold', color='white')

names = [t["symbol"].replace("USDT", "") for t in top20]
vols = [float(t["quoteVolume"]) / 1e6 for t in top20]
changes = [float(t.get("priceChangePercent", 0)) for t in top20]
bar_colors = [COLORS[0] if c >= 0 else COLORS[1] for c in changes]

bars = ax.barh(range(len(names)), vols, color=bar_colors, alpha=0.8, edgecolor='#333')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('24h Volume (Million USD)', fontsize=12)
ax.invert_yaxis()
ax.grid(alpha=0.15, axis='x')

# Add change % labels
for i, (v, c) in enumerate(zip(vols, changes)):
    sign = "+" if c >= 0 else ""
    ax.text(v + max(vols)*0.01, i, f'{sign}{c:.1f}%', va='center', fontsize=9,
            color=COLORS[0] if c >= 0 else COLORS[1])

plt.tight_layout()
plt.savefig('figures/03_volume_profile.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 4: Funding Rates — OKX Cross-Asset
# ============================================================
print("Chart 4: OKX funding rates...")
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Perpetual Funding Rate History (OKX)', fontsize=16, fontweight='bold', color='white')

for i, (inst, color) in enumerate(zip(
    ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"],
    COLORS)):
    
    history = okx_funding[inst]["history"]
    if not history:
        continue
    times = [datetime.fromtimestamp(int(h["fundingTime"])/1000) for h in history]
    rates = [float(h["fundingRate"]) * 100 for h in history]  # to percent
    
    ax = axes[i]
    pos = [r if r >= 0 else 0 for r in rates]
    neg = [r if r < 0 else 0 for r in rates]
    ax.bar(times, pos, width=0.08, color=COLORS[0], alpha=0.7)
    ax.bar(times, neg, width=0.08, color=COLORS[1], alpha=0.7)
    ax.axhline(y=0, color='#555', linewidth=0.5)
    ax.set_ylabel(inst.split("-")[0], fontsize=11, color=color, fontweight='bold')
    ax.tick_params(colors='#888')
    ax.grid(alpha=0.15)
    
    # Annotate average
    avg_rate = np.mean(rates)
    ax.axhline(y=avg_rate, color=color, linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(times[0], avg_rate, f'avg: {avg_rate:.4f}%', fontsize=8, color=color)

axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.tight_layout()
plt.savefig('figures/04_okx_funding_rates.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 5: Hyperliquid Funding Rates vs OKX
# ============================================================
print("Chart 5: Cross-exchange funding rate comparison...")
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Funding Rate Comparison: OKX vs Hyperliquid', fontsize=16, fontweight='bold', color='white')

for i, coin in enumerate(["BTC", "ETH", "SOL"]):
    ax = axes[i]
    
    # OKX
    inst = f"{coin}-USDT-SWAP"
    if inst in okx_funding:
        history = okx_funding[inst]["history"]
        times_okx = [datetime.fromtimestamp(int(h["fundingTime"])/1000) for h in history]
        rates_okx = [float(h["fundingRate"]) * 100 for h in history]
        ax.plot(times_okx, rates_okx, color=COLORS[0], linewidth=1, alpha=0.8, label='OKX', marker='.', markersize=3)
    
    # Hyperliquid
    if coin in hl_funding:
        times_hl = [datetime.fromtimestamp(int(h["time"])/1000) for h in hl_funding[coin]]
        rates_hl = [float(h["fundingRate"]) * 100 for h in hl_funding[coin]]
        ax.plot(times_hl, rates_hl, color=COLORS[2], linewidth=1, alpha=0.8, label='Hyperliquid', marker='.', markersize=3)
    
    ax.axhline(y=0, color='#555', linewidth=0.5)
    ax.set_ylabel(coin, fontsize=12, fontweight='bold', color=COLORS[i])
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.15)

axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
plt.tight_layout()
plt.savefig('figures/05_funding_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 6: Return Distribution (hourly)
# ============================================================
print("Chart 6: Return distributions...")
fig, axes = plt.subplots(1, 5, figsize=(16, 5))
fig.suptitle('Hourly Return Distribution — Last 20 Days', fontsize=16, fontweight='bold', color='white')

for i, (symbol, name, color) in enumerate([
    ("BTCUSDT", "BTC", COLORS[0]), ("ETHUSDT", "ETH", COLORS[1]),
    ("SOLUSDT", "SOL", COLORS[3]), ("DOGEUSDT", "DOGE", COLORS[2]),
    ("XRPUSDT", "XRP", COLORS[4])]):
    
    closes = np.array([float(d[4]) for d in klines[symbol]])
    returns = np.diff(np.log(closes)) * 100
    
    ax = axes[i]
    ax.hist(returns, bins=50, color=color, alpha=0.7, edgecolor='#333', density=True)
    ax.axvline(x=0, color='white', linewidth=0.5)
    ax.set_title(name, fontsize=12, color=color, fontweight='bold')
    ax.set_xlabel('Return (%)', fontsize=9)
    
    # Stats
    mu, sig, skew = np.mean(returns), np.std(returns), 0
    if sig > 0:
        skew = np.mean(((returns - mu) / sig) ** 3)
    kurt = np.mean(((returns - mu) / sig) ** 4) - 3 if sig > 0 else 0
    ax.text(0.05, 0.95, f'μ={mu:.3f}%\nσ={sig:.3f}%\nskew={skew:.2f}\nkurt={kurt:.2f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
    ax.grid(alpha=0.15)

plt.tight_layout()
plt.savefig('figures/06_return_distributions.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 7: Correlation Matrix (hourly returns)
# ============================================================
print("Chart 7: Correlation matrix...")
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
names = ["BTC", "ETH", "SOL", "DOGE", "XRP"]
returns_matrix = []
for s in symbols:
    closes = np.array([float(d[4]) for d in klines[s]])
    returns_matrix.append(np.diff(np.log(closes)))

returns_matrix = np.array(returns_matrix)
corr = np.corrcoef(returns_matrix)

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle('Hourly Return Correlation Matrix', fontsize=16, fontweight='bold', color='white')
im = ax.imshow(corr, cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, fontsize=12, fontweight='bold')
ax.set_yticklabels(names, fontsize=12, fontweight='bold')
for i in range(len(names)):
    for j in range(len(names)):
        color = 'black' if corr[i,j] > 0.5 else 'white'
        ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center', fontsize=12, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig('figures/07_correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 8: Hyperliquid Market Overview
# ============================================================
if hl_meta:
    print("Chart 8: Hyperliquid market overview...")
    universe = hl_meta[0]["universe"]
    asset_ctxs = hl_meta[1]
    
    # Top 20 by open interest
    assets_with_oi = []
    for u, ctx in zip(universe, asset_ctxs):
        try:
            oi_usd = float(ctx.get("openInterest", 0)) * float(ctx.get("markPx", 0))
            funding = float(ctx.get("funding", 0)) * 100
            vol24h = float(ctx.get("dayNtlVlm", 0))
            assets_with_oi.append({
                "name": u["name"],
                "oi_usd": oi_usd,
                "funding": funding,
                "vol24h": vol24h,
                "markPx": float(ctx.get("markPx", 0))
            })
        except:
            pass
    
    assets_with_oi.sort(key=lambda x: x["oi_usd"], reverse=True)
    top = assets_with_oi[:25]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle('Hyperliquid Perpetuals — Market Overview', fontsize=16, fontweight='bold', color='white')
    
    # OI bar chart
    names_hl = [a["name"] for a in top]
    oi_vals = [a["oi_usd"] / 1e6 for a in top]
    funding_vals = [a["funding"] for a in top]
    bar_colors_oi = [COLORS[0] if f >= 0 else COLORS[1] for f in funding_vals]
    
    ax1.barh(range(len(names_hl)), oi_vals, color=bar_colors_oi, alpha=0.8, edgecolor='#333')
    ax1.set_yticks(range(len(names_hl)))
    ax1.set_yticklabels(names_hl, fontsize=9)
    ax1.set_xlabel('Open Interest (Million USD)', fontsize=11)
    ax1.set_title('Open Interest (Top 25)', fontsize=13, color=COLORS[0])
    ax1.invert_yaxis()
    ax1.grid(alpha=0.15, axis='x')
    
    # Funding rate scatter
    vol_vals = [a["vol24h"] / 1e6 for a in top]
    scatter = ax2.scatter(funding_vals, vol_vals, 
                          s=[max(o*2, 20) for o in oi_vals],
                          c=funding_vals, cmap='RdYlGn', alpha=0.7, edgecolors='#555')
    ax2.axvline(x=0, color='#555', linewidth=0.5)
    for a, fv, vv in zip(top[:15], funding_vals[:15], vol_vals[:15]):
        ax2.annotate(a["name"], (fv, vv), fontsize=8, alpha=0.8, color='white',
                    textcoords="offset points", xytext=(5, 5))
    ax2.set_xlabel('Funding Rate (%)', fontsize=11)
    ax2.set_ylabel('24h Volume (Million USD)', fontsize=11)
    ax2.set_title('Funding vs Volume', fontsize=13, color=COLORS[2])
    ax2.grid(alpha=0.15)
    plt.colorbar(scatter, ax=ax2, label='Funding Rate (%)')
    
    plt.tight_layout()
    plt.savefig('figures/08_hyperliquid_overview.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()

# ============================================================
# Chart 9: Intraday Patterns (hourly volume distribution)
# ============================================================
print("Chart 9: Intraday volume patterns...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Intraday Volume Patterns by Hour (UTC)', fontsize=16, fontweight='bold', color='white')

for i, (symbol, name, color) in enumerate([("BTCUSDT", "BTC", COLORS[0]), ("ETHUSDT", "ETH", COLORS[1]), ("SOLUSDT", "SOL", COLORS[3])]):
    data = klines[symbol]
    hourly_vol = {}
    for d in data:
        hour = datetime.fromtimestamp(d[0]/1000).hour
        if hour not in hourly_vol:
            hourly_vol[hour] = []
        hourly_vol[hour].append(float(d[5]))
    
    hours = sorted(hourly_vol.keys())
    avg_vols = [np.mean(hourly_vol[h]) for h in hours]
    
    ax = axes[i]
    bars = ax.bar(hours, avg_vols, color=color, alpha=0.7, edgecolor='#333')
    ax.set_title(name, fontsize=13, color=color, fontweight='bold')
    ax.set_xlabel('Hour (UTC)', fontsize=10)
    ax.set_ylabel('Avg Volume', fontsize=10)
    ax.set_xticks(range(0, 24, 3))
    ax.grid(alpha=0.15, axis='y')
    
    # Highlight peak
    peak_h = hours[np.argmax(avg_vols)]
    ax.annotate(f'Peak: {peak_h}:00', xy=(peak_h, max(avg_vols)),
                fontsize=9, color='white', ha='center',
                xytext=(0, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='white'))

plt.tight_layout()
plt.savefig('figures/09_intraday_patterns.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 10: Cumulative Returns
# ============================================================
print("Chart 10: Cumulative returns...")
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Cumulative Returns — Last 20 Days', fontsize=16, fontweight='bold', color='white')

for i, (symbol, name, color) in enumerate([
    ("BTCUSDT", "BTC", COLORS[0]), ("ETHUSDT", "ETH", COLORS[1]),
    ("SOLUSDT", "SOL", COLORS[3]), ("DOGEUSDT", "DOGE", COLORS[2]),
    ("XRPUSDT", "XRP", COLORS[4])]):
    
    data = klines[symbol]
    times = [datetime.fromtimestamp(d[0]/1000) for d in data]
    closes = np.array([float(d[4]) for d in data])
    cum_ret = (closes / closes[0] - 1) * 100
    
    ax.plot(times, cum_ret, color=color, linewidth=1.5, alpha=0.9, label=f'{name} ({cum_ret[-1]:+.1f}%)')

ax.axhline(y=0, color='#555', linewidth=0.5)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.tight_layout()
plt.savefig('figures/10_cumulative_returns.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 11: BTC Volume-Price Relationship
# ============================================================
print("Chart 11: BTC volume-price...")
fig, ax1 = plt.subplots(figsize=(14, 6))
fig.suptitle('BTC: Price vs Volume Analysis', fontsize=16, fontweight='bold', color='white')

data = klines["BTCUSDT"]
times = [datetime.fromtimestamp(d[0]/1000) for d in data]
closes = [float(d[4]) for d in data]
volumes = [float(d[7]) / 1e6 for d in data]  # quote volume in millions

ax1.plot(times, closes, color=COLORS[0], linewidth=1.2, label='Price')
ax1.set_ylabel('Price (USD)', fontsize=12, color=COLORS[0])
ax1.tick_params(axis='y', labelcolor=COLORS[0])

ax2 = ax1.twinx()
ax2.bar(times, volumes, width=0.03, color=COLORS[2], alpha=0.3, label='Volume')
ax2.set_ylabel('Volume (M USD)', fontsize=12, color=COLORS[2])
ax2.tick_params(axis='y', labelcolor=COLORS[2])

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.grid(alpha=0.15)
ax1.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('figures/11_btc_volume_price.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

# ============================================================
# Chart 12: Funding Rate Annualized Yield
# ============================================================
print("Chart 12: Funding yield analysis...")
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Annualized Funding Yield by Asset (OKX — Last 100 Periods)', fontsize=16, fontweight='bold', color='white')

assets_funding = []
for inst in ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"]:
    history = okx_funding[inst]["history"]
    if history:
        rates = [float(h["fundingRate"]) for h in history]
        avg_8h = np.mean(rates)
        annual = avg_8h * 3 * 365 * 100  # 3 funding per day * 365 days * 100 for %
        std_rate = np.std(rates) * 3 * 365 * 100
        name = inst.split("-")[0]
        assets_funding.append({"name": name, "annual": annual, "std": std_rate})

names_f = [a["name"] for a in assets_funding]
annuals = [a["annual"] for a in assets_funding]
stds = [a["std"] for a in assets_funding]
bar_colors_f = [COLORS[0] if a >= 0 else COLORS[1] for a in annuals]

bars = ax.bar(names_f, annuals, color=bar_colors_f, alpha=0.8, edgecolor='#333')
ax.errorbar(names_f, annuals, yerr=stds, fmt='none', ecolor='#888', capsize=5)
ax.axhline(y=0, color='#555', linewidth=0.5)
ax.set_ylabel('Annualized Yield (%)', fontsize=12)
ax.grid(alpha=0.15, axis='y')

for bar, val in zip(bars, annuals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold',
            color=COLORS[0] if val >= 0 else COLORS[1])

plt.tight_layout()
plt.savefig('figures/12_funding_yield.png', dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
plt.close()

print("\nAll 12 charts generated!")
print(f"Files: {os.listdir('figures')}")
