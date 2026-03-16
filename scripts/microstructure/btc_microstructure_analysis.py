#!/usr/bin/env python3
"""
BTC Microstructure Analysis: Spot vs Perpetual Futures
Real Kaiko order book data analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob, time, json
from datetime import datetime

t0 = time.time()
OUT = "/mnt/work/qr33/comewealth"

# ============================================================
# 1. Load Order Book Data (last 3 months available)
# ============================================================
print("=" * 60)
print("BTC Microstructure Analysis — Spot vs Perpetual Futures")
print("Data: Kaiko consolidated order book L10 (Binance)")
print("=" * 60)

spot_dir = "/mnt/kaiko/consolidated/order_book/kaiko-ob10-v2/Binance/BTCUSDT"
perp_dir = "/mnt/kaiko/consolidated/order_book/kaiko-ob10-v2/Binance Futures/BTCUSDT"

months = ["2023_02", "2023_03", "2023_04"]  # latest 3 months

def load_ob_month(base_dir, month):
    """Load one month of order book snapshots, sample to keep manageable"""
    pattern = os.path.join(base_dir, month, "*.csv.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    
    dfs = []
    # Sample: take every 10th file to keep memory manageable
    sampled = files[::max(1, len(files)//30)]
    for f in sampled:
        try:
            df = pd.read_csv(f, compression='gzip')
            dfs.append(df)
        except Exception as e:
            continue
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# Load spot and perp data
print("\nLoading order book data...")
spot_dfs = []
perp_dfs = []
for m in months:
    print(f"  Loading {m}...")
    s = load_ob_month(spot_dir, m)
    p = load_ob_month(perp_dir, m)
    if not s.empty:
        s['month'] = m
        spot_dfs.append(s)
    if not p.empty:
        p['month'] = m
        perp_dfs.append(p)

spot = pd.concat(spot_dfs, ignore_index=True) if spot_dfs else pd.DataFrame()
perp = pd.concat(perp_dfs, ignore_index=True) if perp_dfs else pd.DataFrame()

print(f"\nSpot: {len(spot):,} rows, columns: {list(spot.columns)}")
print(f"Perp: {len(perp):,} rows, columns: {list(perp.columns)}")

if spot.empty or perp.empty:
    print("ERROR: No data loaded!")
    exit(1)

# Show sample
print(f"\nSpot sample:\n{spot.head(2)}")
print(f"\nPerp sample:\n{perp.head(2)}")

# ============================================================
# 2. Compute Microstructure Metrics
# ============================================================
print("\n" + "=" * 60)
print("Computing Microstructure Metrics")
print("=" * 60)

def compute_metrics(df, label):
    """Compute order book microstructure metrics"""
    # Identify bid/ask columns
    cols = df.columns.tolist()
    print(f"\n[{label}] Columns: {cols}")
    
    # Kaiko OB10 format: typically has ask_price_0..9, ask_amount_0..9, bid_price_0..9, bid_amount_0..9
    ask_price_cols = [c for c in cols if 'ask' in c.lower() and 'price' in c.lower()]
    bid_price_cols = [c for c in cols if 'bid' in c.lower() and 'price' in c.lower()]
    ask_amt_cols = [c for c in cols if 'ask' in c.lower() and ('amount' in c.lower() or 'size' in c.lower() or 'quantity' in c.lower())]
    bid_amt_cols = [c for c in cols if 'bid' in c.lower() and ('amount' in c.lower() or 'size' in c.lower() or 'quantity' in c.lower())]
    
    print(f"  Ask price cols: {ask_price_cols[:3]}...")
    print(f"  Bid price cols: {bid_price_cols[:3]}...")
    print(f"  Ask amount cols: {ask_amt_cols[:3]}...")
    print(f"  Bid amount cols: {bid_amt_cols[:3]}...")
    
    if not ask_price_cols or not bid_price_cols:
        print(f"  WARNING: Cannot identify bid/ask columns!")
        return {}
    
    # Sort columns by level
    ask_price_cols = sorted(ask_price_cols)
    bid_price_cols = sorted(bid_price_cols)
    ask_amt_cols = sorted(ask_amt_cols)
    bid_amt_cols = sorted(bid_amt_cols)
    
    # Best bid/ask (level 0)
    best_ask = df[ask_price_cols[0]].astype(float)
    best_bid = df[bid_price_cols[0]].astype(float)
    mid = (best_ask + best_bid) / 2
    
    # 1. Quoted Spread (bps)
    spread_bps = (best_ask - best_bid) / mid * 10000
    
    # 2. Depth at best (BTC)
    best_ask_size = df[ask_amt_cols[0]].astype(float) if ask_amt_cols else pd.Series(0, index=df.index)
    best_bid_size = df[bid_amt_cols[0]].astype(float) if bid_amt_cols else pd.Series(0, index=df.index)
    
    # 3. Total depth (sum of all 10 levels)
    total_ask_depth = sum(df[c].astype(float) for c in ask_amt_cols)
    total_bid_depth = sum(df[c].astype(float) for c in bid_amt_cols)
    
    # 4. Depth Imbalance (bid - ask) / (bid + ask)
    total_depth = total_bid_depth + total_ask_depth
    imbalance = (total_bid_depth - total_ask_depth) / total_depth.replace(0, np.nan)
    
    # 5. Depth at ±0.1%, ±0.5%, ±1% from mid
    def depth_within_pct(price_cols, amt_cols, mid_prices, pct, side='bid'):
        """Sum amounts within pct of mid price"""
        total = pd.Series(0.0, index=df.index)
        for pc, ac in zip(price_cols, amt_cols):
            p = df[pc].astype(float)
            a = df[ac].astype(float)
            if side == 'bid':
                mask = (mid_prices - p) / mid_prices <= pct
            else:
                mask = (p - mid_prices) / mid_prices <= pct
            total += a * mask
        return total
    
    depth_01_bid = depth_within_pct(bid_price_cols, bid_amt_cols, mid, 0.001, 'bid')
    depth_01_ask = depth_within_pct(ask_price_cols, ask_amt_cols, mid, 0.001, 'ask')
    depth_05_bid = depth_within_pct(bid_price_cols, bid_amt_cols, mid, 0.005, 'bid')
    depth_05_ask = depth_within_pct(ask_price_cols, ask_amt_cols, mid, 0.005, 'ask')
    depth_10_bid = depth_within_pct(bid_price_cols, bid_amt_cols, mid, 0.01, 'bid')
    depth_10_ask = depth_within_pct(ask_price_cols, ask_amt_cols, mid, 0.01, 'ask')
    
    # 6. Price range across 10 levels (bps)
    worst_ask = df[ask_price_cols[-1]].astype(float) if len(ask_price_cols) > 1 else best_ask
    worst_bid = df[bid_price_cols[-1]].astype(float) if len(bid_price_cols) > 1 else best_bid
    ask_range_bps = (worst_ask - best_ask) / mid * 10000
    bid_range_bps = (best_bid - worst_bid) / mid * 10000
    
    metrics = {
        'label': label,
        'n_snapshots': len(df),
        'mid_price_mean': mid.mean(),
        'mid_price_std': mid.std(),
        'spread_bps_mean': spread_bps.mean(),
        'spread_bps_median': spread_bps.median(),
        'spread_bps_p5': spread_bps.quantile(0.05),
        'spread_bps_p95': spread_bps.quantile(0.95),
        'best_bid_size_mean': best_bid_size.mean(),
        'best_ask_size_mean': best_ask_size.mean(),
        'total_bid_depth_mean': total_bid_depth.mean(),
        'total_ask_depth_mean': total_ask_depth.mean(),
        'total_depth_mean_btc': total_depth.mean(),
        'total_depth_mean_usd': (total_depth * mid).mean(),
        'imbalance_mean': imbalance.mean(),
        'imbalance_std': imbalance.std(),
        'depth_01pct_btc': (depth_01_bid + depth_01_ask).mean(),
        'depth_05pct_btc': (depth_05_bid + depth_05_ask).mean(),
        'depth_10pct_btc': (depth_10_bid + depth_10_ask).mean(),
        'ask_range_bps_mean': ask_range_bps.mean(),
        'bid_range_bps_mean': bid_range_bps.mean(),
    }
    
    # Time series for plots
    ts = pd.DataFrame({
        'spread_bps': spread_bps.values,
        'imbalance': imbalance.values,
        'total_depth': total_depth.values,
        'mid': mid.values,
        'best_bid_size': best_bid_size.values,
        'best_ask_size': best_ask_size.values,
    })
    
    return metrics, ts

spot_metrics, spot_ts = compute_metrics(spot, "SPOT")
perp_metrics, perp_ts = compute_metrics(perp, "PERP")

# ============================================================
# 3. Print Results
# ============================================================
print("\n" + "=" * 60)
print("RESULTS: BTC/USDT Spot vs Perpetual (Binance, Feb-Apr 2023)")
print("=" * 60)

def fmt(v, precision=4):
    if isinstance(v, float):
        if abs(v) > 1000:
            return f"{v:,.2f}"
        return f"{v:.{precision}f}"
    return str(v)

print(f"\n{'Metric':<35} {'Spot':>15} {'Perpetual':>15} {'Ratio':>10}")
print("-" * 80)

compare_keys = [
    ('n_snapshots', 'Snapshots', 0),
    ('mid_price_mean', 'Mean Mid Price ($)', 2),
    ('spread_bps_mean', 'Spread Mean (bps)', 4),
    ('spread_bps_median', 'Spread Median (bps)', 4),
    ('spread_bps_p5', 'Spread P5 (bps)', 4),
    ('spread_bps_p95', 'Spread P95 (bps)', 4),
    ('best_bid_size_mean', 'Best Bid Size (BTC)', 4),
    ('best_ask_size_mean', 'Best Ask Size (BTC)', 4),
    ('total_depth_mean_btc', 'Total Depth L10 (BTC)', 2),
    ('total_depth_mean_usd', 'Total Depth L10 ($)', 0),
    ('imbalance_mean', 'Depth Imbalance Mean', 4),
    ('imbalance_std', 'Depth Imbalance Std', 4),
    ('depth_01pct_btc', 'Depth ±0.1% (BTC)', 2),
    ('depth_05pct_btc', 'Depth ±0.5% (BTC)', 2),
    ('depth_10pct_btc', 'Depth ±1.0% (BTC)', 2),
    ('ask_range_bps_mean', 'Ask L10 Range (bps)', 2),
    ('bid_range_bps_mean', 'Bid L10 Range (bps)', 2),
]

for key, name, prec in compare_keys:
    sv = spot_metrics[key]
    pv = perp_metrics[key]
    ratio = pv / sv if sv != 0 else float('nan')
    print(f"{name:<35} {fmt(sv, prec):>15} {fmt(pv, prec):>15} {ratio:>9.2f}x")

# ============================================================
# 4. Generate Figures
# ============================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('BTC/USDT Microstructure: Spot vs Perpetual Futures (Binance, Feb-Apr 2023)', 
             fontsize=14, fontweight='bold')

# (a) Spread distribution
ax = axes[0, 0]
ax.hist(spot_ts['spread_bps'].clip(0, 20), bins=100, alpha=0.6, label='Spot', density=True, color='#2196F3')
ax.hist(perp_ts['spread_bps'].clip(0, 20), bins=100, alpha=0.6, label='Perpetual', density=True, color='#FF5722')
ax.set_xlabel('Quoted Spread (bps)')
ax.set_ylabel('Density')
ax.set_title('(a) Spread Distribution')
ax.legend()

# (b) Depth comparison
ax = axes[0, 1]
categories = ['Best Level', 'Top 5 Levels', 'All 10 Levels']
spot_depths = [spot_metrics['best_bid_size_mean'] + spot_metrics['best_ask_size_mean'],
               spot_metrics['depth_05pct_btc'],
               spot_metrics['total_depth_mean_btc']]
perp_depths = [perp_metrics['best_bid_size_mean'] + perp_metrics['best_ask_size_mean'],
               perp_metrics['depth_05pct_btc'],
               perp_metrics['total_depth_mean_btc']]
x = np.arange(len(categories))
ax.bar(x - 0.15, spot_depths, 0.3, label='Spot', color='#2196F3')
ax.bar(x + 0.15, perp_depths, 0.3, label='Perpetual', color='#FF5722')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Depth (BTC)')
ax.set_title('(b) Order Book Depth')
ax.legend()

# (c) Imbalance distribution
ax = axes[0, 2]
ax.hist(spot_ts['imbalance'].dropna().clip(-1, 1), bins=100, alpha=0.6, label='Spot', density=True, color='#2196F3')
ax.hist(perp_ts['imbalance'].dropna().clip(-1, 1), bins=100, alpha=0.6, label='Perpetual', density=True, color='#FF5722')
ax.set_xlabel('Depth Imbalance (bid-ask)/(bid+ask)')
ax.set_ylabel('Density')
ax.set_title('(c) Order Book Imbalance')
ax.axvline(0, color='k', linestyle='--', alpha=0.3)
ax.legend()

# (d) Spread time series (sampled)
ax = axes[1, 0]
n = min(5000, len(spot_ts), len(perp_ts))
ax.plot(range(n), spot_ts['spread_bps'].iloc[:n].rolling(50).mean(), alpha=0.7, label='Spot', color='#2196F3', linewidth=0.8)
ax.plot(range(n), perp_ts['spread_bps'].iloc[:n].rolling(50).mean(), alpha=0.7, label='Perpetual', color='#FF5722', linewidth=0.8)
ax.set_xlabel('Snapshot Index')
ax.set_ylabel('Spread (bps, 50-MA)')
ax.set_title('(d) Spread Time Series')
ax.legend()

# (e) Depth time series
ax = axes[1, 1]
ax.plot(range(n), spot_ts['total_depth'].iloc[:n].rolling(50).mean(), alpha=0.7, label='Spot', color='#2196F3', linewidth=0.8)
ax.plot(range(n), perp_ts['total_depth'].iloc[:n].rolling(50).mean(), alpha=0.7, label='Perpetual', color='#FF5722', linewidth=0.8)
ax.set_xlabel('Snapshot Index')
ax.set_ylabel('Total Depth (BTC)')
ax.set_title('(e) Depth Time Series')
ax.legend()

# (f) Spread vs Depth scatter
ax = axes[1, 2]
sample_n = min(2000, len(spot_ts), len(perp_ts))
idx_s = np.random.choice(len(spot_ts), sample_n, replace=False)
idx_p = np.random.choice(len(perp_ts), sample_n, replace=False)
ax.scatter(spot_ts['total_depth'].iloc[idx_s], spot_ts['spread_bps'].iloc[idx_s], 
           alpha=0.15, s=5, label='Spot', color='#2196F3')
ax.scatter(perp_ts['total_depth'].iloc[idx_p], perp_ts['spread_bps'].iloc[idx_p], 
           alpha=0.15, s=5, label='Perpetual', color='#FF5722')
ax.set_xlabel('Total Depth (BTC)')
ax.set_ylabel('Spread (bps)')
ax.set_title('(f) Spread vs Depth')
ax.legend()

plt.tight_layout()
fig_path = f"{OUT}/figures/btc_microstructure_20260308.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# ============================================================
# 5. Save Results
# ============================================================
results = {
    'spot': spot_metrics,
    'perp': perp_metrics,
    'data_period': 'Feb-Apr 2023',
    'exchange': 'Binance',
    'pair': 'BTC/USDT',
    'analysis_time': datetime.now().isoformat(),
    'elapsed_seconds': round(time.time() - t0, 1),
}

with open(f"{OUT}/results/btc_microstructure_20260308.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUT}/results/btc_microstructure_20260308.json")
print(f"Total time: {time.time()-t0:.1f}s")
