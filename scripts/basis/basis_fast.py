"""
Fast Basis Monitor: Sample first snapshot per day for midprice
"""
import pandas as pd
import numpy as np
import gzip, os, glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t0 = datetime.now()
OB = "/mnt/kaiko/consolidated/order_book/kaiko-ob10-v2"
RESULTS = "/mnt/work/qr33/comewealth/results"
FIGURES = "/mnt/work/qr33/comewealth/figures"

PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']

def get_midprice_fast(filepath):
    """Read first 200 lines to get first snapshot's best bid/ask"""
    try:
        with gzip.open(filepath, 'rt') as f:
            f.readline()  # header
            best_bid = -1
            best_ask = 1e15
            first_ts = None
            for i, line in enumerate(f):
                if i > 500: break
                parts = line.strip().split(',')
                if len(parts) != 4: continue
                ts, side, price = int(parts[0]), parts[1], float(parts[2])
                if first_ts is None: first_ts = ts
                if ts != first_ts: break  # Only first snapshot
                if side == 'b' and price > best_bid: best_bid = price
                if side == 'a' and price < best_ask: best_ask = price
            if best_bid > 0 and best_ask < 1e15:
                return first_ts, (best_bid + best_ask) / 2
    except:
        pass
    return None, None

all_data = []

for pair in PAIRS:
    print(f"[{datetime.now():%H:%M:%S}] {pair}")
    
    for label, exchange in [('spot', 'Binance'), ('perp', 'Binance Futures')]:
        path = f"{OB}/{exchange}/{pair}"
        if not os.path.exists(path):
            print(f"  {label}: not found")
            continue
        months = sorted(os.listdir(path))
        for month in months:
            files = sorted(glob.glob(f"{path}/{month}/*.csv.gz"))
            for f in files:
                ts, mid = get_midprice_fast(f)
                if mid:
                    date_str = os.path.basename(f).split('_')[-1].replace('.csv.gz','')
                    all_data.append({
                        'pair': pair, 'type': label, 'date': date_str,
                        'timestamp': ts, 'midprice': mid
                    })
        print(f"  {label}: {len([d for d in all_data if d['pair']==pair and d['type']==label])} days")

if all_data:
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Pivot to get spot and perp side by side
    results = []
    for pair in PAIRS:
        spot = df[(df['pair']==pair) & (df['type']=='spot')].set_index('date')['midprice']
        perp = df[(df['pair']==pair) & (df['type']=='perp')].set_index('date')['midprice']
        merged = pd.DataFrame({'spot': spot, 'perp': perp}).dropna()
        merged['basis_bps'] = (merged['perp'] - merged['spot']) / merged['spot'] * 10000
        merged['pair'] = pair
        merged = merged.reset_index()
        results.append(merged)
    
    df_all = pd.concat(results)
    df_all.to_csv(f"{RESULTS}/basis_20260307.csv", index=False)
    
    # Summary
    print("\n=== BASIS SUMMARY ===")
    for pair in df_all['pair'].unique():
        sub = df_all[df_all['pair']==pair]
        mu = sub['basis_bps'].mean()
        std = sub['basis_bps'].std()
        n = len(sub)
        ann = mu / 10000 * 365 * 100
        print(f"{pair:12s} | N={n:4d} | Mean={mu:+7.2f} bps | Std={std:6.2f} bps | Ann={ann:+6.1f}%")
        anomalies = sub[abs(sub['basis_bps'] - mu) > 2*std]
        if len(anomalies) > 0:
            print(f"  ⚠️ {len(anomalies)} anomalies (>2σ)")
    
    # Figure
    pairs_list = sorted(df_all['pair'].unique())
    fig, axes = plt.subplots(len(pairs_list), 1, figsize=(14, 3.5*len(pairs_list)))
    if len(pairs_list) == 1: axes = [axes]
    fig.suptitle('Spot-Perpetual Basis (Binance)\nAgentic Sciences', fontsize=14, fontweight='bold', y=1.01)
    
    for i, pair in enumerate(pairs_list):
        sub = df_all[df_all['pair']==pair].sort_values('date')
        ax = axes[i]
        ax.bar(sub['date'], sub['basis_bps'], width=1, alpha=0.7, color=np.where(sub['basis_bps']>0, '#3b82f6', '#ef4444'))
        mu = sub['basis_bps'].mean()
        ax.axhline(y=mu, color='green', linewidth=1.5, label=f'Mean: {mu:+.1f} bps')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel('Basis (bps)')
        ax.set_title(f'{pair}  |  N={len(sub)}  |  Mean={mu:+.1f} bps  |  Ann={mu/10000*365*100:+.1f}%')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/basis_20260307.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure: basis_20260307.png")
    print(f"Results: basis_20260307.csv ({len(df_all)} rows)")

print(f"\n⏱️ {(datetime.now()-t0).total_seconds():.0f}s")
