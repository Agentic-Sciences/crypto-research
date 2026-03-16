"""
Basis Monitor: Spot-Perpetual basis analysis
Purpose: Calculate BTC/ETH basis across exchanges, annualized rates, detect anomalies
Data: Kaiko consolidated order book (research3)
Output: results/basis_20260307.csv + figures/basis_20260307.png
Server: research3 (local)
"""

import pandas as pd
import numpy as np
import gzip
import os
import glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

t0 = datetime.now()
print(f"[{t0:%H:%M:%S}] Starting basis monitor...")

OB_BASE = "/mnt/kaiko/consolidated/order_book/kaiko-ob10-v2"
RESULTS = "/mnt/work/qr33/comewealth/results"
FIGURES = "/mnt/work/qr33/comewealth/figures"

# Get latest available month for each pair
PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT']
SPOT_EX = 'Binance'
PERP_EX = 'Binance Futures'

def get_latest_months(exchange, pair, n=3):
    """Get the last n available months of data"""
    path = f"{OB_BASE}/{exchange}/{pair}"
    if not os.path.exists(path):
        return []
    months = sorted(os.listdir(path))
    return months[-n:] if len(months) >= n else months

def parse_ob_midprice(filepath):
    """Extract hourly midprices from order book snapshots"""
    midprices = []
    try:
        with gzip.open(filepath, 'rt') as f:
            header = f.readline()  # date,type,price,amount
            current_ts = None
            best_bid = None
            best_ask = None
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 4:
                    continue
                ts, side, price, amount = int(parts[0]), parts[1], float(parts[2]), float(parts[3])
                
                hour_ts = ts // 3600000 * 3600000  # Round to hour
                
                if current_ts is None:
                    current_ts = hour_ts
                
                if hour_ts != current_ts:
                    if best_bid and best_ask:
                        mid = (best_bid + best_ask) / 2
                        midprices.append({'timestamp': current_ts, 'midprice': mid})
                    current_ts = hour_ts
                    best_bid = None
                    best_ask = None
                
                if side == 'b':
                    if best_bid is None or price > best_bid:
                        best_bid = price
                elif side == 'a':
                    if best_ask is None or price < best_ask:
                        best_ask = price
            
            # Last hour
            if best_bid and best_ask:
                mid = (best_bid + best_ask) / 2
                midprices.append({'timestamp': current_ts, 'midprice': mid})
                
    except Exception as e:
        print(f"  Error parsing {os.path.basename(filepath)}: {e}")
    
    return midprices

all_basis = []

for pair in PAIRS:
    print(f"\n[{datetime.now():%H:%M:%S}] Processing {pair}...")
    
    spot_months = get_latest_months(SPOT_EX, pair, n=3)
    perp_months = get_latest_months(PERP_EX, pair, n=3)
    
    if not spot_months or not perp_months:
        print(f"  Skipping {pair}: spot_months={len(spot_months)}, perp_months={len(perp_months)}")
        continue
    
    # Use common months
    common = sorted(set(spot_months) & set(perp_months))
    if not common:
        print(f"  No common months for {pair}")
        continue
    
    print(f"  Common months: {common}")
    
    spot_mids = []
    perp_mids = []
    
    for month in common[-2:]:  # Last 2 common months to keep it manageable
        # Spot
        spot_dir = f"{OB_BASE}/{SPOT_EX}/{pair}/{month}"
        spot_files = sorted(glob.glob(f"{spot_dir}/*.csv.gz"))
        for sf in spot_files[:10]:  # Sample 10 days per month
            mids = parse_ob_midprice(sf)
            for m in mids:
                m['source'] = 'spot'
                m['pair'] = pair
            spot_mids.extend(mids)
        
        # Perp
        perp_dir = f"{OB_BASE}/{PERP_EX}/{pair}/{month}"
        perp_files = sorted(glob.glob(f"{perp_dir}/*.csv.gz"))
        for pf in perp_files[:10]:
            mids = parse_ob_midprice(pf)
            for m in mids:
                m['source'] = 'perp'
                m['pair'] = pair
            perp_mids.extend(mids)
    
    if not spot_mids or not perp_mids:
        print(f"  No data parsed for {pair}")
        continue
    
    df_spot = pd.DataFrame(spot_mids).groupby('timestamp')['midprice'].first().reset_index()
    df_perp = pd.DataFrame(perp_mids).groupby('timestamp')['midprice'].first().reset_index()
    
    df_spot.columns = ['timestamp', 'spot_mid']
    df_perp.columns = ['timestamp', 'perp_mid']
    
    merged = pd.merge(df_spot, df_perp, on='timestamp', how='inner')
    
    if len(merged) == 0:
        print(f"  No overlapping timestamps for {pair}")
        continue
    
    merged['basis_bps'] = (merged['perp_mid'] - merged['spot_mid']) / merged['spot_mid'] * 10000
    merged['basis_ann_pct'] = merged['basis_bps'] / 10000 * 365 * 24 * 100  # Annualized
    merged['pair'] = pair
    merged['datetime'] = pd.to_datetime(merged['timestamp'], unit='ms')
    
    all_basis.append(merged)
    
    print(f"  {pair}: {len(merged)} hourly obs, "
          f"basis mean={merged['basis_bps'].mean():.2f} bps, "
          f"std={merged['basis_bps'].std():.2f} bps")

if all_basis:
    df_all = pd.concat(all_basis, ignore_index=True)
    
    # Save results
    df_all.to_csv(f"{RESULTS}/basis_20260307.csv", index=False)
    print(f"\nSaved {len(df_all)} rows to basis_20260307.csv")
    
    # Summary stats
    summary = df_all.groupby('pair').agg(
        obs=('basis_bps', 'count'),
        mean_bps=('basis_bps', 'mean'),
        std_bps=('basis_bps', 'std'),
        min_bps=('basis_bps', 'min'),
        max_bps=('basis_bps', 'max'),
        median_bps=('basis_bps', 'median'),
        ann_mean_pct=('basis_ann_pct', 'mean'),
    ).round(3)
    print("\n=== Basis Summary ===")
    print(summary.to_string())
    
    # Anomalies (>2 std)
    for pair in df_all['pair'].unique():
        sub = df_all[df_all['pair'] == pair]
        mu, sigma = sub['basis_bps'].mean(), sub['basis_bps'].std()
        anomalies = sub[abs(sub['basis_bps'] - mu) > 2 * sigma]
        if len(anomalies) > 0:
            print(f"\n⚠️ {pair}: {len(anomalies)} anomalies (>2σ)")
            print(anomalies[['datetime', 'basis_bps', 'spot_mid', 'perp_mid']].head(5).to_string())
    
    # === FIGURE ===
    pairs_with_data = df_all['pair'].unique()
    n_pairs = len(pairs_with_data)
    
    fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 4*n_pairs), sharex=False)
    if n_pairs == 1:
        axes = [axes]
    
    fig.suptitle('Spot-Perpetual Basis (Binance)\nAgentic Sciences', fontsize=16, fontweight='bold')
    
    for i, pair in enumerate(sorted(pairs_with_data)):
        sub = df_all[df_all['pair'] == pair].sort_values('datetime')
        ax = axes[i]
        
        ax.plot(sub['datetime'], sub['basis_bps'], linewidth=0.5, alpha=0.7, color='#3b82f6')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        
        mu = sub['basis_bps'].mean()
        sigma = sub['basis_bps'].std()
        ax.axhline(y=mu, color='green', linewidth=1, linestyle='-', alpha=0.5, label=f'Mean: {mu:.1f} bps')
        ax.axhline(y=mu+2*sigma, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(y=mu-2*sigma, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
        
        ax.set_ylabel('Basis (bps)')
        ax.set_title(f'{pair}  |  Mean: {mu:.1f} bps  |  Ann: {mu/10000*365*24*100:.1f}%', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(f"{FIGURES}/basis_20260307.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: basis_20260307.png")

else:
    print("\nNo basis data computed!")

elapsed = (datetime.now() - t0).total_seconds()
print(f"\n⏱️ Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
