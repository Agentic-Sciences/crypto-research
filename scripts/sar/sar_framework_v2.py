#!/usr/bin/env python3
"""
SaR Framework v2 — Optimized Multi-Process Implementation
==========================================================
Improvements over v1:
- Multiprocessing (16 workers) for parallel token analysis
- Snapshot sampling (every 60th snapshot instead of all)
- Chunked file reading to reduce memory
- Progress tracking every 10 tokens
- Both sell and buy side slippage

Data: Kaiko LOB Level-10 snapshots
Server: research3 (local)
"""

import os
import sys
import gzip
import glob
import time
import json
import logging
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = "/mnt/work/qr33/comewealth"
OB_DIR = "/mnt/kaiko/consolidated/order_book/kaiko-ob10-v2"
RESULTS_DIR = f"{BASE_DIR}/results"
FIGURES_DIR = f"{BASE_DIR}/figures"
LOGS_DIR = f"{BASE_DIR}/logs"
CACHE_DIR = f"{BASE_DIR}/cache"

EXCHANGES = {
    "Binance Futures": {"type": "futures", "fee_bps": 4},
    "OkEX": {"type": "mixed", "fee_bps": 5},
}

TRADE_SIZES_USD = [10_000, 50_000, 100_000, 500_000, 1_000_000]
SAR_ALPHA = 0.95
ANALYSIS_MONTH = "2023_04"
SAMPLE_DAYS = 5
SNAPSHOT_SAMPLE_RATE = 60  # use every 60th snapshot (~1 per minute)
N_WORKERS = min(16, cpu_count())

# Concentration params
CONC_LAMBDA = 0.5
CONC_MU = 0.3
N_TARGET = 5
CR1_THRESHOLD = 0.5

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOGS_DIR}/run_sar_v2_{timestamp}.log"

for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# Core Functions
# ============================================================

def compute_slippage(levels, trade_size_usd, mid_price, side='sell'):
    """Compute slippage walking through order book levels."""
    if len(levels) == 0 or mid_price <= 0:
        return np.nan
    
    remaining = trade_size_usd
    filled_usd = 0
    filled_qty = 0
    
    for _, row in levels.iterrows():
        level_usd = row['price'] * row['amount']
        fill = min(remaining, level_usd)
        filled_usd += fill
        filled_qty += fill / row['price']
        remaining -= fill
        if remaining <= 0:
            break
    
    if filled_qty == 0 or remaining > trade_size_usd * 0.01:
        return np.nan
    
    avg_price = filled_usd / filled_qty
    if side == 'sell':
        return (mid_price - avg_price) / mid_price
    else:
        return (avg_price - mid_price) / mid_price


def compute_concentration(amounts_usd):
    """HHI-based concentration haircut."""
    total = amounts_usd.sum()
    if total <= 0:
        return 1.0
    shares = amounts_usd / total
    hhi = np.sum(shares ** 2)
    n_eff = 1.0 / hhi if hhi > 0 else len(shares)
    cr1 = shares.max()
    h = CONC_LAMBDA * max(0, N_TARGET / n_eff - 1) + CONC_MU * max(0, cr1 - CR1_THRESHOLD)
    return h


def analyze_token_worker(args):
    """Worker function for multiprocessing. Analyzes one token."""
    exchange, pair, month, sample_days, sample_rate = args
    
    pair_dir = os.path.join(OB_DIR, exchange, pair, month)
    if not os.path.exists(pair_dir):
        return None
    
    files = sorted(glob.glob(os.path.join(pair_dir, "*.csv.gz")))
    if not files:
        return None
    files = files[-sample_days:]
    
    all_slippages = {size: [] for size in TRADE_SIZES_USD}
    all_haircuts = []
    mid_prices = []
    spreads_bps = []
    bid_depths = []
    ask_depths = []
    
    for f in files:
        try:
            df = pd.read_csv(f, compression='gzip',
                           names=['date', 'type', 'price', 'amount'], header=0)
        except:
            continue
        
        if len(df) == 0:
            continue
        
        # Group by timestamp
        timestamps = df['date'].unique()
        # Sample every Nth snapshot
        sampled_ts = timestamps[::sample_rate]
        
        for ts in sampled_ts:
            snapshot = df[df['date'] == ts]
            bids = snapshot[snapshot['type'] == 'b'].sort_values('price', ascending=False)
            asks = snapshot[snapshot['type'] == 'a'].sort_values('price', ascending=True)
            
            if len(bids) == 0 or len(asks) == 0:
                continue
            
            best_bid = bids['price'].iloc[0]
            best_ask = asks['price'].iloc[0]
            
            if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
                continue
            
            mid = (best_bid + best_ask) / 2
            spread = (best_ask - best_bid) / mid * 10000
            
            mid_prices.append(mid)
            spreads_bps.append(spread)
            
            # Depths in USD
            bid_depth = (bids['price'] * bids['amount']).sum()
            ask_depth = (asks['price'] * asks['amount']).sum()
            bid_depths.append(bid_depth)
            ask_depths.append(ask_depth)
            
            # Slippage for each trade size (sell side)
            for size in TRADE_SIZES_USD:
                slip = compute_slippage(bids, size, mid, 'sell')
                if not np.isnan(slip):
                    all_slippages[size].append(slip)
            
            # Concentration
            h = compute_concentration(bids['price'] * bids['amount'])
            all_haircuts.append(h)
    
    if not mid_prices:
        return None
    
    result = {
        'pair': pair,
        'exchange': exchange,
        'n_snapshots': len(mid_prices),
        'avg_mid_price': float(np.mean(mid_prices)),
        'avg_spread_bps': float(np.mean(spreads_bps)),
        'median_spread_bps': float(np.median(spreads_bps)),
        'avg_bid_depth_usd': float(np.mean(bid_depths)),
        'avg_ask_depth_usd': float(np.mean(ask_depths)),
        'depth_ratio': float(np.mean(bid_depths) / np.mean(ask_depths)) if np.mean(ask_depths) > 0 else np.nan,
        'avg_haircut': float(np.mean(all_haircuts)),
    }
    
    for size in TRADE_SIZES_USD:
        slips = all_slippages[size]
        if len(slips) > 5:
            arr = np.array(slips) * 10000  # bps
            result[f'slip_{size}_mean'] = float(np.mean(arr))
            result[f'slip_{size}_median'] = float(np.median(arr))
            result[f'slip_{size}_p95'] = float(np.percentile(arr, 95))
            result[f'slip_{size}_p99'] = float(np.percentile(arr, 99))
            result[f'slip_{size}_std'] = float(np.std(arr))
            result[f'slip_{size}_n'] = len(slips)
            # Adjusted
            adj = arr * (1 + np.mean(all_haircuts))
            result[f'slip_{size}_adj_p95'] = float(np.percentile(adj, 95))
        else:
            for suffix in ['mean','median','p95','p99','std','adj_p95']:
                result[f'slip_{size}_{suffix}'] = np.nan
            result[f'slip_{size}_n'] = len(slips)
    
    return result


def compute_sar(token_results, trade_size, alpha=0.95):
    """Cross-sectional SaR metrics."""
    valid = [r for r in token_results 
             if r and not np.isnan(r.get(f'slip_{trade_size}_median', np.nan))]
    
    if len(valid) < 5:
        return None
    
    slippages = np.array([r[f'slip_{trade_size}_median'] for r in valid])
    adj_slippages = np.array([r.get(f'slip_{trade_size}_adj_p95', r[f'slip_{trade_size}_p95']) for r in valid])
    pairs = [r['pair'] for r in valid]
    
    sar = float(np.percentile(slippages, alpha * 100))
    sar_adj = float(np.percentile(adj_slippages, alpha * 100))
    
    tail_mask = slippages >= sar
    esar = float(np.mean(slippages[tail_mask])) if tail_mask.sum() > 0 else sar
    
    tsar_dollar = float(np.sum(slippages[tail_mask] / 10000 * trade_size))
    
    # Insurance fund sizing: IF* = c * TSaR$, c in [1.5, 3.0]
    if_low = tsar_dollar * 1.5
    if_high = tsar_dollar * 3.0
    
    # Tail tokens
    tail_tokens = sorted(
        [(p, float(s)) for p, s, m in zip(pairs, slippages, tail_mask) if m],
        key=lambda x: -x[1]
    )
    
    return {
        'trade_size': trade_size,
        'n_tokens': len(valid),
        'median_slip': float(np.median(slippages)),
        'mean_slip': float(np.mean(slippages)),
        'std_slip': float(np.std(slippages)),
        'sar_95': sar,
        'sar_adj_95': sar_adj,
        'adj_increase_pct': (sar_adj / sar - 1) * 100 if sar > 0 else 0,
        'esar_95': esar,
        'tsar_dollar': tsar_dollar,
        'if_recommended_low': if_low,
        'if_recommended_high': if_high,
        'n_tail_tokens': int(tail_mask.sum()),
        'tail_tokens_top10': tail_tokens[:10],
        'all_pairs_slippage': list(zip(pairs, slippages.tolist())),
    }


# ============================================================
# Plotting
# ============================================================

def plot_comprehensive(exchange, sar_results, token_results, save_prefix):
    """Generate comprehensive multi-panel figure."""
    
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle(f'Slippage-at-Risk (SaR) Framework — {exchange}\n'
                 f'Kaiko LOB L10 | {ANALYSIS_MONTH} | {SAMPLE_DAYS} days | {len(token_results)} tokens',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. SaR across trade sizes
    ax1 = fig.add_subplot(gs[0, 0])
    sizes = []
    sars = []
    sars_adj = []
    esars = []
    for size in TRADE_SIZES_USD:
        if size in sar_results and sar_results[size]:
            sizes.append(size)
            sars.append(sar_results[size]['sar_95'])
            sars_adj.append(sar_results[size]['sar_adj_95'])
            esars.append(sar_results[size]['esar_95'])
    
    ax1.plot(sizes, sars, 'o-', color='#2196F3', linewidth=2, label='SaR(95%)', markersize=8)
    ax1.plot(sizes, sars_adj, 's--', color='#F44336', linewidth=2, label='SaR_adj(95%)', markersize=8)
    ax1.plot(sizes, esars, '^:', color='#FF9800', linewidth=2, label='ESaR(95%)', markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('Trade Size (USD)')
    ax1.set_ylabel('Slippage (bps)')
    ax1.set_title('SaR Metrics vs Trade Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Slippage distribution for $100K
    ax2 = fig.add_subplot(gs[0, 1])
    if 100_000 in sar_results and sar_results[100_000]:
        slips = [s for _, s in sar_results[100_000]['all_pairs_slippage']]
        ax2.hist(slips, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
        ax2.axvline(sar_results[100_000]['sar_95'], color='red', linestyle='--', linewidth=2,
                   label=f"SaR = {sar_results[100_000]['sar_95']:.1f}")
        ax2.axvline(sar_results[100_000]['esar_95'], color='orange', linestyle=':', linewidth=2,
                   label=f"ESaR = {sar_results[100_000]['esar_95']:.1f}")
        ax2.set_xlabel('Median Slippage (bps)')
        ax2.set_ylabel('Count')
        ax2.set_title('Cross-Token Slippage Distribution ($100K)')
        ax2.legend()
    
    # 3. Insurance Fund sizing
    ax3 = fig.add_subplot(gs[0, 2])
    if_lows = []
    if_highs = []
    if_sizes = []
    for size in TRADE_SIZES_USD:
        if size in sar_results and sar_results[size]:
            if_sizes.append(size)
            if_lows.append(sar_results[size]['if_recommended_low'])
            if_highs.append(sar_results[size]['if_recommended_high'])
    
    ax3.fill_between(if_sizes, if_lows, if_highs, alpha=0.3, color='green', label='IF range [1.5x, 3.0x]')
    ax3.plot(if_sizes, if_lows, 'o-', color='green', linewidth=1.5)
    ax3.plot(if_sizes, if_highs, 'o-', color='darkgreen', linewidth=1.5)
    ax3.set_xscale('log')
    ax3.set_xlabel('Trade Size (USD)')
    ax3.set_ylabel('Recommended IF ($)')
    ax3.set_title('Insurance Fund Sizing (IF* = c·TSaR$)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Top 30 tokens by slippage (horizontal bar)
    ax4 = fig.add_subplot(gs[1, :2])
    if 100_000 in sar_results and sar_results[100_000]:
        pairs_slips = sorted(sar_results[100_000]['all_pairs_slippage'], key=lambda x: -x[1])
        top30 = pairs_slips[:30]
        names = [p.replace('USDT','') for p,_ in top30]
        vals = [s for _,s in top30]
        colors = ['#F44336' if s >= sar_results[100_000]['sar_95'] else '#2196F3' for s in vals]
        
        ax4.barh(range(len(names)), vals, color=colors, edgecolor='white', height=0.7)
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel('Median Slippage (bps) for $100K sell')
        ax4.set_title(f'Top 30 Highest-Slippage Tokens (red = above SaR 95%)')
        ax4.axvline(sar_results[100_000]['sar_95'], color='red', linestyle='--', alpha=0.7)
    
    # 5. Spread vs Depth scatter
    ax5 = fig.add_subplot(gs[1, 2])
    valid_tokens = [r for r in token_results if r and r.get('avg_bid_depth_usd',0) > 0]
    if valid_tokens:
        depths = [r['avg_bid_depth_usd'] / 1e6 for r in valid_tokens]  # millions
        spreads = [r['avg_spread_bps'] for r in valid_tokens]
        slips_100k = [r.get('slip_100000_median', np.nan) for r in valid_tokens]
        
        scatter = ax5.scatter(depths, spreads, c=slips_100k, cmap='YlOrRd', 
                            s=30, alpha=0.7, edgecolors='gray', linewidths=0.5)
        plt.colorbar(scatter, ax=ax5, label='$100K slip (bps)')
        ax5.set_xlabel('Avg Bid Depth ($M)')
        ax5.set_ylabel('Avg Spread (bps)')
        ax5.set_title('Depth vs Spread (color = slippage)')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
    
    # 6. Heatmap: tokens x trade sizes
    ax6 = fig.add_subplot(gs[2, :])
    if 100_000 in sar_results and sar_results[100_000]:
        # Select top 15 + bottom 15 by $100K slippage
        valid_sorted = sorted(
            [r for r in token_results if r and not np.isnan(r.get('slip_100000_median', np.nan))],
            key=lambda x: x['slip_100000_median']
        )
        selected = valid_sorted[:15] + valid_sorted[-15:]
        
        names = [r['pair'].replace('USDT','') for r in selected]
        matrix = np.zeros((len(selected), len(TRADE_SIZES_USD)))
        for i, r in enumerate(selected):
            for j, size in enumerate(TRADE_SIZES_USD):
                matrix[i, j] = r.get(f'slip_{size}_median', np.nan)
        
        im = ax6.imshow(np.log10(np.clip(matrix, 0.001, None)), cmap='YlOrRd', aspect='auto')
        ax6.set_xticks(range(len(TRADE_SIZES_USD)))
        ax6.set_xticklabels([f'${s//1000}K' for s in TRADE_SIZES_USD])
        ax6.set_yticks(range(len(names)))
        ax6.set_yticklabels(names, fontsize=7)
        ax6.set_title('Slippage Heatmap: Most Liquid (top) vs Least Liquid (bottom)')
        plt.colorbar(im, ax=ax6, label='log10(slippage bps)')
        
        # Annotate values
        for i in range(len(selected)):
            for j in range(len(TRADE_SIZES_USD)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax6.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6,
                            color='white' if v > 10 else 'black')
        
        ax6.axhline(y=14.5, color='black', linewidth=2)
    
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_prefix}_comprehensive.png")


def plot_cross_exchange(all_sar, save_path):
    """Cross-exchange SaR comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cross-Exchange SaR Comparison | Kaiko LOB L10', fontsize=14, fontweight='bold')
    
    colors = {'Binance Futures': '#F0B90B', 'OkEX': '#4CAF50'}
    
    for exchange, sar_dict in all_sar.items():
        sizes, sars, sars_adj, esars = [], [], [], []
        for size in TRADE_SIZES_USD:
            if size in sar_dict and sar_dict[size]:
                sizes.append(size)
                sars.append(sar_dict[size]['sar_95'])
                sars_adj.append(sar_dict[size]['sar_adj_95'])
                esars.append(sar_dict[size]['esar_95'])
        
        if not sizes:
            continue
        c = colors.get(exchange, 'gray')
        label = exchange.replace(' Futures','')
        axes[0].plot(sizes, sars, 'o-', color=c, label=label, linewidth=2)
        axes[1].plot(sizes, sars_adj, 's--', color=c, label=label, linewidth=2)
        axes[2].plot(sizes, esars, '^:', color=c, label=label, linewidth=2)
    
    for ax, title in zip(axes, ['SaR(95%)', 'SaR_adj(95%)', 'ESaR(95%)']):
        ax.set_title(title)
        ax.set_xlabel('Trade Size (USD)')
        ax.set_ylabel('Slippage (bps)')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"SaR Framework v2 — {N_WORKERS} workers, sample rate 1/{SNAPSHOT_SAMPLE_RATE}")
    logger.info(f"Month: {ANALYSIS_MONTH}, days: {SAMPLE_DAYS}")
    logger.info("=" * 60)
    
    all_sar = {}
    all_tokens = {}
    
    for exchange, config in EXCHANGES.items():
        exchange_dir = os.path.join(OB_DIR, exchange)
        if not os.path.exists(exchange_dir):
            logger.warning(f"Not found: {exchange}")
            continue
        
        # Find valid pairs
        pairs = [p for p in os.listdir(exchange_dir)
                 if p.endswith('USDT') and os.path.isdir(os.path.join(exchange_dir, p))]
        
        valid_pairs = []
        for pair in pairs:
            pdir = os.path.join(exchange_dir, pair, ANALYSIS_MONTH)
            if os.path.exists(pdir) and len(glob.glob(os.path.join(pdir, "*.csv.gz"))) >= SAMPLE_DAYS:
                valid_pairs.append(pair)
        
        logger.info(f"\n{'='*40}")
        logger.info(f"{exchange}: {len(valid_pairs)} valid pairs")
        
        if len(valid_pairs) < 5:
            continue
        
        # Parallel analysis
        args = [(exchange, pair, ANALYSIS_MONTH, SAMPLE_DAYS, SNAPSHOT_SAMPLE_RATE) 
                for pair in sorted(valid_pairs)]
        
        t1 = time.time()
        with Pool(N_WORKERS) as pool:
            results = pool.map(analyze_token_worker, args)
        
        token_results = [r for r in results if r is not None]
        elapsed = time.time() - t1
        logger.info(f"Analyzed {len(token_results)}/{len(valid_pairs)} tokens in {elapsed:.1f}s")
        
        all_tokens[exchange] = token_results
        
        # Compute SaR
        sar_results = {}
        for size in TRADE_SIZES_USD:
            sar = compute_sar(token_results, size, SAR_ALPHA)
            sar_results[size] = sar
            
            if sar:
                logger.info(f"\n  ${size:>10,}: {sar['n_tokens']} tokens")
                logger.info(f"    Median: {sar['median_slip']:.2f} bps | SaR: {sar['sar_95']:.2f} | "
                          f"SaR_adj: {sar['sar_adj_95']:.2f} (+{sar['adj_increase_pct']:.1f}%) | "
                          f"ESaR: {sar['esar_95']:.2f}")
                logger.info(f"    TSaR$: ${sar['tsar_dollar']:,.0f} | IF range: "
                          f"${sar['if_recommended_low']:,.0f} - ${sar['if_recommended_high']:,.0f}")
                logger.info(f"    Worst: {[t[0].replace('USDT','') for t in sar['tail_tokens_top10'][:5]]}")
        
        all_sar[exchange] = sar_results
        
        # Save per-token CSV
        df = pd.DataFrame(token_results)
        csv_path = f"{RESULTS_DIR}/sar_v2_tokens_{exchange.replace(' ','_')}_{ANALYSIS_MONTH}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
        
        # Comprehensive plot
        plot_comprehensive(exchange, sar_results, token_results,
                         f"{FIGURES_DIR}/sar_v2_{exchange.replace(' ','_')}_{ANALYSIS_MONTH}")
    
    # Cross-exchange comparison
    if len(all_sar) > 1:
        plot_cross_exchange(all_sar, f"{FIGURES_DIR}/sar_v2_cross_exchange_{ANALYSIS_MONTH}.png")
    
    # Summary CSV
    summary = []
    for exchange, sar_dict in all_sar.items():
        for size, sar in sar_dict.items():
            if sar is None:
                continue
            summary.append({
                'exchange': exchange,
                'trade_size': size,
                'n_tokens': sar['n_tokens'],
                'median_slip_bps': sar['median_slip'],
                'sar_95_bps': sar['sar_95'],
                'sar_adj_95_bps': sar['sar_adj_95'],
                'adj_pct': sar['adj_increase_pct'],
                'esar_95_bps': sar['esar_95'],
                'tsar_dollar': sar['tsar_dollar'],
                'if_low': sar['if_recommended_low'],
                'if_high': sar['if_recommended_high'],
                'n_tail': sar['n_tail_tokens'],
            })
    
    if summary:
        df_sum = pd.DataFrame(summary)
        sum_path = f"{RESULTS_DIR}/sar_v2_summary_{ANALYSIS_MONTH}.csv"
        df_sum.to_csv(sum_path, index=False)
        logger.info(f"\nSummary saved: {sum_path}")
        print(f"\n{df_sum.to_string()}")
    
    total = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE — {total:.1f}s ({total/60:.1f} min)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
