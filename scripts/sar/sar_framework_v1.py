#!/usr/bin/env python3
"""
Slippage-at-Risk (SaR) Framework Implementation
================================================
Based on: "Slippage-at-Risk (SaR): A Forward-Looking Liquidity Risk Framework 
           for Perpetual Futures Exchanges" (arXiv 2603.09164)

Data: Kaiko LOB Level-10 snapshots (Binance Futures, OkEX)
Server: research3 (local)

Key metrics computed:
1. Per-token slippage distributions (various trade sizes)
2. SaR(α) — α-quantile of slippage across tokens  
3. ESaR(α) — Expected Shortfall of slippage
4. TSaR$ — Total dollar slippage from tail tokens
5. Concentration haircut (HHI-based, using level distribution as proxy)
6. Cross-exchange SaR comparison (our unique extension)

Output: results/sar_*.csv, figures/sar_*.png, logs/run_sar_*.log

Author: ComeWealth / Agentic Sciences
Date: 2026-03-14
"""

import os
import sys
import gzip
import glob
import time
import json
import logging
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

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

# Exchanges to analyze
EXCHANGES = {
    "Binance Futures": {"type": "futures", "fee_bps": 4},  # taker fee
    "OkEX": {"type": "mixed", "fee_bps": 5},
    "Bybit": {"type": "futures", "fee_bps": 6},
    "Huobi Derivative Market": {"type": "futures", "fee_bps": 4},
}

# Trade sizes to simulate (in USD)
TRADE_SIZES_USD = [10_000, 50_000, 100_000, 500_000, 1_000_000]

# SaR parameters
SAR_ALPHA = 0.95  # 95th percentile
CONCENTRATION_LAMBDA = 0.5  # HHI penalty weight
CONCENTRATION_MU = 0.3  # CR1 penalty weight
N_TARGET = 5  # target number of liquidity providers
CR1_THRESHOLD = 0.5  # threshold for top provider share

# Analysis period: use latest available month
ANALYSIS_MONTH = "2023_04"  # latest in dataset
SAMPLE_DAYS = 3  # use 3 days for faster execution
SNAPSHOT_SAMPLE_RATE = 10  # sample every 10th snapshot

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{LOGS_DIR}/run_sar_{timestamp}.log"
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

def load_ob_snapshot(filepath):
    """Load a single day's LOB snapshots from gzipped CSV."""
    try:
        df = pd.read_csv(filepath, compression='gzip',
                         names=['date', 'type', 'price', 'amount'],
                         header=0)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        return df
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def compute_slippage_from_snapshot(bids, asks, trade_size_usd, side='sell'):
    """
    Compute slippage for a market order of given USD size.
    
    For a market SELL: walks down the bid side
    For a market BUY: walks up the ask side
    
    Slippage = (mid_price - avg_execution_price) / mid_price  [for sells]
    """
    if len(bids) == 0 or len(asks) == 0:
        return np.nan
    
    best_bid = bids['price'].max()
    best_ask = asks['price'].min()
    mid_price = (best_bid + best_ask) / 2
    
    if mid_price <= 0:
        return np.nan
    
    if side == 'sell':
        # Walk down bids (sorted descending by price)
        levels = bids.sort_values('price', ascending=False)
    else:
        # Walk up asks (sorted ascending by price)
        levels = asks.sort_values('price', ascending=True)
    
    remaining_usd = trade_size_usd
    total_filled_usd = 0
    total_filled_qty = 0
    
    for _, row in levels.iterrows():
        level_usd = row['price'] * row['amount']
        fill_usd = min(remaining_usd, level_usd)
        fill_qty = fill_usd / row['price']
        
        total_filled_usd += fill_usd
        total_filled_qty += fill_qty
        remaining_usd -= fill_usd
        
        if remaining_usd <= 0:
            break
    
    if total_filled_qty == 0 or total_filled_usd == 0:
        return np.nan
    
    # If we couldn't fill the entire order, that's extreme slippage
    if remaining_usd > trade_size_usd * 0.01:  # >1% unfilled
        return np.nan  # mark as too illiquid
    
    avg_price = total_filled_usd / total_filled_qty
    
    if side == 'sell':
        slippage = (mid_price - avg_price) / mid_price
    else:
        slippage = (avg_price - mid_price) / mid_price
    
    return slippage


def compute_concentration_haircut(levels_df):
    """
    Compute concentration haircut using level-based HHI as proxy.
    
    Since we don't have account-level attribution in CEX LOB data,
    we use the distribution of volume across price levels as a proxy
    for liquidity concentration.
    
    h = λ · max(0, N_target/N_eff - 1) + μ · max(0, CR₁ - threshold)
    """
    if len(levels_df) == 0:
        return 1.0  # maximum haircut
    
    amounts = levels_df['amount'].values * levels_df['price'].values  # USD amounts
    total = amounts.sum()
    
    if total <= 0:
        return 1.0
    
    shares = amounts / total
    
    # HHI and effective N
    hhi = np.sum(shares ** 2)
    n_eff = 1.0 / hhi if hhi > 0 else len(shares)
    
    # CR1: share of largest level
    cr1 = shares.max()
    
    # Haircut formula
    h = (CONCENTRATION_LAMBDA * max(0, N_TARGET / n_eff - 1) + 
         CONCENTRATION_MU * max(0, cr1 - CR1_THRESHOLD))
    
    return h


def analyze_token(exchange, pair, month, sample_days=7):
    """
    Analyze a single token: compute slippage distributions and concentration.
    
    Returns dict with slippage stats per trade size.
    """
    pair_dir = os.path.join(OB_DIR, exchange, pair, month)
    if not os.path.exists(pair_dir):
        return None
    
    files = sorted(glob.glob(os.path.join(pair_dir, "*.csv.gz")))
    if not files:
        return None
    
    # Use last N days
    files = files[-sample_days:]
    
    all_slippages = {size: [] for size in TRADE_SIZES_USD}
    all_haircuts = []
    mid_prices = []
    spreads_bps = []
    
    for f in files:
        df = load_ob_snapshot(f)
        if df is None or len(df) == 0:
            continue
        
        # Group by timestamp (each snapshot) with sampling
        snapshot_count = 0
        for ts, snapshot in df.groupby('date'):
            snapshot_count += 1
            if snapshot_count % SNAPSHOT_SAMPLE_RATE != 0:
                continue
            bids = snapshot[snapshot['type'] == 'b']
            asks = snapshot[snapshot['type'] == 'a']
            
            if len(bids) == 0 or len(asks) == 0:
                continue
            
            best_bid = bids['price'].max()
            best_ask = asks['price'].min()
            
            if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
                continue
            
            mid = (best_bid + best_ask) / 2
            spread = (best_ask - best_bid) / mid * 10000  # bps
            
            mid_prices.append(mid)
            spreads_bps.append(spread)
            
            # Compute slippage for each trade size (sell side)
            for size in TRADE_SIZES_USD:
                slip = compute_slippage_from_snapshot(bids, asks, size, side='sell')
                if not np.isnan(slip):
                    all_slippages[size].append(slip)
            
            # Concentration (bid side)
            h = compute_concentration_haircut(bids)
            all_haircuts.append(h)
    
    if not mid_prices:
        return None
    
    result = {
        'pair': pair,
        'exchange': exchange,
        'n_snapshots': len(mid_prices),
        'avg_mid_price': np.mean(mid_prices),
        'avg_spread_bps': np.mean(spreads_bps),
        'median_spread_bps': np.median(spreads_bps),
        'avg_haircut': np.mean(all_haircuts),
    }
    
    for size in TRADE_SIZES_USD:
        slips = all_slippages[size]
        if len(slips) > 10:
            arr = np.array(slips) * 10000  # convert to bps
            result[f'slip_{size}_mean_bps'] = np.mean(arr)
            result[f'slip_{size}_median_bps'] = np.median(arr)
            result[f'slip_{size}_p95_bps'] = np.percentile(arr, 95)
            result[f'slip_{size}_p99_bps'] = np.percentile(arr, 99)
            result[f'slip_{size}_std_bps'] = np.std(arr)
            result[f'slip_{size}_n'] = len(slips)
            # Adjusted slippage (with concentration haircut)
            adj = arr * (1 + np.mean(all_haircuts))
            result[f'slip_{size}_adj_p95_bps'] = np.percentile(adj, 95)
        else:
            result[f'slip_{size}_mean_bps'] = np.nan
            result[f'slip_{size}_n'] = len(slips)
    
    return result


def compute_sar_metrics(token_results, trade_size, alpha=0.95):
    """
    Compute portfolio-level SaR metrics across all tokens.
    
    SaR(α) = α-quantile of {slippage_i} across tokens
    ESaR(α) = E[slippage | slippage > SaR(α)]
    TSaR$ = Σ slippage_i · notional_i for tail tokens
    """
    valid = [r for r in token_results if not np.isnan(r.get(f'slip_{trade_size}_p95_bps', np.nan))]
    
    if len(valid) < 5:
        return None
    
    # Use median slippage per token as the cross-sectional distribution
    slippages = np.array([r[f'slip_{trade_size}_median_bps'] for r in valid])
    adj_slippages = np.array([r.get(f'slip_{trade_size}_adj_p95_bps', r[f'slip_{trade_size}_p95_bps']) for r in valid])
    pairs = [r['pair'] for r in valid]
    
    # SaR(α)
    sar = np.percentile(slippages, alpha * 100)
    sar_adj = np.percentile(adj_slippages, alpha * 100)
    
    # ESaR(α) — expected shortfall
    tail_mask = slippages >= sar
    esar = np.mean(slippages[tail_mask]) if tail_mask.sum() > 0 else sar
    
    # TSaR$ — total dollar slippage assuming equal notional
    notional_per_token = trade_size
    tsar_dollar = np.sum(slippages[tail_mask] / 10000 * notional_per_token)
    
    # Adjusted versions
    tail_mask_adj = adj_slippages >= sar_adj
    esar_adj = np.mean(adj_slippages[tail_mask_adj]) if tail_mask_adj.sum() > 0 else sar_adj
    
    # Identify tail tokens
    tail_tokens = [(p, s) for p, s, m in zip(pairs, slippages, tail_mask) if m]
    tail_tokens.sort(key=lambda x: -x[1])
    
    return {
        'trade_size_usd': trade_size,
        'n_tokens': len(valid),
        'sar_bps': sar,
        'sar_adj_bps': sar_adj,
        'sar_adj_increase_pct': (sar_adj / sar - 1) * 100 if sar > 0 else 0,
        'esar_bps': esar,
        'esar_adj_bps': esar_adj,
        'tsar_dollar': tsar_dollar,
        'median_slippage_bps': np.median(slippages),
        'mean_slippage_bps': np.mean(slippages),
        'slippage_std_bps': np.std(slippages),
        'tail_tokens': tail_tokens[:10],  # top 10 worst
        'all_slippages': list(zip(pairs, slippages.tolist())),
    }


# ============================================================
# Plotting Functions
# ============================================================

def plot_slippage_distribution(sar_results, exchange, save_path):
    """Plot cross-sectional slippage distribution with SaR/ESaR markers."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Slippage-at-Risk Distribution — {exchange}\n'
                 f'Kaiko LOB L10, {ANALYSIS_MONTH}, last {SAMPLE_DAYS} days',
                 fontsize=14, fontweight='bold')
    
    for idx, (size, result) in enumerate(sar_results.items()):
        if result is None:
            continue
        ax = axes[idx // 3, idx % 3]
        
        slippages = [s for _, s in result['all_slippages']]
        
        ax.hist(slippages, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(result['sar_bps'], color='red', linestyle='--', linewidth=2,
                   label=f"SaR(95%) = {result['sar_bps']:.1f} bps")
        ax.axvline(result['sar_adj_bps'], color='darkred', linestyle='-', linewidth=2,
                   label=f"SaR_adj = {result['sar_adj_bps']:.1f} bps")
        ax.axvline(result['esar_bps'], color='orange', linestyle=':', linewidth=2,
                   label=f"ESaR = {result['esar_bps']:.1f} bps")
        
        ax.set_title(f'Trade Size: ${size:,.0f}', fontsize=11)
        ax.set_xlabel('Slippage (bps)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
    
    # Remove empty subplot if odd number
    if len(sar_results) < 6:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_sar_by_trade_size(sar_results_by_exchange, save_path):
    """Plot SaR curves across trade sizes for multiple exchanges."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SaR Framework: Cross-Exchange Comparison\n'
                 'Kaiko LOB L10 Data', fontsize=14, fontweight='bold')
    
    colors = {'Binance Futures': '#F0B90B', 'OkEX': '#4CAF50', 
              'Bybit': '#FF6B00', 'Huobi Derivative Market': '#2196F3'}
    
    for exchange, sar_dict in sar_results_by_exchange.items():
        sizes = []
        sars = []
        sars_adj = []
        esars = []
        
        for size in TRADE_SIZES_USD:
            if size in sar_dict and sar_dict[size] is not None:
                sizes.append(size)
                sars.append(sar_dict[size]['sar_bps'])
                sars_adj.append(sar_dict[size]['sar_adj_bps'])
                esars.append(sar_dict[size]['esar_bps'])
        
        if not sizes:
            continue
        
        color = colors.get(exchange, 'gray')
        label = exchange.replace(' Derivative Market', '')
        
        axes[0].plot(sizes, sars, 'o-', color=color, label=label, linewidth=2)
        axes[1].plot(sizes, sars_adj, 's--', color=color, label=label, linewidth=2)
        axes[2].plot(sizes, esars, '^:', color=color, label=label, linewidth=2)
    
    for ax, title in zip(axes, ['SaR(95%)', 'SaR_adj(95%) [+concentration]', 'ESaR(95%)']):
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Trade Size (USD)')
        ax.set_ylabel('Slippage (bps)')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_tail_tokens_heatmap(token_results, exchange, save_path):
    """Heatmap of slippage across tokens and trade sizes."""
    # Get top 30 most liquid + bottom 30 least liquid
    valid = [r for r in token_results 
             if not np.isnan(r.get(f'slip_{TRADE_SIZES_USD[0]}_mean_bps', np.nan))]
    
    if len(valid) < 10:
        return
    
    # Sort by $100K slippage
    size_key = 100_000
    valid.sort(key=lambda x: x.get(f'slip_{size_key}_median_bps', 999))
    
    # Take top 15 most liquid + bottom 15 least liquid
    selected = valid[:15] + valid[-15:]
    
    # Build matrix
    pairs = [r['pair'].replace('USDT', '') for r in selected]
    matrix = np.zeros((len(selected), len(TRADE_SIZES_USD)))
    
    for i, r in enumerate(selected):
        for j, size in enumerate(TRADE_SIZES_USD):
            matrix[i, j] = r.get(f'slip_{size}_median_bps', np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Log scale for better visualization
    log_matrix = np.log10(np.clip(matrix, 0.01, None))
    
    sns.heatmap(log_matrix, ax=ax, xticklabels=[f'${s//1000}K' for s in TRADE_SIZES_USD],
                yticklabels=pairs, cmap='YlOrRd', annot=matrix, fmt='.1f',
                cbar_kws={'label': 'log10(Slippage bps)'})
    
    ax.set_title(f'Slippage Heatmap — {exchange}\n'
                 f'Top 15 most liquid ↑ | Bottom 15 least liquid ↓',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Trade Size')
    ax.set_ylabel('Token')
    
    # Draw separator line
    ax.axhline(y=15, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


# ============================================================
# Main Execution
# ============================================================

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SaR Framework v1 — Starting Analysis")
    logger.info(f"Analysis month: {ANALYSIS_MONTH}, sample days: {SAMPLE_DAYS}")
    logger.info("=" * 60)
    
    # Ensure output dirs exist
    for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
    
    all_exchange_results = {}
    all_exchange_sar = {}
    
    for exchange, config in EXCHANGES.items():
        exchange_dir = os.path.join(OB_DIR, exchange)
        if not os.path.exists(exchange_dir):
            logger.warning(f"Exchange dir not found: {exchange}")
            continue
        
        # Get all USDT pairs
        pairs = [p for p in os.listdir(exchange_dir) 
                 if p.endswith('USDT') and os.path.isdir(os.path.join(exchange_dir, p))]
        
        # Check which pairs have data for the target month
        valid_pairs = []
        for pair in pairs:
            pair_month_dir = os.path.join(exchange_dir, pair, ANALYSIS_MONTH)
            if os.path.exists(pair_month_dir):
                files = glob.glob(os.path.join(pair_month_dir, "*.csv.gz"))
                if len(files) >= SAMPLE_DAYS:
                    valid_pairs.append(pair)
        
        logger.info(f"\n{'='*40}")
        logger.info(f"Exchange: {exchange}")
        logger.info(f"Total USDT pairs: {len(pairs)}, with {ANALYSIS_MONTH} data: {len(valid_pairs)}")
        
        if len(valid_pairs) < 5:
            logger.warning(f"Too few pairs for {exchange}, skipping")
            continue
        
        # Analyze each token
        token_results = []
        for i, pair in enumerate(sorted(valid_pairs)):
            if (i + 1) % 10 == 0:
                logger.info(f"  Processing {i+1}/{len(valid_pairs)}: {pair}")
                sys.stdout.flush()
            
            result = analyze_token(exchange, pair, ANALYSIS_MONTH, SAMPLE_DAYS)
            if result is not None:
                token_results.append(result)
        
        logger.info(f"Successfully analyzed {len(token_results)}/{len(valid_pairs)} tokens")
        
        if len(token_results) < 5:
            continue
        
        all_exchange_results[exchange] = token_results
        
        # Compute SaR metrics for each trade size
        sar_results = {}
        for size in TRADE_SIZES_USD:
            sar = compute_sar_metrics(token_results, size, SAR_ALPHA)
            sar_results[size] = sar
            
            if sar:
                logger.info(f"\n  Trade Size ${size:>10,}:")
                logger.info(f"    Tokens: {sar['n_tokens']}")
                logger.info(f"    Median slippage: {sar['median_slippage_bps']:.2f} bps")
                logger.info(f"    SaR(95%): {sar['sar_bps']:.2f} bps")
                logger.info(f"    SaR_adj(95%): {sar['sar_adj_bps']:.2f} bps (+{sar['sar_adj_increase_pct']:.1f}%)")
                logger.info(f"    ESaR(95%): {sar['esar_bps']:.2f} bps")
                logger.info(f"    TSaR$: ${sar['tsar_dollar']:,.0f}")
                logger.info(f"    Worst tokens: {[t[0] for t in sar['tail_tokens'][:5]]}")
        
        all_exchange_sar[exchange] = sar_results
        
        # Save per-token results
        df_tokens = pd.DataFrame(token_results)
        token_csv = f"{RESULTS_DIR}/sar_tokens_{exchange.replace(' ', '_')}_{ANALYSIS_MONTH}.csv"
        df_tokens.to_csv(token_csv, index=False)
        logger.info(f"Saved: {token_csv}")
        
        # Plot slippage distribution
        plot_slippage_distribution(
            sar_results, exchange,
            f"{FIGURES_DIR}/sar_distribution_{exchange.replace(' ', '_')}_{ANALYSIS_MONTH}.png"
        )
        
        # Plot heatmap
        plot_tail_tokens_heatmap(
            token_results, exchange,
            f"{FIGURES_DIR}/sar_heatmap_{exchange.replace(' ', '_')}_{ANALYSIS_MONTH}.png"
        )
    
    # Cross-exchange comparison plot
    if len(all_exchange_sar) > 1:
        plot_sar_by_trade_size(
            all_exchange_sar,
            f"{FIGURES_DIR}/sar_cross_exchange_{ANALYSIS_MONTH}.png"
        )
    
    # Summary table
    summary_rows = []
    for exchange, sar_dict in all_exchange_sar.items():
        for size, sar in sar_dict.items():
            if sar is None:
                continue
            summary_rows.append({
                'exchange': exchange,
                'trade_size_usd': size,
                'n_tokens': sar['n_tokens'],
                'median_slip_bps': sar['median_slippage_bps'],
                'sar_95_bps': sar['sar_bps'],
                'sar_adj_95_bps': sar['sar_adj_bps'],
                'adj_increase_pct': sar['sar_adj_increase_pct'],
                'esar_95_bps': sar['esar_bps'],
                'tsar_dollar': sar['tsar_dollar'],
            })
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_csv = f"{RESULTS_DIR}/sar_summary_{ANALYSIS_MONTH}.csv"
        df_summary.to_csv(summary_csv, index=False)
        logger.info(f"\nSaved summary: {summary_csv}")
        logger.info(f"\n{df_summary.to_string()}")
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"SaR Analysis Complete — {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*60}")
    
    return all_exchange_sar, all_exchange_results


if __name__ == "__main__":
    sar_results, token_results = main()
