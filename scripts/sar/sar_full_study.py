#!/usr/bin/env python3
"""
SaR Full Study — Complete Cross-Exchange, Cross-Period Analysis
================================================================
Exchanges: Binance Futures, OkEX, FTX, Bybit, Huobi DM
Periods:
  - Stress events: COVID crash (2020-03), China ban (2021-05), 
    Terra/LUNA (2022-05), FTX collapse (2022-11)
  - Normal periods: 2021-11 (bull peak), 2022-09 (range), 
    2023-01 (recovery), 2023-04 (normal)

Output: comprehensive research-grade figures + CSV + report
"""

import os, sys, gzip, glob, time, json, logging, warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
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
BASE_DIR = "/mnt/work/qr33/comewealth"
OB_DIR = "/mnt/kaiko/consolidated/order_book/kaiko-ob10-v2"
RESULTS_DIR = f"{BASE_DIR}/results"
FIGURES_DIR = f"{BASE_DIR}/figures"
LOGS_DIR = f"{BASE_DIR}/logs"

TRADE_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
SAR_ALPHA = 0.95
SAMPLE_DAYS = 5
SNAPSHOT_RATE = 60
N_WORKERS = min(16, cpu_count())
CONC_LAMBDA, CONC_MU, N_TARGET, CR1_THR = 0.5, 0.3, 5, 0.5

# Define study periods
PERIODS = {
    # Stress events
    'COVID_crash':    {'month': '2020_03', 'label': 'COVID Crash\n2020-03',    'type': 'stress', 'color': '#d32f2f'},
    'China_ban':      {'month': '2021_05', 'label': 'China Ban\n2021-05',      'type': 'stress', 'color': '#e64a19'},
    'Terra_LUNA':     {'month': '2022_05', 'label': 'Terra/LUNA\n2022-05',     'type': 'stress', 'color': '#c62828'},
    'FTX_collapse':   {'month': '2022_11', 'label': 'FTX Collapse\n2022-11',   'type': 'stress', 'color': '#b71c1c'},
    # Normal/other
    'Bull_peak':      {'month': '2021_11', 'label': 'Bull Peak\n2021-11',      'type': 'normal', 'color': '#2e7d32'},
    'Range_bound':    {'month': '2022_09', 'label': 'Range Bound\n2022-09',    'type': 'normal', 'color': '#1565c0'},
    'Recovery':       {'month': '2023_01', 'label': 'Recovery\n2023-01',       'type': 'normal', 'color': '#0277bd'},
    'Normal_2023':    {'month': '2023_04', 'label': 'Normal\n2023-04',         'type': 'normal', 'color': '#00838f'},
}

EXCHANGES = {
    'Binance Futures':        {'fee': 4,  'color': '#F0B90B', 'short': 'Binance'},
    'OkEX':                   {'fee': 5,  'color': '#4CAF50', 'short': 'OKX'},
    'FTX':                    {'fee': 4,  'color': '#5DADE2', 'short': 'FTX'},
    'Bybit':                  {'fee': 6,  'color': '#FF6B00', 'short': 'Bybit'},
    'Huobi Derivative Market':{'fee': 4,  'color': '#2196F3', 'short': 'Huobi'},
}

for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(f"{LOGS_DIR}/run_sar_full_{ts}.log"),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ============================================================
# Worker (same as v2 but returns more metadata)
# ============================================================
def analyze_worker(args):
    exchange, pair, month, sample_days, sample_rate = args
    pair_dir = os.path.join(OB_DIR, exchange, pair, month)
    if not os.path.exists(pair_dir):
        return None
    
    files = sorted(glob.glob(os.path.join(pair_dir, "*.csv.gz")))
    if not files:
        return None
    files = files[-sample_days:]
    
    slippages = {s: [] for s in TRADE_SIZES}
    haircuts, mids, spreads_bps, bid_depths, ask_depths = [], [], [], [], []
    
    for f in files:
        try:
            df = pd.read_csv(f, compression='gzip',
                           names=['date','type','price','amount'], header=0)
        except:
            continue
        if len(df) == 0:
            continue
        
        timestamps = df['date'].unique()[::sample_rate]
        for t in timestamps:
            snap = df[df['date'] == t]
            bids = snap[snap['type'] == 'b'].sort_values('price', ascending=False)
            asks = snap[snap['type'] == 'a'].sort_values('price', ascending=True)
            if len(bids) == 0 or len(asks) == 0:
                continue
            
            bb, ba = bids['price'].iloc[0], asks['price'].iloc[0]
            if bb <= 0 or ba <= 0 or ba < bb:
                continue
            
            mid = (bb + ba) / 2
            mids.append(mid)
            spreads_bps.append((ba - bb) / mid * 10000)
            
            bd = (bids['price'] * bids['amount']).sum()
            ad = (asks['price'] * asks['amount']).sum()
            bid_depths.append(bd)
            ask_depths.append(ad)
            
            # Slippage
            for size in TRADE_SIZES:
                rem = size
                filled_usd = filled_qty = 0
                for _, r in bids.iterrows():
                    lu = r['price'] * r['amount']
                    fill = min(rem, lu)
                    filled_usd += fill
                    filled_qty += fill / r['price']
                    rem -= fill
                    if rem <= 0:
                        break
                if filled_qty > 0 and rem <= size * 0.01:
                    avg_p = filled_usd / filled_qty
                    slippages[size].append((mid - avg_p) / mid)
            
            # Concentration
            amts = bids['price'].values * bids['amount'].values
            total = amts.sum()
            if total > 0:
                shares = amts / total
                hhi = np.sum(shares**2)
                n_eff = 1/hhi if hhi > 0 else len(shares)
                cr1 = shares.max()
                h = CONC_LAMBDA * max(0, N_TARGET/n_eff - 1) + CONC_MU * max(0, cr1 - CR1_THR)
                haircuts.append(h)
    
    if not mids:
        return None
    
    res = {
        'pair': pair, 'exchange': exchange,
        'n_snap': len(mids),
        'mid_price': float(np.mean(mids)),
        'spread_bps': float(np.median(spreads_bps)),
        'bid_depth': float(np.mean(bid_depths)),
        'ask_depth': float(np.mean(ask_depths)),
        'haircut': float(np.mean(haircuts)) if haircuts else 0,
    }
    
    for size in TRADE_SIZES:
        s = slippages[size]
        if len(s) > 5:
            arr = np.array(s) * 10000
            adj = arr * (1 + res['haircut'])
            res.update({
                f's{size}_med': float(np.median(arr)),
                f's{size}_p95': float(np.percentile(arr, 95)),
                f's{size}_adj95': float(np.percentile(adj, 95)),
                f's{size}_n': len(s),
            })
        else:
            res.update({f's{size}_med': np.nan, f's{size}_p95': np.nan,
                       f's{size}_adj95': np.nan, f's{size}_n': len(s)})
    return res


def compute_sar(results, size, alpha=0.95):
    valid = [r for r in results if r and not np.isnan(r.get(f's{size}_med', np.nan))]
    if len(valid) < 3:
        return None
    slips = np.array([r[f's{size}_med'] for r in valid])
    adj = np.array([r.get(f's{size}_adj95', r[f's{size}_p95']) for r in valid])
    sar = float(np.percentile(slips, alpha*100))
    sar_adj = float(np.percentile(adj, alpha*100))
    tail = slips >= sar
    esar = float(np.mean(slips[tail])) if tail.sum() > 0 else sar
    tsar = float(np.sum(slips[tail] / 10000 * size))
    return {
        'n': len(valid), 'median': float(np.median(slips)), 'mean': float(np.mean(slips)),
        'sar': sar, 'sar_adj': sar_adj, 'esar': esar, 'tsar': tsar,
        'adj_pct': (sar_adj/sar-1)*100 if sar > 0 else 0,
        'if_low': tsar*1.5, 'if_high': tsar*3.0,
        'tail_tokens': sorted([(r['pair'],float(s)) for r,s,m in zip(valid,slips,tail) if m], key=lambda x:-x[1])[:10],
    }


# ============================================================
# Main Analysis Loop
# ============================================================
def main():
    t0 = time.time()
    log.info("="*70)
    log.info("SaR FULL STUDY — Cross-Exchange, Cross-Period")
    log.info(f"Exchanges: {list(EXCHANGES.keys())}")
    log.info(f"Periods: {list(PERIODS.keys())}")
    log.info(f"Workers: {N_WORKERS}, Sample rate: 1/{SNAPSHOT_RATE}")
    log.info("="*70)
    
    # Master results: [exchange][period] -> {sar_results, token_results}
    master = defaultdict(dict)
    all_rows = []  # for summary CSV
    
    for period_key, period_info in PERIODS.items():
        month = period_info['month']
        log.info(f"\n{'='*50}")
        log.info(f"Period: {period_key} ({month})")
        log.info(f"{'='*50}")
        
        for exchange, exch_info in EXCHANGES.items():
            exch_dir = os.path.join(OB_DIR, exchange)
            if not os.path.exists(exch_dir):
                continue
            
            # Find valid pairs
            pairs = []
            for p in os.listdir(exch_dir):
                if p.endswith('USDT') and os.path.isdir(os.path.join(exch_dir, p)):
                    pdir = os.path.join(exch_dir, p, month)
                    if os.path.exists(pdir) and len(glob.glob(os.path.join(pdir, "*.csv.gz"))) >= min(SAMPLE_DAYS, 3):
                        pairs.append(p)
            
            if len(pairs) < 3:
                log.info(f"  {exchange}: {len(pairs)} pairs — skipping")
                continue
            
            log.info(f"  {exchange}: {len(pairs)} pairs — analyzing...")
            
            args = [(exchange, p, month, SAMPLE_DAYS, SNAPSHOT_RATE) for p in sorted(pairs)]
            t1 = time.time()
            with Pool(N_WORKERS) as pool:
                results = pool.map(analyze_worker, args)
            token_results = [r for r in results if r]
            elapsed = time.time() - t1
            
            log.info(f"    Analyzed {len(token_results)}/{len(pairs)} in {elapsed:.1f}s")
            
            # Compute SaR for each trade size
            sar_dict = {}
            for size in TRADE_SIZES:
                sar = compute_sar(token_results, size)
                sar_dict[size] = sar
                if sar:
                    all_rows.append({
                        'period': period_key, 'month': month,
                        'event_type': period_info['type'],
                        'exchange': exch_info['short'],
                        'trade_size': size, 'n_tokens': sar['n'],
                        'median_bps': sar['median'], 'sar_95': sar['sar'],
                        'sar_adj_95': sar['sar_adj'], 'adj_pct': sar['adj_pct'],
                        'esar_95': sar['esar'], 'tsar_usd': sar['tsar'],
                        'if_low': sar['if_low'], 'if_high': sar['if_high'],
                    })
            
            master[exchange][period_key] = {
                'sar': sar_dict, 'tokens': token_results,
                'n_pairs': len(pairs), 'n_valid': len(token_results),
            }
            
            # Quick log
            s100k = sar_dict.get(100_000)
            if s100k:
                log.info(f"    $100K: SaR={s100k['sar']:.1f} adj={s100k['sar_adj']:.1f} "
                        f"ESaR={s100k['esar']:.1f} TSaR$={s100k['tsar']:,.0f}")
    
    # Save master summary
    df = pd.DataFrame(all_rows)
    csv_path = f"{RESULTS_DIR}/sar_full_study_{ts}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\nSaved summary: {csv_path}")
    
    # ============================================================
    # Generate Research Figures
    # ============================================================
    log.info("\nGenerating research figures...")
    
    # ---- Figure 1: SaR Time Series (stress vs normal) ----
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Slippage-at-Risk Across Market Regimes\nKaiko LOB L10 | 2020-2023',
                 fontsize=16, fontweight='bold')
    
    for ax_idx, size in enumerate([10_000, 100_000, 500_000, 1_000_000]):
        ax = axes[ax_idx//2, ax_idx%2]
        
        for exchange, exch_info in EXCHANGES.items():
            if exchange not in master:
                continue
            periods_x = []
            sar_vals = []
            sar_adj_vals = []
            colors_pts = []
            
            for pk, pi in PERIODS.items():
                if pk in master[exchange]:
                    s = master[exchange][pk]['sar'].get(size)
                    if s:
                        periods_x.append(pi['label'])
                        sar_vals.append(s['sar'])
                        sar_adj_vals.append(s['sar_adj'])
                        colors_pts.append(pi['color'])
            
            if not periods_x:
                continue
            
            x = range(len(periods_x))
            ax.plot(x, sar_vals, 'o-', color=exch_info['color'], 
                   linewidth=2, markersize=8, label=exch_info['short'])
            ax.plot(x, sar_adj_vals, 's--', color=exch_info['color'],
                   linewidth=1, markersize=6, alpha=0.6)
            
            if exchange == list(EXCHANGES.keys())[0]:
                ax.set_xticks(x)
                ax.set_xticklabels(periods_x, fontsize=8, rotation=0)
        
        # Shade stress periods
        for i, (pk, pi) in enumerate(PERIODS.items()):
            found = False
            for exch in master:
                if pk in master[exch]:
                    found = True
                    break
            if found and pi['type'] == 'stress':
                positions = []
                for j, (pk2, _) in enumerate(PERIODS.items()):
                    if pk2 == pk:
                        positions.append(j)
                for pos in positions:
                    ax.axvspan(pos-0.4, pos+0.4, alpha=0.1, color='red')
        
        ax.set_title(f'Trade Size: ${size:,}', fontsize=12, fontweight='bold')
        ax.set_ylabel('SaR(95%) — bps')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    fig1_path = f"{FIGURES_DIR}/sar_full_regimes_{ts}.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {fig1_path}")
    
    # ---- Figure 2: Stress Amplification ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('Stress Amplification: How Much Worse is Slippage During Crises?\n'
                 'Ratio of Stress SaR to Normal SaR', fontsize=14, fontweight='bold')
    
    # For each exchange, compute stress/normal ratio
    for ax_idx, size in enumerate([10_000, 100_000, 1_000_000]):
        ax = axes[ax_idx]
        
        stress_events = ['COVID_crash', 'China_ban', 'Terra_LUNA', 'FTX_collapse']
        normal_key = 'Normal_2023'
        
        bar_data = {}
        for exchange, exch_info in EXCHANGES.items():
            if exchange not in master:
                continue
            normal_sar = master[exchange].get(normal_key, {}).get('sar', {}).get(size)
            if not normal_sar:
                # Try other normal periods
                for nk in ['Recovery', 'Range_bound', 'Bull_peak']:
                    normal_sar = master[exchange].get(nk, {}).get('sar', {}).get(size)
                    if normal_sar:
                        break
            if not normal_sar:
                continue
            
            ratios = []
            labels = []
            for se in stress_events:
                stress_sar = master[exchange].get(se, {}).get('sar', {}).get(size)
                if stress_sar:
                    ratio = stress_sar['sar'] / normal_sar['sar'] if normal_sar['sar'] > 0 else 0
                    ratios.append(ratio)
                    labels.append(se.replace('_', '\n'))
            
            if ratios:
                bar_data[exch_info['short']] = (labels, ratios, exch_info['color'])
        
        if bar_data:
            n_exch = len(bar_data)
            width = 0.8 / n_exch
            for i, (name, (labels, ratios, color)) in enumerate(bar_data.items()):
                x = np.arange(len(labels))
                ax.bar(x + i*width - (n_exch-1)*width/2, ratios, width, 
                      label=name, color=color, alpha=0.8, edgecolor='white')
            
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Stress / Normal ratio')
            ax.set_title(f'${size:,}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    fig2_path = f"{FIGURES_DIR}/sar_full_stress_amp_{ts}.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {fig2_path}")
    
    # ---- Figure 3: Exchange Liquidity Landscape ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Exchange Liquidity Landscape\nSpread, Depth, Slippage & Concentration',
                 fontsize=14, fontweight='bold')
    
    # Use latest period with most data
    for exchange, exch_info in EXCHANGES.items():
        if exchange not in master:
            continue
        
        # Find period with most tokens
        best_pk = None
        best_n = 0
        for pk in master[exchange]:
            n = master[exchange][pk]['n_valid']
            if n > best_n:
                best_n = n
                best_pk = pk
        
        if not best_pk:
            continue
        
        tokens = master[exchange][best_pk]['tokens']
        valid = [t for t in tokens if t and not np.isnan(t.get('s100000_med', np.nan))]
        if not valid:
            continue
        
        spreads = [t['spread_bps'] for t in valid]
        depths = [t['bid_depth']/1e6 for t in valid]
        slips = [t['s100000_med'] for t in valid]
        haircuts = [t['haircut'] for t in valid]
        
        c = exch_info['color']
        label = f"{exch_info['short']} ({best_pk}, n={len(valid)})"
        
        # Spread distribution
        axes[0,0].hist(spreads, bins=50, alpha=0.5, color=c, label=label)
        # Depth distribution
        axes[0,1].hist([d for d in depths if d < 50], bins=50, alpha=0.5, color=c, label=label)
        # Slippage distribution
        axes[1,0].hist([s for s in slips if s < 200], bins=50, alpha=0.5, color=c, label=label)
        # Depth vs Slippage scatter
        axes[1,1].scatter(depths, slips, c=c, alpha=0.4, s=20, label=label)
    
    axes[0,0].set_xlabel('Spread (bps)'); axes[0,0].set_title('Spread Distribution')
    axes[0,0].set_xlim(0, 50); axes[0,0].legend(fontsize=8)
    axes[0,1].set_xlabel('Bid Depth ($M)'); axes[0,1].set_title('Depth Distribution')
    axes[0,1].legend(fontsize=8)
    axes[1,0].set_xlabel('$100K Slippage (bps)'); axes[1,0].set_title('Slippage Distribution')
    axes[1,0].legend(fontsize=8)
    axes[1,1].set_xlabel('Bid Depth ($M)'); axes[1,1].set_ylabel('$100K Slip (bps)')
    axes[1,1].set_title('Depth vs Slippage'); axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log'); axes[1,1].legend(fontsize=8)
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3_path = f"{FIGURES_DIR}/sar_full_landscape_{ts}.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {fig3_path}")
    
    # ---- Figure 4: Insurance Fund Adequacy ----
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Insurance Fund Adequacy: Required IF by Exchange and Regime\n'
                 'IF* = c · TSaR$ (c ∈ [1.5, 3.0])', fontsize=14, fontweight='bold')
    
    # Heatmap: exchange × period for $100K
    exch_list = [e['short'] for e in EXCHANGES.values() if list(EXCHANGES.keys())[list(EXCHANGES.values()).index(e)] in master]
    period_list = [pk for pk in PERIODS if any(pk in master.get(e,{}) for e in master)]
    
    if exch_list and period_list:
        matrix = np.full((len(exch_list), len(period_list)), np.nan)
        for i, eshort in enumerate(exch_list):
            efull = [k for k,v in EXCHANGES.items() if v['short']==eshort][0]
            for j, pk in enumerate(period_list):
                s = master.get(efull,{}).get(pk,{}).get('sar',{}).get(100_000)
                if s:
                    matrix[i,j] = s['if_high']  # conservative estimate
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(period_list)))
        ax.set_xticklabels([PERIODS[pk]['label'] for pk in period_list], fontsize=9)
        ax.set_yticks(range(len(exch_list)))
        ax.set_yticklabels(exch_list, fontsize=11)
        plt.colorbar(im, ax=ax, label='Required IF ($) for $100K trades')
        
        for i in range(len(exch_list)):
            for j in range(len(period_list)):
                v = matrix[i,j]
                if not np.isnan(v):
                    ax.text(j, i, f'${v:,.0f}', ha='center', va='center', fontsize=8,
                           color='white' if v > np.nanmedian(matrix) else 'black')
    
    plt.tight_layout()
    fig4_path = f"{FIGURES_DIR}/sar_full_insurance_{ts}.png"
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {fig4_path}")
    
    # ---- Figure 5: Concentration Impact ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Concentration Haircut Impact on SaR\n'
                 'How much does liquidity concentration inflate risk estimates?',
                 fontsize=14, fontweight='bold')
    
    # Left: adj% by trade size across exchanges
    for exchange, exch_info in EXCHANGES.items():
        if exchange not in master:
            continue
        # Use latest period
        best_pk = max(master[exchange].keys(), key=lambda pk: master[exchange][pk]['n_valid'])
        sizes_plot = []
        adj_pcts = []
        for size in TRADE_SIZES:
            s = master[exchange][best_pk]['sar'].get(size)
            if s:
                sizes_plot.append(size)
                adj_pcts.append(s['adj_pct'])
        if sizes_plot:
            axes[0].plot(sizes_plot, adj_pcts, 'o-', color=exch_info['color'],
                        label=exch_info['short'], linewidth=2, markersize=8)
    
    axes[0].set_xlabel('Trade Size (USD)')
    axes[0].set_ylabel('Concentration Adjustment (%)')
    axes[0].set_title('Concentration Premium by Trade Size')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Right: adj% across periods for Binance $100K
    binance_key = 'Binance Futures'
    if binance_key in master:
        pks = []
        adjs = []
        colors = []
        for pk, pi in PERIODS.items():
            if pk in master[binance_key]:
                s = master[binance_key][pk]['sar'].get(100_000)
                if s:
                    pks.append(pi['label'])
                    adjs.append(s['adj_pct'])
                    colors.append(pi['color'])
        
        if pks:
            axes[1].bar(range(len(pks)), adjs, color=colors, edgecolor='white', alpha=0.8)
            axes[1].set_xticks(range(len(pks)))
            axes[1].set_xticklabels(pks, fontsize=8)
            axes[1].set_ylabel('Concentration Adjustment (%)')
            axes[1].set_title('Binance $100K: Concentration Premium by Regime')
            axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig5_path = f"{FIGURES_DIR}/sar_full_concentration_{ts}.png"
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved: {fig5_path}")
    
    # Print final summary table
    log.info(f"\n{'='*70}")
    log.info("FINAL SUMMARY — $100K Trade Size")
    log.info(f"{'='*70}")
    
    summary_100k = df[df['trade_size']==100_000].sort_values(['exchange','month'])
    if len(summary_100k) > 0:
        log.info(f"\n{summary_100k[['exchange','period','n_tokens','median_bps','sar_95','sar_adj_95','adj_pct','esar_95','tsar_usd']].to_string()}")
    
    total = time.time() - t0
    log.info(f"\n{'='*70}")
    log.info(f"FULL STUDY COMPLETE — {total:.1f}s ({total/60:.1f} min)")
    log.info(f"Output: {csv_path}")
    log.info(f"Figures: {FIGURES_DIR}/sar_full_*_{ts}.png")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
