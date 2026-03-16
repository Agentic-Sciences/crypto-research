#!/usr/bin/env python3
"""
Funding Rate Dynamics: Hyperliquid (DEX) vs OKX (CEX)
Cross-Exchange Comparison & Predictability Analysis

Data sources:
- Hyperliquid API: hourly funding rates (2023-05 → 2026-03)
- OKX API: 8-hourly funding rates (2025-12 → 2026-03)
- Hyperliquid daily candles: 50 coins × 3 years

Output:
- figures/fr_research_*.png
- results/fr_research_*.csv

Author: Qihong Ruan, Cornell / Agentic Sciences
Date: 2026-03-15
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === Style ===
plt.style.use('dark_background')
COLORS = ['#6366f1','#34d399','#f87171','#fbbf24','#60a5fa','#a78bfa',
          '#fb923c','#e879f9','#22d3ee','#84cc16']

BASE = '/mnt/work/qr33/comewealth'

# ============================
# LOAD DATA
# ============================
print("Loading data...")

# Funding rates
with open(f'{BASE}/cache/hyperliquid/funding_rates_hl_okx.json') as f:
    fr_data = json.load(f)

# Daily candles
with open(f'{BASE}/cache/hyperliquid/hl_data.json') as f:
    candle_data = json.load(f)

# --- Parse Hyperliquid funding rates ---
hl_frames = {}
for coin, records in fr_data['hyperliquid'].items():
    if not records:
        continue
    rows = []
    for r in records:
        rows.append({
            'timestamp': pd.Timestamp(r['time'], unit='ms', tz='UTC'),
            'funding_rate': float(r['fundingRate']),
            'premium': float(r['premium'])
        })
    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    hl_frames[coin] = df

# --- Parse OKX funding rates ---
okx_frames = {}
for coin, records in fr_data['okx'].items():
    if not records:
        continue
    rows = []
    for r in records:
        rows.append({
            'timestamp': pd.Timestamp(int(r['fundingTime']), unit='ms', tz='UTC'),
            'funding_rate': float(r['fundingRate']),
            'realized_rate': float(r.get('realizedRate', r['fundingRate']))
        })
    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    okx_frames[coin] = df

# --- Parse daily candles ---
candle_frames = {}
for coin, records in candle_data['daily_candles'].items():
    if not records:
        continue
    rows = []
    for r in records:
        rows.append({
            'timestamp': pd.Timestamp(r['t'], unit='ms', tz='UTC'),
            'open': float(r['o']),
            'high': float(r['h']),
            'low': float(r['l']),
            'close': float(r['c']),
            'volume': float(r['v'])
        })
    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    df['volume_usd'] = df['volume'] * df['close']
    candle_frames[coin] = df

print(f"Hyperliquid FR: {len(hl_frames)} coins")
print(f"OKX FR: {len(okx_frames)} coins")
print(f"Daily candles: {len(candle_frames)} coins")
for coin in ['BTC','ETH','SOL']:
    if coin in hl_frames:
        print(f"  HL {coin}: {len(hl_frames[coin])} records, {hl_frames[coin].index[0].date()} → {hl_frames[coin].index[-1].date()}")

# ============================
# FIGURE 1: BTC FUNDING RATE TIME SERIES (HL vs OKX)
# ============================
print("\nGenerating Figure 1: BTC Funding Rate Comparison...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

# Panel A: Full Hyperliquid BTC FR history
ax1 = fig.add_subplot(gs[0, :])
btc_hl = hl_frames['BTC']
# Resample to daily for smoother plot
btc_daily_fr = btc_hl['funding_rate'].resample('1D').mean()
btc_7d_fr = btc_hl['funding_rate'].resample('1D').mean().rolling(7).mean()
btc_30d_fr = btc_hl['funding_rate'].resample('1D').mean().rolling(30).mean()

ax1.fill_between(btc_daily_fr.index, btc_daily_fr.values * 100,
                  alpha=0.3, color=COLORS[0], label='Daily avg')
ax1.plot(btc_7d_fr.index, btc_7d_fr.values * 100,
         color=COLORS[1], lw=1.5, label='7-day MA')
ax1.plot(btc_30d_fr.index, btc_30d_fr.values * 100,
         color=COLORS[3], lw=2, label='30-day MA')
ax1.axhline(0, color='white', alpha=0.3, ls='--', lw=0.5)
ax1.set_ylabel('Funding Rate (%)')
ax1.set_title('A. Hyperliquid BTC Perpetual Funding Rate (Hourly → Daily Avg)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.set_xlim(btc_daily_fr.index[0], btc_daily_fr.index[-1])

# Panel B: HL vs OKX overlap period
ax2 = fig.add_subplot(gs[1, :])
# Find overlap period
btc_okx = okx_frames.get('BTC', pd.DataFrame())
if len(btc_okx) > 0:
    overlap_start = btc_okx.index[0]
    overlap_end = btc_okx.index[-1]
    
    hl_overlap = btc_hl.loc[overlap_start:overlap_end, 'funding_rate']
    # Resample HL to 8h to match OKX
    hl_8h = hl_overlap.resample('8h').mean()
    
    ax2.plot(btc_okx.index, btc_okx['funding_rate'].values * 100,
             color=COLORS[2], lw=1, alpha=0.8, label='OKX (8h)')
    ax2.plot(hl_8h.index, hl_8h.values * 100,
             color=COLORS[0], lw=1, alpha=0.8, label='Hyperliquid (8h avg)')
    
    # Spread
    merged = pd.merge(hl_8h.rename('hl'), btc_okx['funding_rate'].rename('okx'),
                       left_index=True, right_index=True, how='inner')
    spread = (merged['hl'] - merged['okx']) * 100
    ax2.fill_between(merged.index, spread.values, alpha=0.2, color=COLORS[3],
                      label=f'Spread (HL-OKX): μ={spread.mean():.4f}%')
    
    ax2.axhline(0, color='white', alpha=0.3, ls='--', lw=0.5)
    ax2.set_ylabel('Funding Rate (%)')
    ax2.set_title(f'B. Hyperliquid vs OKX BTC Funding Rate ({overlap_start.date()} → {overlap_end.date()})',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Panel C: Distribution comparison
ax3 = fig.add_subplot(gs[2, 0])
if len(btc_okx) > 0 and len(merged) > 0:
    ax3.hist(merged['hl'].values * 100, bins=80, alpha=0.6, color=COLORS[0],
             label=f'Hyperliquid (μ={merged["hl"].mean()*100:.4f}%)', density=True)
    ax3.hist(merged['okx'].values * 100, bins=80, alpha=0.6, color=COLORS[2],
             label=f'OKX (μ={merged["okx"].mean()*100:.4f}%)', density=True)
    ax3.set_xlabel('Funding Rate (%)')
    ax3.set_ylabel('Density')
    ax3.set_title('C. Funding Rate Distribution (Overlap Period)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

# Panel D: Autocorrelation
ax4 = fig.add_subplot(gs[2, 1])
max_lag = 48  # 48 hours
acf_vals = []
for lag in range(1, max_lag+1):
    acf_vals.append(btc_hl['funding_rate'].autocorr(lag))
ax4.bar(range(1, max_lag+1), acf_vals, color=COLORS[0], alpha=0.7)
ax4.axhline(0, color='white', alpha=0.3)
ax4.axhline(1.96/np.sqrt(len(btc_hl)), color=COLORS[3], ls='--', alpha=0.5, label='95% CI')
ax4.axhline(-1.96/np.sqrt(len(btc_hl)), color=COLORS[3], ls='--', alpha=0.5)
ax4.set_xlabel('Lag (hours)')
ax4.set_ylabel('Autocorrelation')
ax4.set_title('D. BTC Funding Rate Autocorrelation (Hourly)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)

fig.suptitle('Funding Rate Dynamics: Hyperliquid (DEX) vs OKX (CEX)',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{BASE}/figures/fr_research_btc_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("  Saved: fr_research_btc_comparison.png")

# ============================
# FIGURE 2: CROSS-COIN FUNDING RATE ANALYSIS
# ============================
print("Generating Figure 2: Cross-Coin Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel A: Cross-coin average FR over time
ax = axes[0, 0]
for i, coin in enumerate(['BTC','ETH','SOL','DOGE','AVAX','XRP']):
    if coin in hl_frames and len(hl_frames[coin]) > 100:
        daily = hl_frames[coin]['funding_rate'].resample('1D').mean().rolling(14).mean()
        ax.plot(daily.index, daily.values * 100, color=COLORS[i], lw=1.2, label=coin, alpha=0.8)
ax.axhline(0, color='white', alpha=0.3, ls='--')
ax.set_ylabel('14-day MA Funding Rate (%)')
ax.set_title('A. Cross-Coin Funding Rates (Hyperliquid)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Panel B: Annualized carry by coin (latest 3 months)
ax = axes[0, 1]
carry_data = []
for coin in hl_frames:
    df = hl_frames[coin]
    if len(df) < 100:
        continue
    recent = df.tail(min(2160, len(df)))  # last ~90 days of hourly
    ann_fr = recent['funding_rate'].mean() * 8760 * 100  # annualized %
    vol = candle_frames[coin]['return'].tail(90).std() * np.sqrt(365) * 100 if coin in candle_frames else np.nan
    carry_data.append({'coin': coin, 'ann_carry': ann_fr, 'vol': vol})

carry_df = pd.DataFrame(carry_data).dropna().sort_values('ann_carry', ascending=True)
colors_bar = [COLORS[1] if x > 0 else COLORS[2] for x in carry_df['ann_carry']]
ax.barh(carry_df['coin'], carry_df['ann_carry'], color=colors_bar, alpha=0.7)
ax.axvline(0, color='white', alpha=0.3)
ax.set_xlabel('Annualized Funding Rate (%)')
ax.set_title('B. Annualized Carry (Recent)', fontsize=12, fontweight='bold')

# Panel C: FR-Return correlation
ax = axes[1, 0]
# For BTC: lagged FR vs next-day return
btc_candles = candle_frames.get('BTC', pd.DataFrame())
if len(btc_candles) > 50:
    btc_fr_daily = hl_frames['BTC']['funding_rate'].resample('1D').mean()
    merged_btc = pd.merge(btc_fr_daily.rename('fr'),
                           btc_candles['return'].rename('ret'),
                           left_index=True, right_index=True, how='inner')
    # Lagged: today's FR → tomorrow's return
    merged_btc['ret_next'] = merged_btc['ret'].shift(-1)
    merged_btc = merged_btc.dropna()
    
    ax.scatter(merged_btc['fr'] * 100, merged_btc['ret_next'] * 100,
               alpha=0.15, s=8, color=COLORS[0])
    # Bin analysis
    merged_btc['fr_bin'] = pd.qcut(merged_btc['fr'], 10, labels=False, duplicates='drop')
    bin_means = merged_btc.groupby('fr_bin').agg({'fr': 'mean', 'ret_next': 'mean'})
    ax.scatter(bin_means['fr'] * 100, bin_means['ret_next'] * 100,
               s=80, color=COLORS[3], zorder=5, edgecolor='white', lw=1.5,
               label='Decile means')
    
    # Regression
    slope, intercept, r, p, se = stats.linregress(merged_btc['fr'], merged_btc['ret_next'])
    x_line = np.linspace(merged_btc['fr'].min(), merged_btc['fr'].max(), 100)
    ax.plot(x_line * 100, (slope * x_line + intercept) * 100, color=COLORS[2], ls='--',
            label=f'β={slope:.2f}, p={p:.4f}')
    
    ax.axhline(0, color='white', alpha=0.2)
    ax.axvline(0, color='white', alpha=0.2)
    ax.set_xlabel('Today\'s Avg Funding Rate (%)')
    ax.set_ylabel('Next-Day Return (%)')
    ax.set_title('C. Funding Rate → Next-Day Return (BTC)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

# Panel D: Cross-coin FR correlation heatmap
ax = axes[1, 1]
coins_with_data = [c for c in ['BTC','ETH','SOL','DOGE','AVAX','XRP','ADA','LINK','DOT','LTC']
                   if c in hl_frames and len(hl_frames[c]) > 500]

if len(coins_with_data) >= 3:
    # Use daily average FR
    fr_daily_dict = {}
    for coin in coins_with_data:
        fr_daily_dict[coin] = hl_frames[coin]['funding_rate'].resample('1D').mean()
    
    fr_panel = pd.DataFrame(fr_daily_dict).dropna(how='all')
    # Only keep dates where at least 3 coins have data
    fr_panel = fr_panel.dropna(thresh=3)
    corr = fr_panel.corr()
    
    im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-0.3, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if not np.isnan(corr.values[i, j]):
                ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(corr.values[i,j]) < 0.5 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('D. Cross-Coin Funding Rate Correlation', fontsize=12, fontweight='bold')

fig.suptitle('Cross-Coin Funding Rate Analysis on Hyperliquid',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{BASE}/figures/fr_research_cross_coin.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("  Saved: fr_research_cross_coin.png")

# ============================
# FIGURE 3: PREDICTABILITY & TRADING STRATEGY
# ============================
print("Generating Figure 3: Predictability & Strategy...")

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

# Panel A: FR regime analysis (positive vs negative)
ax = fig.add_subplot(gs[0, 0])
btc_fr_hourly = hl_frames['BTC']['funding_rate']
# Rolling regime
btc_fr_24h = btc_fr_hourly.rolling(24).mean()
positive_pct = (btc_fr_24h > 0).resample('1M').mean() * 100

ax.bar(positive_pct.index, positive_pct.values, width=20, color=COLORS[1], alpha=0.7,
       label='% hours with positive FR')
ax.bar(positive_pct.index, 100 - positive_pct.values, bottom=positive_pct.values,
       width=20, color=COLORS[2], alpha=0.7, label='% hours with negative FR')
ax.axhline(50, color='white', ls='--', alpha=0.3)
ax.set_ylabel('Percentage (%)')
ax.set_title('A. Monthly FR Regime (24h MA)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Panel B: Cumulative carry strategy
ax = fig.add_subplot(gs[0, 1])
btc_cum_fr = btc_fr_hourly.cumsum() * 100
ax.plot(btc_cum_fr.index, btc_cum_fr.values, color=COLORS[0], lw=1.5)
ax.fill_between(btc_cum_fr.index, btc_cum_fr.values, alpha=0.2, color=COLORS[0])
ax.set_ylabel('Cumulative Funding (%)')
ax.set_title('B. Cumulative BTC Funding (Short Perp Carry)', fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Panel C: Mean reversion test — binned FR → next period FR
ax = fig.add_subplot(gs[1, 0])
btc_fr = btc_fr_hourly.to_frame('fr')
btc_fr['fr_next'] = btc_fr['fr'].shift(-1)
btc_fr['fr_8h'] = btc_fr['fr'].rolling(8).mean()
btc_fr = btc_fr.dropna()

# Bin by current 8h average FR
btc_fr['bin'] = pd.qcut(btc_fr['fr_8h'], 20, labels=False, duplicates='drop')
bin_analysis = btc_fr.groupby('bin').agg(
    current_fr=('fr_8h', 'mean'),
    next_fr=('fr_next', 'mean'),
    count=('fr', 'count')
)
ax.scatter(bin_analysis['current_fr'] * 100, bin_analysis['next_fr'] * 100,
           s=bin_analysis['count'] / 20, color=COLORS[0], alpha=0.8, edgecolor='white')
slope_mr, intercept_mr, r_mr, p_mr, se_mr = stats.linregress(
    bin_analysis['current_fr'], bin_analysis['next_fr'])
x_mr = np.linspace(bin_analysis['current_fr'].min(), bin_analysis['current_fr'].max(), 100)
ax.plot(x_mr * 100, (slope_mr * x_mr + intercept_mr) * 100, color=COLORS[2], ls='--',
        label=f'β={slope_mr:.3f} (p={p_mr:.2e})')
# 45-degree line
ax.plot(x_mr * 100, x_mr * 100, color=COLORS[3], ls=':', alpha=0.5, label='45° (persistence)')
ax.set_xlabel('Current 8h Avg FR (%)')
ax.set_ylabel('Next-Hour FR (%)')
ax.set_title('C. FR Persistence: 8h Avg → Next Hour', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# Panel D: Volatility clustering in FR
ax = fig.add_subplot(gs[1, 1])
btc_fr_abs = btc_fr_hourly.abs().resample('1D').mean()
btc_rv = candle_frames['BTC']['return'].abs() if 'BTC' in candle_frames else pd.Series()

if len(btc_rv) > 50:
    merged_vol = pd.merge(btc_fr_abs.rename('fr_vol'),
                           btc_rv.rename('ret_vol'),
                           left_index=True, right_index=True, how='inner')
    ax.scatter(merged_vol['ret_vol'] * 100, merged_vol['fr_vol'] * 100,
               alpha=0.3, s=12, color=COLORS[0])
    
    corr_val = merged_vol['fr_vol'].corr(merged_vol['ret_vol'])
    ax.set_xlabel('Daily |Return| (%)')
    ax.set_ylabel('Daily Avg |FR| (%)')
    ax.set_title(f'D. FR Volatility vs Price Volatility (ρ={corr_val:.3f})',
                 fontsize=12, fontweight='bold')

# Panel E: Hour-of-day FR pattern
ax = fig.add_subplot(gs[2, 0])
btc_hourly = btc_fr_hourly.copy()
btc_hourly_df = btc_hourly.to_frame('fr')
btc_hourly_df['hour'] = btc_hourly_df.index.hour
hourly_pattern = btc_hourly_df.groupby('hour')['fr'].agg(['mean', 'std', 'count'])
hourly_pattern['se'] = hourly_pattern['std'] / np.sqrt(hourly_pattern['count'])

ax.bar(hourly_pattern.index, hourly_pattern['mean'] * 100, 
       yerr=hourly_pattern['se'] * 100 * 1.96,
       color=COLORS[0], alpha=0.7, capsize=2)
ax.axhline(0, color='white', alpha=0.3, ls='--')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Avg Funding Rate (%)')
ax.set_title('E. Intraday FR Pattern (Hour of Day)', fontsize=12, fontweight='bold')
ax.set_xticks(range(0, 24, 2))

# Panel F: FR distribution — heavy tails
ax = fig.add_subplot(gs[2, 1])
fr_vals = btc_fr_hourly.values * 100
ax.hist(fr_vals, bins=200, density=True, alpha=0.7, color=COLORS[0], label='Empirical')
# Fit normal
mu, sigma = np.mean(fr_vals), np.std(fr_vals)
x_norm = np.linspace(fr_vals.min(), fr_vals.max(), 200)
ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), color=COLORS[2], lw=2,
        label=f'Normal (μ={mu:.4f}, σ={sigma:.4f})')
kurtosis = stats.kurtosis(fr_vals)
skewness = stats.skew(fr_vals)
ax.set_xlabel('Funding Rate (%)')
ax.set_ylabel('Density')
ax.set_title(f'F. FR Distribution (kurt={kurtosis:.1f}, skew={skewness:.2f})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(-0.1, 0.1)

fig.suptitle('Funding Rate Predictability & Microstructure',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{BASE}/figures/fr_research_predictability.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("  Saved: fr_research_predictability.png")

# ============================
# FIGURE 4: HL vs OKX CROSS-COIN COMPARISON
# ============================
print("Generating Figure 4: HL vs OKX Cross-Coin...")

# For overlap period, compare all 16 coins
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel A: Scatter of avg FR (HL vs OKX) per coin
ax = axes[0, 0]
scatter_data = []
for coin in okx_frames:
    if coin in hl_frames:
        okx_df = okx_frames[coin]
        hl_df = hl_frames[coin]
        overlap_start = okx_df.index[0]
        overlap_end = okx_df.index[-1]
        hl_overlap = hl_df.loc[overlap_start:overlap_end]
        if len(hl_overlap) > 10:
            hl_mean = hl_overlap['funding_rate'].mean() * 8760 * 100
            okx_mean = okx_df['funding_rate'].mean() * 3 * 365 * 100  # 8h intervals, 3x/day
            scatter_data.append({'coin': coin, 'hl': hl_mean, 'okx': okx_mean})

if scatter_data:
    sdf = pd.DataFrame(scatter_data)
    ax.scatter(sdf['okx'], sdf['hl'], s=80, color=COLORS[0], edgecolor='white', lw=1, zorder=5)
    for _, row in sdf.iterrows():
        ax.annotate(row['coin'], (row['okx'], row['hl']), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
    
    lim = max(abs(sdf[['hl','okx']].values).max() * 1.2, 10)
    ax.plot([-lim, lim], [-lim, lim], 'w--', alpha=0.3, label='45° (perfect alignment)')
    ax.set_xlabel('OKX Annualized FR (%)')
    ax.set_ylabel('Hyperliquid Annualized FR (%)')
    ax.set_title('A. Avg Funding Rate: Hyperliquid vs OKX', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

# Panel B: FR spread histogram (HL - OKX) across all coins
ax = axes[0, 1]
all_spreads = []
for coin in okx_frames:
    if coin in hl_frames:
        okx_df = okx_frames[coin]
        hl_df = hl_frames[coin]
        hl_8h = hl_df['funding_rate'].resample('8h').mean()
        merged = pd.merge(hl_8h.rename('hl'), okx_df['funding_rate'].rename('okx'),
                           left_index=True, right_index=True, how='inner')
        if len(merged) > 10:
            spread = (merged['hl'] - merged['okx']) * 100
            all_spreads.extend(spread.values)

if all_spreads:
    ax.hist(all_spreads, bins=100, density=True, alpha=0.7, color=COLORS[0])
    ax.axvline(np.mean(all_spreads), color=COLORS[3], ls='--',
               label=f'Mean spread: {np.mean(all_spreads):.4f}%')
    ax.set_xlabel('FR Spread (HL - OKX) (%)')
    ax.set_ylabel('Density')
    ax.set_title('B. Cross-Exchange FR Spread Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

# Panel C: Time-varying correlation between HL and OKX
ax = axes[1, 0]
if 'BTC' in hl_frames and 'BTC' in okx_frames:
    btc_okx = okx_frames['BTC']
    btc_hl_8h = hl_frames['BTC']['funding_rate'].resample('8h').mean()
    merged_btc = pd.merge(btc_hl_8h.rename('hl'), btc_okx['funding_rate'].rename('okx'),
                           left_index=True, right_index=True, how='inner')
    if len(merged_btc) > 20:
        rolling_corr = merged_btc['hl'].rolling(30).corr(merged_btc['okx'])
        ax.plot(rolling_corr.index, rolling_corr.values, color=COLORS[0], lw=1.5)
        ax.axhline(merged_btc['hl'].corr(merged_btc['okx']), color=COLORS[3], ls='--',
                   label=f'Full-sample ρ={merged_btc["hl"].corr(merged_btc["okx"]):.3f}')
        ax.set_ylabel('Rolling 30-period Correlation')
        ax.set_title('C. HL-OKX BTC FR Correlation (Rolling)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Panel D: Summary stats table
ax = axes[1, 1]
ax.axis('off')
stats_rows = []
for coin in ['BTC','ETH','SOL','XRP','DOGE','LINK','ADA','DOT']:
    if coin in hl_frames:
        hl_df = hl_frames[coin]
        hl_mean = hl_df['funding_rate'].mean() * 8760 * 100
        hl_std = hl_df['funding_rate'].std() * np.sqrt(8760) * 100
        hl_kurt = stats.kurtosis(hl_df['funding_rate'].values)
        n = len(hl_df)
        stats_rows.append([coin, f'{n:,}', f'{hl_mean:.2f}%', f'{hl_std:.1f}%', f'{hl_kurt:.1f}'])

if stats_rows:
    table = ax.table(cellText=stats_rows,
                     colLabels=['Coin', 'N (hourly)', 'Ann. FR', 'Ann. Vol', 'Kurtosis'],
                     cellLoc='center', loc='center',
                     colColours=['#1a1a2e']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#2a2a3e')
        if row == 0:
            cell.set_facecolor('#2a2a3e')
            cell.set_text_props(fontweight='bold', color='white')
        else:
            cell.set_facecolor('#12121a')
            cell.set_text_props(color='#e4e4ef')
    ax.set_title('D. Hyperliquid FR Summary Statistics', fontsize=12, fontweight='bold', pad=20)

fig.suptitle('DEX vs CEX Funding Rate Comparison',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{BASE}/figures/fr_research_dex_vs_cex.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0f')
plt.close()
print("  Saved: fr_research_dex_vs_cex.png")

# ============================
# SAVE RESULTS CSV
# ============================
print("\nSaving results CSV...")

# Summary statistics
all_stats = []
for coin in hl_frames:
    df = hl_frames[coin]
    if len(df) < 100:
        continue
    fr = df['funding_rate']
    all_stats.append({
        'exchange': 'Hyperliquid',
        'coin': coin,
        'n_records': len(df),
        'start_date': str(df.index[0].date()),
        'end_date': str(df.index[-1].date()),
        'mean_fr': fr.mean(),
        'std_fr': fr.std(),
        'ann_fr_pct': fr.mean() * 8760 * 100,
        'ann_vol_pct': fr.std() * np.sqrt(8760) * 100,
        'skewness': stats.skew(fr.values),
        'kurtosis': stats.kurtosis(fr.values),
        'pct_positive': (fr > 0).mean() * 100,
        'max_fr': fr.max(),
        'min_fr': fr.min(),
        'autocorr_1h': fr.autocorr(1),
        'autocorr_8h': fr.autocorr(8),
        'autocorr_24h': fr.autocorr(24),
    })

for coin in okx_frames:
    df = okx_frames[coin]
    if len(df) < 10:
        continue
    fr = df['funding_rate']
    all_stats.append({
        'exchange': 'OKX',
        'coin': coin,
        'n_records': len(df),
        'start_date': str(df.index[0].date()),
        'end_date': str(df.index[-1].date()),
        'mean_fr': fr.mean(),
        'std_fr': fr.std(),
        'ann_fr_pct': fr.mean() * 3 * 365 * 100,
        'ann_vol_pct': fr.std() * np.sqrt(3*365) * 100,
        'skewness': stats.skew(fr.values),
        'kurtosis': stats.kurtosis(fr.values),
        'pct_positive': (fr > 0).mean() * 100,
        'max_fr': fr.max(),
        'min_fr': fr.min(),
        'autocorr_1h': np.nan,
        'autocorr_8h': fr.autocorr(1),
        'autocorr_24h': fr.autocorr(3),
    })

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(f'{BASE}/results/fr_research_stats.csv', index=False)
print(f"  Saved: fr_research_stats.csv ({len(stats_df)} rows)")

# ============================
# KEY FINDINGS
# ============================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

btc_hl = hl_frames['BTC']['funding_rate']
print(f"\n1. BTC Funding Rate (Hyperliquid, {len(btc_hl):,} hourly obs):")
print(f"   Mean: {btc_hl.mean()*100:.5f}% per hour ({btc_hl.mean()*8760*100:.2f}% annualized)")
print(f"   Std:  {btc_hl.std()*100:.5f}% per hour ({btc_hl.std()*np.sqrt(8760)*100:.1f}% annualized)")
print(f"   Positive: {(btc_hl>0).mean()*100:.1f}% of hours")
print(f"   Kurtosis: {stats.kurtosis(btc_hl.values):.1f} (excess)")
print(f"   AC(1h)={btc_hl.autocorr(1):.3f}, AC(8h)={btc_hl.autocorr(8):.3f}, AC(24h)={btc_hl.autocorr(24):.3f}")

if 'BTC' in okx_frames and len(merged_btc) > 10:
    print(f"\n2. BTC HL-OKX Comparison ({len(merged_btc)} overlapping obs):")
    print(f"   Correlation: {merged_btc['hl'].corr(merged_btc['okx']):.3f}")
    spread = merged_btc['hl'] - merged_btc['okx']
    print(f"   Spread (HL-OKX): mean={spread.mean()*100:.5f}%, std={spread.std()*100:.4f}%")
    print(f"   Spread > 0: {(spread>0).mean()*100:.1f}% of time")

print("\nDone!")
