# ComeWealth — Autonomous Crypto Market Research Agent

> 🏆 **Synthesis Hackathon 2026** submission by [Agentic Sciences](https://agentic-sciences.github.io/agentic-sciences.github.io/)

## What is ComeWealth?

ComeWealth (来财) is a **fully autonomous quantitative research agent** that runs 24/7 on Cornell research servers, analyzing cryptocurrency market microstructure using institutional-grade data. It generates daily research reports, discovers trading signals, and produces publication-quality analysis — all without human intervention.

**The agent is real.** It has been running continuously since March 2026, producing original research across crypto and traditional finance. This is not a demo or prototype — it's a production research system.

## 🔑 Core Capabilities

### 1. Slippage-at-Risk (SaR) Framework
*A novel liquidity risk measure for crypto markets*
- Quantifies execution cost risk across **5 exchanges × 8 market regimes × 5 trade sizes**
- Key finding: Binance SaR(95%) = 37.5 bps @ $100K; OKX = 299 bps (**8x worse**)
- FTX collapse: SaR jumped to 1,667 bps (184% stress amplification)
- Full study: 2,600+ token-period combinations analyzed

📁 `scripts/sar/` · 📊 `figures/sar/`

### 2. Cross-Exchange Funding Rate Analysis
*DEX vs CEX funding rate dynamics*
- Hyperliquid BTC annualized FR: 15.9% (88.7% positive, autocorrelation 0.88)
- OKX BTC: only 3.4% annualized — massive DEX premium
- Cross-coin patterns: funding rate predictability analysis

📁 `scripts/funding_rate/` · 📊 `figures/funding_rate/`

### 3. Market Microstructure Analysis
*Order book dynamics, price impact, and market quality*
- Bid-ask spreads, order flow imbalance, Kyle's lambda estimation
- Cross-exchange spread analysis and arbitrage opportunity detection
- Order book wall detection and depth profiling

📁 `scripts/microstructure/` · 📊 `figures/microstructure/`

### 4. Spot-Futures Basis Monitoring
*Real-time basis tracking across exchanges*
- Annualized basis rates for BTC/ETH perpetual futures
- Cross-exchange basis arbitrage signal generation
- Historical basis distribution and regime detection

📁 `scripts/basis/` · 📊 `figures/basis/`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│               ComeWealth Agent                   │
│          (OpenClaw + Claude Opus 4.6)            │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │  research3    │    │  research1 (SSH)     │   │
│  │  - Kaiko HF   │◄──►│  - TAQ (US equities) │   │
│  │  - Order books │    │  - WRDS databases    │   │
│  │  - Trades      │    │  - NASDAQ data       │   │
│  └──────────────┘    └──────────────────────┘   │
│         │                      │                 │
│         ▼                      ▼                 │
│  ┌──────────────────────────────────────────┐   │
│  │           Analysis Engine                 │   │
│  │  - SaR (Slippage-at-Risk)                │   │
│  │  - Funding Rate Research                  │   │
│  │  - Microstructure Analysis                │   │
│  │  - Basis Monitoring                       │   │
│  │  - Cross-Asset Correlation                │   │
│  └──────────────────────────────────────────┘   │
│         │                                        │
│         ▼                                        │
│  ┌──────────────────────────────────────────┐   │
│  │  Output: Scripts, Figures, CSVs, Papers   │   │
│  │  → GitHub Pages · Telegram · Google Drive │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
└─────────────────────────────────────────────────┘
```

## 🤖 Agent Identity

- **Name**: ComeWealth (来财)
- **Harness**: OpenClaw
- **Model**: Claude Opus 4.6
- **ERC-8004 Identity**: [Base Mainnet](https://basescan.org/tx/0xe646502e085c7cb4301d18bfdd820d6c7270e18e7525df6251505884e5aa5725)
- **Runs on**: Cornell Johnson School research servers (24/7)
- **Human**: Qihong Ruan (Cornell PhD, Quantitative Finance)

## 📊 Research Output (Sample)

| Research Area | Scripts | Figures | Data Points |
|---|---|---|---|
| SaR Framework | 5 | 7 | 2,600+ exchange-period combos |
| Funding Rates | 1 | 4 | Cross-exchange, multi-coin |
| Microstructure | 1 | 1 | Order book + trade analysis |
| Basis Monitoring | 2 | 2 | Daily cross-exchange |
| **Total** | **9+** | **14+** | **Terabytes processed** |

## 🔬 Data Sources

- **Kaiko** (institutional crypto data): Spot trades, futures, order book snapshots across 50+ exchanges
- **TAQ** (NYSE): US equity high-frequency tick data
- **WRDS**: Compustat, CRSP, IBES, RavenPack
- **Hyperliquid**: DEX perpetual futures and funding rates
- **OKX/Binance/Bybit/Deribit**: CEX market data via Kaiko

## 🏃 How to Run

```bash
# SaR Quick Analysis (requires Kaiko data)
python scripts/sar/sar_framework_v1.py

# Funding Rate Research
python scripts/funding_rate/funding_rate_research.py

# BTC Microstructure
python scripts/microstructure/btc_microstructure_analysis.py

# Basis Monitor
python scripts/basis/basis_monitor.py
```

**Dependencies**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

## 📝 Conversation Log

ComeWealth has been operating autonomously since early March 2026. Key milestones:

1. **Mar 3-4**: Initial Kaiko data exploration, first order book analysis
2. **Mar 7**: Spot-futures basis monitoring system built
3. **Mar 8**: Full microstructure paper generated (BTC cross-exchange analysis)
4. **Mar 9**: Registered for Synthesis hackathon via ERC-8004
5. **Mar 13**: SaR Phase 1 — stress event analysis (Terra/Luna, FTX collapse)
6. **Mar 14**: SaR Full Study — 5 exchanges × 8 periods × 5 trade sizes
7. **Mar 15**: Funding rate research — DEX vs CEX comparison
8. **Mar 16**: GitHub sync and hackathon submission

The agent operates on a daily research loop:
- Morning: arXiv scan for new papers
- Continuous: Market monitoring (basis, funding rates, liquidity)
- Evening: Results collection, figure generation, memory update
- Proactive: Discovers and investigates anomalies independently

## 📜 License

MIT

## 🔗 Links

- **Research Lab**: [Agentic Sciences](https://agentic-sciences.github.io/agentic-sciences.github.io/)
- **Live Research**: [Crypto Research Dashboard](https://agentic-sciences.github.io/crypto-research/)
- **SaR Research**: [Slippage-at-Risk](https://agentic-sciences.github.io/crypto-research/sar-research/)
- **Prediction Markets**: [Cross-Platform Analysis](https://agentic-sciences.github.io/prediction-markets/)
