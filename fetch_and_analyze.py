#!/usr/bin/env python3
"""
Crypto Market Microstructure Research
Fetch real data from Binance, OKX, Hyperliquid and generate charts
"""
import json, urllib.request, os, sys
from datetime import datetime, timedelta
import ssl

ctx = ssl.create_default_context()

def fetch_json(url, method="GET", data=None, headers=None):
    """Fetch JSON from URL"""
    if headers is None:
        headers = {"User-Agent": "Mozilla/5.0"}
    if data:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={**headers, "Content-Type": "application/json"}, method=method)
    else:
        req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=30, context=ctx)
        return json.loads(resp.read())
    except Exception as e:
        print(f"  Error fetching {url[:80]}: {e}")
        return None

print("=" * 60)
print("CRYPTO MARKET DATA COLLECTION")
print("=" * 60)

os.makedirs("data", exist_ok=True)

# ============================================================
# 1. BINANCE — Klines (OHLCV) for major assets
# ============================================================
print("\n[1/6] Binance: Fetching historical klines...")
assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
binance_data = {}
for symbol in assets:
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=500"
    data = fetch_json(url)
    if data:
        binance_data[symbol] = data
        print(f"  {symbol}: {len(data)} hourly candles")

with open("data/binance_klines.json", "w") as f:
    json.dump(binance_data, f)

# ============================================================
# 2. BINANCE — 24hr ticker stats
# ============================================================
print("\n[2/6] Binance: Fetching 24hr tickers...")
tickers = fetch_json("https://data-api.binance.vision/api/v3/ticker/24hr")
if tickers:
    usdt_tickers = [t for t in tickers if t["symbol"].endswith("USDT")]
    usdt_tickers.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    with open("data/binance_tickers.json", "w") as f:
        json.dump(usdt_tickers[:100], f)
    print(f"  Got {len(usdt_tickers)} USDT pairs, saved top 100")

# ============================================================
# 3. OKX — Funding rates (current + history)
# ============================================================
print("\n[3/6] OKX: Fetching funding rates...")
okx_instruments = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP", "XRP-USDT-SWAP"]
okx_funding = {}
for inst in okx_instruments:
    # Current funding rate
    url = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst}"
    current = fetch_json(url)
    # Historical funding rates
    url2 = f"https://www.okx.com/api/v5/public/funding-rate-history?instId={inst}&limit=100"
    history = fetch_json(url2)
    okx_funding[inst] = {
        "current": current.get("data", []) if current else [],
        "history": history.get("data", []) if history else []
    }
    if history and history.get("data"):
        print(f"  {inst}: {len(history['data'])} historical funding rates")

with open("data/okx_funding.json", "w") as f:
    json.dump(okx_funding, f)

# ============================================================
# 4. OKX — Open Interest
# ============================================================
print("\n[4/6] OKX: Fetching open interest...")
okx_oi = {}
for inst in okx_instruments:
    url = f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={inst}"
    data = fetch_json(url)
    if data and data.get("data"):
        okx_oi[inst] = data["data"]
        print(f"  {inst}: OI = {data['data'][0].get('oi', 'N/A')}")

with open("data/okx_oi.json", "w") as f:
    json.dump(okx_oi, f)

# ============================================================
# 5. OKX — Historical candlesticks
# ============================================================
print("\n[5/6] OKX: Fetching mark price candles...")
okx_candles = {}
for inst in okx_instruments:
    url = f"https://www.okx.com/api/v5/market/mark-price-candles?instId={inst}&bar=1H&limit=300"
    data = fetch_json(url)
    if data and data.get("data"):
        okx_candles[inst] = data["data"]
        print(f"  {inst}: {len(data['data'])} candles")

with open("data/okx_candles.json", "w") as f:
    json.dump(okx_candles, f)

# ============================================================
# 6. Hyperliquid — Meta + Asset Contexts
# ============================================================
print("\n[6/6] Hyperliquid: Fetching via Tencent Cloud relay...")
# Hyperliquid is blocked from Cornell, use Tencent Cloud
import subprocess
try:
    result = subprocess.run([
        "ssh", "-i", os.path.expanduser("~/.ssh/tencent_cloud"), 
        "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        "root@170.106.73.110",
        "curl -s -X POST https://api.hyperliquid.xyz/info -H 'Content-Type: application/json' -d '{\"type\":\"metaAndAssetCtxs\"}'"
    ], capture_output=True, text=True, timeout=30)
    if result.returncode == 0 and result.stdout:
        hl_data = json.loads(result.stdout)
        with open("data/hyperliquid_meta.json", "w") as f:
            json.dump(hl_data, f)
        n_assets = len(hl_data[0]["universe"]) if isinstance(hl_data, list) and len(hl_data) > 0 else 0
        print(f"  Got {n_assets} perpetual contracts")
    else:
        print(f"  SSH failed: {result.stderr[:100]}")
except Exception as e:
    print(f"  Error: {e}")

# Also get Hyperliquid funding history for top assets
try:
    for coin in ["BTC", "ETH", "SOL"]:
        result = subprocess.run([
            "ssh", "-i", os.path.expanduser("~/.ssh/tencent_cloud"),
            "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            "root@170.106.73.110",
            f'curl -s -X POST https://api.hyperliquid.xyz/info -H "Content-Type: application/json" -d \'{{"type":"fundingHistory","coin":"{coin}","startTime":{int((datetime.utcnow()-timedelta(days=7)).timestamp()*1000)}}}\''
        ], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout:
            hl_funding = json.loads(result.stdout)
            with open(f"data/hyperliquid_funding_{coin.lower()}.json", "w") as f:
                json.dump(hl_funding, f)
            print(f"  {coin} funding history: {len(hl_funding)} entries")
except Exception as e:
    print(f"  Funding history error: {e}")

print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETE")
print("=" * 60)
