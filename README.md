# One-Switch Bot v16.1 - Micro-Capital Mode

Streamlit-based trading bot optimized for small accounts ($70-$100 starting capital).

## 🇺🇸 USA Regulatory Compliant

**MADE FOR US TRADERS - HARD TO FIND, FULLY COMPLIANT**

This bot is specifically designed to meet US regulatory requirements:
- ✅ **SPOT TRADING ONLY** (no futures, no derivatives)
- ✅ **LONG POSITIONS ONLY** (no shorting)
- ✅ **NO LEVERAGE** (100% compliant with US regulations)
- ✅ **US EXCHANGES ONLY** (Binance US)
- ✅ **Regulatory Compliant** for US retail traders

**Perfect for US traders with micro-capital accounts!** One of the few truly USA-compliant automated trading bots available.

---

## Features

- **Micro-Capital Optimized**: Tuned for accounts under $100
- **Streamlit UI**: Beautiful web interface
- **Paper Trading**: Safe testing mode (default)
- **Live Trading**: Toggle when ready
- **Auto-Scaling**: Grows risk limits as account grows
- **Multi-Indicator**: RSI, Donchian, ADX, ATR, Volume
- **Partial Exits**: 50/30/20 splits at 1R/1.5R/2R
- **Trailing Stops**: ATR-based after first TP

## Configuration

- **Timeframe**: 5m candles
- **Scan**: Top 20 USDT pairs by volume
- **Risk**: 0.35% - 1.5% per trade
- **Min Trade**: $0.50
- **Max Trade**: $6.00 (auto-scales up)
- **Entry Limit**: 1 per 30s cycle

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run one_switch_app.py
```

Or double-click `RunBot.bat` (Windows)

Then:
1. Enter API keys in sidebar
2. Toggle "Live Mode" when ready (starts in paper mode)
3. Monitor positions in real-time

## Entry Conditions

- RSI > 55
- Price breaks 20-period Donchian high
- ADX > 18 (trending)
- Volume > 1.5x average
- Spread < 0.20%
- Top liquidity > $10,000

## Exit Strategy

- **TP1** (50%): 1.0R (1x ATR)
- **TP2** (30%): 1.5R
- **TP3** (20%): 2.0R
- **Trailing**: 1.0 ATR after TP1
- **Stop Loss**: ATR-based

## Safety Features

- Paper mode default
- One position per base asset
- Max 1 entry per cycle
- Spread validation
- Liquidity checks
- Fee and slippage modeling

## File Structure

- `data/positions.csv` - Active positions
- `data/paper_trades.csv` - Trade ledger
- `data/equity_peak.json` - Peak tracking
- `data/equity_baseline.json` - Baseline tracking

## ⚠️ Warning

For micro-capital accounts only. Use paper mode first!

## License

Personal use only.
