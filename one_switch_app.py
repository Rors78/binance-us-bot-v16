
import os, time, json, math, random
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="One-Switch — USDT Only (v16.1: Micro-Capital Mode)", layout="wide")
st.title("🦅 One-Switch — USDT Only (v16.1: Micro-Capital Mode)")

# -------------- Sidebar: keys + live only --------------
with st.sidebar:
    st.header("🔐 Keys")
    api_key = st.text_input("BINANCEUS_KEY", type="password")
    api_secret = st.text_input("BINANCEUS_SECRET", type="password")
    live_mode = st.toggle("Live Mode", value=False, help="Paper by default. Flip when ready.")
    st.caption("Micro mode tuned for ~$70–$100 bankroll. Auto-scales as it grows.")

# -------------- Imports --------------
try:
    import ccxt
    import keyring
except Exception:
    st.error("Missing dependency: ccxt. Install with: `pip install -r requirements.txt`")
    st.stop()

# -------------- File paths --------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
PEAK_FILE = os.path.join(DATA_DIR, "equity_peak.json")
BASELINE_FILE = os.path.join(DATA_DIR, "equity_baseline.json")
POSITIONS_CSV = os.path.join(DATA_DIR, "positions.csv")
LEDGER_CSV = os.path.join(DATA_DIR, "paper_trades.csv")

# -------------- Tuned constants (no UI knobs) --------------
TF = "5m"
LIMIT = 500
TOP_N_SCAN = 20                # micro mode: scan top-20 USDT by 24h quote volume
RSI_MIN = 55
DON_LB = 20
ADX_LEN = 14
ADX_MIN = 18
ATR_LEN = 14
ATR_K = 1.5
VOL_K = 1.5
TP_SPLITS = (0.5, 0.3, 0.2)    # 50/30/20
TP_R = (1.0, 1.5, 2.0)         # 1R/1.5R/2R for initial targets
TRAIL_ATR = 1.0                # trailing distance after TP1

TAKER_FEE = 0.0010             # 0.10%
SLIPPAGE = 0.0005              # 0.05%
SPREAD_MAX = 0.20              # a bit more permissive but still sane (%)
MIN_TOPLIQ_USD = 10000         # micro accounts: lower depth threshold

# Dynamic risk & clamps for micro
MIN_RISK_PCT = 0.0035          # 0.35% base floor
MAX_RISK_PCT = 0.015           # 1.5% max
MIN_RISK_USD_HARD = 0.50       # never risk less than 50 cents per entry
MAX_RISK_USD_HARD = 6.00       # cap single-trade risk (auto grows later)

ONE_PER_BASE = True
MAX_NEW_ENTRIES_PER_CYCLE = 1  # micro: 1 entry max per 30s cycle

# -------------- Utils --------------
def ema(s: pd.Series, span: int): return s.ewm(span=span, adjust=False).mean()
def rsi(series: pd.Series, period: int = 14):
    d = series.diff(); g = (d.where(d>0,0)).rolling(period).mean(); l = (-d.where(d<0,0)).rolling(period).mean()
    rs = g / (l.replace(0, np.nan)); return 100 - (100 / (1 + rs))
def atr(df: pd.DataFrame, length: int = 14):
    h,l,c = df["high"], df["low"], df["close"]; pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(length).mean()
def don_hi(df: pd.DataFrame, lb: int): return df["high"].shift(1).rolling(lb).max()
def vol_sma(s: pd.Series, n=20): return s.rolling(n).mean()

def dm_pm(df: pd.DataFrame):
    up = df["high"].diff(); dn = -df["low"].diff()
    plus_dm = np.where((up>dn) & (up>0), up, 0.0)
    minus_dm = np.where((dn>up) & (dn>0), dn, 0.0)
    return pd.Series(plus_dm, index=df.index), pd.Series(minus_dm, index=df.index)

def adx(df: pd.DataFrame, length: int = 14):
    trv = atr(df, length) * length
    plus_dm, minus_dm = dm_pm(df)
    plus_di = 100 * (plus_dm.rolling(length).sum() / trv.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(length).sum() / trv.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(length).mean()

def backoff_sleep(i): time.sleep(min(6 + 2*i, 20) + random.random())

def with_retry(fn, *args, retries=5, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == retries - 1:
                raise
            backoff_sleep(i)

# -------------- Exchange & market --------------
def ex(api_key, api_secret):
    return ccxt.binanceus({
        "apiKey": api_key or None,
        "secret": api_secret or None,
        "enableRateLimit": True,
        "timeout": 35000,
    })

def fetch_df(e, sym, tf=TF, limit=LIMIT):
    o = with_retry(e.fetch_ohlcv, sym, timeframe=tf, limit=limit)
    df = pd.DataFrame(o, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

def compute(df):
    df = df.copy()
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr"] = atr(df, ATR_LEN)
    df["don_hi"] = don_hi(df, DON_LB)
    df["vol_sma20"] = vol_sma(df["volume"], 20)
    df["adx"] = adx(df, ADX_LEN)
    return df

def signal(df):
    if len(df) < 210: return False
    L = df.iloc[-1]
    trend = (L["close"] > L["ema200"]) and (L["ema50"] > L["ema200"])
    momo  = L["rsi14"] >= RSI_MIN
    brk   = L["close"] > L["don_hi"]
    volok = (L["volume"] > L["vol_sma20"] * VOL_K) if not np.isnan(L["vol_sma20"]) else True
    regime = (df["adx"].iloc[-1] >= ADX_MIN) if not np.isnan(df["adx"].iloc[-1]) else True
    return bool(trend and momo and brk and volok and regime)

def fetch_ticker(e, sym):
    try:
        return with_retry(e.fetch_ticker, sym)
    except Exception:
        return {}

def spread_depth_ok(e, sym, last):
    try:
        ob = with_retry(e.fetch_order_book, sym, limit=10)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not bid or not ask: return False
        spread = (ask - bid) / ((ask + bid)/2) * 100.0
        if spread > SPREAD_MAX: return False
        top_bids = sum(p*q for p,q in ob["bids"][:3])
        top_asks = sum(p*q for p,q in ob["asks"][:3])
        if (top_bids + top_asks) * (last or 1) < MIN_TOPLIQ_USD: return False
        return True
    except Exception:
        return True

# -------------- Equity, risk, P&L --------------
def balances(e):
    try:
        b = with_retry(e.fetch_balance)
        tot = b.get("total", {})
        wallet = float(tot.get("USD", 0.0)) + float(tot.get("USDT", 0.0))
        coins = {k:v for k,v in tot.items() if k not in ("USD","USDT") and v and v>0}
        return wallet, coins
    except Exception:
        return 100.0, {}

def load_peak():
    try:
        return float(json.load(open(PEAK_FILE)).get("peak_equity", 0.0))
    except Exception:
        return 0.0

def save_peak(val: float):
    try:
        json.dump({"peak_equity": float(val)}, open(PEAK_FILE, "w"))
    except Exception:
        pass

def base_risk(eq):
    # Micro tilt: slightly higher base % in the $70–$100 range but still conservative
    if eq <= 2000: return 0.0075  # 0.75%
    if eq <= 10000: return 0.01
    if eq <= 50000: return 0.01
    if eq <= 150000: return 0.0075
    return 0.005

def dyn_risk_pct(eq, peak):
    if peak <= 0: peak = eq
    dd = max(0.0, 1.0 - (eq/peak))
    rpct = base_risk(eq) * (1 - 0.5*dd)
    return float(min(MAX_RISK_PCT, max(MIN_RISK_PCT, rpct)))

def dynamic_daily_risk_cap(eq):
    # At least $2; roughly 2% of equity as it grows; max $20
    return float(min(20.0, max(2.0, round(eq * 0.02, 2))))

def session_baseline(equity_now):
    try:
        if os.path.exists(BASELINE_FILE):
            b = json.load(open(BASELINE_FILE))
        else:
            b = {}
        if "baseline" not in b or b.get("reset", False):
            b = {"baseline": equity_now, "reset": False}
            json.dump(b, open(BASELINE_FILE, "w"))
        return float(b["baseline"])
    except Exception:
        return equity_now

def daily_risk_in_play():
    if not os.path.exists(LEDGER_CSV): return 0.0
    try:
        df = pd.read_csv(LEDGER_CSV)
        if df.empty: return 0.0
        df["d"] = pd.to_datetime(df["time"]).dt.date.astype(str)
        t = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today = df[df["d"] == t]
        if today.empty: return 0.0
        risk = ((today["entry"] - today["sl"]).abs() * today["qty"]).sum()
        return float(risk)
    except Exception:
        return 0.0

# -------------- Position mgmt --------------
def ensure_positions():
    if not os.path.exists(POSITIONS_CSV):
        pd.DataFrame(columns=[
            "symbol","entry","qty_total","qty_open","sl","tp1","tp2","tp3",
            "atr","r","state","tp1_hit","tp2_hit","tp3_hit","mode"
        ]).to_csv(POSITIONS_CSV, index=False)

def load_positions():
    ensure_positions()
    try:
        df = pd.read_csv(POSITIONS_CSV)
        return df if not df.empty else pd.DataFrame(columns=[
            "symbol","entry","qty_total","qty_open","sl","tp1","tp2","tp3",
            "atr","r","state","tp1_hit","tp2_hit","tp3_hit","mode"
        ])
    except Exception:
        return pd.DataFrame(columns=[
            "symbol","entry","qty_total","qty_open","sl","tp1","tp2","tp3",
            "atr","r","state","tp1_hit","tp2_hit","tp3_hit","mode"
        ])

def save_positions(df): df.to_csv(POSITIONS_CSV, index=False)

def place_market(e, sym, side, qty):
    try:
        return with_retry(e.create_order, symbol=sym, type="market", side=side, amount=qty)
    except Exception as exn:
        return {"error": str(exn)}

def manage_positions(e):
    df = load_positions()
    if df.empty: return
    changed = False
    for i,row in df.iterrows():
        if row.get("state","open") != "open": continue
        tk = fetch_ticker(e, row["symbol"])
        price = tk.get("last") or 0.0
        if price <= 0: continue

        # TP1 → 50% out, BE shift, start trailing
        if not row["tp1_hit"] and price >= row["tp1"]:
            row["tp1_hit"] = True
            sell = row["qty_total"] * TP_SPLITS[0]
            row["qty_open"] = max(0.0, row["qty_open"] - sell)
            row["sl"] = max(row["sl"], row["entry"])
            changed = True
            if row.get("mode") == "live":
                place_market(e, row["symbol"], "sell", sell)

        # TP2 → 30% out
        if not row["tp2_hit"] and price >= row["tp2"]:
            row["tp2_hit"] = True
            sell = row["qty_total"] * TP_SPLITS[1]
            row["qty_open"] = max(0.0, row["qty_open"] - sell)
            changed = True
            if row.get("mode") == "live":
                place_market(e, row["symbol"], "sell", sell)

        # TP3 → close remainder
        if not row["tp3_hit"] and price >= row["tp3"]:
            row["tp3_hit"] = True
            sell = row["qty_open"]
            row["qty_open"] = 0.0
            row["state"] = "closed"
            changed = True
            if row.get("mode") == "live" and sell > 0:
                place_market(e, row["symbol"], "sell", sell)

        # Trail after TP1
        if row["tp1_hit"] and row["atr"] > 0:
            trail_sl = price - TRAIL_ATR * row["atr"]
            if trail_sl > row["sl"]:
                row["sl"] = trail_sl
                changed = True

        # Stop hit
        if price <= row["sl"] and row["qty_open"] > 0:
            sell = row["qty_open"]
            row["qty_open"] = 0.0
            row["state"] = "stopped"
            changed = True
            if row.get("mode") == "live":
                place_market(e, row["symbol"], "sell", sell)

        df.iloc[i] = row

    if changed: save_positions(df)

# -------------- Universe selection --------------
def all_usdt_symbols(e):
    tks = with_retry(e.fetch_tickers)
    syms = []
    for s, t in tks.items():
        if not s.endswith("/USDT"): continue
        last = t.get("last") or 0.0
        qv = t.get("quoteVolume") or (t.get("baseVolume") or 0.0) * last
        syms.append((s, qv or 0.0))
    syms.sort(key=lambda x: x[1], reverse=True)
    e.load_markets()
    ranked = [s for s,_ in syms if s in e.markets]
    return ranked

# -------------- Main scan --------------
def scan_and_trade(e, live_mode):
    manage_positions(e)

    wallet, _ = balances(e)
    peak = load_peak()
    if wallet > peak:
        peak = wallet; save_peak(peak)

    # Dynamic daily cap
    daily_cap = dynamic_daily_risk_cap(wallet)
    if daily_risk_in_play() >= daily_cap:
        return wallet, [], f"Paused new entries: daily risk cap reached (~${daily_cap})."

    ranked = all_usdt_symbols(e)[:TOP_N_SCAN]

    open_bases = set()
    new_entries = []
    for sym in ranked:
        if len(new_entries) >= MAX_NEW_ENTRIES_PER_CYCLE:
            break
        base = sym.split("/")[0]
        if ONE_PER_BASE and base in open_bases:
            continue

        tk = fetch_ticker(e, sym)
        last = tk.get("last") or 0.0
        if last <= 0: continue
        if not spread_depth_ok(e, sym, last):
            continue

        df = compute(fetch_df(e, sym, TF, LIMIT))
        if not signal(df):
            continue

        L = df.iloc[-1]
        px = float(L["close"]); atrv = float(L["atr"] or 0.0)
        if atrv <= 0:
            continue

        # Micro risk sizing with auto-scale
        rpct = dyn_risk_pct(wallet, peak)
        sl = px - ATR_K * atrv
        risk_per_unit = max(px - sl, 1e-9) * (1 + TAKER_FEE + SLIPPAGE)
        # Risk dollars: at least 50c; as wallet grows, increases naturally via rpct*wallet, capped for micro
        risk_usd = max(MIN_RISK_USD_HARD, wallet * rpct)
        # allow more risk if wallet grows; cap softly for micro accounts, cap lifts as wallet increases
        dynamic_cap = max(2.0, min(MAX_RISK_USD_HARD, wallet * 0.03))  # ~3% soft cap up to $6
        risk_usd = min(risk_usd, dynamic_cap)
        qty = risk_usd / risk_per_unit
        if qty <= 0:
            continue

        note = "paper"; order_id = None
        if live_mode and api_key and api_secret:
            res = place_market(e, sym, "buy", qty)
            order_id = res.get("id") if isinstance(res, dict) else None
            note = "live" if not ("error" in res) else f"live_error: {res.get('error')}"

        # Register position
        ensure_positions()
        pos = pd.DataFrame([{
            "symbol": sym, "entry": round(px,8),
            "qty_total": round(qty,8), "qty_open": round(qty,8),
            "sl": round(sl,8),
            "tp1": round(px + (px - sl)*TP_R[0],8),
            "tp2": round(px + (px - sl)*TP_R[1],8),
            "tp3": round(px + (px - sl)*TP_R[2],8),
            "atr": round(atrv,8), "r": round(px - sl,8),
            "state": "open", "tp1_hit": False, "tp2_hit": False, "tp3_hit": False,
            "mode": note
        }])
        existing = load_positions()
        allp = pd.concat([existing, pos], ignore_index=True)
        save_positions(allp)

        # Ledger row
        row = {
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": sym, "side": "buy", "qty": round(qty,8),
            "entry": round(px,8), "sl": round(sl,8),
            "tp1": round(px + (px - sl)*TP_R[0],8),
            "tp2": round(px + (px - sl)*TP_R[1],8),
            "tp3": round(px + (px - sl)*TP_R[2],8),
            "exit_price": None, "exit_reason": None, "pnl_usd": None,
            "note": note
        }
        header = not os.path.exists(LEDGER_CSV)
        pd.DataFrame([row]).to_csv(LEDGER_CSV, mode="a", header=header, index=False)

        new_entries.append({"symbol": sym, "price": px, "qty": qty, "risk_usd": risk_usd, "mode": note, "order_id": order_id})
        open_bases.add(base)

    return wallet, new_entries, None



# ---- Health monitor & failsafes ----
RECOVERY_HEALTH_OKS = 3  # auto-resume after N consecutive healthy checks
HEALTH_FILE = os.path.join(DATA_DIR, "health.json")

def _load_health():
    try:
        return json.load(open(HEALTH_FILE))
    except Exception:
        return {"fail_streak": 0, "recovery_streak": 0, "paused": False, "reason": ""}

def _save_health(h):
    try:
        json.dump(h, open(HEALTH_FILE, "w"))
    except Exception:
        pass

def health_check(e):
    """
    Returns (ok: bool, reason: str). Also manages fail_streak and pause state.
    """
    h = _load_health()
    try:
        # Basic market reachability
        tks = with_retry(e.fetch_tickers)()
        if not isinstance(tks, dict) or not tks:
            raise RuntimeError("tickers empty")
        # BTC/USDT sanity: OHLCV recency and price non-zero
        sym = "BTC/USDT" if "BTC/USDT" in tks else next((s for s in tks if s.endswith("/USDT")), None)
        if not sym:
            raise RuntimeError("no USDT symbols")
        ohlcv = e.fetch_ohlcv(sym, timeframe="5m", limit=3)
        if not ohlcv or not isinstance(ohlcv, list):
            raise RuntimeError("ohlcv empty")
        last_ts = int(ohlcv[-1][0])  # ms
        now_ms = int(time.time()*1000)
        # recency: last candle within 30 minutes
        if (now_ms - last_ts) > (30*60*1000):
            raise RuntimeError("stale candles >30m")
        # price sanity
        last_close = float(ohlcv[-1][4] or 0.0)
        if not (last_close > 0):
            raise RuntimeError("bad price <= 0")
        # orderbook sanity (spread check)
        ob = e.fetch_order_book(sym, limit=5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if (bid is None) or (ask is None) or (ask <= bid):
            raise RuntimeError("orderbook invalid")
        spread_pct = (ask - bid) / ((ask + bid)/2) * 100.0
        if spread_pct > 0.5:  # BTC should be tighter than 0.5%
            raise RuntimeError("btc spread too wide")
        # Passed all checks
        h["fail_streak"] = 0
        h["recovery_streak"] = int(h.get("recovery_streak", 0)) + 1
        # auto-resume once we have enough clean cycles
        if h.get("paused") and h["recovery_streak"] >= RECOVERY_HEALTH_OKS:
            h["paused"] = False
            h["reason"] = ""
        _save_health(h)
        return True, ""
    except Exception as ex:
        h["fail_streak"] = int(h.get("fail_streak", 0)) + 1
        h["recovery_streak"] = 0
        reason = f"{type(ex).__name__}: {ex}"
        # Auto-pause if too many consecutive failures
        if h["fail_streak"] >= 3:
            h["paused"] = True
            h["reason"] = f"Health check failed {h['fail_streak']}x: {reason}"
        _save_health(h)
        return False, reason

def health_state():
    h = _load_health()
    return bool(h.get("paused", False)), h.get("reason", ""), int(h.get("fail_streak", 0)), int(h.get("recovery_streak", 0))

def health_reset():
    _save_health({"fail_streak": 0, "paused": False, "reason": ""})

# -------------- App flow --------------
def main():
    # sanitize inputs
    global api_key, api_secret
    api_key = sanitize_key(api_key)
    api_secret = sanitize_key(api_secret)
    # persist if asked
    try:
        if 'remember' in globals() and remember and api_key and api_secret:
            save_keys(api_key, api_secret)
    except Exception:
        pass
    e = ex(api_key, api_secret)

    wallet, coins = balances(e)
    baseline = session_baseline(wallet)
    session_pnl = wallet - baseline

    try:
        wallet2, entries, paused_msg = scan_and_trade(e, live_mode)
    except Exception as ex:
        wallet2, entries, paused_msg = wallet, [], f"Temporary issue: {type(ex).__name__}. Auto-retrying next cycle."
    upnl, open_rows = estimate_unrealized_pnl(e)

    # UI
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P&L (session)", f"${session_pnl:,.2f}")
    m2.metric("CurrentWallet (USD/USDT)", f"${wallet2:,.2f}")
    m3.metric("Open uPnL (est.)", f"${upnl:,.2f}")
    wr, w_ct, l_ct = compute_win_rate()
    m4.metric("Win Rate", f"{wr:.1f}%", f"W:{w_ct} / L:{l_ct}")
    if paused_msg:
        st.warning(paused_msg)

    st.subheader("CurrentCoins")
    if coins:
        st.dataframe(pd.DataFrame([{"asset": k, "amount": v} for k,v in coins.items()]))
    else:
        st.info("No non-zero coin balances detected.")

    st.subheader("New Entries (this cycle)")
    if entries:
        st.dataframe(pd.DataFrame(entries))
    else:
        st.write("No new entries this cycle.")

    st.subheader("Open Positions")
    posdf = load_positions()
    if not posdf.empty:
        st.dataframe(posdf)
    else:
        st.write("No open positions.")

    paused, why, streak, rec = health_state()
    if paused:
        st.error(f"Health: PAUSED — {why} (fail streak: {streak})")
    elif rec > 0:
        st.info(f"Health: recovering ({rec}/{RECOVERY_HEALTH_OKS} clean cycles)…")
    else:
        st.success("Health: OK")
    st.caption("Auto-refresh ~30s. Ctrl+C in terminal to stop.")
    time.sleep(30)
    st.experimental_rerun()

# ---- PnL estimation ----
def get_price(e, sym):
    tk = fetch_ticker(e, sym); return float(tk.get("last") or 0.0)

def estimate_unrealized_pnl(e):
    df = load_positions()
    if df.empty: return 0.0, []
    upnl = 0.0; rows = []
    for _, r in df.iterrows():
        if r.get("state","open") != "open" or r.get("qty_open",0) <= 0:
            continue
        price = get_price(e, r["symbol"])
        if price <= 0: continue
        pnl = (price - r["entry"]) * r["qty_open"]
        upnl += pnl
        rows.append({"symbol": r["symbol"], "qty_open": r["qty_open"], "entry": r["entry"], "last": price, "uPnL": pnl})
    return upnl, rows


# ---- Win rate (stealth simple) ----
def compute_win_rate():
    try:
        if not os.path.exists(POSITIONS_CSV):
            return 0.0, 0, 0
        df = pd.read_csv(POSITIONS_CSV)
        if df.empty:
            return 0.0, 0, 0
        wins = (df["state"] == "closed").sum()
        losses = (df["state"] == "stopped").sum()
        total = int(wins + losses)
        if total == 0:
            return 0.0, 0, 0
        wr = float(wins) / total * 100.0
        return wr, int(wins), int(losses)
    except Exception:
        return 0.0, 0, 0

if __name__ == "__main__":
    main()
