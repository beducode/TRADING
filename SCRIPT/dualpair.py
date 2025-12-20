import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================
SYMBOLS = ["XAUUSDm", "BTCUSDm"]

TIMEFRAME_SIGNAL  = mt5.TIMEFRAME_M15
TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M1

LOT = 0.01
RR = 2.0
MAX_POSITIONS = 1
POLL_SECONDS = 1.0

SMA_LOW_PERIOD   = 30
SMA_HIGH_PERIOD  = 60
SMA_TREND_PERIOD = 100
SMA_DISTANCE = 0.5

BREAKEVENT_START = 2
TRAIL_START_PROFIT = 4

RSI_PERIOD = 8
RSI_MID = 50

ADX_PERIOD = 8
ADX_THRESHOLD = 30
ADX_SHIFT = 1

MAGIC = 234000

# ==================================================
# LOGGING
# ==================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("SCALPER")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "MULTI_PAIR.log"),
    maxBytes=5*1024*1024,
    backupCount=3
)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

# ==================================================
# STATE PER SYMBOL
# ==================================================
symbol_state = {
    sym: {
        "buy_signal": False,
        "sell_signal": False,
        "prev_profit": 0,
        "breakevent": False,
        "sma_locked": False
    } for sym in SYMBOLS
}

# ==================================================
# INDICATORS
# ==================================================
def sma(series, period):
    return series.rolling(period).mean()

def calculate_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(ADX_SHIFT)).abs(),
        (low - close.shift(ADX_SHIFT)).abs()
    ], axis=1).max(axis=1)

    dm_plus = np.where((high - high.shift()) > (low.shift() - low),
                       np.maximum(high - high.shift(), 0), 0)
    dm_minus = np.where((low.shift() - low) > (high - high.shift()),
                        np.maximum(low.shift() - low, 0), 0)

    tr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    dp_s = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean()
    dm_s = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean()

    di_plus = 100 * dp_s / tr_s
    di_minus = 100 * dm_s / tr_s
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

# ==================================================
# SIGNAL FUNCTIONS
# ==================================================
def get_sma_signal(df):
    close = df['close']
    open_ = df['open']

    fast = sma(close, SMA_LOW_PERIOD)
    slow = sma(close, SMA_HIGH_PERIOD)
    trend = sma(close, SMA_TREND_PERIOD)

    p_fast, c_fast = fast.iloc[-3], fast.iloc[-2]
    p_slow, c_slow = slow.iloc[-3], slow.iloc[-2]

    curr = open_.iloc[-2]
    dist = abs(p_fast - p_slow)

    if c_slow > trend.iloc[-2] and c_fast > c_slow and curr > c_fast and dist >= SMA_DISTANCE:
        return "BUY"

    if c_slow < trend.iloc[-2] and c_fast < c_slow and curr < c_fast and dist >= SMA_DISTANCE:
        return "SELL"

    return "WAIT"

def get_rsi_signal(df):
    rsi = calculate_rsi(df['close'], RSI_PERIOD)
    val = rsi.iloc[-2]
    if val > RSI_MID:
        return "BUY"
    if val < RSI_MID:
        return "SELL"
    return "WAIT"

def get_adx_signal(df):
    adx = calculate_adx(df, ADX_PERIOD)
    return adx.iloc[-2] > ADX_THRESHOLD

# ==================================================
# MT5 HELPERS
# ==================================================
def get_rates(symbol, tf, n=500):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None:
        return None
    return pd.DataFrame(rates)

def positions_for_symbol(symbol):
    pos = mt5.positions_get(symbol=symbol)
    return [] if pos is None else list(pos)

def place_order(symbol, order_type, price, sl, tp):
    mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "type": order_type,
        "volume": LOT,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC,
        "comment": f"{symbol} ENTRY"
    })

# ==================================================
# POSITION MANAGEMENT
# ==================================================
def manage_position(symbol, df, state):
    pos = positions_for_symbol(symbol)
    if not pos:
        state["sma_locked"] = False
        state["prev_profit"] = 0
        state["breakevent"] = False
        return

    p = pos[0]
    tick = mt5.symbol_info_tick(symbol)

    if p.type == mt5.POSITION_TYPE_BUY:
        profit = tick.bid - p.price_open
        if profit >= BREAKEVENT_START and not state["breakevent"]:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": p.price_open,
                "tp": p.tp
            })
            state["breakevent"] = True

        if profit >= TRAIL_START_PROFIT and profit > state["prev_profit"]:
            new_sl = p.sl + (profit - state["prev_profit"])
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": new_sl,
                "tp": p.tp
            })
        state["prev_profit"] = profit

    if p.type == mt5.POSITION_TYPE_SELL:
        profit = p.price_open - tick.ask
        if profit >= BREAKEVENT_START and not state["breakevent"]:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": p.price_open,
                "tp": p.tp
            })
            state["breakevent"] = True

        if profit >= TRAIL_START_PROFIT and profit > state["prev_profit"]:
            new_sl = p.sl - (profit - state["prev_profit"])
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": new_sl,
                "tp": p.tp
            })
        state["prev_profit"] = profit

# ==================================================
# MAIN
# ==================================================
def main():
    if not mt5.initialize():
        logger.error("MT5 INIT FAILED")
        return

    logger.info("MULTI-PAIR SCALPER STARTED")

    while True:
        for symbol in SYMBOLS:
            state = symbol_state[symbol]

            df_sig = get_rates(symbol, TIMEFRAME_SIGNAL)
            df_con = get_rates(symbol, TIMEFRAME_CONFIRM)
            if df_sig is None or df_con is None:
                continue

            positions = positions_for_symbol(symbol)
            manage_position(symbol, df_con, state)

            if len(positions) >= MAX_POSITIONS or state["sma_locked"]:
                continue

            sma_sig = get_sma_signal(df_sig)
            rsi_sig = get_rsi_signal(df_sig)
            adx_ok  = get_adx_signal(df_sig)

            tick = mt5.symbol_info_tick(symbol)
            bid, ask = tick.bid, tick.ask

            if sma_sig == rsi_sig == "BUY" and adx_ok:
                slow = sma(df_sig['close'], SMA_HIGH_PERIOD).iloc[-2]
                sl = slow + 0.01
                tp = ask + (ask - sl) * RR
                place_order(symbol, mt5.ORDER_TYPE_BUY, ask, sl, tp)
                state["sma_locked"] = True

            if sma_sig == rsi_sig == "SELL" and adx_ok:
                slow = sma(df_sig['close'], SMA_HIGH_PERIOD).iloc[-2]
                sl = slow - 0.01
                tp = bid - (sl - bid) * RR
                place_order(symbol, mt5.ORDER_TYPE_SELL, bid, sl, tp)
                state["sma_locked"] = True

        time.sleep(POLL_SECONDS)

# ==================================================
if __name__ == "__main__":
    main()
