"""
Scalper BTCUSDm/XAUUSDmm M1
- Only read ONE SMA21xSMA50 cross per trading-cycle (locked until all positions closed)
- SMA cross lock implemented (sma_cross_locked)
- Stochastic + RSI + ADX confirmation retained
- Auto Trailing / Break-Even (simple, based on USD profit thresholds)
- Auto Close All when net profit on symbol reached
- Logging to file (rotating)

Requirements:
- MetaTrader5 package
- pandas, numpy
- MT5 terminal open & logged in
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

# ===========================
# USER CONFIGURATION
# ===========================
SYMBOL = "XAUUSDm"            # symbol chosen BTCUSDm
TIMEFRAME = mt5.TIMEFRAME_M1
RR = 1.0

if SYMBOL == "XAUUSDm":
    # VOLUME
    LOT = 0.01
    # SL/TP
    SL_IN_POINTS = 2
    TP_IN_POINTS = 4
    # Trailing / Break-even / Auto-close settings
    TRAIL_START_PROFIT = 1    # USD - start trailing after this profit
    TRAIL_DISTANCE = 0.25      # USD - trailing distance from current price
    SWING_LOOKBACK = 3        # Candle low/high reference for SL update (not used heavily here)
    AUTO_CLOSE_ALL_NET_PROFIT = 50.0  # USD - close all positions on SYMBOL when net profit reaches this
    POLL_SECONDS = 1.0           # loop sleep
    # SMA CROSS
    MAX_SMA_DISTANCE = 0.5
    MIN_SMA_DISTANCE = 0.25
else:
    # VOLUME
    LOT = 0.02
    SL_IN_POINTS = 200
    TP_IN_POINTS = 400
    # Trailing / Break-even / Auto-close settings
    TRAIL_START_PROFIT = 1    # USD - start trailing after this profit
    TRAIL_DISTANCE = 0.25      # USD - trailing distance from current price
    SWING_LOOKBACK = 3        # Candle low/high reference for SL update (not used heavily here)
    AUTO_CLOSE_ALL_NET_PROFIT = 50.0  # USD - close all positions on SYMBOL when net profit reaches this
    POLL_SECONDS = 1.0           # loop sleep
    # SMA CROSS
    MAX_SMA_DISTANCE = 2
    MIN_SMA_DISTANCE = 1


BODY_CANDLE_DIFF = 2
DISTANCE_EMA_HIGH = 0.01
RISK_PERCENT = None          # optional: if set (0.5 = 0.5%) compute LOT; else use LOT
RSI_PERIOD = 8
MAX_RSI_THRESHOLD = 80
MIDDLE_RSI_THRESHOLD = 50
MIN_RSI_THRESHOLD = 20
ADX_PERIOD = 8
ADX_THRESHOLD = 30

ATR_PERIOD = 3
ATR_MULT_XAUUSD = 1.5
ATR_MULT_BTCUSD = 2

SMA_LOW_PERIOD = 30
SMA_HIGH_PERIOD = 60
SMA_TREND_PERIOD = 100

# Aggressive trade limiting
MAX_POSITIONS = 1            # max concurrent positions on symbol (avoid overtrading)

# Shift
SMA_LOW_SHIFT = 3            # number of candles to look back for signal confirmation
LOW_SHIFT = 1
HIGH_SHIFT = 1

# Signal flags
buy_signal = False
sell_signal = False
signal = "WAIT"
last_signal = "WAIT"         # prevent repetitive entries
last_rsi = "WAIT"
stoch_buy_setup = False
stoch_sell_setup = False
prev_profit = 0

# LOCK: only one cross read per cycle
sma_cross_locked = True
candle_check = False
breakevent = False

# Logging setup
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, f"{SYMBOL}_M1.log")

logger = logging.getLogger("scalper")
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
# also print to stdout
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# ===========================
# Helper indicator functions
# ===========================
def is_market_closed(symbol):
    info = mt5.symbol_info(symbol)
    tick = get_tick(symbol)

    if info is None:
        return True

    # Jika symbol trading disabled → market tutup
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return True

    # Jika tick tidak ada → market tutup
    if tick is None:
        return True

    # Jika bid/ask 0 → market tutup
    if tick.bid == 0 or tick.ask == 0:
        return True

    return False


def calculate_stochastic(df, k_period=14, k_smooth=3, d_period=3):
    df = df.copy()
    df['HH'] = df['high'].rolling(window=k_period).max()
    df['LL'] = df['low'].rolling(window=k_period).min()
    df['%K_raw'] = ((df['close'] - df['LL']) / (df['HH'] - df['LL'])) * 100
    df['%K'] = df['%K_raw'].rolling(window=k_smooth).mean()
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df


def stochastic_signals(df):
    global stoch_buy_setup, stoch_sell_setup

    df2 = calculate_stochastic(df)
    if len(df2) < 3:
        return "WAIT"

    k_prev = df2['%K'].iloc[-2]
    d_prev = df2['%D'].iloc[-2]
    k_now  = df2['%K'].iloc[-1]
    d_now  = df2['%D'].iloc[-1]

    # SETUP CONDITIONS
    if k_prev > 70 and d_prev > 70:
        stoch_sell_setup = True
    if k_prev < 30 and d_prev < 30:
        stoch_buy_setup = True

    sell_cross_down = k_now < d_now
    sell_break_below_70 = k_now < 70 and d_now < 70

    if stoch_sell_setup and sell_cross_down and sell_break_below_70:
        stoch_sell_setup = False
        stoch_buy_setup = False
        return "SELL"

    buy_cross_up = k_now > d_now
    buy_break_above_30 = k_now > 30 and d_now > 30

    if stoch_buy_setup and buy_cross_up and buy_break_above_30:
        stoch_buy_setup = False
        stoch_sell_setup = False
        return "BUY"

    return "WAIT"


# def sma(series: pd.Series, period: int) -> pd.Series:
#     return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

# ---------------- TREND FILTER ----------------
def get_trend_position(df):
    low = df['low']
    close = df['close']
    prev = close.iloc[-3]
    curr = close.iloc[-2]
    low_candle = low.iloc[-2]

    fast = sma(close, SMA_LOW_PERIOD)
    slow = sma(close, SMA_HIGH_PERIOD)
    trend = sma(close, SMA_TREND_PERIOD)

    trend_buy = (low_candle > fast.iloc[-2]) and (fast.iloc[-2] > slow.iloc[-2]) and (slow.iloc[-2] > trend.iloc[-2]) and (curr > prev)
    trend_sel = (low_candle < fast.iloc[-2]) and (fast.iloc[-2] < slow.iloc[-2]) and (slow.iloc[-2] < trend.iloc[-3]) and (curr < prev)

    if np.isnan(curr) or np.isnan(curr):
        return 'SIDEWAYS'
    if trend_buy:
        return 'BUY'
    if trend_sel:
        return 'SELL'
    
    return 'SIDEWAYS'

# ---------------- EARLY GOLDEN CROSS ----------------
def get_sinyal_sma_cross(df):
    low = df['low']
    close = df['close']
    prev = close.iloc[-3]
    curr = close.iloc[-2]
    low_candle = low.iloc[-2]
    fast = sma(close, SMA_LOW_PERIOD)
    slow = sma(close, SMA_HIGH_PERIOD)

    p_fast = fast.iloc[-3]
    p_slow = slow.iloc[-3]
    c_fast = fast.iloc[-2]
    c_slow = slow.iloc[-2]

    cross_distance = abs(p_slow - p_fast)

    cross_buy = (c_fast > p_fast) and (c_fast > c_slow) and (cross_distance > MAX_SMA_DISTANCE)
    cross_sel = (c_fast < p_fast) and (c_fast < c_slow) and (cross_distance > MAX_SMA_DISTANCE)

    if cross_buy:
        return "BUY"
    elif cross_sel:
        return "SELL"
    else:
        return "SIDEWAYS"

# ---------------- ADX FILTER ----------------
def get_adx_position(df):
    close = df['close']
    curr = close.iloc[-2]
    adx = calculate_adx(df, ADX_PERIOD)
    curr_adx = adx.iloc[-2]
    prev_adx = adx.iloc[-3]

    adx_strong = curr_adx > ADX_THRESHOLD

    if np.isnan(curr) or np.isnan(curr):
        return False
    if adx_strong:
        return True
    
    return False

# ---------------- RSI FILTER ----------------
def get_rsi_position(df,trend):
    close = df['close']
    curr = close.iloc[-2]
    rsi = calculate_rsi(close, RSI_PERIOD)

    prev_rsi = rsi.iloc[-3]
    curr_rsi = rsi.iloc[-2]
    rsi_real = rsi.iloc[-1]

    rsi_buy = prev_rsi > MIDDLE_RSI_THRESHOLD and rsi_real < MAX_RSI_THRESHOLD
    rsi_sel = prev_rsi < MIDDLE_RSI_THRESHOLD and rsi_real > MIN_RSI_THRESHOLD

    if np.isnan(curr) or np.isnan(curr):
        return 'SIDEWAYS'
    if rsi_buy and trend == 'BUY':
        return 'BUY'
    if rsi_sel and trend == 'SELL':
        return 'SELL'
    
    return 'SIDEWAYS'

# ---------------- CANDLE FILTER ----------------
def valid_body(df):
    candle = df.iloc[-2]
    body = abs(candle['close'] - candle['open'])
    total = candle['high'] - candle['low']
    return body / total >= 0.6 if total != 0 else False


def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    close = close.astype(float).reset_index(drop=True)
    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    rsi = np.full(len(close), np.nan)
    if len(close) <= period:
        return pd.Series(rsi)
    first_gain = gain.iloc[1:period+1].mean()
    first_loss = loss.iloc[1:period+1].mean()
    avg_gain[period] = first_gain
    avg_loss[period] = first_loss
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss.iloc[i]) / period
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_vals = 100 - (100 / (1 + rs))
        rsi = rsi_vals
    return pd.Series(rsi)


def calculate_adx(df: pd.DataFrame, period: int) -> pd.Series:
    df = df.copy()
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = (df['high'] - df['close'].shift(LOW_SHIFT)).abs()
    df['tr2'] = (df['low'] - df['close'].shift(LOW_SHIFT)).abs()
    df['TR'] = df[['tr0','tr1','tr2']].max(axis=1)
    df['up_move'] = df['high'] - df['high'].shift(LOW_SHIFT)
    df['down_move'] = df['low'].shift(LOW_SHIFT) - df['low']
    df['DM_plus'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['DM_minus'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    TR_smooth = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    DMp_smooth = df['DM_plus'].ewm(alpha=1/period, adjust=False).mean()
    DMm_smooth = df['DM_minus'].ewm(alpha=1/period, adjust=False).mean()
    df['DI_plus'] = 100 * (DMp_smooth / TR_smooth)
    df['DI_minus'] = 100 * (DMm_smooth / TR_smooth)
    df['DX'] = ( (df['DI_plus'] - df['DI_minus']).abs() / (df['DI_plus'] + df['DI_minus']) ) * 100
    adx = df['DX'].ewm(alpha=1/period, adjust=False).mean()
    return adx

# ===========================
# MT5 helper functions
# ===========================
def init_mt5():
    if not mt5.initialize():
        logger.error("MT5 initialize() failed, error: %s", mt5.last_error())
        raise SystemExit("MT5 init failed")

def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception as e:
        logger.exception("Error shutting down MT5: %s", e)

def get_rates(symbol, timeframe, n=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_tick(symbol):
    try:
        return mt5.symbol_info_tick(symbol)
    except Exception:
        return None

def symbol_info_check(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"{symbol} not found in Market Watch")
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return info

def place_market_order(symbol, order_type, price, volume, sl_price, tp_price, deviation=20, magic=234000, comment="Market Order"):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("place_market_order: symbol_info None")
        return None
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl_price) if sl_price is not None else None,
        "tp": float(tp_price) if tp_price is not None else None,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_filling": mt5.ORDER_FILLING_FOK if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL else mt5.ORDER_FILLING_IOC
    }
    try:
        result = mt5.order_send(request)
        if result is None:
            logger.error("order_send returned None")
            return None
        if hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("Berhasil buka posisi! Order: %s Deal: %s", getattr(result, 'order', None), getattr(result, 'deal', None))
        else:
            # Some brokers return different retcodes; log full result
            logger.warning("Gagal buka posisi. retcode=%s comment=%s result=%s", getattr(result, 'retcode', None), getattr(result, 'comment', None), result)
        return result
    except Exception as e:
        logger.exception("place_market_order exception: %s", e)
        return None

def set_position_sl_tp(position_ticket, sl, tp):
    try:
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(position_ticket),
            "sl": float(sl) if sl is not None else None,
            "tp": float(tp) if tp is not None else None,
        }
        res = mt5.order_send(req)
        logger.debug("set_position_sl_tp response: %s", res)
        return res
    except Exception as e:
        logger.exception("set_position_sl_tp exception: %s", e)
        return None

def close_position(pos):
    try:
        symbol = pos.symbol
        volume = pos.volume
        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "position": int(pos.ticket),
            "deviation": 50,
            "magic": 234000,
            "comment": "AutoClose",
            "type_filling": mt5.ORDER_FILLING_IOC
        }
        res = mt5.order_send(request)
        logger.info("close_position result: %s", res)
        return res
    except Exception as e:
        logger.exception("close_position exception: %s", e)
        return None
    
def manage_position(df):
    global breakevent
    close = df['close']
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return
    
    fast = sma(close, SMA_LOW_PERIOD)
    pos = positions[0]
    price = mt5.symbol_info_tick(SYMBOL)
    entry = pos.price_open
    sl = pos.sl
    tp = pos.tp

    # ================= BUY =================
    if pos.type == mt5.ORDER_TYPE_BUY:
        current = price.bid
        risk = entry - sl
        profit = current - entry

        # --- BREAK EVEN ---
        if profit >= risk * 0.5 and sl < entry and not breakevent:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": entry + 0.1 * risk,
                "tp": tp
            })
            breakevent = True

        # --- TRAILING ---
        if profit >= risk * 0.5:
            new_sl = fast.iloc[-2]
            if new_sl > sl and new_sl < current:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": tp
                })

    # ================= SELL =================
    if pos.type == mt5.ORDER_TYPE_SELL:
        current = price.ask
        risk = sl - entry
        profit = entry - current

        # --- BREAK EVEN ---
        if profit >= risk * 0.8 and sl > entry:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": entry - 0.1 * risk,
                "tp": tp
            })

        # --- TRAILING ---
        if profit >= risk * 1.2:
            new_sl = fast.iloc[-2]
            if new_sl < sl and new_sl > current:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": tp
                })

def positions_for_symbol(symbol):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None or len(pos) == 0:
        return []
    return list(pos)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ===========================
# Main loop
# ===========================
def main():
    global buy_signal, sell_signal, last_signal, last_rsi, prev_profit, sma_cross_locked, candle_check

    init_mt5()
    try:
        symbol_info_check(SYMBOL)
    except Exception as e:
        logger.exception("Symbol check error: %s", e)
        shutdown_mt5()
        return

    clear_screen()
    logger.info("Starting scalper for %s M1 — %s", SYMBOL, datetime.now().isoformat())

    while True:
        try:
            # Cek market closed
            if is_market_closed(SYMBOL):
                logger.warning("Market CLOSED for %s — waiting 3 hours...", SYMBOL)
                time.sleep(10800)  # 2 jam = 7200 detik
                continue

            df = get_rates(SYMBOL, TIMEFRAME, n=500)
            if df is None or df.empty:
                logger.warning("No rates retrieved, retrying...")
                time.sleep(POLL_SECONDS)
                continue

            # check current positions
            position = positions_for_symbol(SYMBOL)
            num_position = len(position)

            tick = get_tick(SYMBOL)
            if tick is None:
                time.sleep(POLL_SECONDS)
                continue
            bid = tick.bid
            ask = tick.ask

            if sma_cross_locked:
                manage_position(df)
                if num_position == 0:
                    sma_cross_locked = False
                continue

            trend_position = get_trend_position(df)
            sma_cross = get_sinyal_sma_cross(df)
            rsi_position = get_rsi_position(df,trend_position)
            adx_position = get_adx_position(df)
            candle_check = valid_body(df)

            if trend_position == 'BUY' and sma_cross == 'BUY' and rsi_position == 'BUY' and adx_position and candle_check:
                buy_signal = True
                sell_signal = False
            if trend_position == 'SELL' and sma_cross == 'SELL' and rsi_position == 'SELL' and adx_position and candle_check:
                sell_signal = True
                buy_signal = False

            if buy_signal and num_position < MAX_POSITIONS:
                close = df['close']
                slow = sma(close, SMA_HIGH_PERIOD)
                sma60 = slow.iloc[-2]
                volume = LOT
                price = ask
  
                sl_position = sma60
                tp_position = price + (price - sl_position) * RR

                res = place_market_order(SYMBOL, mt5.ORDER_TYPE_BUY, price, volume, sl_position, tp_position, comment="Entry Buy Position")
                buy_signal = False
                prev_profit = 0
                sma_cross_locked = True
                time.sleep(0.2)

            if sell_signal and num_position < MAX_POSITIONS:
                close = df['close']
                slow = sma(close, SMA_HIGH_PERIOD)
                sma60 = slow.iloc[-2]
                volume = LOT
                price = bid

                sl_position = sma60
                tp_position = price - (sl_position - price) * RR

                res = place_market_order(SYMBOL, mt5.ORDER_TYPE_SELL, price, volume, sl_position, tp_position, comment="Entry Sell Position")
                sell_signal = False
                prev_profit = 0
                sma_cross_locked = True
                time.sleep(0.2)
                
            # Logging
            if num_position < MAX_POSITIONS:
                logger.info("TREND: %s | SMA CROSS: %s | CANDLE: %s | RSI: %s | ADX: %s",trend_position, sma_cross, candle_check, rsi_position, adx_position)
            else:
                logger.info("Positions open: %s", num_position)

            # Reset SMA lock when there are no positions (i.e., cycle complete)
            if num_position == 0 and sma_cross_locked:
                buy_signal = False
                sell_signal = False
                prev_profit = 0

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down.")
            break
        except Exception as e:
            logger.exception("Exception in main loop: %s", e)
            time.sleep(1)

    shutdown_mt5()

if __name__ == "__main__":
    main()
