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
SYMBOLXAU = "XAUUSDm" 
SYMBOLBTC = "BTCUSDm"
TIMEFRAME_SIGNAL = mt5.TIMEFRAME_M5   # timeframe signal utama
TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M1   # timeframe entry / konfirmasi
RR = 2.0
# VOLUME
LOT = 0.01
LOTBTC = 0.02

SL_PRICE = 2
SL_PRICEBTC = 200
POLL_SECONDS = 1.0
EMA_DISTANCE = 0.5

EMA_LOW_PERIOD = 8
EMA_HIGH_PERIOD = 20
EMA_TREND_PERIOD = 50
MAX_POSITIONS = 1

BREAKEVENT_START = 2
BREAKEVENT_STARTBTC = 200
TRAIL_START_PROFIT = 3

RSI_PERIOD = 8
MIDDLE_RSI_THRESHOLD = 50

## CONFIG CONFIRM
EMA_FAST_CONFIRM = 9
EMA_SLOW_CONFIRM = 21
EMA_TREND_CONFIRM = 50
RSI_CONFIRM_PERIOD = 7
ADX_CONFIRM_PERIOD = 7
ADX_CONFIRM_THRESHOLD = 20

## CONFIG SINYAL
EMA_FAST_SINYAL = 50
EMA_SLOW_SINYAL = 200
RSI_SINYAL_PERIOD = 14
ADX_SINYAL_PERIOD = 14
BUY_RSI_SINYAL = 55
SELL_RSI_SINYAL = 45
ADX_SINYAL_THRESHOLD = 20

ADX_SHIFT = 1
ADX_PERIOD = 14
ADX_THRESHOLD = 20

# ===========================
# PROFIT TARGET (USD)
# ===========================
TP_USD_XAU = 5.0
SL_USD_XAU = -5.0

TP_USD_BTC = 5.0
SL_USD_BTC = -5.0

# Signal flags
buy_signal = False
sell_signal = False
signal = "WAIT"
last_signal = "WAIT"         # prevent repetitive entries
last_rsi = "WAIT"
stoch_buy_setup = False
stoch_sell_setup = False
prev_profit = 0
profit = 0

# Signal flags
buy_signalbtc = False
sell_signalbtc = False
signalbtc = "WAIT"
last_signalbtc = "WAIT"         # prevent repetitive entries
last_rsibtc = "WAIT"
stoch_buy_setupbtc = False
stoch_sell_setupbtc = False
prev_profitbtc = 0

# LOCK: only one cross read per cycle
ema_cross_locked = False
breakevent = False
ema_cross_lockedbtc = False
breakeventbtc = False

# Logging setup
logger = logging.getLogger("scalper")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
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

def manage_profit_usd(symbol, tp_usd, sl_usd):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    total_profit = sum(p.profit for p in positions)

    if total_profit >= tp_usd:
        logger.info("TP USD HIT %s | Profit: %.2f USD", symbol, total_profit)
        for p in positions:
            close_position(p)

    if total_profit <= sl_usd:
        logger.warning("SL USD HIT %s | Loss: %.2f USD", symbol, total_profit)
        for p in positions:
            close_position(p)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# ---------------- TREND SINYAL TF BESAR ----------------
def get_signal(df):
    close = df['close']
    fast = ema(close, EMA_FAST_SINYAL)
    slow = ema(close, EMA_SLOW_SINYAL)

    ema_fast = fast.iloc[-2]
    ema_slow = slow.iloc[-2]

    rsi = calculate_rsi(close, RSI_SINYAL_PERIOD)

    curr_rsi = rsi.iloc[-2]

    rsi_buy = curr_rsi > BUY_RSI_SINYAL
    rsi_sel = curr_rsi < SELL_RSI_SINYAL

    adx = calculate_adx(df, ADX_SINYAL_PERIOD)
    curr_adx = adx.iloc[-2]

    adx_strong = curr_adx > ADX_SINYAL_THRESHOLD

    ema_valid = ema_trend_valid(df)

    cross_buy = (ema_fast > ema_slow) and rsi_buy and adx_strong and not ema_valid
    cross_sell = (ema_fast < ema_slow) and rsi_sel and adx_strong and not ema_valid

    if cross_buy:
        return "BUY"
    elif cross_sell:
        return "SELL"
    else:
        return "WAIT"
    

def ema_recent_cross(ema_fast, ema_slow, lookback=5):
    for i in range(1, lookback + 1):
        prev = ema_fast.iloc[-i-1] - ema_slow.iloc[-i-1]
        curr = ema_fast.iloc[-i] - ema_slow.iloc[-i]
        if prev * curr < 0:
            return True  # ada cross
    return False

def ema_too_close(ema_fast, ema_slow, price, threshold=0.001):
    distance = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1])
    return distance < price.iloc[-1] * threshold

def ema_flat(ema_slow, lookback=10, slope_min=0.0001):
    slope = abs(ema_slow.iloc[-1] - ema_slow.iloc[-lookback])
    return slope < slope_min

def ema_trend_valid(df):
    close = df['close']

    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    if ema_recent_cross(ema50, ema200):
        return False

    if ema_too_close(ema50, ema200, close):
        return False

    if ema_flat(ema200):
        return False

    return True
   
def get_confirm(df, direction):
    close = df['close']
    low = df['low']
    high = df['high']
    fast = ema(close, EMA_FAST_CONFIRM)
    slow = ema(close, EMA_SLOW_CONFIRM)
    tren = ema(close, EMA_TREND_CONFIRM)

    ema_fast = fast.iloc[-2]
    ema_slow = slow.iloc[-2]
    ema_trend = slow.iloc[-2]

    candle_low = low.iloc[-2]
    candle_high = high.iloc[-2]

    get_pullback = pullback_signal(df)
    buy_candle = (candle_low < tren)
    sell_candle = (candle_high < tren)
    get_rsi = rsi_signal(df, direction)

    adx = calculate_adx(df, ADX_CONFIRM_PERIOD)
    curr_adx = adx.iloc[-2]

    adx_strong = curr_adx > ADX_CONFIRM_THRESHOLD
    strong_candle_buy = strong_bullish_close(df)
    strong_candle_sell = strong_bearish_close(df)

    cross_buy = (ema_fast > ema_slow and ema_slow > ema_trend) and get_pullback == 'BUY' and buy_candle and get_rsi and adx_strong and strong_candle_buy
    cross_sell = (ema_fast < ema_slow and ema_slow < ema_trend) and get_pullback == 'SELL' and sell_candle and get_rsi and adx_strong and strong_candle_sell

    if cross_buy:
        return "BUY"
    elif cross_sell:
        return "SELL"
    else:
        return "WAIT"
    

def ema_trend(df):
    ema9 = ema(df['close'], EMA_FAST_CONFIRM)
    ema21 = ema(df['close'], EMA_SLOW_CONFIRM)

    if ema9.iloc[-2] > ema21.iloc[-2]:
        return "BUY"
    elif ema9.iloc[-2] < ema21.iloc[-2]:
        return "SELL"
    return "NONE"

def candle_touch_ema_zone(df):
    ema9 = ema(df['close'], EMA_FAST_CONFIRM)
    ema21 = ema(df['close'], EMA_SLOW_CONFIRM)

    high = df['high'].iloc[-2]
    low = df['low'].iloc[-2]

    ema_high = max(ema9.iloc[-2], ema21.iloc[-2])
    ema_low  = min(ema9.iloc[-2], ema21.iloc[-2])

    return low <= ema_high and high >= ema_low

def pullback_rejection(df, direction):
    open_ = df['open'].iloc[-2]
    close = df['close'].iloc[-2]

    ema9 = ema(df['close'], EMA_FAST_CONFIRM)

    if direction == "BUY":
        return close > ema9.iloc[-2] and close > open_
    elif direction == "SELL":
        return close < ema9.iloc[-2] and close < open_

    return False

def pullback_signal(df):
    trend = ema_trend(df)

    if trend == "NONE":
        return None

    if not candle_touch_ema_zone(df):
        return None

    if not pullback_rejection(df, trend):
        return None

    return trend

def buy_rsi_rising_from_zone(df):
    rsi = calculate_rsi(df['close'], RSI_CONFIRM_PERIOD)

    rsi_prev = rsi.iloc[-3]
    rsi_curr = rsi.iloc[-2]

    # Area pullback ideal
    in_zone = 30 <= rsi_prev <= 45

    # Momentum naik
    rising = rsi_curr > rsi_prev

    # Belum overbought
    safe = rsi_curr < 70

    return in_zone and rising and safe

def rsi_strong_rise(df, min_slope=2):
    rsi = calculate_rsi(df['close'], RSI_CONFIRM_PERIOD)

    slope = rsi.iloc[-2] - rsi.iloc[-4]
    return slope >= min_slope

def sell_rsi_falling_from_zone(df):
    rsi = calculate_rsi(df['close'], RSI_CONFIRM_PERIOD)

    rsi_prev = rsi.iloc[-3]
    rsi_curr = rsi.iloc[-2]

    in_zone = 55 <= rsi_prev <= 70
    falling = rsi_curr < rsi_prev
    safe = rsi_curr > 30

    return in_zone and falling and safe

def rsi_signal(df, direction):
    if direction == 'BUY':
        if not buy_rsi_rising_from_zone(df):
            return False
    if direction == 'SELL':
        if not buy_rsi_rising_from_zone(df):
            return False
    
    return False

def strong_bullish_close(df):
    o = df['open'].iloc[-2]
    c = df['close'].iloc[-2]
    h = df['high'].iloc[-2]
    l = df['low'].iloc[-2]

    body = abs(c - o)
    range_ = h - l

    if range_ == 0:
        return False

    bullish = c > o
    strong_body = body >= range_ * 0.6
    close_near_high = (h - c) <= range_ * 0.15

    return bullish and strong_body and close_near_high

def strong_bearish_close(df):
    o = df['open'].iloc[-2]
    c = df['close'].iloc[-2]
    h = df['high'].iloc[-2]
    l = df['low'].iloc[-2]

    body = abs(c - o)
    range_ = h - l

    bearish = c < o
    strong_body = body >= range_ * 0.6
    close_near_low = (c - l) <= range_ * 0.15

    return bearish and strong_body and close_near_low

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
    df['tr1'] = (df['high'] - df['close'].shift(ADX_SHIFT)).abs()
    df['tr2'] = (df['low'] - df['close'].shift(ADX_SHIFT)).abs()
    df['TR'] = df[['tr0','tr1','tr2']].max(axis=1)
    df['up_move'] = df['high'] - df['high'].shift(ADX_SHIFT)
    df['down_move'] = df['low'].shift(ADX_SHIFT) - df['low']
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

# ---------------- RSI FILTER ----------------
def get_rsi_position(df):
    close = df['close']
    curr = close.iloc[-2]
    rsi = calculate_rsi(close, RSI_PERIOD)

    curr_rsi = rsi.iloc[-2]
    real_rsi = rsi.iloc[-1]

    rsi_buy = curr_rsi > MIDDLE_RSI_THRESHOLD and real_rsi > MIDDLE_RSI_THRESHOLD
    rsi_sel = curr_rsi < MIDDLE_RSI_THRESHOLD and real_rsi < MIDDLE_RSI_THRESHOLD

    if np.isnan(curr) or np.isnan(curr):
        return 'SID'
    if rsi_buy:
        return 'BUY'
    if rsi_sel:
        return 'SEL'
    
    return 'SID'

# ---------------- ADX FILTER ----------------
def get_adx_position(df):
    close = df['close']
    curr = close.iloc[-2]
    adx = calculate_adx(df, ADX_PERIOD)
    curr_adx = adx.iloc[-2]
    real_adx = adx.iloc[-1]

    adx_strong = curr_adx > ADX_THRESHOLD

    if np.isnan(curr) or np.isnan(curr):
        return False
    if adx_strong:
        return True
    
    return False


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

def symbol_info_check2(symbol):
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
    global buy_signal, sell_signal, last_signal, last_rsi, prev_profit, breakevent, ema_cross_locked 
    global buy_signalbtc, sell_signalbtc, last_signalbtc, last_rsibtc, prev_profitbtc, breakeventbtc, ema_cross_lockedbtc

    init_mt5()
    try:
        symbol_info_check(SYMBOLXAU)
    except Exception as e:
        logger.exception("SYMBOLXAU CHECK ERROR: %s", e)
        shutdown_mt5()
        return
    
    try:
        symbol_info_check(SYMBOLBTC)
    except Exception as e:
        logger.exception("SYMBOLBTC CHECK ERROR: %s", e)
        shutdown_mt5()
        return

    clear_screen()
    logger.info("STARTING SCALPER FOR %s M1 — %s", "XAU & BTC", datetime.now().isoformat())

    while True:
        try:
            # Cek market closed
            if is_market_closed(SYMBOLXAU):
                logger.warning("MARKET CLOSED FOR %s — PASS...", SYMBOLXAU)
                pass

            if is_market_closed(SYMBOLBTC):
                logger.warning("MARKET CLOSED FOR %s — PASS...", SYMBOLBTC)
                pass

            df_signal = get_rates(SYMBOLXAU, TIMEFRAME_SIGNAL, n=500)  # M5
            df_confirm = get_rates(SYMBOLXAU, TIMEFRAME_CONFIRM, n=500)  # M1

            if df_signal is None or df_signal.empty or df_confirm is None or df_confirm.empty:
                logger.warning("NO RATES RETRIEVED, RETRYING...")
                time.sleep(POLL_SECONDS)
                continue
            
            df_signalbtc = get_rates(SYMBOLBTC, TIMEFRAME_SIGNAL, n=500)  # M5 BTC
            df_confirmbtc = get_rates(SYMBOLBTC, TIMEFRAME_CONFIRM, n=500)  # M1 BTC

            if df_signalbtc is None or df_signalbtc.empty or df_confirmbtc is None or df_confirmbtc.empty:
                logger.warning("NO RATES RETRIEVED, RETRYING...")
                time.sleep(POLL_SECONDS)
                continue


            # check current positions
            position = positions_for_symbol(SYMBOLXAU)
            num_position = len(position)

            positionbtc = positions_for_symbol(SYMBOLBTC)
            num_positionbtc = len(positionbtc)

            if num_position >= MAX_POSITIONS:
                ema_cross_locked = True

            if num_positionbtc >= MAX_POSITIONS:
                ema_cross_lockedbtc = True

            # get tick
            tick = get_tick(SYMBOLXAU)
            if tick is None:
                time.sleep(POLL_SECONDS)
                continue
            bid = tick.bid
            ask = tick.ask

            # get tick
            tickbtc = get_tick(SYMBOLBTC)
            if tickbtc is None:
                time.sleep(POLL_SECONDS)
                continue
            bidbtc = tickbtc.bid
            askbtc = tickbtc.ask

            signal = get_signal(df_signal)
            confirm = get_confirm(df_confirm, signal)
            signalbtc = get_signal(df_signalbtc)
            confirmbtc = get_confirm(df_confirmbtc, signalbtc)

            if signal == 'BUY' and confirm == 'BUY':
                buy_signal = True
                sell_signal = False
            if signal == 'SELL' and confirm == 'SELL':
                sell_signal = True
                buy_signal = False

            if signalbtc == 'BUY' and confirmbtc == 'BUY':
                buy_signalbtc = True
                sell_signalbtc = False
            if signalbtc == 'SELL' and confirmbtc == 'SELL':
                sell_signalbtc = True
                buy_signalbtc = False

            # BUY ENTRY
            if buy_signal and num_position < MAX_POSITIONS and not ema_cross_locked:
                close = df_signal['close']
                slow = ema(close, EMA_FAST_SINYAL)
                v_ema = slow.iloc[-2]
                volume = LOT
                price = ask
  
                sl_position = v_ema
                tp_position = price + (price - sl_position) * RR

                res = place_market_order(SYMBOLXAU, mt5.ORDER_TYPE_BUY, price, volume, sl_position, tp_position, comment="ENTRY BUY")
                buy_signal = False
                prev_profit = 0
                breakevent = False
                ema_cross_locked = True
                time.sleep(0.1)

            if buy_signalbtc and num_positionbtc < MAX_POSITIONS and not ema_cross_lockedbtc:
                closebtc = df_signalbtc['close']
                slowbtc = ema(closebtc, EMA_FAST_SINYAL)
                v_emabtc = slowbtc.iloc[-2]
                volumebtc = LOTBTC
                pricebtc = askbtc
  
                sl_positionbtc = v_emabtc
                tp_positionbtc = pricebtc + (pricebtc - sl_positionbtc) * RR

                res = place_market_order(SYMBOLBTC, mt5.ORDER_TYPE_BUY, pricebtc, volumebtc, sl_positionbtc, tp_positionbtc, comment="ENTRY BUY")
                buy_signalbtc = False
                prev_profitbtc = 0
                breakeventbtc = False
                ema_cross_lockedbtc = True
                time.sleep(0.1)

            # SELL ENTRY
            if sell_signal and num_position < MAX_POSITIONS and not ema_cross_locked:
                close = df_signal['close']
                slow = ema(close, EMA_FAST_SINYAL)
                v_ema = slow.iloc[-2]
                volume = LOT
                price = bid

                sl_position = v_ema
                tp_position = price - (sl_position - price) * RR

                res = place_market_order(SYMBOLXAU, mt5.ORDER_TYPE_SELL, price, volume, sl_position, tp_position, comment="ENTRY SEL")
                sell_signal = False
                prev_profit = 0
                breakevent = False
                ema_cross_locked = True
                time.sleep(0.1)

            if sell_signalbtc and num_positionbtc < MAX_POSITIONS and not ema_cross_lockedbtc:
                closebtc = df_signalbtc['close']
                slowbtc = ema(closebtc, EMA_FAST_SINYAL)
                v_emabtc = slowbtc.iloc[-2]
                volumebtc = LOTBTC
                pricebtc = bidbtc

                sl_positionbtc = v_emabtc
                tp_positionbtc = pricebtc - (sl_positionbtc - pricebtc) * RR

                res = place_market_order(SYMBOLBTC, mt5.ORDER_TYPE_SELL, pricebtc, volumebtc, sl_positionbtc, tp_positionbtc, comment="ENTRY SEL")
                sell_signalbtc = False
                prev_profitbtc = 0
                breakeventbtc = False
                ema_cross_lockedbtc = True
                time.sleep(0.1)
                
            # # Logging
            if num_position < MAX_POSITIONS:
                logger.info("PAIR: %s | SIGNAL: %s | CONFIRM: %s",SYMBOLXAU ,signal, confirm)

            if num_positionbtc < MAX_POSITIONS:
                logger.info("PAIR: %s | SIGNAL: %s | CONFIRM: %s",SYMBOLBTC ,signalbtc, confirmbtc)

            # # ===== PROFIT MANAGEMENT USD =====
            manage_profit_usd(SYMBOLXAU, TP_USD_XAU, SL_USD_XAU)
            manage_profit_usd(SYMBOLBTC, TP_USD_BTC, SL_USD_BTC)

            if num_position == 0:
                buy_signal = False
                sell_signal = False
                prev_profit = 0
                breakevent = False
                ema_cross_locked = False

            if num_positionbtc == 0:
                buy_signalbtc = False
                sell_signalbtc = False
                prev_profitbtc = 0
                breakeventbtc = False
                ema_cross_lockedbtc = False

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            logger.info("INTERRUPTED BY USER. SHUTTING DOWN.")
            break
        except Exception as e:
            logger.exception("EXCEPTION IN MAIN LOOP: %s", e)
            time.sleep(1)

    shutdown_mt5()

if __name__ == "__main__":
    main()
