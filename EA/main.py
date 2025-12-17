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
TIMEFRAME_SIGNAL = mt5.TIMEFRAME_M15   # timeframe signal utama
TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M1   # timeframe entry / konfirmasi
RR = 2.0
# VOLUME
LOT = 0.01
LOTBTC = 0.02
POLL_SECONDS = 1.0
SMA_DISTANCE = 0.5

SMA_LOW_PERIOD = 30
SMA_HIGH_PERIOD = 60
SMA_TREND_PERIOD = 100
MAX_POSITIONS = 1

BREAKEVENT_START = 2
TRAIL_START_PROFIT = 4

RSI_PERIOD = 8
MIDDLE_RSI_THRESHOLD = 50

ADX_SHIFT = 1
ADX_PERIOD = 8
ADX_THRESHOLD = 30

# Signal flags
buy_signal = False
sell_signal = False
signal = "WAIT"
last_signal = "WAIT"         # prevent repetitive entries
last_rsi = "WAIT"
stoch_buy_setup = False
stoch_sell_setup = False
prev_profit = 0

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
sma_cross_locked = False
breakevent = False
sma_cross_lockedbtc = False
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
    close = df['close']
    open = df['open']
    fast = sma(close, SMA_LOW_PERIOD)
    slow = sma(close, SMA_HIGH_PERIOD)
    tren = sma(close, SMA_TREND_PERIOD)

    p_fast = fast.iloc[-3]
    p_slow = slow.iloc[-3]
    c_fast = fast.iloc[-2]
    c_slow = slow.iloc[-2]

    tren_ema = tren.iloc[-2]
    current = open.iloc[-2]

    cross_distance = abs(p_fast - p_slow)
    trendup = (c_slow > tren_ema and c_fast > c_slow and c_slow > tren_ema)
    trenddw = (c_slow < tren_ema and c_fast < c_slow and c_slow < tren_ema)

    cross_buy = (c_fast > p_fast) and trendup and (cross_distance >= SMA_DISTANCE) and (current > c_fast)
    cross_sel = (c_fast < p_fast) and trenddw and (cross_distance >= SMA_DISTANCE) and (current < c_fast)

    if cross_buy:
        return "BUY"
    elif cross_sel:
        return "SELL"
    else:
        return "SIDEWAYS"
    
def get_confirm_sma_cross(df):
    close = df['close']
    open = df['open']
    fast = sma(close, SMA_LOW_PERIOD)
    slow = sma(close, SMA_HIGH_PERIOD)
    tren = sma(close, SMA_TREND_PERIOD)

    p_fast = fast.iloc[-3]
    p_slow = slow.iloc[-3]
    c_fast = fast.iloc[-2]
    c_slow = slow.iloc[-2]

    tren_ema = tren.iloc[-2]
    current = open.iloc[-2]

    cross_distance = abs(p_fast - p_slow)
    trendup = (c_slow > tren_ema and c_fast > c_slow and c_slow > tren_ema)
    trenddw = (c_slow < tren_ema and c_fast < c_slow and c_slow < tren_ema)
    
    cross_buy = (c_fast > p_fast) and trendup and (cross_distance >= SMA_DISTANCE) and (current > c_fast)
    cross_sel = (c_fast < p_fast) and trenddw and (cross_distance >= SMA_DISTANCE) and (current < c_fast)

    if cross_buy:
        return "BUY"
    elif cross_sel:
        return "SELL"
    else:
        return "SIDEWAYS"

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

    rsi_buy = curr_rsi > MIDDLE_RSI_THRESHOLD
    rsi_sel = curr_rsi < MIDDLE_RSI_THRESHOLD

    if np.isnan(curr) or np.isnan(curr):
        return 'SIDEWAYS'
    if rsi_buy:
        return 'BUY'
    if rsi_sel:
        return 'SELL'
    
    return 'SIDEWAYS'

# ---------------- ADX FILTER ----------------
def get_adx_position(df):
    close = df['close']
    curr = close.iloc[-2]
    adx = calculate_adx(df, ADX_PERIOD)
    curr_adx = adx.iloc[-2]

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

def trail_position(df, symbol):
    global prev_profit
    close = df['close']
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    fast = sma(close, SMA_LOW_PERIOD)
    pos = positions[0]
    price = mt5.symbol_info_tick(symbol)
    entry = pos.price_open
    sl = pos.sl
    tp = pos.tp

    # ================= BUY =================
    if pos.type == mt5.ORDER_TYPE_BUY:
        current = price.bid
        risk = entry - sl
        profit = current - entry

        # --- TRAILING ---
        if profit >= TRAIL_START_PROFIT and profit > prev_profit:
            increasesltp = (profit-prev_profit)
            new_sl = sl + increasesltp
            if new_sl > sl:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": tp
                })

    if profit > 0 and profit > prev_profit:
        prev_profit = profit

    # ================= SELL =================
    if pos.type == mt5.ORDER_TYPE_SELL:
        current = price.ask
        risk = sl - entry
        profit = entry - current

        # --- TRAILING ---
        if profit >= TRAIL_START_PROFIT and profit > prev_profit:
            increasesltp = (profit-prev_profit)
            new_sl = sl - increasesltp
            if new_sl < sl:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": tp
                })

        if profit > 0 and profit > prev_profit:
            prev_profit = profit

def break_position(df, symbol):
    global breakevent, prev_profit
    close = df['close']
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    
    fast = sma(close, SMA_LOW_PERIOD)
    pos = positions[0]
    price = mt5.symbol_info_tick(symbol)
    entry = pos.price_open
    sl = pos.sl
    tp = pos.tp

    # ================= BUY =================
    if pos.type == mt5.ORDER_TYPE_BUY:
        current = price.bid
        risk = entry - sl
        profit = current - entry

        # --- BREAK EVEN ---
        if profit >= BREAKEVENT_START and sl < entry and not breakevent:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": entry + 0.5,
                "tp": (entry + 0.5) + TRAIL_START_PROFIT
            })
            breakevent = True

        if profit > 0 and profit > prev_profit:
            prev_profit = profit

    # ================= SELL =================
    if pos.type == mt5.ORDER_TYPE_SELL:
        current = price.ask
        risk = sl - entry
        profit = entry - current

        # --- BREAK EVEN ---
        if profit >= BREAKEVENT_START and sl > entry and not breakevent:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": entry - 0.5,
                "tp": (entry - 0.5) - TRAIL_START_PROFIT
            })
            breakevent = True

        if profit > 0 and profit > prev_profit:
            prev_profit = profit


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
    global buy_signal, sell_signal, last_signal, last_rsi, prev_profit, breakevent, sma_cross_locked 
    global buy_signalbtc, sell_signalbtc, last_signalbtc, last_rsibtc, prev_profitbtc, breakeventbtc, sma_cross_lockedbtc

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

            df_signal = get_rates(SYMBOLXAU, TIMEFRAME_SIGNAL, n=500)  # M15
            df_confirm = get_rates(SYMBOLXAU, TIMEFRAME_CONFIRM, n=500)  # M1

            if df_signal is None or df_signal.empty or df_confirm is None or df_confirm.empty:
                logger.warning("NO RATES RETRIEVED, RETRYING...")
                time.sleep(POLL_SECONDS)
                continue
            
            df_signalbtc = get_rates(SYMBOLBTC, TIMEFRAME_SIGNAL, n=500)  # M15 BTC
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
                sma_cross_locked = True

            if num_positionbtc >= MAX_POSITIONS:
                sma_cross_lockedbtc = True

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

            if sma_cross_locked:
                break_position(df_confirm, SYMBOLXAU)
                if breakevent:
                    trail_position(df_confirm, SYMBOLXAU)
                if num_position == 0:
                    sma_cross_locked = False
                pass

            if sma_cross_lockedbtc:
                break_position(df_confirmbtc, SYMBOLBTC)
                if breakeventbtc:
                    trail_position(df_confirmbtc, SYMBOLBTC)
                if num_positionbtc == 0:
                    sma_cross_lockedbtc = False
                pass

            sma_signal = get_sinyal_sma_cross(df_signal)
            sma_confirm = get_confirm_sma_cross(df_confirm)
            rsi_signal = get_rsi_position(df_signal)
            adx_check = get_adx_position(df_signal)

            sma_signalbtc = get_sinyal_sma_cross(df_signalbtc)
            sma_confirmbtc = get_confirm_sma_cross(df_confirmbtc)
            rsi_signalbtc = get_rsi_position(df_signalbtc)
            adx_checkbtc = get_adx_position(df_signalbtc)

            if sma_signal == 'BUY' and sma_confirm == 'BUY' and rsi_signal == 'BUY' and adx_check:
                buy_signal = True
                sell_signal = False
            if sma_signal == 'SELL' and sma_confirm == 'SELL' and rsi_signal == 'SELL' and adx_check:
                sell_signal = True
                buy_signal = False

            if sma_signalbtc == 'BUY' and sma_confirmbtc == 'BUY' and rsi_signalbtc == 'BUY' and adx_checkbtc:
                buy_signalbtc = True
                sell_signalbtc = False
            if sma_signalbtc == 'SELL' and sma_confirmbtc == 'SELL' and rsi_signalbtc == 'SELL' and adx_checkbtc:
                sell_signalbtc = True
                buy_signalbtc = False


            if buy_signal and num_position < MAX_POSITIONS and not sma_cross_locked:
                close = df_signal['close']
                slow = sma(close, SMA_TREND_PERIOD) + 0.01
                v_sma = slow.iloc[-2]
                volume = LOT
                price = ask
  
                sl_position = v_sma
                tp_position = price + (price - sl_position) * RR

                res = place_market_order(SYMBOLXAU, mt5.ORDER_TYPE_BUY, price, volume, sl_position, tp_position, comment="ENTRY BUY")
                buy_signal = False
                prev_profit = 0
                breakevent = False
                sma_cross_locked = True
                time.sleep(0.1)

            if buy_signalbtc and num_positionbtc < MAX_POSITIONS and not sma_cross_lockedbtc:
                closebtc = df_signalbtc['close']
                slowbtc = sma(closebtc, SMA_TREND_PERIOD) + 0.01
                v_smabtc = slowbtc.iloc[-2]
                volumebtc = LOTBTC
                pricebtc = askbtc
  
                sl_positionbtc = v_smabtc
                tp_positionbtc = pricebtc + (pricebtc - sl_positionbtc) * RR

                res = place_market_order(SYMBOLBTC, mt5.ORDER_TYPE_BUY, pricebtc, volumebtc, sl_positionbtc, tp_positionbtc, comment="ENTRY BUY")
                buy_signalbtc = False
                prev_profitbtc = 0
                breakeventbtc = False
                sma_cross_lockedbtc = True
                time.sleep(0.1)

            
            if sell_signal and num_position < MAX_POSITIONS and not sma_cross_locked:
                close = df_signal['close']
                slow = sma(close, SMA_TREND_PERIOD) - 0.01
                v_sma = slow.iloc[-2]
                volume = LOT
                price = bid

                sl_position = v_sma
                tp_position = price - (sl_position - price) * RR

                res = place_market_order(SYMBOLXAU, mt5.ORDER_TYPE_SELL, price, volume, sl_position, tp_position, comment="ENTRY SELL")
                sell_signal = False
                prev_profit = 0
                breakevent = False
                sma_cross_locked = True
                time.sleep(0.1)

            if sell_signalbtc and num_positionbtc < MAX_POSITIONS and not sma_cross_lockedbtc:
                closebtc = df_signalbtc['close']
                slowbtc = sma(closebtc, SMA_TREND_PERIOD) - 0.01
                v_smabtc = slowbtc.iloc[-2]
                volumebtc = LOTBTC
                pricebtc = bidbtc

                sl_positionbtc = v_smabtc
                tp_positionbtc = pricebtc - (sl_positionbtc - pricebtc) * RR

                res = place_market_order(SYMBOLBTC, mt5.ORDER_TYPE_SELL, pricebtc, volumebtc, sl_positionbtc, tp_positionbtc, comment="ENTRY SELL")
                sell_signalbtc = False
                prev_profitbtc = 0
                breakeventbtc = False
                sma_cross_lockedbtc = True
                time.sleep(0.1)
                
            # Logging
            if num_position < MAX_POSITIONS:
                logger.info("PAIR: %s | SMA SIGNAL: %s | SMA CONFIRM: %s | RSI: %s | ADX: %s",SYMBOLXAU ,sma_signal, sma_confirm, rsi_signal, adx_check)
            else:
                logger.info("POSITIONS OPEN: %s | PAIR: %s ", num_position, SYMBOLXAU)

            if num_positionbtc < MAX_POSITIONS:
                logger.info("PAIR: %s | SMA SIGNAL: %s | SMA CONFIRM: %s | RSI: %s | ADX: %s",SYMBOLBTC ,sma_signalbtc, sma_confirmbtc, rsi_signalbtc, adx_checkbtc)
            else:
                logger.info("POSITIONS OPEN:  %s | PAIR: %s", num_positionbtc, SYMBOLBTC)


            if num_position == 0 and sma_cross_locked:
                buy_signal = False
                sell_signal = False
                prev_profit = 0
                breakevent = False

            if num_positionbtc == 0 and sma_cross_lockedbtc:
                buy_signalbtc = False
                sell_signalbtc = False
                prev_profitbtc = 0
                breakeventbtc = False

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
