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
LOT = 0.01                   # default volume
if SYMBOL == "XAUUSDm":
    # SL/TP
    TP_IN_POINTS = 2
    SL_IN_POINTS = 1
    # SMA CROSS
    MIN_SMA_DISTANCE = 1
    MAX_SMA_DISTANCE = 0.25
    # CANDLE
    BODY_CANDLE_DIFF = 0.25
else:
    TP_IN_POINTS = 100
    SL_IN_POINTS = 200
    MIN_SMA_DISTANCE = 10
    MAX_SMA_DISTANCE = 20
    BODY_CANDLE_DIFF = 2.5

RISK_PERCENT = None          # optional: if set (0.5 = 0.5%) compute LOT; else use LOT
RSI_PERIOD = 3
BUY_RSI_THRESHOLD = 20
MIDDLE_RSI_THRESHOLD = 50
SELL_RSI_THRESHOLD = 80
ADX_PERIOD = 3
ADX_THRESHOLD = 50

ATR_PERIOD = 3
ATR_MULT_XAUUSD = 1.5
ATR_MULT_BTCUSD = 2

SMA_LOW_PERIOD = 8
SMA_HIGH_PERIOD = 21
SMA_TREND_PERIOD = 100

# Trailing / Break-even / Auto-close settings
BREAK_EVEN_PROFIT = 0.5   # USD - when position profit >= this, move SL to break-even (price_open)
TRAIL_START_PROFIT = 0.5    # USD - start trailing after this profit
TRAIL_DISTANCE = 1.0      # USD - trailing distance from current price
SWING_LOOKBACK = 3        # Candle low/high reference for SL update (not used heavily here)
AUTO_CLOSE_ALL_NET_PROFIT = 50.0  # USD - close all positions on SYMBOL when net profit reaches this
POLL_SECONDS = 1.0           # loop sleep

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
sma_cross_locked = False
rsi_cross_locked = False
candle_check = True

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


def get_sma_trend(df):
    close = df['close']
    price_close = df['close'].iloc[-1]
    sma_low = sma(close, SMA_LOW_PERIOD).iloc[-1]
    sma_high = sma(close, SMA_HIGH_PERIOD).iloc[-1]
    sma_distance_buy = sma_high + MIN_SMA_DISTANCE
    sma_distance_sell =  sma_high - MIN_SMA_DISTANCE

    market_uptrend = (sma_low > sma_high) and sma_low > sma_distance_buy and (price_close > sma_low)
    market_downtrend = (sma_low < sma_high) and sma_low < sma_distance_sell and (price_close < sma_low)

    if market_uptrend:
        return "UP"
    elif market_downtrend:
        return "DOWN"
    else:
        return "SIDEWAYS"


def get_rsi_position(p,c):
    if np.isnan(p) or np.isnan(c):
        return 'SIDEWAYS'
    if p < BUY_RSI_THRESHOLD and c > BUY_RSI_THRESHOLD:
        return 'BUY'
    if p > SELL_RSI_THRESHOLD and c < SELL_RSI_THRESHOLD:
        return 'SELL'
    
    return 'WAIT'


def get_adx_position(c):
    if np.isnan(c):
        return False
    if c > ADX_THRESHOLD:
        return True
    return False


def get_sinyal_sma_cross(df):
    close = df['close']
    sma_low = sma(close, SMA_LOW_PERIOD)
    sma_high = sma(close, SMA_HIGH_PERIOD)

    prev_sma_low = sma_low.iloc[-2]
    prev_sma_high = sma_high.iloc[-2]
    curr_sma_low = sma_low.iloc[-1]
    curr_sma_high = sma_high.iloc[-1]

    curr_sma_high_distance_buy = curr_sma_high + MAX_SMA_DISTANCE
    curr_sma_high_distance_sell = curr_sma_high - MAX_SMA_DISTANCE

    if prev_sma_low > prev_sma_high and curr_sma_low > curr_sma_high_distance_buy:
        return "BUY"
    elif prev_sma_low < prev_sma_high and curr_sma_low < curr_sma_high_distance_sell:
        return "SELL"
    else:
        return "WAIT"

def avoid_small_candle_signal_direction(df, min_body_pips, signal):
    close = df['close'].iloc[-1]
    open_ = df['open'].iloc[-1]
    body = abs(close - open_)

    # BUY signal → hanya valid jika candle hijau
    if signal == "BUY":
        if close <= open_:
            return True  # candle tidak hijau → hindari entry
        if body < min_body_pips:
            return True  # body kecil → hindari entry
        return False

    # SELL signal → hanya valid jika candle merah
    if signal == "SELL":
        if close >= open_:
            return True  # candle tidak merah → hindari entry
        if body < min_body_pips:
            return True  # body kecil → hindari entry
        return False

    return True  # jika signal tidak dikenali → hindari entry

def check_body_candle(df, smalow, signal):
    """
    signal = 'BUY' atau 'SELL'
    
    Return:
      True  → valid
      False → tidak valid
    """
    open_ = df['open'].iloc[-1]
    close = df['close'].iloc[-1]

    upper_body = max(open_, close)
    lower_body = min(open_, close)

    # BUY → body harus di atas SMA8
    if signal == "BUY":
        return lower_body > smalow     # body entirely above

    # SELL → body harus di bawah SMA8
    if signal == "SELL":
        return upper_body < smalow     # body entirely below

    return False


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

def positions_for_symbol(symbol):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None or len(pos) == 0:
        return []
    return list(pos)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ===========================
# Signals & Compute
# ===========================
def compute_signals(df):
    close = df['close']
    df['SMA_LOW'] = sma(close, SMA_LOW_PERIOD)
    df['SMA_HIGH'] = sma(close, SMA_HIGH_PERIOD)
    df['SMA_TREND'] = sma(close, SMA_TREND_PERIOD)
    df['RSI'] = calculate_rsi(close, RSI_PERIOD)
    df['ADX'] = calculate_adx(df, ADX_PERIOD)
    df['prev_low'] = df['SMA_LOW'].shift(HIGH_SHIFT)
    df['prev_high'] = df['SMA_HIGH'].shift(HIGH_SHIFT)
    df['cross_up_prev_candle'] = (df['SMA_LOW'].shift(HIGH_SHIFT) > df['SMA_HIGH'].shift(HIGH_SHIFT)) & (df['SMA_LOW'].shift(SMA_LOW_SHIFT) <= df['SMA_HIGH'].shift(SMA_LOW_SHIFT))
    df['cross_down_prev_candle'] = (df['SMA_LOW'].shift(HIGH_SHIFT) < df['SMA_HIGH'].shift(HIGH_SHIFT)) & (df['SMA_LOW'].shift(SMA_LOW_SHIFT) >= df['SMA_HIGH'].shift(SMA_LOW_SHIFT))
    df['sma_distance'] = abs(df['SMA_LOW'] - df['SMA_HIGH'])
    df['valid_cross_up'] = df['cross_up_prev_candle'] & (df['sma_distance'] >= MIN_SMA_DISTANCE)
    df['valid_cross_down'] = df['cross_down_prev_candle'] & (df['sma_distance'] >= MIN_SMA_DISTANCE)
    return df

# ===========================
# Main loop
# ===========================
def main():
    global buy_signal, sell_signal, last_signal, last_rsi, prev_profit, sma_cross_locked, rsi_cross_locked, candle_check

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
            
            # Detect SMA cross but only if not locked
            sinyal_sma_cross = get_sinyal_sma_cross(df)
            if not sma_cross_locked and sinyal_sma_cross in ("BUY","SELL"):
                last_signal = sinyal_sma_cross
                # rsi_cross_locked = False
                sma_cross_locked = True

            
            df_sign = compute_signals(df)
            last_prev = df_sign.iloc[-3]
            prev = df_sign.iloc[-2]
            current = df_sign.iloc[-1]

            rsi_position = get_rsi_position(prev['RSI'], current['RSI'])
            if not rsi_cross_locked and rsi_position in ("BUY","SELL"):
                last_rsi = rsi_position
                rsi_cross_locked = True

            adx_position = get_adx_position(current['ADX'])

            if last_signal == "BUY":
                candle_check = avoid_small_candle_signal_direction(df, BODY_CANDLE_DIFF, last_signal)
            
            if last_signal == "SELL":
                candle_check = avoid_small_candle_signal_direction(df, BODY_CANDLE_DIFF, last_signal)

            tick = get_tick(SYMBOL)
            if tick is None:
                time.sleep(POLL_SECONDS)
                continue
            bid = tick.bid
            ask = tick.ask

            check_body = check_body_candle(df, current['SMA_LOW'], last_signal)

            # Only allow entry when SMA cross is locked and its direction matches trend + confirmations
            if sma_cross_locked and last_signal == 'BUY':
                if last_rsi == 'BUY' and adx_position and check_body and not candle_check:
                    buy_signal = True
                    sell_signal = False

            if sma_cross_locked and last_signal == 'SELL':
                if last_rsi == 'SELL' and adx_position and check_body and not candle_check:
                    sell_signal = True
                    buy_signal = False

            # check current positions
            position = positions_for_symbol(SYMBOL)
            num_position = len(position)

            # ENTRY: only once per locked cross, and enforce MAX_POSITIONS
            if buy_signal and num_position < MAX_POSITIONS:
                sma_high = current['SMA_HIGH']
                volume = LOT
                price = ask
                # SL di bawah SMA60
                sl_distance = abs(price - sma_high)
                sl_position = sma_high - 0.01  # agar pasti berada sedikit di bawah

                # Risk Reward 1:3
                tp_position = price + (sl_distance * 3)

                # sl_position = price - SL_IN_POINTS
                # tp_position = price + TP_IN_POINTS
                res = place_market_order(SYMBOL, mt5.ORDER_TYPE_BUY, price, volume, sl_position, tp_position, comment="Entry Buy Position")
                buy_signal = False
                prev_profit = 0
                time.sleep(0.2)

            if sell_signal and num_position < MAX_POSITIONS:
                sma_high = current['SMA_HIGH']
                volume = LOT
                price = bid
                # SL di atas SMA60
                sl_distance = abs(price - sma_high)
                sl_position = sma_high + 0.01  # agar pasti sedikit di atas

                # Risk Reward 1:2
                tp_position = price - (sl_distance * 2)
                # sl_position = price + SL_IN_POINTS
                # tp_position = price - TP_IN_POINTS
                res = place_market_order(SYMBOL, mt5.ORDER_TYPE_SELL, price, volume, sl_position, tp_position, comment="Entry Sell Position")
                sell_signal = False
                prev_profit = 0
                time.sleep(0.2)
                
            # # Logging
            if num_position < MAX_POSITIONS:
                logger.info("SMA CROSS: %s | RSI CROSS: %s | ADX CROSS: %s | CANDLE BODY: %s | CANDLE PASS: %s", sinyal_sma_cross, last_rsi, adx_position, check_body, candle_check)
            else:
                logger.info("Positions open: %s", num_position)

            # Manage open positions: trailing, break-even, and check auto close net profit
            if position:
                net_profit = sum([p.profit for p in position])
                # Auto-close all if net profit target reached
                if net_profit >= AUTO_CLOSE_ALL_NET_PROFIT:
                    logger.info("AUTO CLOSE ALL triggered. Net profit: %s >= %s", net_profit, AUTO_CLOSE_ALL_NET_PROFIT)
                    for p in position:
                        close_position(p)
                    # after closing reset lock & signals
                    sma_cross_locked = False
                    rsi_cross_locked = False
                    last_signal = "WAIT"
                    last_rsi = "WAIT"
                    buy_signal = False
                    sell_signal = False
                    prev_profit = 0
                    time.sleep(0.5)
                    continue

                # Per position trailing/breakeven
                for pos in position:
                    try:
                        # detect if profit increased from last stored prev_profit
                        if pos.profit > 0 and prev_profit == 0:
                            prev_profit = pos.profit

                        # Break-even: move SL to entry price when profit >= BREAK_EVEN_PROFIT
                        if pos.profit >= BREAK_EVEN_PROFIT:
                            # set SL to price_open (break-even) if not already
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                desired_sl = pos.price_open
                                if pos.sl is None or pos.sl < desired_sl:
                                    set_position_sl_tp(pos.ticket, desired_sl, pos.tp)
                                    logger.info("Set break-even SL for BUY pos %s -> %s", pos.ticket, desired_sl)
                            elif pos.type == mt5.POSITION_TYPE_SELL:
                                desired_sl = pos.price_open
                                if pos.sl is None or pos.sl > desired_sl:
                                    set_position_sl_tp(pos.ticket, desired_sl, pos.tp)
                                    logger.info("Set break-even SL for SELL pos %s -> %s", pos.ticket, desired_sl)

                        # Trailing: if profit passes TRAIL_START_PROFIT, move SL to follow price with TRAIL_DISTANCE
                        if pos.profit >= TRAIL_START_PROFIT:
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                new_sl = bid - TRAIL_DISTANCE
                                # only move sl forward
                                if pos.sl is None or new_sl > pos.sl:
                                    set_position_sl_tp(pos.ticket, new_sl, pos.tp)
                                    logger.debug("Trailing BUY pos %s set SL -> %s", pos.ticket, new_sl)
                            elif pos.type == mt5.POSITION_TYPE_SELL:
                                new_sl = ask + TRAIL_DISTANCE
                                if pos.sl is None or new_sl < pos.sl:
                                    set_position_sl_tp(pos.ticket, new_sl, pos.tp)
                                    logger.debug("Trailing SELL pos %s set SL -> %s", pos.ticket, new_sl)

                    except Exception as e:
                        logger.exception("Exception managing position %s: %s", getattr(pos,'ticket',None), e)

            # Reset SMA lock when there are no positions (i.e., cycle complete)
            if num_position == 0 and sma_cross_locked:
                sma_cross_locked = False
                last_signal = "WAIT"
                buy_signal = False
                sell_signal = False
                prev_profit = 0

            # reset RSI lock when there are no positions
            if num_position == 0 and last_rsi == "BUY" and (current["SMA_LOW"] < prev['SMA_LOW']):
                rsi_cross_locked = False
                last_rsi = "WAIT"
            else:
                pass
            
            # reset RSI lock when there are no positions
            if num_position == 0 and last_rsi == "SELL" and (current["SMA_LOW"] > prev['SMA_LOW']):
                rsi_cross_locked = False
                last_rsi = "WAIT"
            else:
                pass

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
