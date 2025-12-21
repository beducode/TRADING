import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

# ===========================
# LOAD CONFIG JSON
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# ===========================
# MAP TIMEFRAME STRING → MT5
# ===========================
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1
}

# ===========================
# USER CONFIGURATION
# ===========================
SYMBOL = CONFIG["symbol"]

TIMEFRAME_ENTRY = TF_MAP[CONFIG["timeframe"]["entry"]]
TIMEFRAME_CONFIRM = TF_MAP[CONFIG["timeframe"]["confirm"]]

RR = CONFIG["trading"]["rr"]
LOT = CONFIG["trading"]["lot"]
POLL_SECONDS = CONFIG["trading"]["poll_seconds"]
MAX_POSITIONS = CONFIG["trading"]["max_positions"]

# ===========================
# EMA CONFIG
# ===========================
EMA_FAST = CONFIG["ema"]["fast"]
EMA_SLOW = CONFIG["ema"]["slow"]
EMA_TREND = CONFIG["ema"]["trend"]

# ===========================
# ATR CONFIG
# ===========================
ATR_PERIOD = CONFIG["atr"]["period"]

# ===========================
# PROFIT TRAILING
# ===========================
PROFIT_TRIGGER = CONFIG["profit_trailing"]["profit_trigger"]

# ===========================
# GLOBAL PARAM
# ===========================
buy_signal = False
sell_signal = False
signal_locked = False
profit_peak = {}

# ===========================
# LOGING SETUP
# ===========================
logger = logging.getLogger("scalper")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# ===========================
# MT5 INDIKATOR
# ===========================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# ===========================
# MT5 SIGNAL
# ===========================
def get_trend_signal(df):
    close = df['close']
    open = df['open']
    fast = ema(close, EMA_FAST)
    slow = ema(close, EMA_SLOW)
    trend = ema(close, EMA_TREND)

    price_open = open.iloc[-2]
    emafast = fast.iloc[-2]
    emaslow = slow.iloc[-2]
    ematrend = trend.iloc[-2]
    
    market_bullish = emafast > emaslow > ematrend and price_open > ematrend
    market_bearish = emafast < emaslow < ematrend and price_open < ematrend

    if market_bullish:
        return 'BUY'
    elif market_bearish:
        return 'SELL'
    else:
        return 'WAIT'
    
def atr(df):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean()
    
def strong_ema_cross(df, atr):
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    ema8 = ema(close, EMA_FAST)
    ema21 = ema(close, EMA_SLOW)

    # nilai candle sebelumnya & sekarang
    ema8_prev, ema8_curr = ema8.iloc[-2], ema8.iloc[-1]
    ema21_prev, ema21_curr = ema21.iloc[-2], ema21.iloc[-1]

    close_curr = close.iloc[-1]
    open_curr = open_.iloc[-1]
    high_curr = high.iloc[-1]
    low_curr = low.iloc[-1]

    # ===== CROSS =====
    bullish_cross = ema8_prev < ema21_prev and ema8_curr > ema21_curr and close_curr > ema21_curr
    bearish_cross = ema8_prev > ema21_prev and ema8_curr < ema21_curr and close_curr < ema21_curr

    # ===== SLOPE EMA21 =====
    ema21_slope = ema21_curr - ema21_prev

    # ===== JARAK EMA =====
    ema_distance = abs(ema8_curr - ema21_curr)

    # ===== BODY CANDLE =====
    body = abs(close_curr - open_curr)
    range_ = high_curr - low_curr
    body_ratio = body / range_ if range_ > 0 else 0

    # ===== FILTER =====
    strong_body = body_ratio > 0.6
    enough_distance = ema_distance > (round(atr,2) * 0.1)
    if bullish_cross and ema21_slope > 0 and strong_body and enough_distance:
        return 'BUY'

    if bearish_cross and ema21_slope < 0 and strong_body and enough_distance:
        return 'SELL'

    return 'WAIT'

def detect_swings(df, left=2, right=2):
    swings_high = []
    swings_low = []

    for i in range(left, len(df) - right):
        high = df['high']
        low = df['low']

        if high[i] == max(high[i-left:i+right+1]):
            swings_high.append((i, high[i]))

        if low[i] == min(low[i-left:i+right+1]):
            swings_low.append((i, low[i]))

    return swings_high, swings_low

def is_higher_high_higher_low(swings_high, swings_low):
    if len(swings_high) < 2 or len(swings_low) < 2:
        return False

    prev_high = swings_high[-2][1]
    curr_high = swings_high[-1][1]

    prev_low = swings_low[-2][1]
    curr_low = swings_low[-1][1]

    HHHL = curr_high > prev_high and curr_low > prev_low

    return HHHL

def is_lower_high_lower_low(swings_high, swings_low):
    if len(swings_high) < 2 or len(swings_low) < 2:
        return False

    prev_high = swings_high[-2][1]
    curr_high = swings_high[-1][1]

    prev_low = swings_low[-2][1]
    curr_low = swings_low[-1][1]

    LHLL = curr_high < prev_high and curr_low < prev_low

    return LHLL


# ===========================
# MT5 HELPER FUNCTIONS
# ===========================
def init_mt5():
    if not mt5.initialize():
        logger.error("MT5 INITIALIZE() FAILED, ERROR: %s", mt5.last_error())
        raise SystemExit("MT5 INIT FAILED")

def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception as e:
        logger.exception("ERROR SHUTTING DOWN MT5: %s", e)

def place_market_order(symbol, order_type, price, volume, sl_price, tp_price, deviation=20, magic=234000, comment="Market Order"):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("PLACE_MARKET_ORDER: SYMBOL_INFO NONE")
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
            logger.error("ORDER_SEND RETURNED NONE")
            return None
        if hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("BERHASIL BUKA POSISI! ORDER: %s DEAL: %s", getattr(result, 'order', None), getattr(result, 'deal', None))
        else:
            # Some brokers return different retcodes; log full result
            logger.warning("GAGAL BUKA POSISI. RETCODE=%s COMMENT=%s RESULT=%s", getattr(result, 'retcode', None), getattr(result, 'comment', None), result)
        return result
    except Exception as e:
        logger.exception("PLACE_MARKET_ORDER EXCEPTION: %s", e)
        return None

def is_market_closed(symbol):
    info = mt5.symbol_info(symbol)
    tick = get_tick(symbol)

    if info is None:
        return True

    # JIKA SYMBOL TRADING DISABLED → MARKET TUTUP
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return True

    # JIKA TICK TIDAK ADA → MARKET TUTUP
    if tick is None:
        return True

    # JIKA BID/ASK 0 → MARKET TUTUP
    if tick.bid == 0 or tick.ask == 0:
        return True

    return False

def close_position(position):
    tick = get_tick(position.symbol)
    if tick is None:
        return False

    if position.type == mt5.ORDER_TYPE_BUY:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": position.magic,
        "comment": "CLOSE PROFIT RETRACE"
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("POSITION CLOSED | TICKET=%s | PROFIT=%.2f", position.ticket, position.profit)
        return True
    else:
        logger.warning("FAILED CLOSE POSITION | TICKET=%s", position.ticket)
        return False
    
def close_by_retrace(position):
    global signal_locked
    # ===========================
    # PROFIT RETRACEMENT CHECK
    # ===========================
    for pos in position:
        ticket = pos.ticket
        current_profit = pos.profit

        # SIMPAN PROFIT TERTINGGI
        if ticket not in profit_peak:
            profit_peak[ticket] = current_profit
        else:
            profit_peak[ticket] = max(profit_peak[ticket], current_profit)

        # KONDISI CLOSE
        if profit_peak[ticket] > PROFIT_TRIGGER and current_profit < PROFIT_TRIGGER:
            logger.info(
                "PROFIT RETRACE DETECTED | TICKET=%s | PEAK=%.2f | NOW=%.2f",
                ticket, profit_peak[ticket], current_profit
            )
            close_position(pos)
            profit_peak.pop(ticket, None)
            signal_locked = False

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
        raise RuntimeError(f"{symbol} NOT FOUND IN MARKET WATCH")
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return info

def positions_for_symbol(symbol):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None or len(pos) == 0:
        return []
    return list(pos)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ===========================
# MAIN LOOP
# ===========================
def main():
    global buy_signal, sell_signal, signal_locked

    init_mt5()
    try:
        symbol_info_check(SYMBOL)
    except Exception as e:
        logger.exception("SYMBOL CHECK ERROR: %s", e)
        shutdown_mt5()
        return

    clear_screen()
    logger.info("STARTING SCALPER FOR %s M1 — %s", SYMBOL, datetime.now().isoformat())

    while True:
        try:
            # CEK MARKET CLOSE
            if is_market_closed(SYMBOL):
                logger.warning("MARKET CLOSED FOR %s — WAITING 1 HOURS...", SYMBOL)
                time.sleep(3600)  # 1 jam = 3600 detik
                continue
            
            # GET DATA RATES
            df = get_rates(SYMBOL, TIMEFRAME_ENTRY, n=500)
            dfmain = get_rates(SYMBOL, TIMEFRAME_CONFIRM, n=500) 
            if df is None or df.empty or dfmain is None or dfmain.empty:
                logger.warning("NO RATES RETRIEVED, RETRYING...")
                time.sleep(POLL_SECONDS)
                continue

            # CHECK CURRENT POSITIONS
            position = positions_for_symbol(SYMBOL)
            num_position = len(position)

            tick = get_tick(SYMBOL)
            if tick is None:
                time.sleep(POLL_SECONDS)
                continue
            bid = tick.bid
            ask = tick.ask

            if signal_locked or num_position > 0:
                close_by_retrace(position)
                continue

            # GET SIGNAL DARI INDIKATO
            swings_high, swings_low = detect_swings(dfmain)

            #---- SIGNAL MAIN
            if is_higher_high_higher_low(swings_high, swings_low):
                trendmain = "UPTREND"
            elif is_lower_high_lower_low(swings_high, swings_low):
                trendmain = "DOWNTREND"
            else:
                trendmain = "WAIT"

            #---- SIGNAL ENTRY
            trendentry = get_trend_signal(df)

            #---- CROSS EMA
            atr_value = atr(df).iloc[-2]
            signal_ema_cross = strong_ema_cross(df, atr_value)

            # VALIDASI SIGNAL
            if trendmain == 'UPTREND' and trendentry == 'BUY' and signal_ema_cross == 'BUY':
                buy_signal = True
                sell_signal = False
            if trendmain == 'DOWNTREND' and trendentry == 'SELL' and signal_ema_cross == 'SELL':
                sell_signal = True
                buy_signal = False

            # ENTRY SIGNAL BUY
            if buy_signal and num_position < MAX_POSITIONS:
                close = df['close']
                v_ema = ema(close, EMA_SLOW)
                emasl = v_ema.iloc[-1]
                volume = LOT
                price = ask
  
                sl_position = emasl
                tp_position = price + (price - sl_position) * RR
                place_market_order(SYMBOL, mt5.ORDER_TYPE_BUY, price, volume, sl_position, tp_position, comment="ENTRY BUY")
                buy_signal = False
                signal_locked = True
                time.sleep(POLL_SECONDS)

            # VALIDASI SIGNAL SELL
            if sell_signal and num_position < MAX_POSITIONS:
                close = df['close']
                v_ema = ema(close, EMA_SLOW)
                emasl = v_ema.iloc[-1]
                volume = LOT
                price = ask
  
                sl_position = emasl
                tp_position = price + (price - sl_position) * RR
                place_market_order(SYMBOL, mt5.ORDER_TYPE_SELL, price, volume, sl_position, tp_position, comment="ENTRY SELL")
                sell_signal = False
                signal_locked = True
                time.sleep(POLL_SECONDS)

            # PRINT RUNNING
            if num_position < MAX_POSITIONS:
                logger.info("PAIR : %s | TM : %s | TE : %s | CROSS : %s | ATR : %s",SYMBOL, trendmain, trendentry, signal_ema_cross, round(atr_value,2))
            else:
                logger.info("POSITIONS OPEN : %s | PAIR : %s", num_position, SYMBOL)
                clear_screen()
            
            # RESET GLOBAL PARAM
            if num_position == 0 and signal_locked:
                buy_signal = False
                sell_signal = False
                signal_locked = False
                profit_peak.clear()

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
