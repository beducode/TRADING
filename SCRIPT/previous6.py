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
SYMBOL = "XAUUSDm"  # symbol chosen BTCUSDm
TIMEFRAME_SIGNAL = mt5.TIMEFRAME_M15   # timeframe signal utama
TIMEFRAME_CONFIRM = mt5.TIMEFRAME_M1   # timeframe entry / konfirmasi
RR = 1.0
LOT = 0.01
POLL_SECONDS = 1.0 
STOCHASTIC_MAX = 70
STOCHASTIC_MID = 50
STOCHASTIC_MIN = 30
EMA_TREND_PERIOD = 100
MAX_POSITIONS = 1

# Signal flags
buy_signal = False
sell_signal = False
signal = "WAIT"
stoch_buy_setup = False
stoch_sell_setup = False
prev_profit = 0

# LOCK: only one cross read per cycle
signal_locked = False
candle_signal = False
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

# ===========================
# MT5 Indikator
# ===========================
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
    if len(df2) < 20:
        return "SIDEWAYS"

    k_prev = df2['%K'].iloc[-3]
    d_prev = df2['%D'].iloc[-3]
    k_now  = df2['%K'].iloc[-2]
    d_now  = df2['%D'].iloc[-2]

    # SETUP CONDITIONS
    if k_prev > k_now and d_prev > d_now and (k_prev > STOCHASTIC_MID and d_prev > STOCHASTIC_MID):
        stoch_sell_setup = True
    if k_prev < k_now and d_prev < d_now and (k_prev < STOCHASTIC_MID and d_prev < STOCHASTIC_MID):
        stoch_buy_setup = True

    sell_cross_down = k_now < d_now
    sell_break_below = k_now < STOCHASTIC_MAX and d_now < STOCHASTIC_MAX

    if stoch_sell_setup and sell_cross_down and sell_break_below:
        stoch_sell_setup = False
        stoch_buy_setup = False
        return "SELL"

    buy_cross_up = k_now > d_now
    buy_break_above = k_now > STOCHASTIC_MIN and d_now > STOCHASTIC_MIN

    if stoch_buy_setup and buy_cross_up and buy_break_above:
        stoch_buy_setup = False
        stoch_sell_setup = False
        return "BUY"

    return "SIDEWAYS"

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

# ---------------- TREND FILTER ----------------
def get_trend_position(df):
    low = df['low']
    high = df['high']
    close = df['close']

    # pastikan data cukup
    if len(df) < 20:
        return 'SIDEWAYS'

    trend = ema(close, EMA_TREND_PERIOD)

    # candle sinyal = candle yang sudah close
    low_signal = low.iloc[-2]
    high_signal = high.iloc[-2]
    ema_signal = trend.iloc[-2]

    prev_ema = trend.iloc[-3]
    curr_ema = trend.iloc[-2]

    # Cek 15 Candle Sebelumnya
    last_low = low.iloc[-17:-2]
    last_high = high.iloc[-17:-2]
    last_ema = trend.iloc[-17:-2]

    # BUY: semua low candle di atas EMA
    valid_buy_trend = (last_low > last_ema).all()

    # SELL: semua high candle di bawah EMA
    valid_sell_trend = (last_high < last_ema).all()

    # === KONDISI FINAL ===
    if valid_buy_trend and low_signal > ema_signal and curr_ema > prev_ema:
        return 'BUY'

    if valid_sell_trend and high_signal < ema_signal and curr_ema < prev_ema:
        return 'SELL'

    return 'SIDEWAYS'

def get_trend_signal(df):
    low = df['low']
    high = df['high']
    close = df['close']

    trend = ema(close, EMA_TREND_PERIOD)

    # candle sinyal = candle yang sudah close
    low_signal = low.iloc[-1]
    high_signal = high.iloc[-1]
    ema_signal = trend.iloc[-1]

    # === KONDISI FINAL ===
    if low_signal > ema_signal:
        return 'BUY'

    if high_signal < ema_signal:
        return 'SELL'

    return 'SIDEWAYS'

# ---------------- CANDLE FILTER ----------------
def valid_body(df):
    candle = df.iloc[-2]
    body = abs(candle['close'] - candle['open'])
    total = candle['high'] - candle['low']
    return body / total >= 0.6 if total != 0 else False

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
    
    v_ema = ema(close, EMA_TREND_PERIOD)
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
            new_sl = v_ema.iloc[-2]
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
            new_sl = v_ema.iloc[-2]
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
    global buy_signal, sell_signal, last_rsi, prev_profit, signal_locked, candle_signal

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

            df_signal = get_rates(SYMBOL, TIMEFRAME_SIGNAL, n=500)  # M30
            df_confirm = get_rates(SYMBOL, TIMEFRAME_CONFIRM, n=500)    # M1
            if df_signal is None or df_signal.empty or df_confirm is None or df_confirm.empty:
                logger.warning("NO RATES RETRIEVED, RETRYING...")
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

            if signal_locked:
                manage_position(df_signal)
                if num_position == 0:
                    signal_locked = False
                continue

            # ===== SIGNAL DARI TF 30 =====
            trend_signal = get_trend_signal(df_signal)
            # ===== SIGNAL DARI TF 1 =====
            trend_entry = get_trend_position(df_confirm)
            stoch_signal = stochastic_signals(df_confirm)
            candle_signal = valid_body(df_confirm)

            # ===== SIGNAL TREND DARI TF 30 & CONFIRM DARI TF 1 =====
            if trend_signal == 'BUY' and trend_entry == 'BUY' and stoch_signal == 'BUY' and candle_signal:
                buy_signal = True
                sell_signal = False
            if trend_signal == 'SELL' and trend_entry == 'BUY' and stoch_signal == 'SELL' and candle_signal:
                sell_signal = True
                buy_signal = False

            if buy_signal and num_position < MAX_POSITIONS:
                close = df_signal['close']
                v_ema = ema(close, EMA_TREND_PERIOD) - 0.20
                emasl = v_ema.iloc[-2]
                volume = LOT
                price = ask
  
                sl_position = emasl
                tp_position = price + (price - sl_position) * RR

                res = place_market_order(SYMBOL, mt5.ORDER_TYPE_BUY, price, volume, sl_position, tp_position, comment="Entry Buy")
                buy_signal = False
                prev_profit = 0
                signal_locked = True
                time.sleep(0.2)

            if sell_signal and num_position < MAX_POSITIONS:
                close = df_signal['close']
                v_ema = ema(close, EMA_TREND_PERIOD) + 0.20
                emasl = v_ema.iloc[-2]
                volume = LOT
                price = bid

                sl_position = emasl
                tp_position = price - (sl_position - price) * RR

                res = place_market_order(SYMBOL, mt5.ORDER_TYPE_SELL, price, volume, sl_position, tp_position, comment="Entry Sell")
                sell_signal = False
                prev_profit = 0
                signal_locked = True
                time.sleep(0.2)
                
            # Logging
            if num_position < MAX_POSITIONS:
                logger.info("TREND: %s | ENTRY: %s | CROSS STOCH: %s | CANDLE CHECK: %s",trend_signal, trend_entry, stoch_signal, candle_signal)
            else:
                logger.info("POSITIONS OPEN: %s", num_position)

            # Reset EMA lock when there are no positions (i.e., cycle complete)
            if num_position == 0 and signal_locked:
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
