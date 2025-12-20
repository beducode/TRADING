import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

# ===========================
# USER CONFIGURATION
# ===========================
SYMBOL = "BTCUSDm"
TIMEFRAME_ENTRY = mt5.TIMEFRAME_M1   # TREND ENTRY
RR = 1.0
if SYMBOL == "BTCUSDm":
    LOT = 0.03
else:
    LOT = 0.01
POLL_SECONDS = 1.0
MAX_POSITIONS = 1

# ===========================
# INDIKATOR CONFIGURATION
# ===========================

# ---- EMA SETUP ----
EMA_FAST = 8
EMA_SLOW = 21
EMA_TREND = 200

# ---- RSI SETUP ----
RSI_PERIOD = 14
MID_RSI_LEVEL = 50
RSI_LIMIT_BUY = 65
RSI_LIMIT_SELL = 35

# ===========================
# GLOBAL PARAM
# ===========================
buy_signal = False
sell_signal = False
signal_locked = False

# ===========================
# PROFIT TRAILING PARAM
# ===========================
PROFIT_TRIGGER = 1.0
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

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    close = close.astype(float).reset_index(drop=True)
    delta = close.diff()

    gain = delta.clip(lower=0).fillna(0)
    loss = -delta.clip(upper=0).fillna(0)

    # prepare arrays
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    rsi = np.full(len(close), np.nan)

    if len(close) <= period:
        return pd.Series(rsi)

    first_gain = gain.iloc[1:period+1].mean()
    first_loss = loss.iloc[1:period+1].mean()

    avg_gain[period] = first_gain
    avg_loss[period] = first_loss

    # Wilder smoothing for subsequent values
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss.iloc[i]) / period

    # compute RSI where avg_loss != 0
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_vals = 100 - (100 / (1 + rs))
        rsi = rsi_vals

    return pd.Series(rsi)

# ===========================
# MT5 SIGNAL
# ===========================
def get_trend_signal(df):
    close = df['close']
    open = df['open']
    filter = ema(close, EMA_TREND)

    price_open = open.iloc[-12:-2]
    filterema = filter.iloc[-12:-2]

    market_bullish = (price_open > filterema).all()
    market_bearish = (price_open < filterema).all()

    if market_bullish:
        return 'BUY'
    elif market_bearish:
        return 'SELL'
    else:
        return 'WAIT'

def get_rsi_signal(df):
    rsi = calculate_rsi(df['close'], RSI_PERIOD)

    rsi_buy = rsi.iloc[-2] > MID_RSI_LEVEL and rsi.iloc[-2] < RSI_LIMIT_BUY
    rsi_sell = rsi.iloc[-2] > RSI_LIMIT_SELL and rsi.iloc[-2] < MID_RSI_LEVEL

    rsi_valid_buy = rsi.iloc[-2] < rsi.iloc[-1]
    rsi_valid_sell = rsi.iloc[-2] > rsi.iloc[-1]

    if rsi_buy and rsi_valid_buy:
        return 'BUY'
    elif rsi_sell and rsi_valid_sell:
        return 'SELL'
    else:
        return 'WAIT'

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
            df = get_rates(SYMBOL, TIMEFRAME_ENTRY, n=500)    # M1
            if df is None or df.empty:
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

            # GET SIGNAL DARI INDIKATOR
            signal_tren = get_trend_signal(df)
            signal_rsi = get_rsi_signal(df)

            # VALIDASI SIGNAL
            if signal_tren == 'BUY':
                buy_signal = True
                sell_signal = False
            if signal_tren == 'SELL':
                sell_signal = True
                buy_signal = False

            # # ENTRY SIGNAL BUY
            # if buy_signal and num_position < MAX_POSITIONS:
            #     close = df['close']
            #     v_ema = ema(close, EMA_SLOW)
            #     emasl = v_ema.iloc[-1]
            #     volume = LOT
            #     price = ask
  
            #     sl_position = emasl
            #     tp_position = price + (price - sl_position) * RR
            #     place_market_order(SYMBOL, mt5.ORDER_TYPE_BUY, price, volume, sl_position, tp_position, comment="ENTRY BUY")
            #     buy_signal = False
            #     signal_locked = True
            #     time.sleep(POLL_SECONDS)

            # # VALIDASI SIGNAL SELL
            # if sell_signal and num_position < MAX_POSITIONS:
            #     close = df['close']
            #     v_ema = ema(close, EMA_SLOW)
            #     emasl = v_ema.iloc[-1]
            #     volume = LOT
            #     price = ask
  
            #     sl_position = emasl
            #     tp_position = price + (price - sl_position) * RR
            #     place_market_order(SYMBOL, mt5.ORDER_TYPE_SELL, price, volume, sl_position, tp_position, comment="ENTRY SELL")
            #     sell_signal = False
            #     signal_locked = True
            #     time.sleep(POLL_SECONDS)

            # PRINT RUNNING
            if num_position < MAX_POSITIONS:
                logger.info("PAIR : %s | TREND : %s | RSI : %s",SYMBOL, signal_tren, signal_rsi)
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
