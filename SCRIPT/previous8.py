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
TIMEFRAME_UTAMA = mt5.TIMEFRAME_M5   # TREND UTAMA
RR = 2.0
if SYMBOL == "BTCUSDm":
    LOT = 0.05
else:
    LOT = 0.01
POLL_SECONDS = 1.0
MAX_POSITIONS = 1

# ===========================
# INDIKATOR CONFIGURATION
# ===========================

# ---- EMA PERIODE ----
EMA_TREND = 100

# ---- STOCH PERIODE ----
STOCH_PERIODE = 8
K_SMOOTH = 3
D_PERIODE = 3
STOCH_HIGH_LEVEL = 70
STOCH_LOW_LEVEL = 30

# ---- ATR PERIODE ----
ATR_PERIODE = 8
ATR_HIGH_LEVEL = 30
ATR_LOW_LEVEL = 10
ATR_MOM_HIGH_LEVEL = 80
ATR_MOM_LOW_LEVEL = 60

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

def calculate_stochastic(df, k_period, k_smooth, d_period):
    df = df.copy()
    df['HH'] = df['high'].rolling(window=k_period).max()
    df['LL'] = df['low'].rolling(window=k_period).min()
    df['%K_raw'] = ((df['close'] - df['LL']) / (df['HH'] - df['LL'])) * 100
    df['%K'] = df['%K_raw'].rolling(window=k_smooth).mean()
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df

def atr_calculation(df):
    high = df['high']
    low = df['low']
    close = df['close']

    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )
    return tr.rolling(ATR_PERIODE).mean()

# ===========================
# MT5 SIGNAL
# ===========================
def get_trend_signal(df):
    close = df['close']
    open = df['open']
    trendema = ema(close, EMA_TREND)

    # LIHAT TREND EMA NAIK ATAU TURUN
    prev_ema = trendema.iloc[-3]
    curr_ema = trendema.iloc[-2]
    real_ema = trendema.iloc[-1]

    trend_buy = (real_ema > curr_ema > prev_ema)
    trend_sell = (real_ema < curr_ema < prev_ema)

    # KONDISI BEBERAPA CANDLE TERAKHIR
    last_open = open.iloc[-11:-1]
    last_ema = trendema.iloc[-11:-1]

    buy_validation = (last_open > last_ema).all()
    sell_validation = (last_open < last_ema).all()

    if trend_buy and buy_validation:
        return 'BUY'
    elif trend_sell and sell_validation:
        return 'SELL'
    else:
        return 'WAIT'
    
def stochastic_signal(df):
    if len(df) < STOCH_PERIODE:
        return "WAIT"

    stockbuy1 = (df['%K'].iloc[-2] < STOCH_LOW_LEVEL) and df['%K'].iloc[-1] > STOCH_LOW_LEVEL
    stockbuy2 = (df['%K'].iloc[-3] < STOCH_LOW_LEVEL) and df['%K'].iloc[-1] > STOCH_LOW_LEVEL
    stockbuy3 = (df['%K'].iloc[-4] < STOCH_LOW_LEVEL) and df['%K'].iloc[-1] > STOCH_LOW_LEVEL
    stockbuy4 = (df['%K'].iloc[-5] < STOCH_LOW_LEVEL) and df['%K'].iloc[-1] > STOCH_LOW_LEVEL

    stocksell1 = (df['%K'].iloc[-2] > STOCH_HIGH_LEVEL) and df['%K'].iloc[-1] < STOCH_HIGH_LEVEL
    stocksell2 = (df['%K'].iloc[-3] > STOCH_HIGH_LEVEL) and df['%K'].iloc[-1] < STOCH_HIGH_LEVEL
    stocksell3 = (df['%K'].iloc[-4] > STOCH_HIGH_LEVEL) and df['%K'].iloc[-1] < STOCH_HIGH_LEVEL
    stocksell4 = (df['%K'].iloc[-5] > STOCH_HIGH_LEVEL) and df['%K'].iloc[-1] < STOCH_HIGH_LEVEL

    buy_stoch = (stockbuy1 or stockbuy2 or stockbuy3 or stockbuy4) and (df['%K'].iloc[-1] >= df['%D'].iloc[-1])
    sell_stoch = (stocksell1 or stocksell2 or stocksell3 or stocksell4) and (df['%K'].iloc[-1] <= df['%D'].iloc[-1])

    if buy_stoch:
        return 'BUY'
    elif sell_stoch:
        return 'SELL'
    else:
        return 'WAIT'
    
def get_atr_point(df):
    data_atr = atr_calculation(df)

    atr = data_atr.iloc[-1]

    return atr

def atr_position(df):
    data_atr = atr_calculation(df)

    atr = data_atr.iloc[-1]

    if (round(atr) > ATR_LOW_LEVEL) and (round(atr) < ATR_HIGH_LEVEL):
        return True

    return False

def atr_momentum(df):
    data_atr = atr_calculation(df)

    atrmom = data_atr.iloc[-1]

    if (round(atrmom) > ATR_MOM_LOW_LEVEL) and (round(atrmom) < ATR_MOM_HIGH_LEVEL):
        return True

    return False

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
            dfmain = get_rates(SYMBOL, TIMEFRAME_UTAMA, n=500)    # M5
            if df is None or df.empty and dfmain is None or dfmain.empty:
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

            # GET DATA INDIKATOR
            data_stoch = calculate_stochastic(df, STOCH_PERIODE, K_SMOOTH, D_PERIODE)

            # GET SIGNAL DARI INDIKATOR
            signal_tren = get_trend_signal(df)
            signal_stoch = stochastic_signal(data_stoch)
            signal_atr = atr_position(df)
            atr_point = get_atr_point(df)
            

            # VALIDASI SIGNAL
            if signal_tren == 'BUY' and signal_stoch == 'BUY' and signal_atr:
                buy_signal = True
                sell_signal = False
            if signal_tren == 'SELL' and signal_stoch == 'SELL' and signal_atr:
                sell_signal = True
                buy_signal = False

            # ENTRY SIGNAL BUY
            if buy_signal and num_position < MAX_POSITIONS:
                close = df['close']
                v_ema = ema(close, EMA_TREND)
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
                v_ema = ema(close, EMA_TREND)
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
                logger.info("PAIR : %s | TREND : %s | STOCH CROSS : %s | ATR : %s",SYMBOL, signal_tren, signal_stoch, signal_atr)
            else:
                logger.info("POSITIONS OPEN : %s | PAIR : %s", num_position, SYMBOL)
            
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
