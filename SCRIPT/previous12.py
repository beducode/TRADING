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
SYMBOLS = CONFIG["symbols"]

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
# RSI CONFIG
# ===========================
RSI_PERIOD = CONFIG["rsi"]["period"]
RSI_HIGH = CONFIG["rsi"]["rsi_high"]
RSI_LOW = CONFIG["rsi"]["rsi_low"]

# ===========================
# ADX CONFIG
# ===========================
ADX_PERIOD = CONFIG["adx"]["period"]

# ===========================
# PROFIT TRAILING
# ===========================
PROFIT_TRIGGER = CONFIG["profit_trailing"]["profit_trigger"]

# ===========================
# TRAILING STOP CONFIG
# ===========================
TRAIL_ENABLE = CONFIG["trailing_stop"]["enable"]
TRAIL_ATR_MULT = CONFIG["trailing_stop"]["atr_multiplier"]
TRAIL_START_PROFIT = CONFIG["trailing_stop"]["start_profit"]


# ===========================
# GLOBAL PARAM
# ===========================
symbol_state = {}

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
# STATE
# ==========================
def init_symbol_state(symbol):
    if symbol not in symbol_state:
        symbol_state[symbol] = {
            "buy_signal": False,
            "sell_signal": False,
            "signal_locked": False,
            "profit_peak": {}
        }

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

def rsi(series):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def rsi_bias(df):
    r = rsi(df['close'])

    bullish = r.iloc[-2] > RSI_HIGH and r.iloc[-2] > r.iloc[-3]
    bearish = r.iloc[-2] < RSI_LOW and r.iloc[-2] < r.iloc[-3]

    return bullish, bearish

def calculate_adx(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr2'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['tr0','tr1','tr2']].max(axis=1)
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['DM_plus'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['DM_minus'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    TR_smooth = df['TR'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    DMp_smooth = df['DM_plus'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    DMm_smooth = df['DM_minus'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    df['DI_plus'] = 100 * (DMp_smooth / TR_smooth)
    df['DI_minus'] = 100 * (DMm_smooth / TR_smooth)
    df['DX'] = ( (df['DI_plus'] - df['DI_minus']).abs() / (df['DI_plus'] + df['DI_minus']) ) * 100
    adx = df['DX'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
    return adx

def adx_trend(df):
    adx_val = calculate_adx(df).iloc[-2]
    return adx_val > 20   # trend kuat

def volume_confirmation(df):
    vol = df['tick_volume']
    vol_ma = vol.rolling(20).mean()

    return vol.iloc[-2] > vol_ma.iloc[-2] * 1.3

    
def strong_ema_cross(df, atr):
    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']

    emafast = ema(close, EMA_FAST)
    emaslow = ema(close, EMA_SLOW)

    close_curr = close.iloc[-2]
    open_curr = open_.iloc[-2]
    high_curr = high.iloc[-2]
    low_curr = low.iloc[-2]

    # ===== CROSS =====
    bullish_cross = (emafast.iloc[-5] < emaslow.iloc[-5] or (emafast.iloc[-4] < emaslow.iloc[-4]) or (emafast.iloc[-3] < emaslow.iloc[-3])) and emafast.iloc[-2] > emaslow.iloc[-2] and close_curr > emaslow.iloc[-2]
    bearish_cross = (emafast.iloc[-5] > emaslow.iloc[-5] or (emafast.iloc[-4] > emaslow.iloc[-4]) or (emafast.iloc[-3] > emaslow.iloc[-3])) and emafast.iloc[-2] < emaslow.iloc[-2] and close_curr < emaslow.iloc[-2]

    # ===== SLOPE EMA21 =====
    ema21_slope = emaslow.iloc[-2] - emaslow.iloc[-3]

    # ===== JARAK EMA =====
    ema_distance = abs(emafast.iloc[-2] - emaslow.iloc[-2])

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
    
def close_by_retrace(symbol, positions):
    state = symbol_state[symbol]

    for pos in positions:
        ticket = pos.ticket
        current_profit = pos.profit

        if ticket not in state["profit_peak"]:
            state["profit_peak"][ticket] = current_profit
        else:
            state["profit_peak"][ticket] = max(
                state["profit_peak"][ticket],
                current_profit
            )

        if (
            state["profit_peak"][ticket] > PROFIT_TRIGGER
            and current_profit < PROFIT_TRIGGER
        ):
            logger.info(
                "[%s] PROFIT RETRACE | TICKET=%s | PEAK=%.2f | NOW=%.2f",
                symbol, ticket,
                state["profit_peak"][ticket],
                current_profit
            )
            close_position(pos)
            state["profit_peak"].pop(ticket, None)
            state["signal_locked"] = False


def modify_sl(position, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl": float(new_sl),
        "tp": position.tp,
        "magic": position.magic,
        "comment": "TRAILING STOP"
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            "TRAIL SL MOVED | TICKET=%s | NEW SL=%.2f",
            position.ticket, new_sl
        )
        return True
    return False

def apply_trailing_stop(position, atr_value):
    if not TRAIL_ENABLE:
        return

    tick = get_tick(position.symbol)
    if tick is None:
        return

    # START TRAILING JIKA PROFIT SUDAH CUKUP
    if position.profit < TRAIL_START_PROFIT:
        return

    trail_distance = atr_value * TRAIL_ATR_MULT

    if position.type == mt5.ORDER_TYPE_BUY:
        new_sl = tick.bid - trail_distance
        if position.sl is None or new_sl > position.sl:
            modify_sl(position, new_sl)

    elif position.type == mt5.ORDER_TYPE_SELL:
        new_sl = tick.ask + trail_distance
        if position.sl is None or new_sl < position.sl:
            modify_sl(position, new_sl)


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

def process_symbol(symbol):
    init_symbol_state(symbol)
    state = symbol_state[symbol]

    if is_market_closed(symbol):
        logger.warning("[%s] MARKET CLOSED", symbol)
        return

    df = get_rates(symbol, TIMEFRAME_ENTRY, 500)
    dfmain = get_rates(symbol, TIMEFRAME_CONFIRM, 500)

    if df is None or df.empty or dfmain is None or dfmain.empty:
        return

    positions = positions_for_symbol(symbol)
    num_position = len(positions)

    tick = get_tick(symbol)
    if tick is None:
        return

    bid, ask = tick.bid, tick.ask

    # ===== MANAGE POSISI =====
    if num_position > 0:
        atr_value = atr(df).iloc[-2]

        for pos in positions:
            apply_trailing_stop(pos, atr_value)

        close_by_retrace(symbol, positions)
        return

    # ===== SIGNAL =====
    trendentry = get_trend_signal(df)
    atr_value = atr(df).iloc[-2]
    signal_cross = strong_ema_cross(df, atr_value)
    bull_rsi, bear_rsi = rsi_bias(df)
    adx_ok = adx_trend(df)
    vol_ok = volume_confirmation(df)

    bullish_valid = (
        trendentry == 'BUY'
        and signal_cross == 'BUY'
        and bull_rsi
        and adx_ok
        and vol_ok
    )

    bearish_valid = (
        trendentry == 'SELL'
        and signal_cross == 'SELL'
        and bear_rsi
        and adx_ok
        and vol_ok
    )

    if bullish_valid:
        state["buy_signal"] = True
        state["sell_signal"] = False

    if bearish_valid:
        state["sell_signal"] = True
        state["buy_signal"] = False

    # ===== ENTRY BUY =====
    if state["buy_signal"] and num_position < MAX_POSITIONS:
        emasl = ema(df['close'], EMA_TREND).iloc[-1]
        price = ask
        sl = emasl
        tp = price + (price - sl) * RR

        place_market_order(symbol, mt5.ORDER_TYPE_BUY, price, LOT, sl, tp, comment="BUY")
        state["buy_signal"] = False
        state["signal_locked"] = True

    # ===== ENTRY SELL =====
    if state["sell_signal"] and num_position < MAX_POSITIONS:
        emasl = ema(df['close'], EMA_TREND).iloc[-1]
        price = bid
        sl = emasl
        tp = price + (price - sl) * RR

        place_market_order(symbol, mt5.ORDER_TYPE_SELL, price, LOT, sl, tp, comment="SELL")
        state["sell_signal"] = False
        state["signal_locked"] = True

    logger.info("[%s] TREND :%s | CROSS :%s",symbol, trendentry, signal_cross)


# ===========================
# MAIN LOOP
# ===========================
def main():
    init_mt5()

    for symbol in SYMBOLS:
        symbol_info_check(symbol)

    logger.info("MULTI PAIR BOT STARTED: %s", ", ".join(SYMBOLS))

    while True:
        try:
            for symbol in SYMBOLS:
                process_symbol(symbol)

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("MAIN LOOP ERROR: %s", e)

    shutdown_mt5()


if __name__ == "__main__":
    clear_screen()
    main()
