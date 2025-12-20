import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

SYMBOL = "BTCUSDm"
LOT = 0.01
TIMEFRAME = mt5.TIMEFRAME_M1
RR = 2.0
MAGIC = 30060

logger = logging.getLogger("scalper")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

sma_cross_locked = False

# ---------------- MT5 INIT ----------------
mt5.initialize()

# ---------------- INDICATOR ----------------
def sma(series, p):
    return series.rolling(p).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

# ---------------- DATA ----------------
def get_data():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 150)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# ---------------- CANDLE FILTER ----------------
def valid_body(candle):
    body = abs(candle['close'] - candle['open'])
    total = candle['high'] - candle['low']
    return body / total >= 0.6 if total != 0 else False

# ---------------- ORDER ----------------
def send_order(direction, price, sl, tp):
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": MAGIC,
        "deviation": 20,
        "comment": "PRE_GOLDEN",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    return mt5.order_send(request)

# ---------------- MAIN LOOP ----------------
while True:
    df = get_data()

    df['sma30'] = sma(df['close'], 30)
    df['sma60'] = sma(df['close'], 60)
    df['sma100'] = sma(df['close'], 100)
    df['rsi'] = rsi(df['close'])
    df['adx'] = adx(df)

    c = df.iloc[-2]
    p = df.iloc[-3]

    if sma_cross_locked:
        if not mt5.positions_get(symbol=SYMBOL):
            sma_cross_locked = False
        time.sleep(1)
        continue

    # BUY
    if (
        c['close'] > c['sma100'] and
        c['sma30'] < c['sma60'] and
        (c['sma60'] - c['sma30']) < (p['sma60'] - p['sma30']) and
        c['rsi'] > 52 and p['rsi'] < 52 and
        c['adx'] > 22 and c['adx'] > p['adx'] and
        valid_body(c)
    ):
        price = mt5.symbol_info_tick(SYMBOL).ask
        sl = c['sma60']
        tp = price + (price - sl) * RR
        send_order("BUY", price, sl, tp)
        sma_cross_locked = True

    # SELL
    if (
        c['close'] < c['sma100'] and
        c['sma30'] > c['sma60'] and
        (c['sma30'] - c['sma60']) < (p['sma30'] - p['sma60']) and
        c['rsi'] < 48 and p['rsi'] > 48 and
        c['adx'] > 22 and c['adx'] > p['adx'] and
        valid_body(c)
    ):
        price = mt5.symbol_info_tick(SYMBOL).bid
        sl = c['sma60']
        tp = price - (sl - price) * RR
        send_order("SELL", price, sl, tp)
        sma_cross_locked = True

    logger.info("ADX: %s", c['adx'])

    time.sleep(1)
