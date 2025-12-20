import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime

# ================== CONFIG ==================
SYMBOLS = {
    "XAUUSDm": {"lot": 0.01, "magic": 90001, "rr": 2.0},
    "BTCUSDm": {"lot": 0.01, "magic": 90002, "rr": 1.5}
}

TF_ENTRY = mt5.TIMEFRAME_M1
TF_TREND = mt5.TIMEFRAME_M5

ATR_PERIOD = 14
ATR_MULT_SL = 1.0
ATR_MULT_TRAIL = 0.8

MAX_TRADE_PER_SYMBOL = 1
PARTIAL_CLOSE_PCT = 0.5
MAX_DD_PERCENT = 5

SESSION_LONDON = (7, 16)
SESSION_NY = (13, 21)

NEWS_BLOCK = [(12, 13), (19, 20)]
# ============================================


# ================= MT5 INIT =================
if not mt5.initialize():
    raise RuntimeError("MT5 NOT CONNECTED")


# =============== INDICATORS =================
def ema(series, p):
    return series.ewm(span=p, adjust=False).mean()

def atr(df, p=ATR_PERIOD):
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def rsi(series, p=14):
    d = series.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = -d.where(d < 0, 0).rolling(p).mean()
    rs = g / l
    return 100 - (100 / (1 + rs))

def adx(df, p=14):
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().abs()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = atr(df, 1)
    plus_di = 100 * pd.Series(plus_dm).rolling(p).sum() / tr.rolling(p).sum()
    minus_di = 100 * pd.Series(minus_dm).rolling(p).sum() / tr.rolling(p).sum()
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    return dx.rolling(p).mean()


# ============ MARKET STRUCTURE ==============
def detect_swings(df, l=1, r=1):
    sh, sl = [], []
    for i in range(l, len(df) - r):
        if df['high'][i] == max(df['high'][i-l:i+r+1]):
            sh.append(df['high'][i])
        if df['low'][i] == min(df['low'][i-l:i+r+1]):
            sl.append(df['low'][i])
    return sh, sl

def uptrend(df):
    sh, sl = detect_swings(df)
    return len(sh) > 1 and len(sl) > 1 and sh[-1] > sh[-2] and sl[-1] > sl[-2]

def downtrend(df):
    sh, sl = detect_swings(df)
    return len(sh) > 1 and len(sl) > 1 and sh[-1] < sh[-2] and sl[-1] < sl[-2]


# ================= FILTER ===================
def session_ok():
    h = datetime.now().hour
    return (SESSION_LONDON[0] <= h <= SESSION_LONDON[1] or
            SESSION_NY[0] <= h <= SESSION_NY[1])

def news_block():
    h = datetime.now().hour
    return any(start <= h <= end for start, end in NEWS_BLOCK)

def equity_guard():
    acc = mt5.account_info()
    if not acc:
        return False
    dd = (acc.balance - acc.equity) / acc.balance * 100
    return dd < MAX_DD_PERCENT


# ================ ENTRY =====================
def pullback_buy(df):
    e9, e21 = ema(df['close'], 9), ema(df['close'], 21)
    return df['low'].iloc[-2] <= e21.iloc[-2] and df['close'].iloc[-2] > e9.iloc[-2]

def pullback_sell(df):
    e9, e21 = ema(df['close'], 9), ema(df['close'], 21)
    return df['high'].iloc[-2] >= e21.iloc[-2] and df['close'].iloc[-2] < e9.iloc[-2]

def bullish_rejection(o, c, l):
    return (o - l) > abs(c - o) * 1.5

def bearish_rejection(o, c, h):
    return (h - o) > abs(c - o) * 1.5

def mtf_ok(df5, side):
    r = rsi(df5['close']).iloc[-1]
    a = adx(df5).iloc[-1]
    if side == 'buy':
        return uptrend(df5) and r > 50 and a > 20
    return downtrend(df5) and r < 50 and a > 20


# =============== TRADING ====================
def get_positions(symbol, magic):
    pos = mt5.positions_get(symbol=symbol)
    return [] if pos is None else [p for p in pos if p.magic == magic]

def send_order(symbol, side, df, cfg):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == 'buy' else tick.bid
    a = atr(df).iloc[-1]

    sl = price - a * ATR_MULT_SL if side == 'buy' else price + a * ATR_MULT_SL
    tp = price + a * cfg['rr'] if side == 'buy' else price - a * cfg['rr']

    mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": cfg['lot'],
        "type": mt5.ORDER_TYPE_BUY if side == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": cfg['magic'],
        "comment": "MS_MULTI_FULL",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    })


def manage_positions(symbol, df, cfg):
    a = atr(df).iloc[-1]
    tick = mt5.symbol_info_tick(symbol)

    for p in get_positions(symbol, cfg['magic']):
        price = tick.bid if p.type == 0 else tick.ask
        profit = abs(price - p.price_open)

        # Breakeven
        if profit >= a and abs(p.sl - p.price_open) > a * 0.1:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": p.price_open,
                "tp": p.tp
            })

        # Partial TP
        if profit >= a * cfg['rr'] and p.volume > cfg['lot'] * 0.6:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": p.ticket,
                "volume": p.volume * PARTIAL_CLOSE_PCT,
                "type": mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 20,
                "magic": cfg['magic']
            })

        # Trailing SL
        if p.type == 0:
            new_sl = price - a * ATR_MULT_TRAIL
            if new_sl > p.sl:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": p.ticket,
                    "sl": new_sl,
                    "tp": p.tp
                })
        else:
            new_sl = price + a * ATR_MULT_TRAIL
            if new_sl < p.sl:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": p.ticket,
                    "sl": new_sl,
                    "tp": p.tp
                })


# ============== EXECUTION ===================
def execute_symbol(symbol, cfg):
    if not session_ok() or news_block() or not equity_guard():
        return

    df1 = pd.DataFrame(mt5.copy_rates_from_pos(symbol, TF_ENTRY, 0, 200))
    df5 = pd.DataFrame(mt5.copy_rates_from_pos(symbol, TF_TREND, 0, 200))

    if df1.empty or df5.empty:
        return

    positions = get_positions(symbol, cfg['magic'])
    if len(positions) >= MAX_TRADE_PER_SYMBOL:
        manage_positions(symbol, df1, cfg)
        return

    o, c, h, l = df1[['open','close','high','low']].iloc[-2]

    if mtf_ok(df5, 'buy') and pullback_buy(df1) and bullish_rejection(o, c, l):
        send_order(symbol, 'buy', df1, cfg)

    if mtf_ok(df5, 'sell') and pullback_sell(df1) and bearish_rejection(o, c, h):
        send_order(symbol, 'sell', df1, cfg)


# ================= MAIN =====================
last_candle = {}

while True:
    for symbol, cfg in SYMBOLS.items():
        rates = mt5.copy_rates_from_pos(symbol, TF_ENTRY, 0, 2)
        if rates is None:
            continue

        t = rates[-1]['time']
        if last_candle.get(symbol) != t:
            last_candle[symbol] = t
            execute_symbol(symbol, cfg)

    time.sleep(1)
