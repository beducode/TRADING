import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import joblib
import logging
from xgboost import XGBClassifier

# ================= CONFIG =================
SYMBOLS = ["ETHUSDm", "BTCUSDm"]
TIMEFRAME_BIAS  = mt5.TIMEFRAME_M5
TIMEFRAME_ENTRY = mt5.TIMEFRAME_M1

BIAS_THRESHOLD = 0.65     # M5 bias
TRAIL_START_ATR = 1.0     # mulai trailing setelah profit 1 ATR

LOOKBACK = 400

EMA_FAST = 9
EMA_MID  = 21
EMA_SLOW = 50

RSI_PERIOD = 7
ATR_PERIOD = 14
ADX_PERIOD = 5

RISK_PERCENT = 1.0
AI_THRESHOLD = 0.70

MODEL_PATH = "xgb_ai_model.pkl"
MAGIC = 777777

FEATURES = [
    'ema9_dist','ema21_dist','ema50_dist',
    'rsi','adx','atr'
]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42
}

# ================= LOGGING =================
logging.basicConfig(
    filename="ai_mt5_bot.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= MT5 INIT =================
if not mt5.initialize():
    raise RuntimeError("❌ MT5 gagal connect")

for sym in SYMBOLS:
    mt5.symbol_select(sym, True)

# ================= INDICATORS =================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100 / (1 + rs))

def atr(df, period):
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        )
    )
    return tr.rolling(period).mean()

def adx(df, period):
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    tr = atr(df, 1)
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / tr.rolling(period).sum()
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / tr.rolling(period).sum()
    return abs(plus_di - minus_di)

# ================= DATA & FEATURES =================
def get_data(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, LOOKBACK)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def build_features(df):
    df['ema9']  = ema(df['close'], EMA_FAST)
    df['ema21'] = ema(df['close'], EMA_MID)
    df['ema50'] = ema(df['close'], EMA_SLOW)

    df['rsi'] = rsi(df['close'], RSI_PERIOD)
    df['atr'] = atr(df, ATR_PERIOD)
    df['adx'] = adx(df, ADX_PERIOD)

    df['ema9_dist']  = df['close'] - df['ema9']
    df['ema21_dist'] = df['close'] - df['ema21']
    df['ema50_dist'] = df['close'] - df['ema50']

    df['trend_buy']  = (df['ema9'] > df['ema21']) & (df['ema21'] > df['ema50'])
    df['trend_sell'] = (df['ema9'] < df['ema21']) & (df['ema21'] < df['ema50'])

    return df.dropna()

# ================= PRICE ACTION =================
def pa_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if last['trend_buy'] and prev['close'] < prev['ema21'] and last['close'] > last['ema21']:
        return "BUY"

    if last['trend_sell'] and prev['close'] > prev['ema21'] and last['close'] < last['ema21']:
        return "SELL"

    return None

# ================= AI MODEL =================
def train_ai(df):
    df = df.copy()
    df['future'] = df['close'].shift(-5)
    df['target'] = (df['future'] > df['close']).astype(int)

    X = df[FEATURES]
    y = df['target']

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X[:-5], y[:-5], verbose=False)

    joblib.dump(model, MODEL_PATH)
    logging.info("✅ XGBoost model trained & saved")

    return model

def ai_predict(model, row):
    X = pd.DataFrame([row[FEATURES].to_dict()])
    prob = model.predict_proba(X)[0]
    return prob[1], prob[0]  # buy_prob, sell_prob

# ================= RISK MANAGEMENT =================
def calc_lot(symbol, sl_points):
    acc = mt5.account_info()
    risk_money = acc.balance * (RISK_PERCENT / 100)

    info = mt5.symbol_info(symbol)
    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size

    cost_per_point = tick_value / tick_size
    lot = risk_money / (sl_points * cost_per_point)

    return round(max(info.volume_min, lot), 2)

# ================= ORDER =================
def send_order(symbol, side, atr_val):
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if side == "BUY" else tick.bid

    sl = price - atr_val * 2 if side == "BUY" else price + atr_val * 2
    tp = price + atr_val * 3 if side == "BUY" else price - atr_val * 3

    lot = calc_lot(symbol, atr_val)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "XGB_AI_BOT",
        "type_time": mt5.ORDER_TIME_GTC
    }

    result = mt5.order_send(request)
    logging.info(f"{symbol} {side} | lot={lot} | retcode={result.retcode}")

# ================= MAIN LOOP =================
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def ai_bias(model, df_m5):
    row = df_m5.iloc[-1]
    X = pd.DataFrame([row[FEATURES].to_dict()])
    prob = model.predict_proba(X)[0]

    buy_prob = prob[1]
    sell_prob = prob[0]

    if buy_prob >= BIAS_THRESHOLD:
        return "BUY", buy_prob
    elif sell_prob >= BIAS_THRESHOLD:
        return "SELL", sell_prob
    else:
        return None, max(buy_prob, sell_prob)
    
def entry_with_bias(df_m1, bias):
    signal = pa_signal(df_m1)
    if signal == bias:
        return signal
    return None

def modify_sl(position, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl": new_sl,
        "tp": position.tp,
        "symbol": position.symbol,
        "magic": MAGIC,
        "comment": "AI_TRAILING"
    }
    mt5.order_send(request)
    logging.info(f"Trailing SL updated {position.symbol} -> {new_sl}")


def ai_trailing_stop(symbol, model):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    pos = positions[0]
    df = build_features(get_data(symbol, TIMEFRAME_ENTRY))
    row = df.iloc[-1]

    # AI confidence
    buy_p, sell_p = ai_predict(model, row)
    confidence = buy_p if pos.type == mt5.ORDER_TYPE_BUY else sell_p

    atr = row['atr']
    price = mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask

    # profit sudah > X ATR
    if abs(price - pos.price_open) < atr * TRAIL_START_ATR:
        return

    # SL multiplier adaptif dari AI
    sl_mult = max(0.8, 2.0 - confidence)

    if pos.type == mt5.ORDER_TYPE_BUY:
        new_sl = price - atr * sl_mult
        if pos.sl == 0 or new_sl > pos.sl:
            modify_sl(pos, new_sl)

    if pos.type == mt5.ORDER_TYPE_SELL:
        new_sl = price + atr * sl_mult
        if pos.sl == 0 or new_sl < pos.sl:
            modify_sl(pos, new_sl)


model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

while True:
    try:
        for symbol in SYMBOLS:

            # ===== BIAS TF M5 =====
            df_m5 = build_features(get_data(symbol, TIMEFRAME_BIAS))

            if model is None:
                model = train_ai(df_m5)
                continue

            bias, bias_prob = ai_bias(model, df_m5)
            if not bias:
                continue

            # ===== ENTRY TF M1 =====
            df_m1 = build_features(get_data(symbol, TIMEFRAME_ENTRY))
            entry = entry_with_bias(df_m1, bias)

            if entry:
                atr_val = df_m1.iloc[-1]['atr']
                send_order(symbol, entry, atr_val)
                logging.info(f"{symbol} ENTRY {entry} | bias={bias_prob:.2f}")

            # ===== AI TRAILING =====
            ai_trailing_stop(symbol, model)

        time.sleep(60)

    except Exception as e:
        logging.error(str(e))
        time.sleep(60)

