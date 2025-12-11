import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time

PAIR = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
RSI_PERIOD = 3
BARS_HISTORY = 500  # keep plenty of history to initialize Wilder smoothing
INCLUDE_CURRENT_CANDLE = True  # set False to use last closed candle (pos=1)

def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """
    Wilder RSI implementation (same method MT4/MT5 uses).
    Returns a pandas Series aligned with `close`.
    """
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

    # first average = simple mean of first `period` gains/losses (starting after first diff)
    # indices: use 1..period inclusive for deltas (delta[0] is NaN)
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

# ----- MT5 initialization -----
if not mt5.initialize():
    print("❌ MT5 init failed:", mt5.last_error())
    raise SystemExit

try:
    while True:
        pos = 0 if INCLUDE_CURRENT_CANDLE else 1
        bars = mt5.copy_rates_from_pos(PAIR, TIMEFRAME, pos, BARS_HISTORY)
        if bars is None or len(bars) == 0:
            print("⚠ no data from MT5, retrying...")
            time.sleep(1)
            continue

        df = pd.DataFrame(bars)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # Make sure we use 'close' (applied price)
        closes = df['close']

        # compute Wilder RSI
        df['rsi_wilder'] = rsi_wilder(closes, RSI_PERIOD)

        # latest RSI (most recent candle we requested)
        latest_rsi = df['rsi_wilder'].iloc[-1]
        # also print last few closes for visual check
        print(f"{pd.Timestamp.now()} | {PAIR} | latest close: {closes.iloc[-1]:.5f} | RSI({RSI_PERIOD}) = {np.nan if np.isnan(latest_rsi) else round(latest_rsi, 2)}")

        # OPTIONAL: show last 6 values for visual comparison
        # print(df[['time', 'close', 'rsi_wilder']].tail(1).to_string(index=False))
        # print("RSI ({latest_rsi})")
        time.sleep(0.5)
finally:
    mt5.shutdown()
