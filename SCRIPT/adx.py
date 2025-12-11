import pandas as pd
import numpy as np

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Wilder ADX implementation - Same as MT4/MT5.
    Requires DataFrame with ['high','low','close']
    Returns ADX, +DI, -DI aligned with input.
    """

    high = df['high'].astype(float).reset_index(drop=True)
    low = df['low'].astype(float).reset_index(drop=True)
    close = df['close'].astype(float).reset_index(drop=True)

    size = len(df)
    if size <= period:
        df['ADX'] = np.nan
        df['+DI'] = np.nan
        df['-DI'] = np.nan
        return df

    # True Range (TR)
    tr = np.zeros(size)
    dm_plus = np.zeros(size)
    dm_minus = np.zeros(size)

    for i in range(1, size):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]

        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

        dm_plus[i] = high_diff if (high_diff > low_diff and high_diff > 0) else 0
        dm_minus[i] = low_diff if (low_diff > high_diff and low_diff > 0) else 0

    # Wilder smoothing
    atr = np.zeros(size)
    pDM = np.zeros(size)
    mDM = np.zeros(size)

    atr[period] = tr[1:period+1].mean()
    pDM[period] = dm_plus[1:period+1].mean()
    mDM[period] = dm_minus[1:period+1].mean()

    for i in range(period+1, size):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        pDM[i] = (pDM[i-1] * (period - 1) + dm_plus[i]) / period
        mDM[i] = (mDM[i-1] * (period - 1) + dm_minus[i]) / period

    # DI calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = (pDM / atr) * 100
        minus_di = (mDM / atr) * 100

    # DX calculation
    dx = np.zeros(size)
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    # ADX
    adx = np.full(size, np.nan)
    adx[period*2] = dx[period+1:period*2+1].mean()  # initial ADX

    for i in range(period*2+1, size):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['ADX'] = adx

    return df
