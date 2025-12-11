import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
import sys

# ========================
# CONFIG BOT
# ========================
PAIR = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
VOLUME = 0.01  # default lot size (will be recalculated based on risk)
STOP_LOSS = 2
TAKE_PROFIT = 2

# RSI SETTINGS
RSI_PERIOD = 3
RSI_OB = 80
RSI_OS = 20
BARS_HISTORY = 500  # keep plenty of history to initialize Wilder smoothing
INCLUDE_CURRENT_CANDLE = True  # set False to use last closed candle (pos=1)

# ATR SETTINGS
ATR_PERIOD = 3
ATR_MULT_SL = 2.5
ATR_MULT_TP = 3.5

COOLDOWN_LOSS_TRADES = 1    # cooldown setelah 1 loss (tidak dipakai detail di contoh)
COOLDOWN_TIME = 60          # detik

MAGIC = 4444

# ADX SETTINGS
ADX_PERIOD = 5        # default 5 (ubah kalau mau)
FETCH_BARS = 100      # jumlah bar historis yang diambil setiap perhitungan (cukup untuk stabilitas)
POLL_INTERVAL = 1.0   # detik antar pengecekan (script akan menunggu bar baru; kecil saja)
PRINT_EVERY_BAR = True
ADX_LIMIT = 30       # minimal ADX untuk konfirmasi tren

entry_adx = None
entry_profit_prev = None

# ========================
# INDICATORS
# ========================
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

def calculate_atr(df, period=14):
    # returns ATR in price units (same as price)
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return tr.rolling(period).mean()

def calculate_adx_wilder(df, period=14):
    df = df.copy().reset_index(drop=True).astype(float)
    # True Range
    df['prev_close'] = df['close'].shift(1)
    df['TR'] = np.maximum(df['high'] - df['low'],np.maximum((df['high'] - df['prev_close']).abs(),(df['low'] - df['prev_close']).abs()))
    # Directional Movements
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['DM_plus'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['DM_minus'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)

    # Initialize smoothed series with NaN
    df['sm_TR'] = np.nan
    df['sm_DM_plus'] = np.nan
    df['sm_DM_minus'] = np.nan
    df['DI_plus'] = np.nan
    df['DI_minus'] = np.nan
    df['DX'] = np.nan
    df['ADX'] = np.nan

    if len(df) <= period:
        return df  # not enough data

    # compute initial sums: take rows 1..period (inclusive)
    init_start = 1
    init_end = period  # inclusive index
    sm_TR_init = df.loc[init_start:init_end, 'TR'].sum()
    sm_DM_plus_init = df.loc[init_start:init_end, 'DM_plus'].sum()
    sm_DM_minus_init = df.loc[init_start:init_end, 'DM_minus'].sum()

    df.at[init_end, 'sm_TR'] = sm_TR_init
    df.at[init_end, 'sm_DM_plus'] = sm_DM_plus_init
    df.at[init_end, 'sm_DM_minus'] = sm_DM_minus_init

    # Compute DI and DX at init_end
    if sm_TR_init != 0:
        di_plus = 100.0 * (sm_DM_plus_init / sm_TR_init)
        di_minus = 100.0 * (sm_DM_minus_init / sm_TR_init)
        df.at[init_end, 'DI_plus'] = di_plus
        df.at[init_end, 'DI_minus'] = di_minus
        dx = 100.0 * (abs(di_plus - di_minus) / (di_plus + di_minus)) if (di_plus + di_minus) != 0 else 0.0
        df.at[init_end, 'DX'] = dx
    else:
        df.at[init_end, 'DI_plus'] = 0.0
        df.at[init_end, 'DI_minus'] = 0.0
        df.at[init_end, 'DX'] = 0.0

    # Wilder smoothing for subsequent rows
    for i in range(init_end + 1, len(df)):
        tr = df.at[i, 'TR']
        dm_plus = df.at[i, 'DM_plus']
        dm_minus = df.at[i, 'DM_minus']

        prev_sm_TR = df.at[i - 1, 'sm_TR']
        prev_sm_DM_plus = df.at[i - 1, 'sm_DM_plus']
        prev_sm_DM_minus = df.at[i - 1, 'sm_DM_minus']

        if np.isnan(prev_sm_TR):
            # If previous smoothed was NaN, use the initial at init_end (shouldn't happen after initialization)
            prev_sm_TR = sm_TR_init
            prev_sm_DM_plus = sm_DM_plus_init
            prev_sm_DM_minus = sm_DM_minus_init

        sm_TR = prev_sm_TR - (prev_sm_TR / period) + tr
        sm_DM_plus = prev_sm_DM_plus - (prev_sm_DM_plus / period) + dm_plus
        sm_DM_minus = prev_sm_DM_minus - (prev_sm_DM_minus / period) + dm_minus

        df.at[i, 'sm_TR'] = sm_TR
        df.at[i, 'sm_DM_plus'] = sm_DM_plus
        df.at[i, 'sm_DM_minus'] = sm_DM_minus

        if sm_TR != 0:
            di_plus = 100.0 * (sm_DM_plus / sm_TR)
            di_minus = 100.0 * (sm_DM_minus / sm_TR)
        else:
            di_plus = 0.0
            di_minus = 0.0

        df.at[i, 'DI_plus'] = di_plus
        df.at[i, 'DI_minus'] = di_minus

        dx = 100.0 * (abs(di_plus - di_minus) / (di_plus + di_minus)) if (di_plus + di_minus) != 0 else 0.0
        df.at[i, 'DX'] = dx

    # ADX: first ADX is simple average of first 'period' DX values starting from index init_end .. init_end+period-1
    # Need at least period DX values starting at init_end
    dx_start = init_end
    dx_end = init_end + period - 1
    if dx_end < len(df):
        first_adx = df.loc[dx_start:dx_end, 'DX'].mean()
        df.at[dx_end, 'ADX'] = first_adx

        # Smooth ADX subsequently using Wilder (recursive)
        for j in range(dx_end + 1, len(df)):
            prev_adx = df.at[j - 1, 'ADX']
            if np.isnan(prev_adx):
                prev_adx = first_adx
            curr_dx = df.at[j, 'DX']
            adx = ( (prev_adx * (period - 1)) + curr_dx ) / period
            df.at[j, 'ADX'] = adx

    return df

# ========================
# ORDER / POSITION HELPERS
# ========================
def open_trade(action, adx, volume, sl, tp, comment_suffix="RSI"):
    global entry_adx, entry_profit_prev

    price_tick = mt5.symbol_info_tick(PAIR)
    if price_tick is None:
        print("Cannot get tick")
        return None

    price = price_tick.ask if action == "BUY" else price_tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PAIR,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 30,
        "magic": MAGIC,
        "comment": f"{comment_suffix}_{action}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)

    # Simpan ADX saat entry
    entry_adx = adx  
    entry_profit_prev = 0  # awalnya 0, belum profit

    return result

def has_open_trade(direction=None):
    positions = mt5.positions_get(symbol=PAIR)
    if positions is None:
        return False
    if direction is None:
        return len(positions) > 0
    # MT5 pos.type: 0 = buy, 1 = sell
    return any(p.type == (0 if direction == "BUY" else 1) for p in positions)

# ========================
# UI & HELPER FUNCTIONS
# ========================
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def loading_animation(text="🚀 Siap Memulai Trading", duration=2, interval=0.25):
    end_time = time.time() + duration
    while time.time() < end_time:
        for dots in range(3):
            sys.stdout.write(f"\r{text}{'.'*dots}{' '*(3-dots)}")
            sys.stdout.flush()
            time.sleep(interval)
    clear_screen()

# ========================
# MAIN BOT LOOP
# ========================
def run_bot():
    if not mt5.initialize():
        print("MT5 Connect Failed")
        return

    print("Bot Started...")
    prev_rsi = None
    clear_screen()
    loading_animation()

    while True:
        try:
            # Ambil rates terakhir
            pos = 0 if INCLUDE_CURRENT_CANDLE else 1
            rates = mt5.copy_rates_from_pos(PAIR, TIMEFRAME, pos, BARS_HISTORY)
            if rates is None or len(rates) == 0:
                print("Cannot fetch rates, retrying...")
                time.sleep(1)
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Indicators
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['atr'] = calculate_atr(df, ATR_PERIOD)
            ema50 = df['close'].ewm(span=50, adjust=False).mean()
            adx_df = calculate_adx_wilder(df[['high','low','close']], period=ADX_PERIOD)

            latest_adx_row = adx_df[~adx_df['ADX'].isna()].tail(1)
            if not latest_adx_row.empty:
                latest_adx = latest_adx_row['ADX'].iloc[-1]

            # EMA filter signals
            uptrend = df['close'].iloc[-1] > ema50.iloc[-1]
            downtrend = df['close'].iloc[-1] < ema50.iloc[-1]
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            adx = latest_adx

            # SL/TP (harga)
            if atr and atr > 0:
                sl_buy = price - STOP_LOSS
                tp_buy = price + TAKE_PROFIT
                sl_sell = price + STOP_LOSS
                tp_sell = price - TAKE_PROFIT
            else:
                time.sleep(1)
                continue

            # Anti-Double Position: skip opening new entry if there is any open position and we don't want multiple entries
            if has_open_trade():
                # still allow hedging and break-even logic above; skip new entries
                time.sleep(1)
                continue

            # Confirmations for entries
            if not has_open_trade():
                confirmed_buy = prev_rsi < RSI_OS and rsi > RSI_OS and adx > ADX_LIMIT and uptrend
                confirmed_sell = prev_rsi > RSI_OB and rsi > RSI_OB and adx > ADX_LIMIT and downtrend
                # Find Signal
                print(f" 🔍 Find Sinyal | RSI: {np.nan if np.isnan(rsi) else round(rsi, 2)} | PREV RSI: {round(prev_rsi,2) if not pd.isna(prev_rsi) else 'NA'} | ADX: {round(adx,2) if not pd.isna(adx) else 'NA'}")

            # Entry Logic
            # if prev_rsi is not None:
                # BUY when uptrend and RSI crosses above OS
                if confirmed_buy:
                    print("⚡ BUY SIGNAL CONFIRMED")
                    lot = VOLUME
                    open_trade("BUY", adx, lot, sl_buy, tp_buy, comment_suffix="RSI")
                    prev_rsi = None

                # SELL when downtrend and RSI crosses below OB
                if confirmed_sell:
                    print("⚡SELL SIGNAL CONFIRMED")
                    lot = VOLUME
                    open_trade("SELL", adx, lot, sl_sell, tp_sell, comment_suffix="RSI")
                    prev_rsi = None

            # =========================
            # FORCE CLOSE RULE
            # =========================

            positions = mt5.positions_get(symbol=PAIR)
            if positions:
                pos = positions[0]
                current_profit = pos.profit

                # Jika sudah pernah profit, lalu jadi minus + ADX melemah
                if entry_adx is not None and adx is not None:
                    if current_profit < entry_profit_prev and entry_profit_prev > 0 and adx < entry_adx:
                        print("⚠ CLOSE FORCE: ADX melemah, profit berubah minus!")
                        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(PAIR).bid if pos.type == 0 else mt5.symbol_info_tick(PAIR).ask
                        close_req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": PAIR,
                            "volume": pos.volume,
                            "type": close_type,
                            "position": pos.ticket,
                            "price": price,
                            "deviation": 50,
                            "magic": MAGIC,
                            "comment": "ForceCloseADX",
                        }
                        mt5.order_send(close_req)
                        entry_adx = None
                        entry_profit_prev = None

                # Update history profit
                if current_profit > entry_profit_prev:
                    entry_profit_prev = current_profit

            time.sleep(1)

        except Exception as e:
            print("Exception in main loop:", e)
            time.sleep(1)
            continue

if __name__ == "__main__":
    run_bot()
# ========================