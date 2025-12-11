import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import math
import time

# ========================
# CONFIG BOT
# ========================
PAIR = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
VOLUME = 0.01  # default lot size (will be recalculated based on risk)
STOP_LOSS = 2
TAKE_PROFIT = 4

RSI_PERIOD = 3
RSI_OB = 60
RSI_OS = 40
BARS_HISTORY = 500  # keep plenty of history to initialize Wilder smoothing
INCLUDE_CURRENT_CANDLE = True  # set False to use last closed candle (pos=1)

RISK_PER_TRADE = 0.005      # 0.5% risk per trade
ATR_PERIOD = 3
ATR_MULT_SL = 2.5
ATR_MULT_TP = 3.5

COOLDOWN_LOSS_TRADES = 1    # cooldown setelah 1 loss (tidak dipakai detail di contoh)
COOLDOWN_TIME = 60          # detik

MAGIC = 4444

# Reverse hedging config
MAX_HEDGE_LOT = 0.05        # batas lot untuk hedge agar tidak overexpose
HEDGE_PROFIT_BUFFER = 1.05  # buffer untuk sedikit profit saat menghitung lot hedge (5%)
HEDGE_OPENED_FLAG = False   # global flag untuk mencegah multiple hedge spam

# Break-even / auto close config
BREAK_EVEN_PROFIT = 0.0     # tutup semua posisi jika net profit >= nilai ini (ubah sesuai keinginan)
CLOSE_PROFIT_TARGET = 1.0   # optional: jika ingin menutup saat net profit >= nilai ini (unused if BREAK_EVEN_PROFIT used)

ADX_PERIOD = 5        # default 5 (ubah kalau mau)
FETCH_BARS = 100      # jumlah bar historis yang diambil setiap perhitungan (cukup untuk stabilitas)
POLL_INTERVAL = 1.0   # detik antar pengecekan (script akan menunggu bar baru; kecil saja)
PRINT_EVERY_BAR = True


last_loss_time = None
loss_count = 0

# global hold signals
hold_buy_signal = False
hold_sell_signal = False

# ========================
# INDICATORS
# ========================
def ema_cross_signal(df):
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()

    df['prev_EMA8'] = df['EMA8'].shift(10)
    df['prev_EMA21'] = df['EMA21'].shift(10)

    df['signal'] = None

    # Buy Signal
    buy_signal = (df['prev_EMA8'] <= df['prev_EMA21']) & \
                 (df['EMA8'] > df['EMA21']) & \
                 (df['close'] > df['EMA8']) & \
                 (df['close'] > df['EMA21'])

    df.loc[buy_signal, 'signal'] = "BUY"

    # Sell Signal
    sell_signal = (df['prev_EMA8'] >= df['prev_EMA21']) & \
                  (df['EMA8'] < df['EMA21']) & \
                  (df['close'] < df['EMA8']) & \
                  (df['close'] < df['EMA21'])

    df.loc[sell_signal, 'signal'] = "SELL"

    return df

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
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

def calculate_atr(df, period=14):
    # returns ATR in price units (same as price)
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return tr.rolling(period).mean()

def calculate_adx_wilder(df, period=14):
    """
    df: DataFrame berisi columns ['high','low','close'] secara berurutan (index waktu ascending)
    period: integer, periode ADX (e.g. 5 atau 14)
    returns: df dengan kolom ['TR','DM_plus','DM_minus','sm_TR','sm_DM_plus','sm_DM_minus','DI_plus','DI_minus','DX','ADX']
    Implementation follows Wilder smoothing (first smoothed value = sum of first 'period' values,
    subsequent = prev_smoothed - (prev_smoothed / period) + current_value).
    """
    df = df.copy().reset_index(drop=True).astype(float)
    # True Range
    df['prev_close'] = df['close'].shift(1)
    df['TR'] = np.maximum(df['high'] - df['low'],
                          np.maximum((df['high'] - df['prev_close']).abs(),
                                     (df['low'] - df['prev_close']).abs()))
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

    # Need at least period+1 rows to compute first smoothed values (because TR and DM start from row 1)
    # Compute first smoothed values at index = period (sum of first 'period' TR/DM starting from index 1..period)
    # Note: df rows are 0..N-1; TR at row 0 is NaN because prev_close is NaN. We align smoothing accordingly.
    # We compute sums over rows 1..period inclusive for initial smoothed TR/DM.
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
# RISK LOT CALCULATION
# ========================
def calculate_lot(stop_loss_price_distance):
    """
    Kalkulasi lot berdasarkan risk money dan nilai per pip/tick.
    stop_loss_price_distance: jarak SL dalam satuan harga (misal ATR)
    """
    info = mt5.account_info()
    if info is None:
        return 0.01
    balance = info.balance
    risk_money = balance * RISK_PER_TRADE

    sym = mt5.symbol_info(PAIR)
    if sym is None:
        return 0.01

    # trade_tick_value adalah nilai 1 tick (per kontrak) dalam akun currency
    tick_val = sym.trade_tick_value
    if tick_val is None or stop_loss_price_distance == 0:
        return 0.01

    # Untuk XAUUSD, stop_loss_price_distance sudah dalam price units, gunakan langsung
    lot = risk_money / (stop_loss_price_distance * tick_val)
    # pembatasan minimal dan maksimal lot
    lot = max(0.01, round(lot, 2))
    return lot

def calculate_hedge_lot(loss_money, stop_loss_price_distance):
    """
    Hitung lot hedge untuk menutup floating loss_money,
    memberi buffer kecil untuk kembali ke netral/profit.
    """
    sym = mt5.symbol_info(PAIR)
    if sym is None:
        return 0.01
    tick_val = sym.trade_tick_value
    if tick_val is None or stop_loss_price_distance == 0:
        return 0.01

    required = (loss_money * HEDGE_PROFIT_BUFFER) / (stop_loss_price_distance * tick_val)
    # Batasi dengan MAX_HEDGE_LOT
    hedge_lot = round(min(required, MAX_HEDGE_LOT), 2)
    if hedge_lot < 0.01:
        hedge_lot = 0.01
    return hedge_lot

# ========================
# ORDER / POSITION HELPERS
# ========================
def open_trade(action, volume, sl, tp, comment_suffix="RSI"):
    """
    Open market trade. action = "BUY" or "SELL". volume in lots.
    """
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
    print(f"Order send ({comment_suffix} {action}):", result)
    return result

def has_open_trade(direction=None):
    positions = mt5.positions_get(symbol=PAIR)
    if positions is None:
        return False
    if direction is None:
        return len(positions) > 0
    # MT5 pos.type: 0 = buy, 1 = sell
    return any(p.type == (0 if direction == "BUY" else 1) for p in positions)

def get_positions():
    pos = mt5.positions_get(symbol=PAIR)
    if pos is None:
        return []
    return list(pos)

def close_position(position):
    """
    Close a single position by sending market order in opposite direction with same volume.
    """
    pos_type = position.type  # 0 buy, 1 sell
    volume = position.volume
    ticket = position.ticket

    tick = mt5.symbol_info_tick(PAIR)
    if tick is None:
        print("Cannot get tick to close position")
        return None

    if pos_type == 0:  # buy -> we sell to close
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:  # sell -> buy to close
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PAIR,
        "volume": volume,
        "type": order_type,
        "position": int(ticket),
        "price": price,
        "deviation": 30,
        "magic": MAGIC,
        "comment": "CloseOnBreakEven",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    print("Close order result:", result)
    return result

def close_all_positions():
    """
    Close all positions for PAIR. Return list of results.
    """
    results = []
    positions = get_positions()
    for p in positions:
        res = close_position(p)
        results.append(res)
        # small pause between close orders
        time.sleep(0.3)
    return results

# ========================
# MAIN BOT LOOP
# ========================
def run_bot():
    global last_loss_time, loss_count, HEDGE_OPENED_FLAG
    global hold_buy_signal, hold_sell_signal

    if not mt5.initialize():
        print("MT5 Connect Failed")
        return

    print("Bot Started...")
    prev_rsi = None

    while True:
        try:
            # Cooldown logic (set when loss happened) - contoh sederhana
            if last_loss_time and (time.time() - last_loss_time < COOLDOWN_TIME):
                print("Cooldown setelah loss...")
                time.sleep(1)
                continue

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
            df = ema_cross_signal(df)
            adx_df = calculate_adx_wilder(df[['high','low','close']], period=ADX_PERIOD)

            latest_adx_row = adx_df[~adx_df['ADX'].isna()].tail(1)
            if not latest_adx_row.empty:
                latest_adx = latest_adx_row['ADX'].iloc[-1]
                latest_time_idx = latest_adx_row.index[-1]
                bar_time = df['time'].iloc[latest_time_idx]

            rsi = df['rsi'].iloc[-1]
            price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            adx = latest_adx

            # sinyal filter SMA 30 & SMA 60
            ema_buy = df['BUY'].iloc[-1]
            ema_sell = df['SELL'].iloc[-1]

            if not hold_buy_signal and not hold_buy_signal:
                print(f"{datetime.now()} | Price: {price} | RSI: {np.nan if np.isnan(rsi) else round(rsi, 2)} | ATR: {round(atr,5) if not pd.isna(atr) else 'NA'} | ADX: {round(adx,2) if not pd.isna(adx) else 'NA'}")

            
            # SL/TP (harga)
            if atr and atr > 0:
                sl_buy = price - STOP_LOSS
                tp_buy = price + TAKE_PROFIT
                sl_sell = price + STOP_LOSS
                tp_sell = price - TAKE_PROFIT
            else:
                time.sleep(1)
                continue

            # ---- Reverse Hedging Logic ----
            positions = get_positions()
            net_profit = sum([p.profit for p in positions]) if positions else 0.0

            # Trigger hedge if any single position floating loss less than -2$ and hedge not opened yet
            if positions:
                # evaluate the largest losing position
                worst_pos = min(positions, key=lambda p: p.profit)  # most negative profit
                worst_profit = worst_pos.profit
                pos_direction = "BUY" if worst_pos.type == 0 else "SELL"

                # If worst position loss < -2 and no hedge opened, create hedge
                if worst_profit < -2 and not HEDGE_OPENED_FLAG:
                    print(f"🔥 Reverse Hedging Triggered! Worst position floating: {worst_profit:.2f} USD")

                    # Opposite action for hedge
                    hedge_action = "SELL" if pos_direction == "BUY" else "BUY"

                    # Use ATR as approximate stop_loss distance for hedge calculation
                    sl_distance = atr if atr and atr > 0 else max(0.1, abs(price * 0.001))
                    hedge_lot = calculate_hedge_lot(abs(worst_profit), sl_distance)

                    # Set SL/TP for hedge similar to entry style
                    tick = mt5.symbol_info_tick(PAIR)
                    if tick is None:
                        print("Can't get tick for hedge price")
                        time.sleep(1)
                    else:
                        hedge_price = tick.ask if hedge_action == "BUY" else tick.bid
                        sl = hedge_price - ATR_MULT_SL * atr if hedge_action == "BUY" else hedge_price + ATR_MULT_SL * atr
                        tp = hedge_price + ATR_MULT_TP * atr if hedge_action == "BUY" else hedge_price - ATR_MULT_TP * atr

                        print(f"Opening hedge {hedge_action} lot={hedge_lot} sl={sl:.5f} tp={tp:.5f}")
                        res = open_trade(hedge_action, hedge_lot, sl, tp, comment_suffix="Reverse_Hedge")
                        HEDGE_OPENED_FLAG = True
                        time.sleep(1)

            # Break-even / Auto close all if net profit >= threshold
            # NOTED: adjust BREAK_EVEN_PROFIT to required positive threshold if you want close at small profit
            positions = get_positions()
            # net_profit = sum([p.profit for p in positions]) if positions else 0.0
            # if positions and net_profit >= BREAK_EVEN_PROFIT:
            #     print(f"Net profit {net_profit:.2f} >= {BREAK_EVEN_PROFIT} — Closing all positions (break-even take).")
            #     close_all_positions()
            #     # reset hedge flag & cooldown
            #     HEDGE_OPENED_FLAG = False
            #     last_loss_time = None
            #     loss_count = 0
            #     # short pause
            #     time.sleep(2)
            #     # continue main loop
            #     prev_rsi = None
            #     continue

            # Anti-Double Position: skip opening new entry if there is any open position and we don't want multiple entries
            if has_open_trade():
                # still allow hedging and break-even logic above; skip new entries
                time.sleep(1)
                prev_rsi = df['rsi'].iloc[-1]
                continue

            # SMA cross filtering signals (persistent hold logic)
            if ema_buy and not hold_buy_signal:
                hold_buy_signal = True
                hold_sell_signal = False  # reset opposite side
            if ema_sell and not hold_sell_signal:
                hold_sell_signal = True
                hold_buy_signal = False  # reset opposite side

            if hold_buy_signal:
                print("HOLD BUY SIGNAL (EMA Filter Active)")
            if hold_sell_signal:
                print("HOLD SELL SIGNAL (EMA Filter Active)")

            # Entry Logic (RSI cross)
            if prev_rsi is not None:
                # BUY when RSI crosses above OS and trend is up
                if hold_buy_signal and (prev_rsi < RSI_OS and rsi > RSI_OS) and adx > 25:
                    print("BUY SIGNAL CONFIRMED (EMA + RSI Breakout)")
                    # calculate lot based on ATR distance
                    lot = VOLUME
                    open_trade("BUY", lot, sl_buy, tp_buy, comment_suffix="RSI")

                # SELL when RSI crosses below OB and trend is down
                if hold_sell_signal and (prev_rsi > RSI_OB and rsi < RSI_OB) and adx > 25:
                    print("SELL SIGNAL")
                    lot = VOLUME
                    open_trade("SELL", lot, sl_sell, tp_sell, comment_suffix="RSI")

            prev_rsi = rsi
            time.sleep(1)

        except Exception as e:
            print("Exception in main loop:", e)
            time.sleep(1)
            continue

if __name__ == "__main__":
    run_bot()
# ========================