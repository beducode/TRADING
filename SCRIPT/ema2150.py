"""
Entry = EMA_LOW cross EMA_HIGH (confirmed on previous closed candle)
Trend filter = price vs EMA_TREND
Momentum = RSI(8) threshold
ADX(5) ensures trend strength
Auto break-even and optional auto-close-all-on-net-profit.

Requirements:
- MetaTrader5 package (pip install MetaTrader5)
- pandas, numpy
- Run on machine with MT5 terminal open & logged in
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime, timezone

# ===========================
# USER CONFIGURATION
# ===========================
SYMBOL = "BTCUSD"            # symbol
TIMEFRAME = mt5.TIMEFRAME_M1
LOT = 0.01                   # default volume
TP_IN_POINTS = 100           # take profit in points (price units, adjust to broker digits)
SL_IN_POINTS = 100           # stop loss in points
RISK_PERCENT = None          # optional: if set (0.5 = 0.5%) we can compute LOT based on account risk; else LOT used
RSI_PERIOD = 8
RSI_THRESHOLD = 50
ADX_PERIOD = 8
ADX_THRESHOLD = 30

EMA_LOW_PERIOD = 21
EMA_HIGH_PERIOD = 50
EMA_TREND_PERIOD = 100

BREAK_EVEN_PROFIT = 3.0      # when a position's profit (in account currency) >= this, move SL to break-even
AUTO_CLOSE_ALL_NET_PROFIT = 50.0  # if total net profit for SYMBOL reaches this, close all positions on the SYMBOL
POLL_SECONDS = 1.0           # loop sleep; for M1 scalping a short interval is ok
WAIT_SECONDS = 100           # loop sleep; for M1 scalping a short interval is ok

# Aggressive trade limiting
MAX_POSITIONS = 1            # max concurrent positions on symbol (avoid overtrading)

# Shift
EMA_LOW_SHIFT = 3            # number of candles to look back for signal confirmation
LOW_SHIFT = 1                # number of candles to look back for signal confirmation
HIGH_SHIFT = 1               # number of candles to look back for signal confirmation

# EMA CROSS
MIN_EMA_DISTANCE = 20       # minimum distance between EMAs to consider valid cross (in points)


# Cross Signal EMA
cross_signal_buy = False
cross_signal_sell = False

# ===========================
# Helper indicator functions
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

def calculate_adx(df: pd.DataFrame, period: int) -> pd.Series:
    df = df.copy()
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)

    df['tr0'] = df['high'] - df['low']
    df['tr1'] = (df['high'] - df['close'].shift(LOW_SHIFT)).abs()
    df['tr2'] = (df['low'] - df['close'].shift(LOW_SHIFT)).abs()
    df['TR'] = df[['tr0','tr1','tr2']].max(axis=1)

    df['up_move'] = df['high'] - df['high'].shift(LOW_SHIFT)
    df['down_move'] = df['low'].shift(LOW_SHIFT) - df['low']

    df['DM_plus'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['DM_minus'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)

    # Wilder smoothing
    TR_smooth = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    DMp_smooth = df['DM_plus'].ewm(alpha=1/period, adjust=False).mean()
    DMm_smooth = df['DM_minus'].ewm(alpha=1/period, adjust=False).mean()

    df['DI_plus'] = 100 * (DMp_smooth / TR_smooth)
    df['DI_minus'] = 100 * (DMm_smooth / TR_smooth)
    df['DX'] = ( (df['DI_plus'] - df['DI_minus']).abs() / (df['DI_plus'] + df['DI_minus']) ) * 100
    adx = df['DX'].ewm(alpha=1/period, adjust=False).mean()
    return adx

# ===========================
# MT5 helper functions
# ===========================
def init_mt5():
    if not mt5.initialize():
        print("MT5 initialize() failed, error:", mt5.last_error())
        raise SystemExit("MT5 init failed")
    # optional: set timeout
    # time.sleep(5000)

def shutdown_mt5():
    mt5.shutdown()

def get_rates(symbol, timeframe, n=500):
    # get last n bars
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    # convert time
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_tick(symbol):
    return mt5.symbol_info_tick(symbol)

def symbol_info_check(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"{symbol} not found in Market Watch")
    if not info.visible:
        # try to make it visible
        mt5.symbol_select(symbol, True)
    return info

def calc_lot_by_risk(account_balance, stop_loss_points, risk_percent):
    # very rough lot calc (requires symbol contract size & tick value for precise calculation)
    # we will try to use symbol_info to compute tick value
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return LOT
    # For safety return default LOT if calculation seems off.
    try:
        tick_value = info.trade_tick_value
        tick_size = info.trade_tick_size
        # dollar risk per lot per point = tick_value / tick_size
        dollar_risk_per_point_per_lot = tick_value / tick_size
        risk_amount = account_balance * (risk_percent / 100.0)
        lots = risk_amount / (stop_loss_points * dollar_risk_per_point_per_lot)
        # adjust to minimal lot step
        step = info.volume_step
        if step and step > 0:
            lots = max(step, (round(lots / step) * step))
        if lots <= 0:
            return LOT
        return float(round(lots, 2))
    except Exception as e:
        return LOT

def place_order(symbol, order_type, volume, price, sl, tp, deviation=20, magic=234000):
    # order_type: mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": np.round(float(sl), 2),
        "tp": np.round(float(tp), 2),
        "deviation": deviation,
        "magic": magic,
        "comment": "scalper_ema821",
        "type_filling": mt5.ORDER_FILLING_FOK if mt5.symbol_info(symbol).trade_mode == mt5.SYMBOL_TRADE_MODE_FULL else mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Gagal eksekusi order. Retcode: {result.retcode}")
        print(f"Deskripsi: {mt5.last_error()}")
    return result

def modify_position_sl(ticket, symbol, sl):
    # modify by sending a modify request for the order (position to order)
    # We will use trade request MODIFY_POSITION if available; safer: send trade request to modify order
    # In MT5 Python API, use mt5.order_send with ORDER_ACTION_SLTP is not straightforward — many brokers accept position modification via mt5.order_send with POSITION_ID
    # Fallback: use mt5.order_modify for pending orders only. For opened positions, use mt5.order_send to close+reopen? That's heavy.
    # But mt5.positions_get and mt5.order_modify exist depending on terminal.
    try:
        res = mt5.position_modify(symbol, ticket, sl, None)
        return res
    except Exception:
        return None

# ===========================
# Utility functions
# ===========================
def compute_signals(df):
    close = df['close']
    high = df['high']
    low = df['low']

    df['EMA_LOW'] = ema(close, EMA_LOW_PERIOD)
    df['EMA_HIGH'] = ema(close, EMA_HIGH_PERIOD)
    df['EMA_TREND'] = ema(close, EMA_TREND_PERIOD)
    df['RSI'] = calculate_rsi(close, RSI_PERIOD)
    df['ADX'] = calculate_adx(df, ADX_PERIOD)

    # cross detection uses previous closed candle
    # We'll compute cross on previous closed candle i.e. between EMA_LOW.shift(1) and EMA_HIGH.shift(1)
    df['prev_low'] = df['EMA_LOW'].shift(HIGH_SHIFT)
    df['prev_high'] = df['EMA_HIGH'].shift(HIGH_SHIFT)

    df['prev_cross_up'] = (df['prev_low'] > df['prev_high']) & (df['EMA_LOW'].shift(EMA_LOW_SHIFT) <= df['EMA_HIGH'].shift(EMA_LOW_SHIFT))
    df['prev_cross_down'] = (df['prev_low'] < df['prev_high']) & (df['EMA_LOW'].shift(EMA_LOW_SHIFT) >= df['EMA_HIGH'].shift(EMA_LOW_SHIFT))

    # Simpler: detect if previous candle closed with EMA_LOW > EMA_HIGH and earlier candle EMA_LOW <= EMA_HIGH
    df['cross_up_prev_candle'] = (df['EMA_LOW'].shift(HIGH_SHIFT) > df['EMA_HIGH'].shift(HIGH_SHIFT)) & (df['EMA_LOW'].shift(EMA_LOW_SHIFT) <= df['EMA_HIGH'].shift(EMA_LOW_SHIFT))
    df['cross_down_prev_candle'] = (df['EMA_LOW'].shift(HIGH_SHIFT) < df['EMA_HIGH'].shift(HIGH_SHIFT)) & (df['EMA_LOW'].shift(EMA_LOW_SHIFT) >= df['EMA_HIGH'].shift(EMA_LOW_SHIFT))

    # Tambahkan distance EMA
    df['ema_distance'] = abs(df['EMA_LOW'] - df['EMA_HIGH'])

    # Valid cross (tidak rapat)
    df['valid_cross_up'] = df['cross_up_prev_candle'] & (df['ema_distance'] >= MIN_EMA_DISTANCE)
    df['valid_cross_down'] = df['cross_down_prev_candle'] & (df['ema_distance'] >= MIN_EMA_DISTANCE)

    return df

def positions_for_symbol(symbol):
    pos = mt5.positions_get(symbol=symbol)
    return pos if pos is not None else tuple()

def total_net_profit(symbol):
    pos = positions_for_symbol(symbol)
    total = 0.0
    for p in pos:
        total += p.profit
    return total

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ===========================
# Main loop
# ===========================
def main():
    global cross_signal_buy, cross_signal_sell
    
    init_mt5()
    try:
        symbol_info_check(SYMBOL)
    except Exception as e:
        print("Symbol check error:", e)
        shutdown_mt5()
        return
    
    clear_screen()
    print(f"Starting scalper for {SYMBOL} M1 — {datetime.now().isoformat()}")

    last_bar_time = None

    while True:
        try:
            df = get_rates(SYMBOL, TIMEFRAME, n=500)
            if df is None or df.empty:
                print("No rates retrieved, retrying...")
                time.sleep(POLL_SECONDS)
                continue

            # identify last closed candle time
            # df.iloc[-1] is the current (possibly still-forming) candle; df.iloc[-2] is last closed
            last_closed = df.iloc[-2]
            last_closed_time = last_closed['time']

            # run signals
            df_sign = compute_signals(df)

            # evaluate previous closed candle signals
            prev = df_sign.iloc[-2]  # previous closed candle row
            current = df_sign.iloc[-1]  # current forming candle

            # fetch current market tick
            tick = get_tick(SYMBOL)
            if tick is None:
                time.sleep(POLL_SECONDS)
                continue
            bid = tick.bid
            ask = tick.ask
            point = mt5.symbol_info(SYMBOL).point

            # decide if we have a confirmed cross on previous closed candle
            buy_signal = False
            sell_signal = False

            # Get Cross Signals EMA
            if prev['valid_cross_up'] and cross_signal_buy == False:
                cross_signal_buy = True
                cross_signal_sell = False  # reset opposite signal
            
            if prev['valid_cross_down'] and cross_signal_sell == False:
                cross_signal_sell = True
                cross_signal_buy = False  # reset opposite signal

            # Conditions for BUY:
            if (cross_signal_buy
                and current['EMA_LOW'] > current['EMA_TREND']
                and current['EMA_HIGH'] > current['EMA_TREND']
                and last_closed['close'] > current['EMA_TREND']
                and current['RSI'] > RSI_THRESHOLD
                and current['RSI'] > prev['RSI']
                and current['ADX'] > ADX_THRESHOLD):
                buy_signal = True

            # Conditions for SELL:
            if (cross_signal_sell
                and current['EMA_LOW'] < current['EMA_TREND']
                and current['EMA_HIGH'] < current['EMA_TREND']
                and last_closed['close'] < current['EMA_TREND']
                and current['RSI'] < RSI_THRESHOLD
                and current['RSI'] < prev['RSI']
                and prev['ADX'] > ADX_THRESHOLD):
                sell_signal = True

            print(f"CROSS UP: {cross_signal_buy} | CROSS DOWN: {cross_signal_sell} | LAST PRICE : {prev['close']:.5f} | RSI: {current['RSI']:.2f} | PREV RSI: {prev['RSI']:.2f} | ADX: {current['ADX']:.2f} | PREV ADX: {prev['ADX']:.2f}")

            # Avoid opening too many positions
            current_positions = positions_for_symbol(SYMBOL)
            num_positions = len(current_positions)

            # Place orders only when confirmed (B — confirmation entry)
            if buy_signal and num_positions < MAX_POSITIONS:
                # price to buy = ask
                price = ask
                sl_price = price - SL_IN_POINTS
                tp_price = price + TP_IN_POINTS

                # compute volume if using risk percent
                volume = LOT
                if RISK_PERCENT:
                    account_info = mt5.account_info()
                    if account_info is not None:
                        volume = calc_lot_by_risk(account_info.balance, SL_IN_POINTS, RISK_PERCENT)

                clear_screen()
                print(f"{datetime.now().isoformat()} BUY signal detected (confirmed on prev candle). Placing order...")
                res = place_order(SYMBOL, mt5.ORDER_TYPE_BUY, volume, price, sl_price, tp_price)
                cross_signal_buy = False  # reset after use
                print("Order result:", res)

            if sell_signal and num_positions < MAX_POSITIONS:
                price = bid
                sl_price = price + SL_IN_POINTS
                tp_price = price - TP_IN_POINTS

                volume = LOT
                if RISK_PERCENT:
                    account_info = mt5.account_info()
                    if account_info is not None:
                        volume = calc_lot_by_risk(account_info.balance, SL_IN_POINTS, RISK_PERCENT)

                clear_screen()
                print(f"{datetime.now().isoformat()} SELL signal detected (confirmed on prev candle). Placing order...")
                res = place_order(SYMBOL, mt5.ORDER_TYPE_SELL, volume, price, sl_price, tp_price)
                cross_signal_sell = False  # reset after use
                print("Order result:", res)

            # Auto-close on opposite signal: if existing long(s) and sell_signal True -> close longs
            if sell_signal and num_positions > 0:
                for p in current_positions:
                    if p.type == mt5.POSITION_TYPE_BUY:
                        print("Opposite signal detected -> closing BUY position ticket", p.ticket)
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": p.volume,
                            "type": mt5.ORDER_TYPE_SELL,
                            "position": p.ticket,
                            "price": bid,
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "close_on_opposite_signal"
                        }
                        res = mt5.order_send(close_request)
                        print("Close result:", res)

            if buy_signal and num_positions > 0:
                for p in current_positions:
                    if p.type == mt5.POSITION_TYPE_SELL:
                        print("Opposite signal detected -> closing SELL position ticket", p.ticket)
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": p.volume,
                            "type": mt5.ORDER_TYPE_BUY,
                            "position": p.ticket,
                            "price": ask,
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "close_on_opposite_signal"
                        }
                        res = mt5.order_send(close_request)
                        print("Close result:", res)

            # Break-even management per position
            positions = positions_for_symbol(SYMBOL)
            for p in positions:
                try:
                    entry = p.price_open
                    current_price = ask if p.type == mt5.POSITION_TYPE_BUY else bid
                    profit_points = (current_price - entry) if p.type == mt5.POSITION_TYPE_BUY else (entry - current_price)

                    # Hitung target & level trailing
                    tp_points = TP_IN_POINTS * point
                    trigger_trailing = 0.7 * tp_points
                    sl_new = entry + (0.5 * tp_points) if p.type == mt5.POSITION_TYPE_BUY else entry - (0.5 * tp_points)

                    last_adx = current['ADX']

                    # Jika ADX turun → close semua posisi
                    if last_adx < ADX_THRESHOLD:
                        print(f"ADX turun < {ADX_THRESHOLD}, close semua posisi!!!")
                        for pos in positions:
                            if pos.type == mt5.POSITION_TYPE_BUY:
                                close_type = mt5.ORDER_TYPE_SELL
                                price = bid
                            else:
                                close_type = mt5.ORDER_TYPE_BUY
                                price = ask
                            close_req = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": SYMBOL,
                                "volume": pos.volume,
                                "type": close_type,
                                "position": pos.ticket,
                                "price": price,
                                "deviation": 20,
                                "magic": 234000,
                                "comment": "close_adx_drop"
                            }
                            res = mt5.order_send(close_req)
                            print("Close result:", res)
                        break  # keluar dari loop posisi agar tidak trigger dua kali

                    # Trailing Stop: Profit ≥ 70% TP & ADX masih kuat
                    if profit_points >= trigger_trailing:
                        if (p.type == mt5.POSITION_TYPE_BUY and sl_new > p.sl) or \
                        (p.type == mt5.POSITION_TYPE_SELL and sl_new < p.sl):
                            
                            print(f"Trailing Stop aktif! Ticket {p.ticket} | Profit: {p.profit:.2f} | ADX: {last_adx:.2f}")
                            try:
                                res = mt5.position_modify(p.symbol, p.ticket, sl_new, p.tp)
                                print("Trailing SL modified:", res)
                            except Exception as e:
                                print("Trailing modify failed:", e)

                except Exception as e:
                    print("Trailing exception:", e)

            # Auto-close all if net profit >= threshold
            net = total_net_profit(SYMBOL)
            if net >= AUTO_CLOSE_ALL_NET_PROFIT and net > 0:
                print(f"Auto-close: net profit reached {net:.2f} >= {AUTO_CLOSE_ALL_NET_PROFIT}. Closing all positions on {SYMBOL}.")
                positions = positions_for_symbol(SYMBOL)
                for p in positions:
                    if p is None:
                        continue
                    if p.type == mt5.POSITION_TYPE_BUY:
                        price = bid
                        close_type = mt5.ORDER_TYPE_SELL
                    else:
                        price = ask
                        close_type = mt5.ORDER_TYPE_BUY
                    close_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": SYMBOL,
                        "volume": p.volume,
                        "type": close_type,
                        "position": p.ticket,
                        "price": price,
                        "deviation": 50,
                        "magic": 234000,
                        "comment": "auto_close_net_profit"
                    }
                    res = mt5.order_send(close_req)
                    print("Auto-close result:", res)

            # sleep a bit before next poll
            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print("Interrupted by user. Shutting down.")
            break
        except Exception as e:
            print("Exception in main loop:", e)
            time.sleep(1)

    shutdown_mt5()

if __name__ == "__main__":
    main()
