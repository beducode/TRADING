import MetaTrader5 as mt5
import pandas as pd
import time
import os
import json
import math
from datetime import datetime
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# ===========================
# LOAD CONFIG JSON
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# ======================
# BASIC SETTINGS
# ======================
SYMBOLS = CONFIG["symbols"]
MTL_ATR_LOW  = CONFIG["atr_multiplier"]["sl"]
MTL_ATR_HIGH = CONFIG["atr_multiplier"]["tp"]

# ======================
# SESSION SETTINGS (WIB) 
# ======================
SESSION_ENABLE = CONFIG["session"]["enable"]
LONDON_START = CONFIG["session"]["london"]["start"]
LONDON_END   = CONFIG["session"]["london"]["end"]
NY_START = CONFIG["session"]["newyork"]["start"]
NY_END   = CONFIG["session"]["newyork"]["end"]

# ================= NEWS FILTER (HIGH IMPACT TIME) =================
NEWS_BLOCK_HOURS = CONFIG["news_filter"]["block_hours"]

def tf_map(tf):
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4
    }[tf]

class SwingDetector:
    def __init__(self, lookback=3):
        self.lookback = lookback

    def detect(self, highs, lows):
        swing_high = None
        swing_low = None

        if len(highs) < self.lookback * 2:
            return None, None

        i = -self.lookback - 1

        if highs.iloc[i] == max(highs.iloc[i-self.lookback:i+self.lookback]):
            swing_high = highs.iloc[i]

        if lows.iloc[i] == min(lows.iloc[i-self.lookback:i+self.lookback]):
            swing_low = lows.iloc[i]

        return swing_high, swing_low
    
class AutoSRBot:
    def __init__(self):
        self.risk_percent = CONFIG["risk"]["percent"]

        self.tf_trend = tf_map(CONFIG["timeframe"]["trend"])
        self.tf_slow = tf_map(CONFIG["timeframe"]["sr"])
        self.tf_fast = tf_map(CONFIG["timeframe"]["entry"])

        cfg = CONFIG["indicator"]
        self.ema_fast = cfg["ema_fast"]
        self.ema_slow = cfg["ema_slow"]
        self.rsi_period = cfg["rsi_period"]
        self.atr_period = cfg["atr_period"]

        self.magic = CONFIG["trade"]["magic"]
        self.deviation = CONFIG["trade"]["deviation"]
        self.default_lot = CONFIG["trade"]["default_lot"]

        self.init_mt5()

    # ================= INIT =================
    def init_mt5(self):
        if not mt5.initialize():
            raise RuntimeError("MT5 init failed")
        print("[OK] MT5 Connected")

    # ================= TIME FILTER =================
    def session_allowed(self):
        hour = datetime.now().hour

        london = LONDON_START <= hour <= LONDON_END
        newyork = hour >= NY_START or hour <= NY_END

        return london or newyork

    def news_allowed(self):
        hour = datetime.now().hour
        for start, end in NEWS_BLOCK_HOURS:
            if start <= hour <= end:
                return False
        return True

    # ================= DATA =================
    def get_df(self, symbol, tf, bars=300):
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    # ================= TREND =================
    def trend(self, symbol):
        df = self.get_df(symbol, self.tf_trend, 500)
        if df is None:
            return None

        close = df['close']
        atr_val = AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period).average_true_range()
        atr_avg_20 = AverageTrueRange(df['high'], df['low'], df['close'], 20).average_true_range()
        rsi = RSIIndicator(close, self.rsi_period).rsi()
        ema_fast = EMAIndicator(close, self.ema_fast).ema_indicator()
        ema_slow = EMAIndicator(close, self.ema_slow).ema_indicator()

        ema_distance = abs(ema_fast.iloc[-2] - ema_slow.iloc[-2])

        cross_count = 0
        for i in range(2, 10):
            if (ema_fast.iloc[i] > ema_slow.iloc[i]) != (ema_fast.iloc[i-1]  > ema_slow.iloc[i-1]):
                cross_count += 1

        atr_ratio = atr_val.iloc[-2] / atr_avg_20.iloc[-2]

        is_choppy = (
            ema_distance < 0.2 * atr_val.iloc[-2] or
            cross_count >= 3 or
            atr_ratio < 0.7 or
            (45 < rsi.iloc[-2] < 55)
        )

        if ema_fast.iloc[-2]  > ema_slow.iloc[-2] and not is_choppy:
            return "BUY"
        if ema_fast.iloc[-2] < ema_slow.iloc[-2] and not is_choppy:
            return "SELL"
        return 'WAIT'
    
    def close_position(self, pos):
        try:
            symbol = pos.symbol
            volume = pos.volume
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "price": float(price),
                "position": int(pos.ticket),
                "deviation": 50,
                "magic": 234000,
                "comment": "AutoClose",
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            res = mt5.order_send(request)
            print(f"CLOSE_POSITION RESULT :{res}")
            return res
        except Exception as e:
            print(f"CLOSE_POSITION EXCEPTION :{e}")
            return None

    def fibonacci_levels(self, low, high):
        diff = high - low
        return {
            "0.382": high - diff * 0.382,
            "0.5": high - diff * 0.5,
            "0.618": high - diff * 0.618,
            "1.0": high,
            "1.272": high + diff * 0.272,
            "1.618": high + diff * 0.618,
        }
    
    def trailing_stop_fib_buy(
        price,
        entry,
        current_sl,
        fib_levels
    ):
        new_sl = current_sl

        if price >= fib_levels["1.0"]:
            new_sl = max(new_sl, entry)

        if price >= fib_levels["1.272"]:
            new_sl = max(new_sl, fib_levels["0.382"])

        if price >= fib_levels["1.618"]:
            new_sl = max(new_sl, fib_levels["0.5"])

        return new_sl
    
    def trailing_stop_fib_sell(
        price,
        entry,
        current_sl,
        fib_low,
        fib_high
    ):
        diff = fib_high - fib_low

        tp1 = fib_low
        tp2 = fib_low - diff * 0.272
        tp3 = fib_low - diff * 0.618

        sl_382 = fib_low + diff * 0.382
        sl_5 = fib_low + diff * 0.5

        new_sl = current_sl

        if price <= tp1:
            new_sl = min(new_sl, entry)

        if price <= tp2:
            new_sl = min(new_sl, sl_382)

        if price <= tp3:
            new_sl = min(new_sl, sl_5)

        return new_sl

    

    # ================= SIGNAL =================
    def signal(self, symbol):
        if self.has_open_position(symbol):
            return None, None, None, None, None

        if SESSION_ENABLE:
            if not self.session_allowed() or not self.news_allowed():
                return None, None, None, None, None

        bias = self.trend(symbol)
        if not bias:
            return None, None, None, None, None

        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return None, None, None, None, None
        
        close = df['close'].iloc[-2]
        open = df['open'].iloc[-2]
        high = df['high'].iloc[-2]
        low = df['low'].iloc[-2]

        prev_price = df['close'].iloc[-3]
        price = df['close'].iloc[-1]
        atr_val = AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period).average_true_range()
        rsi = RSIIndicator(df['close'], self.rsi_period).rsi()

        support, resistance = self.get_sr(symbol)

        body = abs(close - open)
        range_ = high - low

        momentum = (
            body / range_ >= 0.7 and
            range_ >= 0.8 * atr_val.iloc[-1]
        )

        bullish = (
            momentum and
            close > open and
            close >= high - (range_ * 0.1) and
            rsi.iloc[-1] > 55
        )

        bearish = (
            momentum and
            close < open and
            close <= low + (range_ * 0.1) and
            rsi.iloc[-1] < 45
        )

        if bias == 'BUY':
            print(f"PAIR : {symbol} | BIAS : {bias} | RSI : {rsi.iloc[-1]:.2f} | MOMENTUM : {momentum} | BULISH : {bullish}")
        else:
            print(f"PAIR : {symbol} | BIAS : {bias} | RSI : {rsi.iloc[-1]:.2f} | MOMENTUM : {momentum} | BEARISH : {bearish}")

        # BUY SETUP
        if (
            bias == "BUY"
            and bullish
        ):
            return "BUY", price, support, resistance, atr_val.iloc[-1]

        # SELL SETUP
        if (
            bias == "SELL"
            and bearish
        ):
            return "SELL", price, support, resistance, atr_val.iloc[-1]

        return None, None, None, None, None
    
    # SUPPORT / RESISTANCE (HTF & LTF)
    def find_sr(self, df):
        support, resistance = [], []

        for i in range(1, len(df) - 1):
            if df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i+1]:
                support.append(df['low'][i])

            if df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i+1]:
                resistance.append(df['high'][i])

        return support, resistance
    
    # MULTI-TIMEFRAME LOGIC
    def get_sr(self, symbol):
        df = self.get_df(symbol, self.tf_fast, 100)
        if df is None:
            return [], [], None

        sup_dt, res_dt = self.find_sr(df)

        sup = min(sup_dt)
        res = max(res_dt)

        return sup, res
    
    # SL & TP (ATR + STRUCTURE)
    def sl_tp(self, signal, price, atr_val, support, resistance):
        if signal == "BUY":
            sl = support - atr_val * 0.5
            tp = price + atr_val * 2
            return sl, tp

        if signal == "SELL":
            sl = resistance + atr_val * 0.5
            tp = price - atr_val * 2
            return sl, tp

        return None, None

    # ================= ORDER =================
    def open_trade(self, symbol, side, sl, tp):
        if self.has_open_position(symbol):
            return

        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if side == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.default_lot,
            "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "AUTO_SR_MTF_JSON"
        }


        mt5.order_send(request)
        print(f"[OPEN] {symbol} {side}")

    def has_open_position(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        return positions is not None and len(positions) > 0

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    # ================= RUN =================
    def run(self):
        print("[TRADE BOT BY BEDUCODE IS RUNNING]")
        print("==================================")
        while True:
            try:
                for symbol in SYMBOLS:
                    side, price, sup, res, atr = self.signal(symbol)
                    if side:
                        sl, tp = self.sl_tp(side, price, atr, sup, res)
                        self.open_trade(symbol, side, sl, tp)
                    
                time.sleep(1)

            except Exception as e:
                print("ERROR:", e)
                time.sleep(1)

# ================= START =================
if __name__ == "__main__":
    bot = AutoSRBot()
    bot.clear_screen()
    bot.run()
