import MetaTrader5 as mt5
import pandas as pd
import time
import os
import json
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
    def trend_bias(self, symbol):
        df = self.get_df(symbol, self.tf_trend, 500)
        if df is None:
            return None

        close = df['close']
        ema_fast = EMAIndicator(close, self.ema_fast).ema_indicator().iloc[-2]
        ema_slow = EMAIndicator(close, self.ema_slow).ema_indicator().iloc[-2]
        prev_ema_slow = EMAIndicator(close, self.ema_slow).ema_indicator().iloc[-3]

        if ema_fast > ema_slow and ema_slow > prev_ema_slow:
            return "BUY"
        if ema_fast < ema_slow and ema_slow < prev_ema_slow:
            return "SELL"
        return 'WAIT'
    
    # SUPPORT / RESISTANCE (HTF & LTF)
    def find_sr(self, df):
        support, resistance = [], []

        for i in range(1, len(df) - 1):
            if df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i+1]:
                support.append(df['low'][i])

            if df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i+1]:
                resistance.append(df['high'][i])

        return support, resistance
    
    def nearest_sr(self, price, supports, resistances):
        sup = max([s for s in supports if s < price], default=None)
        res = min([r for r in resistances if r > price], default=None)
        return sup, res
    
    # MULTI-TIMEFRAME LOGIC
    def multi_tf_sr(self, symbol):
        df_htf = self.get_df(symbol, self.tf_slow, 500)
        df_ltf = self.get_df(symbol, self.tf_fast, 500)
        if df_htf is None or df_ltf is None:
            return [], [], None

        sup_htf, res_htf = self.find_sr(df_htf)
        sup_ltf, res_ltf = self.find_sr(df_ltf)

        price = df_ltf['close'].iloc[-2]

        htf_sup, htf_res = self.nearest_sr(price, sup_htf, res_htf)
        ltf_sup, ltf_res = self.nearest_sr(price, sup_ltf, res_ltf)

        return htf_sup, htf_res, ltf_sup, ltf_res
    
    # TREND STEP CHANNEL (CORE FILTER)
    def step_channel_trend(self, symbol):
        df = self.get_df(symbol, self.tf_slow, 500)
        if df is None:
            return [], [], None
        
        highs = df['high']
        lows = df['low']

        if highs.iloc[-2] > highs.iloc[-3] and lows.iloc[-2] > lows.iloc[-3]:
            return "UP"

        if highs.iloc[-2] < highs.iloc[-3] and lows.iloc[-2] < lows.iloc[-3]:
            return "DOWN"

        return "WAIT"
    
    def breakout_fakeout(self, df, resistance, support, atr_val, vol_period=14, wick_thresh=0.3):
        close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        open_price = df['open'].iloc[-1]
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]

        # Wick rejection (aman divide by zero)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        candle_range = high - low
        if candle_range == 0:
            wick_ok = True
        else:
            wick_ok = (upper_wick / candle_range < wick_thresh) and (lower_wick / candle_range < wick_thresh)
    
        # 2-candle confirmation
        second_confirm = True
        if len(df) >= 3:
            close_2 = df['close'].iloc[-3]
            second_confirm = (close > close_2) if resistance is None else True

        # BREAKOUT BUY
        if resistance and close > resistance + atr_val * 0.2 and wick_ok and second_confirm:
            return "BREAKOUT_BUY"

        # FAKEOUT SELL
        if resistance and prev_close > resistance and close < resistance and wick_ok and second_confirm:
            return "FAKEOUT_SELL"

        # BREAKDOWN SELL
        if support and close < support - atr_val * 0.2 and wick_ok and second_confirm:
            return "BREAKDOWN_SELL"

        # FAKEOUT BUY
        if support and prev_close < support and close > support and wick_ok and second_confirm:
            return "FAKEOUT_BUY"

        return "WAIT"

    
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

    # ================= S/R =================
    def sr_zones(self, symbol):
        df = self.get_df(symbol, self.tf_slow, 500)
        if df is None:
            return [], [], None

        atr = AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period)\
            .average_true_range().iloc[-2]

        supports, resistances = [], []

        for i in range(2, len(df)-2):
            if df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i+1]:
                supports.append(df['low'][i])
            if df['high'][i] > df['high'][i-1] and df['high'][i] > df['high'][i+1]:
                resistances.append(df['high'][i])

        sup_z = [(s-atr*0.3, s+atr*0.3) for s in set(supports)]
        res_z = [(r-atr*0.3, r+atr*0.3) for r in set(resistances)]

        return sup_z, res_z, atr
    
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

    def trailing_management(self, symbol):
        if not CONFIG["trailing"]["enable"]:
            return

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        for pos in positions:
            tick = mt5.symbol_info_tick(symbol)
            atr_val = AverageTrueRange(
                self.get_df(symbol, self.tf_fast, 500)['high'],
                self.get_df(symbol, self.tf_fast, 500)['low'],
                self.get_df(symbol, self.tf_fast, 500)['close'],
                self.atr_period
            ).average_true_range().iloc[-1]

            trail_distance = CONFIG["trailing"]["atr_multiplier"] * atr_val
            min_distance = CONFIG["trailing"]["min_distance"] * mt5.symbol_info(symbol).point

            # BUY POSITION
            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = tick.bid - trail_distance
                if pos.sl is None or new_sl > pos.sl + min_distance:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": new_sl,
                        "tp": pos.tp
                    }
                    mt5.order_send(request)
                    print(f"[TRAILING STOP] {symbol} BUY | Old SL: {pos.sl} | New SL: {new_sl:.5f}")

            # SELL POSITION
            elif pos.type == mt5.POSITION_TYPE_SELL:
                new_sl = tick.ask + trail_distance
                if pos.sl is None or new_sl < pos.sl - min_distance:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": new_sl,
                        "tp": pos.tp
                    }
                    mt5.order_send(request)
                    print(f"[TRAILING STOP] {symbol} SELL | Old SL: {pos.sl} | New SL: {new_sl:.5f}")


    # ================= SIGNAL =================
    def signal(self, symbol):
        if self.has_open_position(symbol):
            return None, None

        if SESSION_ENABLE:
            if not self.session_allowed() or not self.news_allowed():
                return None, None

        bias = self.trend_bias(symbol)
        if not bias:
            return None, None

        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return None, None

        price = df['close'].iloc[-2]
        atr_val = AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period).average_true_range().iloc[-1]
        trend = self.step_channel_trend(symbol)
        htf_sup, htf_res, ltf_sup, ltf_res = self.multi_tf_sr(symbol)
        event = self.breakout_fakeout(df, ltf_res, ltf_sup, atr_val)
        tolerance = atr_val * 0.3

        htf_sup = htf_sup if htf_sup is not None else price
        htf_res = htf_res if htf_res is not None else price

        if bias == 'BUY':
            print(f"PAIR : {symbol} | BIAS : {bias} | TREND : {trend} | SUP : {htf_sup:.2f} | RES : {htf_res:.2f}")
        else:
            print(f"PAIR : {symbol} | BIAS : {bias} | TREND : {trend} | SUP : {htf_sup:.2f} | RES : {htf_res:.2f}")

        # BUY SETUP
        if (
            bias == "BUY"
            and trend == "UP"
            and htf_sup
            and abs(price - htf_sup) <= tolerance
            and event in ["FAKEOUT_BUY", "BREAKOUT_BUY"]
        ):
            return "BUY", price, htf_sup, htf_res, atr_val

        # SELL SETUP
        if (
            bias == "SELL"
            and trend == "DOWN"
            and htf_res
            and abs(price - htf_res) <= tolerance
            and event in ["FAKEOUT_SELL", "BREAKDOWN_SELL"]
        ):
            return "SELL", price, htf_sup, htf_res, atr_val

        return None, None, None, None, None

    # ================= LOT =================
    def calc_lot(self, symbol, atr):
        acc = mt5.account_info()
        sym = mt5.symbol_info(symbol)

        risk_money = acc.balance * (self.risk_percent / 100)
        sl_points = atr / sym.point
        lot = risk_money / (sl_points * sym.trade_tick_value)
        return round(max(sym.volume_min, min(lot, sym.volume_max)), 2)

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

                    # ====== Trailing Stop / Trailing Profit ======
                    self.trailing_management(symbol)

                time.sleep(1)

            except Exception as e:
                print("ERROR:", e)
                time.sleep(1)

# ================= START =================
if __name__ == "__main__":
    bot = AutoSRBot()
    bot.clear_screen()
    bot.run()
