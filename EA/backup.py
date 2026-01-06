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

# ======================
# CANDLE 
# ======================
BODY_RATIO = CONFIG["candle"]["body_ratio_min"]

# ======================
# ATR 
# ======================
ATR_BODY = CONFIG["atr"]["atr_body"]
ATR_EMA_DISTANCE = CONFIG["atr"]["ema_distance_atr_ratio"]
ATR_RATIO_MIN = CONFIG["atr"]["atr_ratio_min"]
CONFIRM_ATR_RATIO = CONFIG["atr"]["confirm_atr_ratio"]

# ======================
# RSI 
# ======================
RSI_PERIOD = CONFIG["rsi"]["rsi_period"]
RSI_BUY = CONFIG["rsi"]["rsi_buy"]
RSI_SELL = CONFIG["rsi"]["rsi_sell"]

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
        self.last_entry_candle = {}
        self.take_profit = CONFIG["sltp"]["take_profit"]

        self.tf_trend = tf_map(CONFIG["timeframe"]["trend"])
        self.tf_slow = tf_map(CONFIG["timeframe"]["sr"])
        self.tf_fast = tf_map(CONFIG["timeframe"]["entry"])

        cfg = CONFIG["indicator"]
        self.ema_fast = cfg["ema_fast"]
        self.ema_slow = cfg["ema_slow"]
        self.ema_trend = cfg["ema_trend"]
        self.atr_period = cfg["atr_period"]
        self.atr_period_high = cfg["atr_period_high"]

        self.magic = CONFIG["trade"]["magic"]
        self.deviation = CONFIG["trade"]["deviation"]
        self.default_lot = CONFIG["trade"]["default_lot"]

        self.init_mt5()

    # =========================
    # INIT
    # =========================
    def init_mt5(self):
        if not mt5.initialize():
            raise RuntimeError("MT5 init failed")
        print("[OK] MT5 Connected")

    # =========================
    # DATA
    # =========================
    def get_df(self, symbol, tf, bars=300):
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    # =========================
    # CALC TAKE PROFIT
    # =========================
    def calc_tp_usd(self, symbol, lot):
        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        tick_value = info.trade_tick_value
        tick_size = info.trade_tick_size

        if tick_value == 0 or tick_size == 0:
            return None

        # jarak harga agar profit = 1 USD
        price_distance = (self.take_profit / (tick_value * lot)) * tick_size
        return price_distance
    
    # =========================
    # GET CLOSE CANDLE TIME
    # =========================
    def get_closed_candle_time(self, symbol):
        df = self.get_df(symbol, self.tf_fast, 3)
        if df is None:
            return None
        return df['time'].iloc[-2]  # candle sudah close
    
    # =========================
    # MULTIPLE TIMEFRAME
    # =========================
    def get_last_candle(self, timeframe, symbol):
        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1
        }

        rates = mt5.copy_rates_from_pos(symbol,
            tf_map[timeframe],
            1,   # candle yang sudah close
            1
        )

        if rates is None or len(rates) == 0:
            return None

        return rates[0]  # dict-like (open, high, low, close)

    # =========================
    # MULTI TREND
    # =========================
    def trend(self, symbol):
        tfs = ["H1", "M30", "M15", "M5"]
        bullish = 0
        bearish = 0
        
        for tf in tfs:
            candle = self.get_last_candle(tf, symbol)
            if candle is None:
                continue

            if candle["close"] > candle["open"]:
                bullish += 1
            elif candle["close"] < candle["open"]:
                bearish += 1

        if bullish >= 3:
            return "BULLISH"
        if bearish >= 3:
            return "BEARISH"

        return "WAIT"
    
    # =========================
    # STRONG CANDLE
    # =========================
    def strong_candle(self, row, atr):
        close = row['close'].iloc[-2]
        open = row['open'].iloc[-2]
        high = row['high'].iloc[-2]
        low = row['low'].iloc[-2]
        rng = high - low
        body = abs(close - open)
        com_body_ratio = body / rng
        body_range = atr * ATR_BODY
        if (rng >= 0) and (com_body_ratio > BODY_RATIO) and (rng > body_range):
            if close > open:
                return 'BULLISH'
            else:
                return 'BEARISH'
        
        return "WAIT"
    
    # =========================
    # ANTI CHOPPY
    # =========================
    def anti_choppy(self, atr_low, atr_high, ema_fast, ema_slow):
        if atr_low < atr_high * ATR_RATIO_MIN:
            return False

        if abs(ema_fast - ema_slow) < atr_low * ATR_EMA_DISTANCE:
            return False

        return True

    # =========================
    # CLOSE POSITION
    # =========================
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

    # =========================
    # FIND SIGNAL
    # =========================
    def signal(self, symbol):
        # =========================
        # CEK CANDLE TIME
        # =========================
        closed_candle_time = self.get_closed_candle_time(symbol)
        if closed_candle_time is None:
            return None
        
        # ===============================
        # CEK: SUDAH ENTRY DI CANDLE INI?
        # ===============================
        last_time = self.last_entry_candle.get(symbol)
        if last_time == closed_candle_time:
            return None

        # =========================
        # GET DATA
        # =========================
        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return None
        
        # =========================
        # DATA INDIKATOR
        # =========================
        atr_low = AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period).average_true_range()
        atr_high = AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period_high).average_true_range()
        rsi = RSIIndicator(df['close'], RSI_PERIOD).rsi()
        ema_fast = EMAIndicator(df["close"], self.ema_fast).ema_indicator()
        ema_slow = EMAIndicator(df["close"], self.ema_slow).ema_indicator()

        # =========================
        # CHECK POSITION
        # =========================
        position = self.has_open_position(symbol)
        if position:
            return None
            
        trend = self.trend(symbol)
        candle_cond = self.strong_candle(df, atr_low.iloc[-2])
        
        antichoppy = self.anti_choppy(atr_low.iloc[-2], atr_high.iloc[-2], ema_fast.iloc[-2], ema_slow.iloc[-2])
        
        tick = mt5.symbol_info_tick(symbol)

        # BUY
        if(
            # trend == "BULLISH" and
            candle_cond == "BULLISH" and
            # antichoppy and
            # df['close'].iloc[-2] > ema_slow.iloc[-2] and
            # rsi.iloc[-2] > RSI_BUY and
            # === BREAK LEVEL ===
            tick.ask > df['high'].iloc[-2] 
            # and

            # # === KONFIRMASI HARGA LANJUT NAIK ===
            # df['high'].iloc[-1] > df['high'].iloc[-2] and
            # tick.ask > df['high'].iloc[-2] + (atr_low.iloc[-2] * CONFIRM_ATR_RATIO)
        ):
            entry = tick.ask
            # SL ADA DI HARGA OPEN CANDLE SEBELUM NYA
            sl = df['open'].iloc[-2]

            # TP SESUAI CALC USD
            tp_distance = self.calc_tp_usd(symbol, self.default_lot)

            return {
                "side": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": entry + tp_distance
            }
        
        # SELL
        if (
            # trend == "BEARISH" and
            candle_cond == "BEARISH" and
            # antichoppy and
            # df['close'].iloc[-2] < ema_slow.iloc[-2] and
            # rsi.iloc[-2] < RSI_SELL and

            # === BREAK LEVEL ===
            tick.bid < df['high'].iloc[-2] 
            # and

            # # === KONFIRMASI HARGA LANJUT TURUN ===
            # df['low'].iloc[-1] > df['low'].iloc[-2] and
            # tick.bid > df['low'].iloc[-2] + (atr_low.iloc[-2] * CONFIRM_ATR_RATIO)    
            ):

            entry = tick.bid
            sl = df['open'].iloc[-2]
            
            # TP SESUAI CALC USD
            tp_distance = self.calc_tp_usd(symbol, self.default_lot)

            return {
                "side": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": entry - tp_distance
            }

        if trend == 'BUY':
            print(f"PAIR : {symbol} | CANDLE : {candle_cond}")
        else:
            print(f"PAIR : {symbol} | CANDLE : {candle_cond}")

        return None
    
    # =========================
    # SL & TP
    # =========================
    def sl_tp(self, signal):
        if signal["side"] == "BUY":
            sl = signal["sl"]
            tp = signal["tp"]
            return sl, tp

        if signal["side"] == "SELL":
            sl = signal["sl"]
            tp = signal["tp"]
            return sl, tp

        return None, None

    # ================= ORDER =================
    def open_trade(self, symbol, side, sl, tp):
        if self.has_open_position(symbol):
            return
        
        closed_candle_time = self.get_closed_candle_time(symbol)
        if closed_candle_time:
            self.last_entry_candle[symbol] = closed_candle_time

        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if side["side"] == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.default_lot,
            "type": mt5.ORDER_TYPE_BUY if side["side"] == "BUY" else mt5.ORDER_TYPE_SELL,
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
                    side = self.signal(symbol)
                    if side:
                        sl, tp = self.sl_tp(side)
                        self.open_trade(symbol, side, sl, tp)
                    
                time.sleep(0.5)

            except Exception as e:
                print("ERROR:", e)
                time.sleep(0.5)

# ================= START =================
if __name__ == "__main__":
    bot = AutoSRBot()
    bot.clear_screen()
    bot.run()
