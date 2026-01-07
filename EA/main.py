import MetaTrader5 as mt5
import pandas as pd
import time
import os
import json
import math
from datetime import datetime
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator

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
# STOCHASTIC 
# ======================
STOCH_PERIODE = CONFIG["stochastic"]["stoch_period"]
K_SMOOTH = CONFIG["stochastic"]["stoch_k"]
D_PERIODE = CONFIG["stochastic"]["stoch_d"]
LEVEL_STOCH_HIGH = CONFIG["stochastic"]["level_stoch_high"]
LEVEL_STOCH_LOW = CONFIG["stochastic"]["level_stoch_low"]
LEVEL_STOCH_MID = CONFIG["stochastic"]["level_stoch_mid"]

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

    def get_tick(self, symbol):
        try:
            return mt5.symbol_info_tick(symbol)
        except Exception:
            return None

    def is_market_closed(self, symbol):
        info = mt5.symbol_info(symbol)
        tick = self.get_tick(symbol)

        if info is None:
            return True

        # Jika symbol trading disabled → market tutup
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return True

        # Jika tick tidak ada → market tutup
        if tick is None:
            return True

        # Jika bid/ask 0 → market tutup
        if tick.bid == 0 or tick.ask == 0:
            return True

        return False
    
    # =========================
    # STOCHASTIC
    # =========================
    def calculate_stochastic(self, df, k_period, k_smooth, d_period):
        df = df.copy()
        df['HH'] = df['high'].rolling(window=k_period).max()
        df['LL'] = df['low'].rolling(window=k_period).min()
        df['%K_raw'] = ((df['close'] - df['LL']) / (df['HH'] - df['LL'])) * 100
        df['%K'] = df['%K_raw'].rolling(window=k_smooth).mean()
        df['%D'] = df['%K'].rolling(window=d_period).mean()
        return df
        

    def stochastic_signal(self, df):
        buy_stoch = df['%K'].iloc[-1] < LEVEL_STOCH_LOW and df['%K'].iloc[-1] > df['%D'].iloc[-1]
        sell_stoch = df['%K'].iloc[-1] > LEVEL_STOCH_HIGH and df['%K'].iloc[-1] < df['%D'].iloc[-1]

        if buy_stoch:
            return 'BUY'
        elif sell_stoch:
            return 'SELL'
        else:
            return 'WAIT'

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
        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return None
        return df['time'].iloc[-2]  # candle sudah close
    
    # =========================
    # MULTIPLE TIMEFRAME
    # =========================
    def get_last_candle(self, timeframe, symbol):
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
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
        tfs = ["H1", "M30", "M15", "M5", "M1"]
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
    # CANDLE PATTERN
    # =========================
    def bullish_engulfing(self, symbol):
        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        return (
            prev["close"] < prev["open"] and
            curr["close"] > curr["open"] and
            curr["close"] > prev["open"] and
            curr["open"] < prev["close"]
        )

    def bearish_engulfing(self, symbol):
        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return False
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        return (
            prev["close"] > prev["open"] and
            curr["close"] < curr["open"] and
            curr["open"] > prev["close"] and
            curr["close"] < prev["open"]
        )

    def hammer(self,symbol):
        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return False
        c = df.iloc[-1]

        body = abs(c["close"] - c["open"])
        lower = min(c["open"], c["close"]) - c["low"]
        upper = c["high"] - max(c["open"], c["close"])
        return lower >= body * 2 and upper <= body

    def shooting_star(self,symbol):
        df = self.get_df(symbol, self.tf_fast, 500)
        if df is None:
            return False
        c = df.iloc[-1]
    
        body = abs(c["close"] - c["open"])
        upper = c["high"] - max(c["open"], c["close"])
        lower = min(c["open"], c["close"]) - c["low"]
        return upper >= body * 2 and lower <= body
    
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

        if (rng >= 0) and (com_body_ratio > BODY_RATIO):
            if close > open:
                return 'BULLISH'
            else:
                return "BEARISH"
        
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
        ema_fast = EMAIndicator(df["close"], self.ema_fast).ema_indicator()
        ema_slow = EMAIndicator(df["close"], self.ema_slow).ema_indicator()
        ema_trend = EMAIndicator(df["close"], self.ema_trend).ema_indicator()

        # =========================
        # CHECK POSITION
        # =========================
        position = self.has_open_position(symbol)
        if position:
            return None
            
        trend_multiframe = self.trend(symbol)
        candle_trend = self.strong_candle(df, atr_low.iloc[-2])
        rsi = RSIIndicator(df['close'], RSI_PERIOD).rsi().iloc[-1]
        data_stoch = self.calculate_stochastic(df, STOCH_PERIODE, K_SMOOTH, D_PERIODE)
        signal_stoch = self.stochastic_signal(data_stoch)
        close_candle = df['close'].iloc[-2]
        
        tick = mt5.symbol_info_tick(symbol)

        # BUY
        if(
            candle_trend == "BULLISH" and
            ema_fast.iloc[-2] > ema_slow.iloc[-2] and
            tick.ask > df['high'].iloc[-2] and
            rsi <= RSI_SELL and
            signal_stoch == 'BUY' and
            (
                self.bullish_engulfing(symbol) or self.hammer(symbol)
            )
        ):
            entry = tick.ask
            open_prev = df['open'].iloc[-2] 
            close_prev = df['close'].iloc[-2] 
            body = abs(close_prev - open_prev) 
            sl = open_prev + (body / 2)

            # RR 1:1
            risk = entry - sl
            tp = entry + risk

            return {
                "side": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": tp
            }
        
        # SELL
        if (
            candle_trend == "BEARISH" and
            ema_fast.iloc[-2] < ema_slow.iloc[-2] and
            tick.bid < df['low'].iloc[-2] and
            rsi >= RSI_BUY and
            signal_stoch == 'SELL' and
            (
                self.bearish_engulfing(symbol) or self.shooting_star(symbol)
            )
        ):

            entry = tick.bid
            open_prev = df['open'].iloc[-2] 
            close_prev = df['close'].iloc[-2] 
            body = abs(close_prev - open_prev) 
            sl = open_prev - (body / 2)

            
            # RR 1:1
            risk = sl - entry
            tp = entry - risk

            return {
                "side": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": tp
            }

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if trend_multiframe == 'BULLISH':
            print(f"[{now}] PAIR : {symbol} | CANDLE : {candle_trend} | STOCH : {signal_stoch} | RSI : {rsi:.2f}")
        else:
            print(f"[{now}] PAIR : {symbol} | CANDLE : {candle_trend} | STOCH : {signal_stoch} | RSI : {rsi:.2f}")

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
                    if self.is_market_closed(symbol):
                        print(f"MARKET CLOSED FOR {symbol} — WAITING 1 HOURS...")
                        time.sleep(3600)  # 1 jam = 3600 detik
                        continue
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
