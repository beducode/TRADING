import MetaTrader5 as mt5
import pandas as pd
import os
import time, sys
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


class MomentumBreakoutStrategy:
    def __init__(self, config):
        self.cfg = config
        self.tp1_hit = False

    # =========================
    # CLEAR SCREEN
    # =========================
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    # =========================
    # WAIT SIGNAL
    # =========================
    def waiting_signal(self, text="WAITING SIGNAL", duration=1, interval=0.25):
        end_time = time.time() + duration
        while time.time() < end_time:
            for dots in range(3):
                sys.stdout.write(f"\r{text}{'.'*dots}{' '*(3-dots)}")
                sys.stdout.flush()
                time.sleep(interval)

    # =========================
    # OPEN BUY
    # =========================
    def open_buy(self, text="OPEN BUY", duration=1, interval=0.25):
        end_time = time.time() + duration
        while time.time() < end_time:
            for dots in range(3):
                sys.stdout.write(f"\r{text}{'.'*dots}{' '*(3-dots)}")
                sys.stdout.flush()
                time.sleep(interval)
    
    # =========================
    # OPEN SELL
    # =========================
    def open_sell(self, text="OPEN SELL", duration=1, interval=0.25):
        end_time = time.time() + duration
        while time.time() < end_time:
            for dots in range(3):
                sys.stdout.write(f"\r{text}{'.'*dots}{' '*(3-dots)}")
                sys.stdout.flush()
                time.sleep(interval)

    def has_open_position(self):
        positions = mt5.positions_get(symbol=self.cfg["symbol"])
        return positions is not None and len(positions) > 0

    # =========================
    # LOAD DATA
    # =========================
    def get_rates(self, n=100):
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15
        }
        rates = mt5.copy_rates_from_pos(
            self.cfg["symbol"],
            tf_map[self.cfg["timeframe"]],
            0,
            n
        )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    
    # =========================
    # MULTIPLE TIMEFRAME
    # =========================
    def get_last_candle(self, timeframe):
        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1
        }

        rates = mt5.copy_rates_from_pos(
            self.cfg["symbol"],
            tf_map[timeframe],
            1,   # candle yang sudah close
            1
        )

        if rates is None or len(rates) == 0:
            return None

        return rates[0]  # dict-like (open, high, low, close)
    
    # =========================
    # TREND MARKET
    # =========================
    def multi_tf_trend(self):
        tfs = ["H1", "M30", "M15", "M5"]
        bullish = 0
        bearish = 0

        for tf in tfs:
            candle = self.get_last_candle(tf)
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

        return "NEUTRAL"

    # =========================
    # INDICATORS
    # =========================
    def apply_indicators(self, df):
        df["ema_fast"] = EMAIndicator(df["close"], self.cfg["ema_fast"]).ema_indicator()

        df["ema_slow"] = EMAIndicator(df["close"], self.cfg["ema_slow"]).ema_indicator()

        df["rsi"] = RSIIndicator(df["close"], self.cfg["rsi_period"]).rsi()

        atr = AverageTrueRange(df["high"], df["low"], df["close"], self.cfg["atr_period"])
        df["atr"] = atr.average_true_range()
        df["atr_avg_20"] = df["atr"].rolling(20).mean()

        return df

    # =========================
    # ANTI CHOPPY
    # =========================
    def anti_choppy(self, row):
        if row["atr"] < row["atr_avg_20"] * self.cfg["anti_choppy"]["atr_ratio_min"]:
            return False

        if abs(row["ema_fast"] - row["ema_slow"]) < row["atr"] * self.cfg["anti_choppy"]["ema_distance_atr_ratio"]:
            return False

        return True

    # =========================
    # STRONG CANDLE
    # =========================
    def strong_candle(self, row):
        rng = row["high"] - row["low"]
        if rng <= 0:
            return None

        body = abs(row["close"] - row["open"])
        if body / rng < self.cfg["body_ratio_min"]:
            return None

        if rng < row["atr"] * self.cfg["atr_multiplier"]:
            return None

        return "BULLISH" if row["close"] > row["open"] else "BEARISH"

    # =========================
    # ENTRY CHECK
    # =========================
    def check_entry(self, df):
        prev = df.iloc[-2]
        last = df.iloc[-1]

        print(prev["close"])
        print(last["ema_slow"])
        print(prev["high"])

        if mt5.positions_total() > 0:
            return None
        
        trend_tf = self.multi_tf_trend()
        if trend_tf == "NEUTRAL":
            return None

        direction = self.strong_candle(prev)
        if not direction:
            return None
        
        antichoppy = self.anti_choppy(prev)
        if not antichoppy:
            return None

        tick = mt5.symbol_info_tick(self.cfg["symbol"])

        # BUY
        if (trend_tf == "BULLISH" and
            direction == "BULLISH" and
            prev["close"] > last["ema_slow"] and
            prev["rsi"] > self.cfg["rsi_buy"] and

            # === BREAK LEVEL ===
            tick.ask > prev["high"] #and
            
            # # === KONFIRMASI HARGA LANJUT NAIK ===
            # last["high"] > prev["high"] and  # follow through
            # tick.ask > prev["high"] + (prev["atr"] * self.cfg["confirm_atr_ratio"])
            ):

            entry = tick.ask
            sl = (prev["open"] + prev["close"]) / 2
            risk = entry - sl

            return {
                "side": "BUY",
                "entry": entry,
                "sl": sl,
                "tp": entry + risk * self.cfg["rr_tp2"]
            }

        # SELL
        if (trend_tf == "BEARISH" and
            direction == "BEARISH" and

            prev["close"] < last["ema_slow"] and
            prev["rsi"] < self.cfg["rsi_sell"] and

            # === BREAK LEVEL ===
            tick.bid < prev["low"] #and

            # # === KONFIRMASI HARGA LANJUT TURUN ===
            # last["low"] < prev["low"] and  # follow through
            # tick.bid < prev["low"] - (prev["atr"] * self.cfg["confirm_atr_ratio"])
            
            ):

            entry = tick.bid
            sl = (prev["open"] + prev["close"]) / 2
            risk = sl - entry

            return {
                "side": "SELL",
                "entry": entry,
                "sl": sl,
                "tp": entry - risk * self.cfg["rr_tp2"]
            }

        return None

    # =========================
    # ORDER SEND
    # =========================
    def send_order(self, signal):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg["symbol"],
            "volume": self.cfg["lot"],
            "type": mt5.ORDER_TYPE_BUY if signal["side"] == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "magic": self.cfg["magic"],
            "deviation": 20,
            "comment": "MomentumBreakout"
        }
        return mt5.order_send(request)

    # =========================
    # POSITION MANAGEMENT
    # =========================
    def manage_position(self, df):
        pos = mt5.positions_get(symbol=self.cfg["symbol"])
        if not pos:
            self.tp1_hit = False
            return

        pos = pos[0]
        prev = df.iloc[-2]

        entry = pos.price_open
        sl = pos.sl
        vol = pos.volume
        price = mt5.symbol_info_tick(self.cfg["symbol"]).bid if pos.type == 0 else \
                mt5.symbol_info_tick(self.cfg["symbol"]).ask

        risk = abs(entry - sl)
        tp1 = entry + risk * self.cfg["rr_tp1"] if pos.type == 0 else entry - risk * self.cfg["rr_tp1"]

        # === PARTIAL CLOSE ===
        if not self.tp1_hit:
            if (pos.type == 0 and price >= tp1) or (pos.type == 1 and price <= tp1):
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": pos.ticket,
                    "symbol": self.cfg["symbol"],
                    "volume": vol * self.cfg["partial_ratio"],
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "price": price,
                    "magic": self.cfg["magic"]
                })

                mt5.order_send({
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": entry,
                    "tp": 0
                })
                self.tp1_hit = True

        # === TRAILING STOP ===
        if self.tp1_hit:
            new_sl = max(sl, prev["low"]) if pos.type == 0 else min(sl, prev["high"])
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": 0
            })