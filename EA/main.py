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

# ====================== 
# PROFIT TARGET (USD) 
# ======================
TP_USD = CONFIG["profit_target_usd"]["tp"]
SL_USD = CONFIG["profit_target_usd"]["sl"]

# ================= NEWS FILTER (HIGH IMPACT TIME) =================
NEWS_BLOCK_HOURS = CONFIG["news_filter"]["block_hours"]

class ConfigManager:
    def __init__(self, config_file, preset_file):
        self.config_file = config_file
        self.preset_file = preset_file
        self.last_config_mtime = 0
        self.last_preset_mtime = 0
        self.load()

    def load(self):
        with open(self.config_file) as f:
            self.config = json.load(f)

        with open(self.preset_file) as f:
            self.presets = json.load(f)

        self.active = self.presets["active_preset"]
        self.preset = self.presets[self.active]

        print(f"[CONFIG] LOADED PRESET : {self.active.upper()}")

    def reload_if_changed(self):
        cfg_m = os.path.getmtime(self.config_file)
        pre_m = os.path.getmtime(self.preset_file)

        if cfg_m != self.last_config_mtime or pre_m != self.last_preset_mtime:
            self.last_config_mtime = cfg_m
            self.last_preset_mtime = pre_m
            self.load()

def tf(tf):
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4
    }[tf]

class AutoSRBot:
    def __init__(self):
        self.cfg = ConfigManager("config.json", "presets.json")
        self.apply_preset()
        self.init_mt5()

        self.risk_percent = CONFIG["risk"]["percent"]

        self.tf_trend = tf(CONFIG["timeframe"]["trend"])
        self.tf_sr    = tf(CONFIG["timeframe"]["sr"])
        self.tf_entry = tf(CONFIG["timeframe"]["entry"])

        ind = CONFIG["indicator"]
        self.ema_fast = ind["ema_fast"]
        self.ema_slow = ind["ema_slow"]
        self.rsi_period = ind["rsi_period"]
        self.atr_period = ind["atr_period"]

        self.magic = CONFIG["trade"]["magic"]
        self.deviation = CONFIG["trade"]["deviation"]
        self.default_lot = CONFIG["trade"]["default_lot"]

    def apply_preset(self):
        p = self.cfg.preset

        self.risk_percent = p["risk_percent"]
        self.min_rsi_buy = p["min_rsi_buy"]
        self.max_rsi_sell = p["max_rsi_sell"]

        self.tf_trend = tf(p["timeframe"]["trend"])
        self.tf_sr    = tf(p["timeframe"]["sr"])
        self.tf_entry = tf(p["timeframe"]["entry"])

        self.atr_sl = p["atr_multiplier"]["sl"]
        self.atr_tp = p["atr_multiplier"]["tp"]
        
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
        df = self.get_df(symbol, self.tf_trend, 200)
        if df is None:
            return None

        close = df['close']
        ema_fast = EMAIndicator(close, self.ema_fast).ema_indicator().iloc[-1]
        ema_slow = EMAIndicator(close, self.ema_slow).ema_indicator().iloc[-1]
        rsi = RSIIndicator(close, self.rsi_period).rsi().iloc[-1]

        if ema_fast > ema_slow and rsi > 50:
            return "BUY"
        if ema_fast < ema_slow and rsi < 50:
            return "SELL"
        return None
    
    # ================== ENTRY CONFIRM (M1) ==================
    def volume_spike(self, df):
        return df['tick_volume'].iloc[-1] > df['tick_volume'].rolling(20).mean().iloc[-1] * MTL_ATR_LOW

    def bullish_rejection(self, df):
        c = df.iloc[-1]
        body = abs(c.close - c.open)
        lower_wick = min(c.open, c.close) - c.low
        return lower_wick > body * MTL_ATR_LOW

    def bearish_rejection(self, df):
        c = df.iloc[-1]
        body = abs(c.close - c.open)
        upper_wick = c.high - max(c.open, c.close)
        return upper_wick > body * MTL_ATR_LOW

    # ================= S/R =================
    def sr_zones(self, symbol):
        df = self.get_df(symbol, self.tf_sr, 200)
        if df is None:
            return [], [], None

        atr = AverageTrueRange(df['high'], df['low'], df['close'], self.atr_period)\
            .average_true_range().iloc[-1]

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
    
    def manage_profit_usd(self, symbol, tp_usd, sl_usd):
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return

        total_profit = sum(p.profit for p in positions)

        if total_profit >= tp_usd:
            print(f"TP USD HIT : {symbol} | PROFIT : {total_profit:.2f} USD")
            for p in positions:
                self.close_position(p)

        if total_profit <= sl_usd:
            print(f"SL USD HIT : {symbol} | LOSS : {total_profit:.2f} USD")
            for p in positions:
                self.close_position(p)

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

        df = self.get_df(symbol, self.tf_entry, 120)
        if df is None:
            return None, None

        price = df['close'].iloc[-1]
        rsi = RSIIndicator(df['close'], self.rsi_period).rsi().iloc[-1]

        sup, res, atr = self.sr_zones(symbol)

        volspike = self.volume_spike(df)
        bullish_reject = self.bullish_rejection(df)
        bearish_reject = self.bearish_rejection(df)

        if bias == 'BUY':
            print(f"PAIR : {symbol} | TREND : {bias} | RSI : {rsi:.2f} | VOLUME : {volspike} | BULLISH : {bullish_reject}")
        else:
            print(f"PAIR : {symbol} | TREND : {bias} | RSI : {rsi:.2f} | VOLUME : {volspike} | BEARISH : {bullish_reject}")
        
        if bias == "BUY":
            for l, h in sup:
                if (
                    l <= price <= h and
                    rsi > self.min_rsi_buy and
                    volspike and
                    bullish_reject
                ):
                    return "BUY", atr

        if bias == "SELL":
            for l, h in res:
                if (
                    l <= price <= h and
                    rsi < self.max_rsi_sell and
                    volspike and
                    bearish_reject
                ):
                    return "SELL", atr

        return None, None

    # ================= LOT =================
    def calc_lot(self, symbol, atr):
        acc = mt5.account_info()
        sym = mt5.symbol_info(symbol)

        risk_money = acc.balance * (self.risk_percent / 100)
        sl_points = atr / sym.point
        lot = risk_money / (sl_points * sym.trade_tick_value)
        return round(max(sym.volume_min, min(lot, sym.volume_max)), 2)

    # ================= ORDER =================
    def open_trade(self, symbol, side, atr):
        if self.has_open_position(symbol):
            self.manage_profit_usd(symbol, TP_USD, SL_USD)
            return

        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if side == "BUY" else tick.bid

        sl = price - atr*MTL_ATR_LOW if side == "BUY" else price + atr*MTL_ATR_LOW
        tp = price + atr*MTL_ATR_HIGH if side == "BUY" else price - atr*MTL_ATR_HIGH

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
        print("=================================")
        while True:
            try:
                self.cfg.reload_if_changed()
                self.apply_preset()

                for symbol in SYMBOLS:
                    side, atr = self.signal(symbol)
                    if side:
                        self.open_trade(symbol, side, atr)

                time.sleep(1)

            except Exception as e:
                print("ERROR:", e)
                time.sleep(1)

# ================= START =================
if __name__ == "__main__":
    bot = AutoSRBot()
    bot.clear_screen()
    bot.run()
