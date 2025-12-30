import MetaTrader5 as mt5
import json
import os
import time
from strategy import MomentumBreakoutStrategy

mt5.initialize()

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

strategy = MomentumBreakoutStrategy(config)
strategy.clear_screen()

while True:

    # FIND SIGNAL
    df = strategy.get_rates()
    df = strategy.apply_indicators(df)
    signal = strategy.check_entry(df)

    # OPEN POSISI STATUS
    if not strategy.has_open_position():
        strategy.waiting_signal()
    else:
        strategy.clear_screen()
        if signal["side"] == "BUY":
            strategy.open_buy()
        else:
            strategy.open_sell()

    # SEND ORDER
    if signal:
        strategy.send_order(signal)

    strategy.manage_position(df)
    time.sleep(1)