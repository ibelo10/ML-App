import ta
import pandas as pd

import pandas as pd
import ta

def add_features(df):
    # Make a copy of the DataFrame to avoid the SettingWithCopyWarning and protect original OHLCV values
    df = df.copy()

    # Store original OHLCV values
    original_ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Calculate technical indicators without altering the original OHLCV data
    df['SMA'] = ta.trend.sma_indicator(df['Close'], window=14)
    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=14)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['Bollinger_Mid'] = ta.volatility.bollinger_mavg(df['Close'], window=20)
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'], window=20)
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'], window=20)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['Stochastic'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    df['Williams_%R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)

    # Manually calculate Donchian Channel
    df['Donchian_Channel_High'] = df['High'].rolling(window=20).max()
    df['Donchian_Channel_Low'] = df['Low'].rolling(window=20).min()

    # Manually calculate Parabolic SAR
    df['Parabolic_SAR'] = parabolic_sar(df['High'], df['Low'], df['Close'])

    # New indicators
    df['Keltner_Channel_High'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'], window=20)
    df['Keltner_Channel_Low'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'], window=20)

    # Fill NaNs with 0, but avoid doing this on OHLCV columns
    df = df.fillna(0)

    # Check if OHLCV columns have changed
    if not df[['Open', 'High', 'Low', 'Close', 'Volume']].equals(original_ohlcv):
        raise ValueError("OHLCV values were altered during feature engineering!")

    # Return the DataFrame with new features and original OHLCV values intact
    return df

def parabolic_sar(high, low, close, step=0.02, max_step=0.2):
    length = len(close)
    psar = close.copy()
    is_uptrend = True
    af = step  # acceleration factor
    ep = high.iloc[0]  # extreme price

    for i in range(1, length):
        if is_uptrend:
            psar.iloc[i] = psar.iloc[i - 1] + af * (ep - psar.iloc[i - 1])
            if low.iloc[i] < psar.iloc[i]:
                is_uptrend = False
                psar.iloc[i] = ep
                af = step
                ep = low.iloc[i]
        else:
            psar.iloc[i] = psar.iloc[i - 1] + af * (ep - psar.iloc[i - 1])
            if high.iloc[i] > psar.iloc[i]:
                is_uptrend = True
                psar.iloc[i] = ep
                af = step
                ep = high.iloc[i]

        if is_uptrend:
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
        else:
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)

    return psar
