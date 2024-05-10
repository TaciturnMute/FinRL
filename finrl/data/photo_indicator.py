import pandas as pd


def calculate_dmi(df, window=14):

    """计算DMI指标，并将其添加到dataframe中"""

    # 计算 TR
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # 计算 +DM 和 -DM
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    # 选择最大的DM
    cond = plus_dm > minus_dm
    plus_dm[~cond] = 0
    minus_dm[cond] = 0

    # 计算平滑的 TR, +DM, -DM
    atr = tr.rolling(window=window).mean()
    smooth_plus_dm = plus_dm.rolling(window=window).mean()
    smooth_minus_dm = minus_dm.rolling(window=window).mean()

    # 计算 +DI 和 -DI
    plus_di = 100 * smooth_plus_dm / atr
    minus_di = 100 * smooth_minus_dm / atr

    # 计算 DX 和 ADX
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = dx.rolling(window=window).mean()

    # 添加到 DataFrame
    df['+DI'] = plus_di
    df['-DI'] = minus_di
    df['ADX'] = adx

    return df


def stochastics(dataframe, low, high, close, k, d ):
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal 
    When the %K crosses below %D, sell signal
    """

    df = dataframe.copy()

    # Set minimum low and maximum high of the k stoch
    low_min  = df[low].rolling( window = k ).min()
    high_max = df[high].rolling( window = k ).max()

    # Fast Stochastic
    df['k_fast'] = 100 * (df[close] - low_min)/(high_max - low_min)
    df['k_fast'].ffill(inplace=True)
    df['d_fast'] = df['k_fast'].rolling(window = d).mean()

    # Slow Stochastic
    df['k_slow'] = df["d_fast"]
    df['d_slow'] = df['k_slow'].rolling(window = d).mean()

    return df