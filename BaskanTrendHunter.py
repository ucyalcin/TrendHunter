import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time

# Sayfa Ayarları
st.set_page_config(page_title="BAŞKAN TREND HUNTER V36 (SNIPER)", layout="wide")

# ==========================================
# 1. AYARLAR
# ==========================================
st.sidebar.header("STRATEJİ AYARLARI")

tf_label = st.sidebar.selectbox("Zaman Dilimi", ("1 Gün", "4 Saat", "1 Saat", "15 Dakika", "5 Dakika"))

tf_map = {
    "1 Gün":    {"interval": "1d", "period": "5y", "custom_4h": False}, 
    "4 Saat":   {"interval": "1h", "period": "2y", "custom_4h": True}, 
    "1 Saat":   {"interval": "1h", "period": "1y", "custom_4h": False},
    "15 Dakika":{"interval": "15m", "period": "1mo", "custom_4h": False},
    "5 Dakika": {"interval": "5m", "period": "1mo", "custom_4h": False}
}
selected_tf = tf_map[tf_label]

st.sidebar.markdown("---")

use_ext_hours = st.sidebar.checkbox("Genişletilmiş Saatleri Dahil Et (Pre/Post Market)", value=False)
if tf_label == "4 Saat" and use_ext_hours:
    st.sidebar.warning("⚠️ 4 Saatlikte Ext. Hours önerilmez.")

use_dema_filter = st.sidebar.checkbox("Fiyat > DEMA Kuralını Kullan", value=True)
dema_len = st.sidebar.number_input("DEMA Uzunluğu", value=200, min_value=5, disabled=not use_dema_filter)

# SUPERTREND
st.sidebar.markdown("### SUPERTREND AYARLARI")
st_atr_len = st.sidebar.number_input("ATR Uzunluğu", value=12)
st_factor = st.sidebar.number_input("Faktör", value=3.0)
freshness = st.sidebar.number_input("Sinyal Tazeliği (Son kaç mum?)", min_value=1, value=20, step=1)

# LAZYBEAR IMPULSE MACD
st.sidebar.markdown("### LAZYBEAR MACD AYARLARI")
lazy_ma = st.sidebar.number_input("Length MA (Vars: 34)", value=34)
lazy_sig = st.sidebar.number_input("Length Signal (Vars: 9)", value=9)

# ADX
st.sidebar.markdown("### DİĞER")
adx_len = st.sidebar.number_input("ADX Uzunluğu", value=14, min_value=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🩻 RÖNTGEN MODU")
debug_symbol = st.sidebar.text_input("Şüpheli Sembolü Yaz (Örn: MRNA)", value="")
btn_debug = st.sidebar.button("RÖNTGENİ ÇEK")

st.sidebar.markdown("---")
use_crypto = st.sidebar.checkbox("KRİPTO (Yahoo)", value=True)
use_us = st.sidebar.checkbox("ABD BORSASI (Galaxy List)", value=True)
manual_input = st.sidebar.text_area("Manuel Semboller", placeholder="Ekstra...")
start_btn = st.sidebar.button("TARAMAYI BAŞLAT", type="primary")

# ==========================================
# 2. ÖZEL MUM MİMARI
# ==========================================
def resample_custom_us_4h(df_1h):
    if df_1h.empty: return df_1h
    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        df_1h.index = df_1h.index.tz_convert('America/New_York')
    df_1h = df_1h.between_time('09:30', '16:00')
    agg_candles = []
    for date, group in df_1h.groupby(df_1h.index.date):
        session1 = group[group.index.hour < 13] 
        if not session1.empty:
            agg_candles.append({
                'time': session1.index[0],
                'open': session1['open'].iloc[0],
                'high': session1['high'].max(),
                'low': session1['low'].min(),
                'close': session1['close'].iloc[-1], 
                'volume': session1['volume'].sum()
            })
        session2 = group[group.index.hour >= 13]
        if not session2.empty:
            agg_candles.append({
                'time': session2.index[0],
                'open': session2['open'].iloc[0],
                'high': session2['high'].max(),
                'low': session2['low'].min(),
                'close': session2['close'].iloc[-1],
                'volume': session2['volume'].sum()
            })
    if not agg_candles: return pd.DataFrame()
    df_4h = pd.DataFrame(agg_candles)
    df_4h.set_index('time', inplace=True)
    return df_4h

# ==========================================
# 3. HESAPLAMA MOTORU (LAZYBEAR ENGINE)
# ==========================================
def calculate_dema(series, length):
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return 2 * ema1 - ema2

def calculate_smma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def calculate_lazybear_macd(df, len_ma, len_sig):
    # LazyBear Matematiği
    src = (df['high'] + df['low'] + df['close']) / 3
    ma_line = calculate_smma(src, len_ma)
    signal_line = calculate_smma(ma_line, len_sig)
    hist = ma_line - signal_line
    
    df['LB_Signal'] = signal_line
    df['LB_Hist'] = hist
    return df

def calculate_adx(df, length=14):
    df = df.copy()
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift(1))
    df['tr2'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

    df['up'] = df['high'] - df['high'].shift(1)
    df['down'] = df['low'].shift(1) - df['low']

    df['plus_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
    df['minus_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)

    df['tr_smooth'] = df['tr'].ewm(alpha=1/length, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=1/length, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=1/length, adjust=False).mean()

    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].ewm(alpha=1/length, adjust=False).mean()
    return df

def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    up = hl2 - (multiplier * atr)
    dn = hl2 + (multiplier * atr)

    trend = np.ones(len(df))
    trend_up = np.zeros(len(df))
    trend_dn = np.zeros(len(df))
    
    close = df['close'].values
    up_val = up.values
    dn_val = dn.values
    
    trend_up[0] = up_val[0]
    trend_dn[0] = dn_val[0]
    
    for i in range(1, len(df)):
        if close[i-1] > trend_up[i-1]:
            trend_up[i] = max(up_val[i], trend_up[i-1])
        else:
            trend_up[i] = up_val[i]
        if close[i-1] < trend_dn[i-1]:
            trend_dn[i] = min(dn_val[i], trend_dn[i-1])
        else:
            trend_dn[i] = dn_val[i]
        if close[i] > trend_dn[i-1]:
            trend[i] = 1
        elif close[i] < trend_up[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

    df['ST_Trend'] = trend
    return df

# ==========================================
# 4. ANALİZ MOTORU (V36 - SNIPER)
# ==========================================
def analyze(df, symbol, dema_len, st_atr, st_fact, fresh, adx_len, use_dema, is_debug=False):
    if len(df) < (dema_len + 50): 
        if is_debug: st.error(f"Yetersiz Veri: {len(df)}")
        return None

    # HESAPLAMALAR
    df['DEMA'] = calculate_dema(df['close'], dema_len)
    df = calculate_supertrend(df, st_atr, st_fact)
    df = calculate_adx(df, adx_len)
    df = calculate_lazybear_macd(df, lazy_ma, lazy_sig)
    
    current = df.iloc[-1]   # T
    prev = df.iloc[-2]      # T-1
    prev2 = df.iloc[-3]     # T-2
    
    # DEBUG EKRANI
    if is_debug:
        st.write(f"### 🧬 {symbol} DETAYLI ANALİZİ (V36 - SNIPER)")
        st.write("Strateji: Signal < 0 + (Dün Yeşil Bar < 0) -> (Bugün Mavi Bar > 0)")
        
        last_20 = df.tail(20).copy()
        last_20['Zaman'] = last_20.index.strftime('%Y-%m-%d %H:%M')
        last_20['Fiyat'] = last_20['close'].round(2)
        last_20['ST'] = np.where(last_20['ST_Trend'] == 1, "🟢", "🔴")
        last_20['LB_Hist'] = last_20['LB_Hist'].round(4)
        
        # Detaylı Durum
        durum_list = []
        for i in range(len(last_20)):
            idx = last_20.index[i]
            # T anındaki veriler
            h_curr = last_20['LB_Hist'].iloc[i]
            # T-1 anındaki veriler (iloc ile güvenli erişim)
            if i > 0:
                h_prev = last_20['LB_Hist'].iloc[i-1]
                # T-2
                if i > 1:
                    h_prev2 = last_20['LB_Hist'].iloc[i-2]
                    
                    # Şartları kontrol et
                    is_crossover = (h_prev < 0) and (h_curr > 0)
                    was_green = (h_prev > h_prev2) # Dün, evvelsi günden büyük mü? (Rising Negative)
                    
                    if is_crossover and was_green:
                        durum_list.append("🎯 TAM İSABET")
                    elif is_crossover:
                        durum_list.append("⚡ Normal Kırılım")
                    else:
                        durum_list.append("-")
                else:
                    durum_list.append("-")
            else:
                durum_list.append("-")
                
        last_20['Durum'] = durum_list
        st.dataframe(last_20[['Zaman', 'Fiyat', 'ST', 'LB_Hist', 'Durum']], use_container_width=True)

    # --- SNIPER FİLTRELERİ ---
    
    # 1. SuperTrend BUY mu?
    if current['ST_Trend'] != 1: return None
    
    # 2. LazyBear Signal < 0 (Dipte miyiz?)
    if current['LB_Signal'] >= 0: return None
    
    # 3. HİSTOGRAM KIRILIMI (Dün Eksi, Bugün Artı)
    hist_crossover = (prev['LB_Hist'] < 0) and (current['LB_Hist'] > 0)
    if not hist_crossover: return None
    
    # 4. HAZIRLIK ŞARTI: "AŞAĞIDA YEŞİL BARLAR UZUYOR MU?"
    # Bu şart: Bir önceki mumun histogramı (ki negatifti), ondan önceki mumdan büyük olmalı.
    # Bu, negatif bölgede YÜKSELİŞ (YEŞİL BAR) olduğunu kanıtlar.
    was_green_foundation = prev['LB_Hist'] > prev2['LB_Hist']
    
    if not was_green_foundation: return None

    # 5. Tazelik
    lookback = int(fresh) + 1
    recent_trends = df['ST_Trend'].tail(lookback).values
    if -1 not in recent_trends: return None 
        
    candles_ago = 0
    found_signal = False
    for i in range(len(recent_trends)-1, 0, -1):
        if recent_trends[i] == 1 and recent_trends[i-1] == -1:
            candles_ago = (len(recent_trends) - 1) - i
            found_signal = True
            break
            
    if not found_signal: return None 

    signal_candle_idx = -(1 + candles_ago)
    signal_candle = df.iloc[signal_candle_idx]
    
    if use_dema:
        if signal_candle['close'] <= signal_candle['DEMA']:
            return None 

    return {
        "Enstrüman": symbol,
        "Fiyat": round(current['close'], 2),
        "Sinyal": f"🔥 {candles_ago} Mum Önce",
        "ADX": round(current['adx'], 2),
        "LB Hist": round(current['LB_Hist'], 4),
        "Setup": "Yeşil(Dip) -> Mavi(Kırılım)",
        "Durum": "🎯 TAM İSABET"
    }

# ==========================================
# 5. VERİ ÇEKME & LİSTELER
# ==========================================
def get_crypto_yahoo():
    return [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", 
        "AVAX-USD", "TRX-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "SHIB-USD",
        "BCH-USD", "ATOM-USD", "XLM-USD", "UNI7083-USD", "NEAR-USD", "APT21794-USD"
    ]

def get_us_universe():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "AVGO", "NFLX",
        "INTC", "QCOM", "CSCO", "DELL", "APP", "TSM", "BIDU", "BABA", "PLTR", "CRWD",
        "RBRK", "LSCC", "BBAI", "ZM", "ZS", "ZETA", "CLS", "PENG", "SOXL",
        "ADBE", "CRM", "NOW", "ORCL", "IBM", "INTU", "UBER", "ABNB", "BKNG", "PANW",
        "FTNT", "SNOW", "SQ", "SHOP", "U", "ROKU", "DKNG", "HOOD", "PYPL", "MU", "TXN",
        "LRCX", "ADI", "KLAC", "ARM", "SMCI", "SNDK", "AMAT", "ON", "MCHP", "CDNS", "SNPS",
        "DDOG", "NET", "MDB", "TEAM", "TTWO", "EA", "PDD", "JD", "OKTA",
        "JPM", "V", "MA", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "SCHW", "USB",
        "TRV", "AIG", "SPGI", "COIN", "MSTR", "BRK-B", "PGR", "CB", "CME", "ICE", "COF", "SYF",
        "BA", "GE", "F", "GM", "CAT", "DE", "HON", "UNP", "UPS", "FDX", "LMT", "RTX",
        "NOC", "GD", "EMR", "MMM", "ETN", "VZ", "T", "TMUS", "CMCSA", "ADP", "CSX", "NSC",
        "WM", "RSG", "RIVN", "LCID",
        "JNJ", "PFE", "MRNA", "REGN", "LLY", "UNH", "ABBV", "AMGN", "BMY", "GILD", "ISRG",
        "SYK", "CVS", "TMO", "DHR", "VRTX", "MOH", "MDT", "BSX", "ZTS", "CI", "HUM",
        "WMT", "COST", "PG", "KO", "PEP", "XOM", "CVX", "DIS", "MCD", "NKE", "SBUX",
        "TGT", "LOW", "HD", "TJX", "LULU", "MDLZ", "PM", "MO", "CL", "KMB", "EL",
        "CMG", "MAR", "KHC", "HSY", "KR",
        "OXY", "SLB", "HAL", "COP", "EOG", "FCX", "NEM", "LIN", "DOW", "SHW", "NEE",
        "DUK", "SO", "MPC", "APD", "ECL", "NUE", "PLD", "AMT", "CCI", "EQIX", "PSA",
        "NVDX", "AAPU", "GGLL", "AMZZ", "METU", "AMZP", "MARA", "QQQT",
        "O", "AGNC", "ORC", "SPHD", "DX", "OXLC", "GLAD", "GAIN", "GOOD", "LAND", "SRET",
        "QYLD", "XYLD", "SDIV", "DIV", "RYLD", "JEPI", "JEPQ", "EFC", "SCM", "PSEC",
        "QQQY", "APLE", "MAIN", "WSR", "ARR", "SBR", "GROW", "HRZN", "LTC", "PNNT",
        "SLG", "ARCC", "HTGC", "SPG", "NLY", "ETV", "PDI", "ARE", "FRT", "SPYI", "WPC",
        "ECC", "OMAH", "QQQI", "ABR", "IIPR", "CIM", "VNM", "RIET", "DLR", "VICI", "OXSQ",
        "OMCL", "POWL", "DXPE", "TLN", "RH", "TOST", "NU", "MOS", "AES", "ASRT", "WRD",
        "CRS", "LUV", "ALL", "AYI", "APTV", "BIIB", "FTI", "VERU", "AZO", "CEG", "NVO",
        "MRK",
        "AGQ", "UGL", "LIT", "QQQ", "TQQQ", "UAMY", "WEAT", "GOOP", "QLD", "YINN",
        "IGM", "SPY", "PFIX", "TLT", "TLTW", "BIL", "VOO", "VTI", "BND", "VYM", "SCHD"
    ]

def fetch_and_analyze(symbol, tf_conf, use_ext, is_debug=False):
    try:
        target_interval = "1h" if tf_conf['custom_4h'] else tf_conf['interval']
        
        df = yf.download(
            symbol, 
            period=tf_conf['period'], 
            interval=target_interval, 
            prepost=use_ext, 
            auto_adjust=False, 
            progress=False
        )
        
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.rename(columns=lambda x: x.lower(), inplace=True)
        
        if tf_conf['custom_4h']:
            is_crypto = symbol.endswith("-USD")
            if is_crypto:
                agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                df = df.resample('4h').agg(agg_dict).dropna()
            else:
                if use_ext:
                    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    df = df.resample('4h').agg(agg_dict).dropna()
                else:
                    df = resample_custom_us_4h(df)
                    if df.empty: return None

        return analyze(df, symbol, dema_len, st_atr_len, st_factor, freshness, adx_len, use_dema_filter, is_debug)
    except Exception as e:
        if is_debug: st.error(f"Hata: {e}")
        return None

# ==========================================
# 6. ARAYÜZ
# ==========================================
st.title("🚀 BAŞKAN TREND HUNTER V36 (SNIPER)")

if btn_debug and debug_symbol:
    st.info(f"🔍 {debug_symbol} Röntgen Çekiliyor...")
    fetch_and_analyze(debug_symbol.strip().upper(), selected_tf, use_ext_hours, is_debug=True)

if start_btn:
    results = []
    tasks = []
    
    if use_crypto: tasks.extend(get_crypto_yahoo())
    if use_us: tasks.extend(get_us_universe())
    
    if manual_input: 
        for m in manual_input.split(','): tasks.append(m.strip())

    bar = st.progress(0)
    status = st.empty()
    
    for i, sym in enumerate(tasks):
        bar.progress((i+1)/len(tasks))
        status.text(f"Analiz: {sym}")
        res = fetch_and_analyze(sym, selected_tf, use_ext_hours)
        if res: results.append(res)

    bar.empty()
    if results:
        st.success(f"Bitti! {len(results)} Sinyal Bulundu.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("Sonuç Yok.")
