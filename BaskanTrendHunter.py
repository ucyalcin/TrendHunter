import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Sayfa Ayarları
st.set_page_config(page_title="BAŞKAN TREND HUNTER V17", layout="wide")

# ==========================================
# 1. AYARLAR
# ==========================================
st.sidebar.header("STRATEJİ AYARLARI")

tf_label = st.sidebar.selectbox("Zaman Dilimi", ("1 Gün", "4 Saat", "1 Saat", "15 Dakika", "5 Dakika"))

# tf_map (Resampling mantığı Extended Hours ile uyumlu hale getirildi)
tf_map = {
    "1 Gün":    {"ccxt": "1d", "yf": "1d", "yf_per": "2y", "resample": False}, 
    "4 Saat":   {"ccxt": "4h", "yf": "1h", "yf_per": "1y", "resample": True}, 
    "1 Saat":   {"ccxt": "1h", "yf": "1h", "yf_per": "6mo", "resample": False},
    "15 Dakika":{"ccxt": "15m", "yf": "15m", "yf_per": "1mo", "resample": False},
    "5 Dakika": {"ccxt": "5m", "yf": "5m", "yf_per": "1mo", "resample": False}
}
selected_tf = tf_map[tf_label]

st.sidebar.markdown("---")

# --- EXTENDED HOURS SEÇENEĞİ (YENİ) ---
use_ext_hours = st.sidebar.checkbox("Genişletilmiş Saatleri (Pre/Post Market) Dahil Et", value=False, help="İşaretlersen TradingView'daki 'EXT' modu gibi gece verilerini de dahil eder. Mum sayıları artar.")

use_dema_filter = st.sidebar.checkbox("Fiyat > DEMA Kuralını Kullan", value=True)
dema_len = st.sidebar.number_input("DEMA Uzunluğu", value=200, min_value=5, disabled=not use_dema_filter)

st_atr_len = st.sidebar.number_input("SuperTrend ATR", value=12)
st_factor = st.sidebar.number_input("SuperTrend Faktör", value=3.0)
freshness = st.sidebar.number_input("Sinyal Tazeliği (Son kaç mum?)", min_value=1, value=20, step=1)

st.sidebar.markdown("---")

st.sidebar.subheader("🩻 RÖNTGEN MODU")
debug_symbol = st.sidebar.text_input("Şüpheli Sembolü Yaz (Örn: WMT)", value="")
btn_debug = st.sidebar.button("RÖNTGENİ ÇEK")

st.sidebar.markdown("---")
use_crypto = st.sidebar.checkbox("KRİPTO (Binance)", value=True)
use_us = st.sidebar.checkbox("ABD BORSASI (Grandmaster List)", value=True)
manual_input = st.sidebar.text_area("Manuel Semboller", placeholder="Ekstra...")
start_btn = st.sidebar.button("GENEL TARAMAYI BAŞLAT", type="primary")

# ==========================================
# 2. HESAPLAMA MOTORU
# ==========================================
def calculate_dema(series, length):
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return 2 * ema1 - ema2

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
    df['ST_Value'] = np.where(trend == 1, trend_dn, trend_up)
    return df

# ==========================================
# 3. ANALİZ
# ==========================================
def analyze(df, symbol, dema_len, st_atr, st_fact, fresh, use_dema, is_debug=False):
    if len(df) < (dema_len + 50): 
        if is_debug: st.error(f"Yetersiz Veri: {len(df)}")
        return None

    df['DEMA'] = calculate_dema(df['close'], dema_len)
    df = calculate_supertrend(df, st_atr, st_fact)
    
    if is_debug:
        st.write(f"### 🧬 {symbol} DETAYLI ANALİZİ")
        last_20 = df.tail(20).copy()
        last_20['Zaman'] = last_20.index
        # Zamanı daha okunur yapalım (US saatine göre)
        last_20['Zaman_Str'] = last_20.index.strftime('%Y-%m-%d %H:%M')
        last_20['Fiyat'] = last_20['close'].round(2)
        last_20['DEMA'] = last_20['DEMA'].round(2)
        last_20['Trend'] = np.where(last_20['ST_Trend'] == 1, "🟢 BUY", "🔴 SELL")
        
        st.dataframe(last_20[['Zaman_Str', 'Fiyat', 'DEMA', 'Trend']], use_container_width=True)
        
        curr = df.iloc[-1]
        st.write(f"**Anlık:** {curr['close']:.2f}")

    current = df.iloc[-1]
    
    if use_dema and current['close'] <= current['DEMA']: return None
    if current['ST_Trend'] != 1: return None
    
    lookback = int(fresh) + 1
    recent = df['ST_Trend'].tail(lookback).values
    if -1 not in recent: return None 
        
    candles_ago = 0
    for t in reversed(recent):
        if t == 1: candles_ago += 1
        else: break
        
    return {
        "Enstrüman": symbol,
        "Fiyat": round(current['close'], 2),
        "DEMA": round(current['DEMA'], 2),
        "Sinyal": f"🔥 {candles_ago} Mum Önce",
        "Durum": "YENİ TREND"
    }

# ==========================================
# 4. VERİ ÇEKME & LİSTELER
# ==========================================
def get_crypto():
    try:
        x = ccxt.binance()
        m = x.load_markets()
        return [s for s in m if s.endswith('/USDT') and 'UP' not in s and 'DOWN' not in s]
    except: return []

def get_us():
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AMD", "AVGO", "NFLX",
        "INTC", "QCOM", "CSCO", "DELL", "APP", "TSM", "BIDU", "BABA", "PLTR", "CRWD",
        "RBRK", "LSCC", "BBAI", "ZM", "ZS", "ZETA", "CLS", "PENG", "SOXL",
        "NVDX", "AAPU", "GGLL", "AMZZ", "METU", "AMZP", "MSTR", "COIN", "MARA", "QQQT",
        "O", "AGNC", "ORC", "SPHD", "DX", "OXLC", "GLAD", "GAIN", "GOOD", "LAND", "SRET",
        "QYLD", "XYLD", "SDIV", "DIV", "RYLD", "JEPI", "JEPQ", "EFC", "SCM", "PSEC",
        "QQQY", "APLE", "MAIN", "WSR", "ARR", "SBR", "GROW", "HRZN", "LTC", "PNNT",
        "SLG", "ARCC", "HTGC", "SPG", "NLY", "ETV", "PDI", "ARE", "FRT", "SPYI", "WPC",
        "ECC", "OMAH", "QQQI", "ABR", "IIPR", "CIM", "VNM", "RIET", "DLR", "VICI", "OXSQ",
        "JPM", "V", "JNJ", "WMT", "PG", "XOM", "KO", "DIS", "CVX", "PFE", "BA", "GE", "F",
        "UBER", "PEP", "COST", "LULU", "MRNA", "REGN", "VZ", "MO", "OMCL", "POWL", "DXPE",
        "TLN", "RH", "TOST", "NU", "MOS", "AES", "OXY", "ASRT", "WRD", "CRS", "LUV",
        "ALL", "AYI", "APTV", "BIIB", "FTI", "VERU", "AZO", "HD", "EL", "CEG", "UPS",
        "NVO", "MRK", "MOH",
        "AGQ", "UGL", "LIT", "QQQ", "TQQQ", "UAMY", "WEAT", "GOOP", "QLD", "YINN",
        "IGM", "SPY", "PFIX", "TLT", "TLTW", "BIL", "VOO", "VTI", "BND", "VYM", "SCHD"
    ]

def convert_4h(df):
    # Resampling yaparken Pre-Market verisi varsa onları da akıllıca dahil etmesi lazım
    # Ancak 4H periyodunda Pre-Market genelde gürültüdür. Yine de kullanıcı isterse dahil edilir.
    return df.resample('4h', offset='30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

def fetch_and_analyze(symbol, typ, tf_conf, use_ext, is_debug=False):
    try:
        df = pd.DataFrame()
        if typ == 'CRYPTO':
            # Kriptoda Extended Hours diye bir şey yok (7/24)
            x = ccxt.binance()
            limit = 1500
            ohlcv = x.fetch_ohlcv(symbol, tf_conf['ccxt'], limit=limit)
            df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
        else:
            # prepost=use_ext BURASI SİHİRLİ NOKTA
            df = yf.download(symbol, period=tf_conf['yf_per'], interval=tf_conf['yf'], prepost=use_ext, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.rename(columns=lambda x: x.lower(), inplace=True)
            if tf_conf['resample']: df = convert_4h(df)
            
        if df.empty: return None
        return analyze(df, symbol, dema_len, st_atr_len, st_factor, freshness, use_dema_filter, is_debug)
    except Exception as e:
        if is_debug: st.error(f"Hata: {e}")
        return None

# ==========================================
# 5. ARAYÜZ
# ==========================================
st.title("🚀 BAŞKAN TREND HUNTER V17 (EXTENDED)")

if btn_debug and debug_symbol:
    st.info(f"🔍 {debug_symbol} Röntgen Çekiliyor... (Extended Hours: {use_ext_hours})")
    typ = 'CRYPTO' if '/' in debug_symbol else 'US'
    fetch_and_analyze(debug_symbol.strip().upper(), typ, selected_tf, use_ext_hours, is_debug=True)

if start_btn:
    results = []
    tasks = []
    
    if use_crypto: tasks.extend([(s, 'CRYPTO') for s in get_crypto()])
    if use_us: tasks.extend([(s, 'US') for s in get_us()])
    if manual_input: 
        for m in manual_input.split(','): tasks.append((m.strip(), 'CRYPTO' if '/' in m else 'US'))

    bar = st.progress(0)
    status = st.empty()
    
    for i, (sym, typ) in enumerate(tasks):
        bar.progress((i+1)/len(tasks))
        status.text(f"Analiz: {sym}")
        res = fetch_and_analyze(sym, typ, selected_tf, use_ext_hours)
        if res: results.append(res)

    bar.empty()
    if results:
        st.success(f"Bitti! {len(results)} Sinyal Bulundu.")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("Sonuç Yok. Tazelik ayarını artırmayı veya 'Genişletilmiş Saatleri Dahil Et' seçeneğini değiştirmeyi dene.")
