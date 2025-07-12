import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import asyncio
import threading
from websockets import WebSocketException, ConnectionClosed, InvalidStatusCode
from websocket_service import WebSocketService, get_websocket_service
import logging
import requests
import os
from typing import Optional, Dict, Any, List
import uuid
import time
from functools import lru_cache
import hashlib
import secrets

st.set_page_config(page_title="Moneyverse AI Trading", layout="wide")

# Production configuration
class Config:
    """Production configuration for the Streamlit app."""
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
    WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://localhost:8000")
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
    MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "300"))  # 5 minutes
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "30"))
    MAX_DISPLAY_SIGNALS = int(os.getenv("MAX_DISPLAY_SIGNALS", "100"))
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    SUPPORTED_PAIRS = [
        "XAU/USD", "GBP/USD", "GBP/JPY", "EUR/USD", "USD/JPY", "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "BNB/USD", "DOGE/USD", "XRP/USD", "LTC/USD", "DOT/USD", "AVAX/USD", "SHIB/USD", "MATIC/USD", "TRX/USD", "LINK/USD", "UNI/USD", "BCH/USD", "EOS/USD", "ATOM/USD", "XMR/USD", "ETC/USD", "FIL/USD", "XTZ/USD", "AAVE/USD", "NEO/USD", "USD/CHF", "USD/CAD", "AUD/USD", "NZD/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/CAD", "EUR/AUD", "EUR/NZD", "GBP/CHF", "GBP/AUD", "GBP/NZD", "AUD/JPY", "AUD/NZD", "AUD/CAD", "AUD/CHF", "NZD/JPY", "NZD/CAD", "NZD/CHF", "CAD/JPY", "CHF/JPY"
    ]
    SUPPORTED_TIMEFRAMES = ["1M", "5M", "15M", "30M", "45M", "1H", "2H", "4H", "8H", "1D", "1W", "1MN"]

# Configure logging based on environment
if Config.DEBUG:
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# logger = logging.getLogger(__name__)
# Use st.write for debugging if needed
if Config.DEBUG:
  st.write(f"Environment: {Config.ENVIRONMENT}")
  st.write(f"API Base URL: {Config.API_BASE_URL}")
  st.write(f"WebSocket URL: {Config.WS_BASE_URL}/ws/signals")

# Security utilities
class SecurityManager:
    """Manages security-related functionality."""
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    @staticmethod
    def generate_session_token() -> str:
        return secrets.token_urlsafe(32)
    @staticmethod
    def validate_session(session_data: Dict) -> bool:
        if not session_data:
            return False
        last_activity = session_data.get("last_activity", 0)
        if time.time() - last_activity > Config.SESSION_TIMEOUT_MINUTES * 60:
            return False
        return True

# Rate limiting
class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self):
        self.requests = {}
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        if key not in self.requests:
            self.requests[key] = []
        self.requests[key] = [req_time for req_time in self.requests[key] if now - req_time < Config.RATE_LIMIT_WINDOW]
        if len(self.requests[key]) < Config.MAX_LOGIN_ATTEMPTS:
            self.requests[key].append(now)
            return True
        return False

rate_limiter = RateLimiter()

# --- SESSION STATE INIT ---
def init_session_state():
    defaults = {
        "signals": [],
        "last_update": None,
        "selected_pair": "BTC/USD",
        "selected_timeframe": "1h",
        "ws_connected": False,
        "license_code": "",
        "auth_token": None,
        "ws": None,
        "historical_data": {},
        "market_data": {},
        "trends": {},
        "active_signals": [],
        "completed_signals": [],
        "license_verified": False,
        "user_email": "",
        "user_full_name": None,
        "license_error": "",
        "session_token": None,
        "last_activity": time.time(),
        "login_attempts": 0,
        "error_count": 0,
        "cache": {},
        "cache_timestamps": {},
        "available_pairs": ["BTC/USD", "ETH/USD", "XAU/USD", "EUR/USD", "USD/JPY"]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    # Ensure websocket service is initialized
    get_websocket_service()

init_session_state()

# Cache management
@lru_cache(maxsize=128)
def get_cached_data(key: str, ttl: int = Config.CACHE_TTL) -> Optional[Any]:
    if key in st.session_state.get("cache", {}):
        timestamp = st.session_state.get("cache_timestamps", {}).get(key, 0)
        if time.time() - timestamp < ttl:
            return st.session_state["cache"][key]
    return None

def set_cached_data(key: str, data: Any):
    if "cache" not in st.session_state:
        st.session_state["cache"] = {}
    if "cache_timestamps" not in st.session_state:
        st.session_state["cache_timestamps"] = {}
    st.session_state["cache"][key] = data
    st.session_state["cache_timestamps"][key] = time.time()

# --- UTILITY FUNCTIONS ---
def format_signal(signal):
    try:
      return {
        "Pair": signal.get("pair", "N/A"),
        "Type": signal.get("type", "N/A"),
            "Direction": signal.get("direction", "N/A"),
            "Entry": signal.get("entry", signal.get("entry_price", "N/A")),
            "TP1": signal.get("tp1", "N/A"),
            "TP2": signal.get("tp2", "N/A"),
            "SL": signal.get("sl", signal.get("stop_loss", "N/A")),
        "Status": signal.get("status", "N/A"),
            "Confidence": signal.get("confidence", "N/A"),
            "Generated Time": signal.get("generated_time", signal.get("timestamp", "N/A")),
            "Reason": signal.get("reason", "N/A"),
            "Last Updated": signal.get("last_updated", "N/A"),
            "Label": signal.get("label", ""),
            "Logic Note": signal.get("logic_note", "")
        }
    except Exception as e:
       st.write(f"Error formatting signal: {e}")
    return {k: "Error" for k in [
            "Pair", "Type", "Direction", "Entry", "TP1", "TP2", "SL", "Status", "Confidence", "Generated Time", "Reason", "Last Updated", "Label", "Logic Note"
        ]}

def create_price_chart(signals, historical_data=None):
    """Create a price chart using Plotly with error handling."""
    try:
        fig = go.Figure()
        if not (historical_data and "candles" in historical_data and historical_data["candles"]):
            return fig
        df_hist = pd.DataFrame(historical_data["candles"])
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], utc=True)
        df_hist = df_hist.sort_values('timestamp').set_index('timestamp')
        fig.add_trace(go.Candlestick(
            x=df_hist.index,
            open=df_hist["open"],
            high=df_hist["high"],
            low=df_hist["low"],
            close=df_hist["close"],
            name="Historical"
        ))
        if signals:
            df_signals = pd.DataFrame(signals)
            # Robustly handle both 'created_at' and 'generated_time' as the signal timestamp
            if 'created_at' in df_signals:
                df_signals['time'] = pd.to_datetime(df_signals['created_at'], utc=True, errors='coerce')
            elif 'generated_time' in df_signals:
                df_signals['time'] = pd.to_datetime(df_signals['generated_time'], utc=True, errors='coerce')
            else:
                df_signals['time'] = pd.NaT
            df_signals['type'] = df_signals.get('direction', None)
            def get_marker_price(signal_row):
                try:
                    loc = df_hist.index.searchsorted(signal_row['time'], side='right')
                    if loc == 0:
                        return None
                    candle = df_hist.iloc[loc - 1]
                    if signal_row['type'] == 'BUY':
                        return candle['low'] * 0.998
                    elif signal_row['type'] == 'SELL':
                        return candle['high'] * 1.002
                except (KeyError, IndexError):
                    return None
                return None
            df_signals['marker_price'] = df_signals.apply(get_marker_price, axis=1)
            df_signals.dropna(subset=['marker_price'], inplace=True)
            buy_signals = df_signals[df_signals['type'] == 'BUY']
            sell_signals = df_signals[df_signals['type'] == 'SELL']
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                        x=buy_signals['time'], y=buy_signals['marker_price'],
                        mode='markers', name='Buy Signal',
                        marker=dict(color='lime', size=10, symbol='triangle-up')
            ))
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                        x=sell_signals['time'], y=sell_signals['marker_price'],
                        mode='markers', name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            fig.update_layout(
            title='Price and Signals',
            xaxis_title='Time', yaxis_title='Price',
            template='plotly_dark', height=500,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return fig
    except Exception as e:
        st.error(f"Error creating price chart: {e}")
        return None

# --- Timeframe Mapping ---
def map_timeframe_to_backend(tf: str) -> str:
    mapping = {
        "1m": "1M",
        "5m": "5M",
        "15m": "15M",
        "30m": "30M",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D"
    }
    return mapping.get(tf, tf)

# --- API FUNCTIONS WITH RETRY LOGIC ---
async def make_api_request(url: str, method: str = "GET", **kwargs) -> Optional[Dict]:
    headers = kwargs.get("headers", {})
    if st.session_state.get("auth_token"):
        headers["Authorization"] = f"Bearer {st.session_state['auth_token']}"
    kwargs["headers"] = headers
    kwargs["timeout"] = Config.REQUEST_TIMEOUT
    for attempt in range(Config.MAX_RETRIES):
        try:
            if method.upper() == "GET":
                response = requests.get(url, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            st.write(f"Request timeout (attempt {attempt + 1}/{Config.MAX_RETRIES})")
            if attempt == Config.MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            st.write(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response:
                if e.response.status_code == 401:
                    st.session_state["auth_token"] = None
                    st.error("Authentication failed. Please log in again.")
                    st.rerun()
                elif e.response.status_code == 429:
                    st.warning("Rate limit exceeded. Please wait before trying again.")
                    return None
            raise

async def fetch_historical_data(pair: str, timeframe: str, limit: int = 100) -> Optional[Dict[str, Any]]:
    """Fetch historical data for a given pair and timeframe."""
    if not st.session_state.get("auth_token"):
        return None
    try:
        base, quote = pair.split('/')
        timeframe_mapped = map_timeframe_to_backend(timeframe)
        url = f"{Config.API_BASE_URL}/market-data/{base}/{quote}/historical?timeframe={timeframe_mapped}&limit={limit}"
        headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"}
        data = await make_api_request(url, headers=headers)
        if data and data.get("error"):
            st.error(f"Historical data error: {data['error']}")
        return data
    except ValueError:
        st.error(f"Invalid pair format: {pair}. Expected format is BASE/QUOTE.")
        return None
    except Exception as e:
        st.error(f"Failed to fetch historical data: {e}")
        return None

async def fetch_market_data(pair: str) -> Optional[Dict[str, Any]]:
    cache_key = f"market_{pair}"
    cached_data = get_cached_data(cache_key, ttl=60)
    if cached_data:
        return cached_data
    try:
        st.write(f"Fetching market data for {pair}")
        data = await make_api_request(f"{Config.API_BASE_URL}/market-data/{pair}")
        if data:
            set_cached_data(cache_key, data)
        st.write(f"Successfully fetched market data for {pair}")
        if data and data.get("error"):
            st.error(f"Market data error: {data['error']}")
        return data
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        return None

async def fetch_trend_analysis(pair: str, timeframe: str) -> Optional[Dict[str, Any]]:
    cache_key = f"trend_{pair}_{timeframe}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        return cached_data
    try:
        st.write(f"Fetching trend analysis for {pair} on {timeframe} timeframe")
        # Split pair into base and quote for backend compatibility
        base, quote = pair.split('/')
        data = await make_api_request(
            f"{Config.API_BASE_URL}/trend-analysis/{base}/{quote}",
            params={"timeframe": timeframe.lower()}
        )
        if data:
            set_cached_data(cache_key, data)
        st.write(f"Successfully fetched trend analysis for {pair}")
        return data
    except Exception as e:
        st.write(f"Error fetching trend analysis: {str(e)}")
        st.error("Failed to fetch trend analysis. Please try again later.")
        return None

async def generate_signal(pair: str, signal_type: str, timeframe: Optional[str] = None) -> Dict:
    try:
        backend_timeframe = map_timeframe_to_backend(timeframe)
        data = await make_api_request(
            f"{Config.API_BASE_URL}/signals/generate",
            method="POST",
            json={
                "pair": pair,
                "type": signal_type,
                "timeframe": backend_timeframe
            }
        )
        return data or {}
    except Exception as e:
        st.write(f"Error generating signal: {e}")
        st.error("Failed to generate signal. Please try again later.")
        return {}

async def get_active_signals() -> List[Dict]:
    try:
        data = await make_api_request(f"{Config.API_BASE_URL}/signals/active")
        return data or []
    except Exception as e:
        st.write(f"Error fetching active signals: {e}")
        st.error("Failed to fetch active signals. Please try again later.")
        return []

async def get_available_pairs() -> List[str]:
    try:
        data = await make_api_request(f"{Config.API_BASE_URL}/market/pairs")
        return data or []
    except Exception as e:
        st.write(f"Error fetching available pairs: {e}")
        st.error("Failed to fetch available pairs. Please try again later.")
        return []

async def register_user(email: str, password: str, full_name: str) -> Dict:
    if not rate_limiter.is_allowed(f"register_{email}"):
        st.error("Too many registration attempts. Please wait before trying again.")
        return {"error": "Rate limit exceeded"}
    try:
        data = await make_api_request(
            f"{Config.API_BASE_URL}/auth/register",
            method="POST",
            json={"email": email, "password": password, "full_name": full_name}
        )
        return data or {"error": "Registration failed"}
    except Exception as e:
        st.write(f"Error registering user: {e}")
        st.error("Failed to register user. Please try again later.")
        return {"error": str(e)}

async def login_user(email: str, password: str) -> Dict:
    if not rate_limiter.is_allowed(f"login_{email}"):
        st.error("Too many login attempts. Please wait before trying again.")
        return {"error": "Rate limit exceeded"}
    try:
        data = await make_api_request(
            f"{Config.API_BASE_URL}/auth/login",
            method="POST",
            data={"username": email, "password": password}
        )
        return data or {"error": "Login failed"}
    except Exception as e:
        st.write(f"Login error: {e}")
        return {"error": str(e)}

async def create_license(user_email: str, expiry_days: int, features: List[str]) -> Dict:
    try:
        data = await make_api_request(
            f"{Config.API_BASE_URL}/licenses",
            method="POST",
            json={"user_email": user_email, "expiry_days": expiry_days, "features": features}
        )
        return data or {"error": "License creation failed"}
    except Exception as e:
        st.write(f"Error creating license: {e}")
        st.error("Failed to create license. Please try again later.")
        return {"error": str(e)}

async def validate_license(license_code: str) -> Dict:
    try:
        data = await make_api_request(
            f"{Config.API_BASE_URL}/licenses/validate",
            method="POST",
            params={"license_code": license_code}
        )
        if data and "error" not in data:
            st.session_state["license_error"] = ""
            return data
        else:
            error_msg = data.get("error", data.get("detail", "License validation failed")) if data else "License validation failed"
            st.error(f"License validation failed: {error_msg}")
            return {"error": error_msg}
    except Exception as e:
        st.session_state["license_error"] = "Failed to validate license code. Please try again later."
        st.error(f"License validation error: {e}")
        return {"error": f"License validation error: {str(e)}"}

async def get_active_licenses() -> List[Dict]:
    try:
        data = await make_api_request(f"{Config.API_BASE_URL}/licenses/active")
        return data or []
    except Exception as e:
        st.write(f"Error getting active licenses: {e}")
        return []

# --- WEBSOCKET FUNCTIONS ---
def send_start_monitoring(pair, timeframe, license_code):
    """Sends a message to the backend to start monitoring a pair/timeframe."""
    ws_service = get_websocket_service()
    if ws_service.is_connected:
        ws_service.send_message({
            'type': 'start_monitoring',
            'pair': pair,
            'timeframe': timeframe
            })
        st.toast(f"Switched to {pair} on {timeframe}")
    else:
        st.error("WebSocket not connected. Cannot change monitoring.")

# --- PAGE NAVIGATION ---
def show_dashboard_page():
    show_dashboard()

def show_signals_page():
    current_language = st.session_state.get("language", "en")
    st.header(get_text("all_signals", current_language))
    signals = asyncio.run(get_active_signals())
    st.dataframe(pd.DataFrame(signals))
    st.write("Raw signals:", signals)

def show_market_data_page():
    st.header("Market Data")
    pair = st.selectbox("Select Pair", Config.SUPPORTED_PAIRS)
    data = asyncio.run(fetch_market_data(pair))
    hist = asyncio.run(fetch_historical_data(pair, "1h")) if pair and st.session_state.auth_token else None
    data_sources = []
    if data and data.get("data_source"):
        data_sources.append(data["data_source"])
    if hist and hist.get("data_source"):
        data_sources.append(hist["data_source"])
    show_demo_mode_banner(data_sources)
    st.write("Market Data:", data)
    if data and data.get("last_updated"):
        st.caption(f"Last updated: {data['last_updated']}")
    if data and data.get("data_source"):
        if data["data_source"] == "unavailable" or data.get("error"):
            st.error("Data unavailable. Please try again later or check your data provider/API key.")
    else:
            st.info(f"Source: {data['data_source'].capitalize()}")
    if hist:
        st.write("Historical Data:", hist)
        if hist.get("last_updated"):
            st.caption(f"Historical data last updated: {hist['last_updated']}")
        if hist.get("data_source"):
            if hist["data_source"] == "unavailable" or hist.get("error"):
                st.error("Historical data unavailable. Please try again later or check your data provider/API key.")
            else:
                st.info(f"Historical data source: {hist['data_source'].capitalize()}")

def show_licenses_page():
    st.header("Licenses")
    license_code = st.text_input("Validate License Code", key="validate_license_input")
    if st.button("Validate License"):
        result = asyncio.run(validate_license(license_code))
        st.json(result)
    licenses = asyncio.run(get_active_licenses())
    st.write("Active Licenses:", licenses)

def show_profile_page():
    st.header(get_text("profile", st.session_state.get("language", "en")))
    st.write(get_text("email", st.session_state.get("language", "en")), st.session_state.get("user_email", ""))
    # --- Preferences Section ---
    st.subheader(get_text("preferences", st.session_state.get("language", "en")))
    # Load preferences from backend
    if "user_preferences" not in st.session_state:
        try:
            url = f"{Config.API_BASE_URL}/auth/me"
            headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"} if st.session_state.get("auth_token") else {}
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                prefs = resp.json().get("preferences", {}) or {}
                st.session_state["user_preferences"] = prefs
            else:
                st.session_state["user_preferences"] = {}
        except Exception:
            st.session_state["user_preferences"] = {}
    prefs = st.session_state["user_preferences"]
    # Notification preferences
    notif_signals = st.checkbox(get_text("notif_signals", st.session_state.get("language", "en")), value=prefs.get("notif_signals", True))
    notif_news = st.checkbox(get_text("notif_news", st.session_state.get("language", "en")), value=prefs.get("notif_news", True))
    # Favorite pairs
    fav_pairs = st.multiselect(get_text("favorite_pairs", st.session_state.get("language", "en")), Config.SUPPORTED_PAIRS, default=prefs.get("favorite_pairs", ["BTC/USD", "XAU/USD"]))
    # Save button
    if st.button(get_text("save_preferences", st.session_state.get("language", "en"))):
        new_prefs = {
            "notif_signals": notif_signals,
            "notif_news": notif_news,
            "favorite_pairs": fav_pairs
        }
        try:
            url = f"{Config.API_BASE_URL}/auth/profile"
            headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"} if st.session_state.get("auth_token") else {}
            payload = {"preferences": new_prefs}
            resp = requests.put(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                st.session_state["user_preferences"] = new_prefs
                st.success(get_text("preferences_saved", st.session_state.get("language", "en")))
            else:
                st.error(get_text("preferences_save_failed", st.session_state.get("language", "en")))
        except Exception as e:
            st.error(f"{get_text('preferences_save_failed', st.session_state.get('language', 'en'))}: {e}")

def show_admin_page():
    st.header("Admin Tools")
    st.write("Create License, View All Users, etc. (implement as needed)")

# --- Modern Chatbot UI ---
def show_chatbot_page():
    st.header("🤖 Trading Assistant (Chatbot)")
    st.markdown("<small>Ask any trading question or request a summary of signals.</small>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_input = st.text_input("Type your message...", "", key="chatbot_input", help="Ask about prices, signals, or trading concepts.")
    model = st.selectbox("Model", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], key="chatbot_model", help="Choose the AI model for your assistant.")
    send_disabled = st.session_state.get("chatbot_waiting", False)
    if st.session_state.get("mobile_mode", False):
        send_clicked = st.button("Send", key="chatbot_send", disabled=send_disabled or not user_input.strip())
    else:
        col1, col2 = st.columns([8,1])
        with col2:
            send_clicked = st.button("Send", key="chatbot_send", disabled=send_disabled or not user_input.strip())
    if send_clicked:
        st.session_state["chatbot_waiting"] = True
        with st.spinner("Waiting for AI response..."):
            try:
                headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"} if st.session_state.get("auth_token") else {}
                data = asyncio.run(make_api_request(
                    f"{Config.API_BASE_URL}/chat",
                    method="POST",
                    json={"message": user_input, "model": model},
                    headers=headers
                ))
                ai_response = data.get("response", "[No response]") if data else "[No response]"
            except Exception as e:
                ai_response = f"[Request error: {e}]"
        st.session_state["chat_history"].append((user_input, ai_response, datetime.now().strftime("%H:%M")))
        st.session_state["chatbot_waiting"] = False
        st.experimental_rerun()
    st.markdown("---")
    chat_container = st.container()
    with chat_container:
        for user_msg, ai_msg, msg_time in st.session_state["chat_history"][-20:]:
            st.markdown(f"<div style='background:#222;border-radius:8px;padding:8px 12px;margin-bottom:4px;'><b>🧑 You [{msg_time}]:</b><br>{user_msg}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#1a3c6b;border-radius:8px;padding:8px 12px;margin-bottom:12px;color:#fff;'><b>🤖 Bot:</b><br>{ai_msg}</div>", unsafe_allow_html=True)
    st.caption("Powered by Groq Llama/Mixtral/Gemma. For educational use only.")

# --- Modern AI Chart Analyzer UI ---
def show_ai_chart_analyzer_page():
    st.header("📊 AI Chart Analyzer")
    st.markdown("<small>Get AI-powered insights for any pair and timeframe, or upload a chart image for instant analysis.</small>", unsafe_allow_html=True)
    pair = st.selectbox("Trading Pair", Config.SUPPORTED_PAIRS, key="ai_chart_pair", help="Choose the market to analyze.")
    timeframe = st.selectbox("Timeframe", Config.SUPPORTED_TIMEFRAMES, key="ai_chart_timeframe", help="Select the timeframe for analysis.")
    analyze_clicked = st.button("Analyze Chart", key="analyze_chart_btn")
    uploaded_file = st.file_uploader("Or upload a chart image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="chart_image_uploader")
    if analyze_clicked:
        with st.spinner("Analyzing chart..."):
            try:
                headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"} if st.session_state.get("auth_token") else {}
                data = asyncio.run(make_api_request(
                    f"{Config.API_BASE_URL}/ai/analyze-chart",
                    method="POST",
                    json={"pair": pair, "timeframe": timeframe},
                    headers=headers
                ))
                insights = data.get("insights", {}) if data else {}
                if data and data.get("error"):
                    st.error(f"AI Chart Analyzer error: {data['error']}")
                st.success(insights.get("summary", "No summary available."))
                with st.container():
                    st.markdown(f"<b>Trend:</b> <span style='color:green;'>{insights.get('trend','')}</span>", unsafe_allow_html=True)
                    st.markdown(f"<b>Support:</b> {insights.get('support','')}")
                    st.markdown(f"<b>Resistance:</b> {insights.get('resistance','')}")
                    st.markdown(f"<b>Patterns:</b> {' | '.join(insights.get('patterns', []))}")
                    st.caption(f"Last analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                st.error(f"Failed to analyze chart: {e}")
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Chart", use_column_width=True)
        with st.spinner("Analyzing uploaded chart image..."):
            try:
                headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"} if st.session_state.get("auth_token") else {}
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                resp = requests.post(f"{Config.API_BASE_URL}/ai/analyze-chart-image", files=files, headers=headers, timeout=30)
                if resp.status_code == 200:
                    result = resp.json().get("result", "No analysis result.")
                    st.success(result)
                elif resp.status_code == 401:
                    st.session_state["auth_token"] = None
                    st.error("Authentication failed. Please log in again.")
                    st.rerun()
                else:
                    st.error(f"Error: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Failed to analyze chart image: {e}")
    st.markdown("---")
    st.caption("AI Chart Analyzer uses technical indicators, pattern recognition, and image analysis for educational purposes.")

# --- Modern News Filter UI ---
def show_news_filter_page():
    st.header("📰 News Filter & Headlines")
    st.markdown("<small>See the latest news and major events for your selected pair.</small>", unsafe_allow_html=True)
    pair = st.selectbox("Trading Pair", Config.SUPPORTED_PAIRS, key="news_pair", help="Choose the market for news.")
    fetch_clicked = st.button("Fetch News", key="fetch_news_btn")
    if fetch_clicked:
        api_pair = pair.replace('/', '-')
        with st.spinner("Fetching news..."):
            try:
                headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"} if st.session_state.get("auth_token") else {}
                data = asyncio.run(make_api_request(
                    f"{Config.API_BASE_URL}/news/{api_pair}",
                    headers=headers
                ))
                if data:
                    if data.get("error"):
                        st.error(f"News error: {data['error']}")
                    headlines = data.get("headlines", [])
                    major_event = data.get("major_event", False)
                    if major_event:
                        st.warning("⚠️ Major economic event detected in news headlines! Trade with caution.")
                    for h in headlines:
                        published = h.get("publishedAt", "")
                        st.markdown(f"<div style='background:#23272f;border-radius:8px;padding:10px 14px;margin-bottom:8px;'><b>📰 <a href='{h['url']}' target='_blank'>{h['title']}</a></b><br><span style='font-size:12px;color:#aaa;'>Published: {published}</span></div>", unsafe_allow_html=True)
                else:
                    st.error("No news data returned.")
            except Exception as e:
                st.error(f"Failed to fetch news: {e}")
    st.markdown("---")
    st.caption("News is for informational purposes only. Always verify with official sources.")

# --- Automatic WebSocket Polling and UI Refresh ---
def poll_websocket_messages():
    ws_service = get_websocket_service()
    for _ in range(10):
        msg = ws_service.get_message()
        if msg:
            handle_websocket_message(msg)
        else:
            break
        
# Force Streamlit to rerun every 5 seconds for real-time updates
st.query_params["_"] = int(time.time() // 5)

# --- Footer ---
def show_footer():
    st.markdown("""
    <hr style='margin-top:2em;margin-bottom:0.5em;border:1px solid #333;'>
    <div style='text-align:center;font-size:13px;color:#888;'>
        Moneyverse A.I. &copy; {year} | <a href='mailto:support@moneyverse.ai'>Support</a> | <a href='https://moneyverse.ai' target='_blank'>Website</a>
    </div>
    """.format(year=datetime.now().year), unsafe_allow_html=True)

# --- Main ---
def main():
    init_session_state()
    poll_websocket_messages()  # Ensure real-time sync with backend
    ws_service = get_websocket_service()
    if not st.session_state.license_verified:
        show_license_screen()
        return
    show_sidebar()  # Ensure sidebar is always shown after license verification
    st.sidebar.title("Navigation")
    
    current_language = st.session_state.get("language", "en")
    navigation_options = [
        get_text("dashboard", current_language),
        get_text("signals", current_language), 
        get_text("market_data", current_language),
        get_text("ai_chart_analyzer", current_language),
        get_text("news_filter", current_language),
        get_text("trading_assistant", current_language),
        get_text("education", current_language),
        get_text("licenses", current_language),
        get_text("profile", current_language),
        get_text("admin", current_language),
        get_text("backtest", current_language)
    ]
    
    page = st.sidebar.radio("Go to", navigation_options)
    
    if page == get_text("dashboard", current_language):
        show_dashboard_page()
    elif page == get_text("signals", current_language):
        show_signals_page()
    elif page == get_text("market_data", current_language):
        show_market_data_page()
    elif page == get_text("ai_chart_analyzer", current_language):
        show_ai_chart_analyzer_page()
    elif page == get_text("news_filter", current_language):
        show_news_filter_page()
    elif page == get_text("trading_assistant", current_language):
        show_chatbot_page()
    elif page == get_text("education", current_language):
        show_education_page()
    elif page == get_text("licenses", current_language):
        show_licenses_page()
    elif page == get_text("profile", current_language):
        show_profile_page()
    elif page == get_text("admin", current_language):
        show_admin_page()
    elif page == get_text("backtest", current_language):
        show_backtest_page()
    show_footer()

def show_license_screen():
    """Display the license validation screen."""
    st.header("Enter Your License Key")
    license_code = st.text_input("License Key", type="password", key="license_input")
    if st.button("Verify License"):
        if license_code:
                    asyncio.run(handle_license_validation(license_code))
        else:
            st.error("Please enter a license key.")

    if st.session_state.license_error:
        st.error(st.session_state.license_error)

def show_dashboard():
    """Display the main dashboard content in a modern, clean layout."""
    st.session_state.signals = asyncio.run(get_active_signals())
    prev_pair = st.session_state.get("prev_pair", st.session_state.selected_pair)
    prev_timeframe = st.session_state.get("prev_timeframe", st.session_state.selected_timeframe)

    # --- DEMO MODE BANNER ---
    # Check both market and historical data sources
    pair = st.session_state.selected_pair
    timeframe = st.session_state.selected_timeframe
    market_data = asyncio.run(fetch_market_data(pair))
    historical_data = asyncio.run(fetch_historical_data(pair, timeframe, 100))
    data_sources = []
    if market_data and market_data.get("data_source"):
        data_sources.append(market_data["data_source"])
    if historical_data and historical_data.get("data_source"):
        data_sources.append(historical_data["data_source"])
    show_demo_mode_banner(data_sources)

    if st.session_state.selected_pair != prev_pair or st.session_state.selected_timeframe != prev_timeframe:
        send_start_monitoring(
            st.session_state.selected_pair, 
            st.session_state.selected_timeframe, 
            st.session_state.license_code
        )
        if st.session_state.selected_timeframe == "1h":
            asyncio.run(fetch_historical_data(st.session_state.selected_pair, st.session_state.selected_timeframe))
        st.session_state.prev_pair = st.session_state.selected_pair
        st.session_state.prev_timeframe = st.session_state.selected_timeframe
        st.rerun()

    st.header("Trading Dashboard")
    # --- Responsive Columns ---
    if st.session_state.get("mobile_mode", False):
        left_col, right_col = st.container(), st.container()
    else:
        left_col, right_col = st.columns([2, 2])
    with left_col:
        # --- Signals Table ---
        signals = st.session_state.signals
        if signals:
            df_signals = pd.DataFrame([
                {
                    "Pair": s.get("pair", "N/A"),
                    "Direction": s.get("direction", "N/A"),
                    "Entry": s.get("entry", s.get("entry_price", "N/A")),
                    "TP1": s.get("tp1", "N/A"),
                    "TP2": s.get("tp2", "N/A"),
                    "Stop Loss": s.get("sl", s.get("stop_loss", "N/A")),
                    "Confidence": f"{float(s.get('confidence', 0)) * 100:.2f}%" if s.get('confidence') is not None else "N/A",
                    "Status": s.get("status", "N/A"),
                    "Created": s.get("created_at", s.get("generated_time", "N/A")),
                }
                for s in signals[:Config.MAX_DISPLAY_SIGNALS]
            ])
            st.dataframe(df_signals, use_container_width=True)
            # --- Signal Details as Expanders ---
            for s in signals[:5]:
                with st.expander(f"{s.get('pair', 'N/A')} - {s.get('direction', 'N/A')} - {s.get('status', 'N/A')}"):
                    if st.session_state.get("mobile_mode", False):
                        st.write(f"**Entry:** {s.get('entry', s.get('entry_price', 'N/A'))}")
                        st.write(f"**TP1:** {s.get('tp1', 'N/A')}")
                        st.write(f"**TP2:** {s.get('tp2', 'N/A')}")
                        st.write(f"**Stop Loss:** {s.get('sl', s.get('stop_loss', 'N/A'))}")
                        st.write(f"**Confidence:** {float(s.get('confidence', 0)) * 100:.2f}%" if s.get('confidence') is not None else "N/A")
                        st.write(f"**Timeframe:** {s.get('timeframe', 'N/A')}")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Entry:** {s.get('entry', s.get('entry_price', 'N/A'))}")
                            st.write(f"**TP1:** {s.get('tp1', 'N/A')}")
                            st.write(f"**TP2:** {s.get('tp2', 'N/A')}")
                        with col2:
                            st.write(f"**Stop Loss:** {s.get('sl', s.get('stop_loss', 'N/A'))}")
                            st.write(f"**Confidence:** {float(s.get('confidence', 0)) * 100:.2f}%" if s.get('confidence') is not None else "N/A")
                            st.write(f"**Timeframe:** {s.get('timeframe', 'N/A')}")
                            st.write(f"**Reason:** {s.get('reason', 'N/A')}")
                            st.write(f"**Created At:** {s.get('created_at', s.get('generated_time', 'N/A'))}")
                    # --- User Feedback and Force Close Button ---
                if s.get('status') in ["ACTIVE", "TP1_HIT"]:
                        st.warning("A signal is still open for this pair/timeframe. New signals will not be generated until it is fully closed (TP2, SL, or manual close).")
                        adv = st.checkbox(f"Show advanced options for {s.get('pair', 'N/A')} {s.get('timeframe', 'N/A')}", key=f"adv_{s.get('id')}")
                        if adv:
                            if st.button(f"Force Close Signal {s.get('id')}", key=f"force_{s.get('id')}"):
                                resp = asyncio.run(make_api_request(f"{Config.API_BASE_URL}/signals/{s.get('id')}/force-close", method="PATCH", headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}))
                                if resp and resp.get("success"):
                                    st.success("Signal force-closed. New signals can now be generated.")
                                    st.rerun()
                                else:
                                    st.error(resp.get("detail") or "Failed to force close signal.")
                else:
                        st.info("No signals available for the selected pair.")
    with right_col:
        # --- Market Data Metrics ---
        if market_data:
            st.write(f"Fetching market data for {pair}")
            st.write(f"Successfully fetched market data for {pair}")
            st.metric("Current Price", f"${market_data.get('price', 0):,.2f}")
            st.metric("24h Change", f"{market_data.get('change_24h', 0):.2f}%")
            if market_data.get("last_updated"):
                st.caption(f"Last updated: {market_data['last_updated']}")
            if market_data.get("data_source"):
                if market_data["data_source"] == "unavailable" or market_data.get("error"):
                    st.error("Data unavailable. Please try again later or check your data provider/API key.")
                else:
                    st.info(f"Source: {market_data['data_source'].capitalize()}")
        else:
            st.info("Loading market data...")
        # --- Candlestick Chart ---
        st.write("")
        st.subheader("Price and Signals")
        if historical_data:
            fig = create_price_chart(signals, historical_data)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not create price chart. Please try again.")
            if historical_data.get("last_updated"):
                st.caption(f"Historical data last updated: {historical_data['last_updated']}")
            if historical_data.get("data_source"):
                if historical_data["data_source"] == "unavailable" or historical_data.get("error"):
                    st.error("Historical data unavailable. Please try again later or check your data provider/API key.")
                else:
                    st.info(f"Historical data source: {historical_data['data_source'].capitalize()}")
        else:
            st.info("No historical data available.")

# --- Sidebar ---
def show_sidebar():
    """Show the sidebar for navigation and controls."""
    with st.sidebar:
        st.header("Controls")
        # --- Mobile Mode Toggle ---
        st.subheader("📱 Mobile Mode")
        if "mobile_mode" not in st.session_state:
            st.session_state["mobile_mode"] = False
        st.session_state["mobile_mode"] = st.checkbox("Enable Mobile Mode (vertical layout)", value=st.session_state["mobile_mode"])
        
        # Language selector
        st.subheader("🌐 Language")
        current_language = st.session_state.get("language", "en")
        selected_language = st.selectbox(
            "Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            index=list(SUPPORTED_LANGUAGES.keys()).index(current_language)
        )
        if selected_language != current_language:
            st.session_state.language = selected_language
            st.rerun()
        
        # Market Selection
        st.subheader("Market Selection")
        try:
            current_pair_index = Config.SUPPORTED_PAIRS.index(st.session_state.selected_pair)
        except ValueError:
            current_pair_index = 0
        st.session_state.selected_pair = st.selectbox(
            get_text("trading_pair", current_language),
            Config.SUPPORTED_PAIRS,
            index=current_pair_index
        )
        # Only allow supported timeframes for historical data
        valid_timeframes = ["1H", "4H", "1D"]
        try:
            current_tf_index = valid_timeframes.index(st.session_state.selected_timeframe)
        except ValueError:
            current_tf_index = 0
        st.session_state.selected_timeframe = st.selectbox(
            get_text("timeframe", current_language),
            valid_timeframes,
            index=current_tf_index
        )
        st.info(f"Connected with license: ...{st.session_state.license_code[-4:]}")
    if st.button(get_text("logout", current_language)):
            ws_service = get_websocket_service()
            ws_service.disconnect()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# --- Signals Section ---
def show_signals_section():
    """Display active and completed signals."""
    st.subheader("Signal Dashboard")
    active_signals = [s for s in st.session_state.signals if s.get('status') == 'ACTIVE']
    completed_signals = [s for s in st.session_state.signals if s.get('status') != 'ACTIVE']
    if active_signals:
        st.subheader("Active Signals")
        df_active = pd.DataFrame([format_signal(s) for s in active_signals])
        st.dataframe(df_active[['Pair', 'Type', 'Direction', 'Entry', 'TP1', 'TP2', 'SL', 'Status', 'Confidence', 'Generated Time', 'Reason', 'Last Updated', 'Label', 'Logic Note']], use_container_width=True)
    else:
        st.info("No active signals at the moment.")
    if completed_signals:
        st.subheader("Completed Signals")
        df_completed = pd.DataFrame([format_signal(s) for s in completed_signals])
        st.dataframe(df_completed[['Pair', 'Type', 'Direction', 'Entry', 'TP1', 'TP2', 'SL', 'Status', 'Confidence', 'Generated Time', 'Reason', 'Last Updated', 'Label', 'Logic Note']], use_container_width=True)

def show_market_data_section():
    """Shows the market data in the right column."""
    st.subheader("Market Data")
    pair = st.session_state.selected_pair
    timeframe = st.session_state.selected_timeframe

    try:
        if st.session_state.market_data and st.session_state.market_data.get('pair') == pair:
            market_data = st.session_state.market_data
            st.metric("Current Price", f"${market_data.get('price', 0):.2f}")
            st.metric("24h Change", f"{market_data.get('change_24h', 0):.2f}%")
        else:
            asyncio.run(fetch_market_data(pair))
            st.info("Loading market data...")
        if timeframe != "1h":
            st.info("Trend analysis is only available for the 1h timeframe.")
        else:
            if st.session_state.trends.get(pair):
                trend_data = st.session_state.trends[pair]
                st.metric("Trend", trend_data.get('trend', 'N/A').capitalize())
            else:
                asyncio.run(fetch_trend_analysis(pair, timeframe))
                st.info("Analyzing trend...")
    except Exception as e:
        logger.error(f"Error in market data section: {e}")
        st.error("Could not load market data.")

def handle_websocket_message(message: Dict):
    """Processes a message from the WebSocket and updates the app state."""
    msg_type = message.get("type")
    data = message.get("data")
    signals = message.get("signals")

    if msg_type == "signal_update":
        # Handle both single and multiple signals
        if signals and isinstance(signals, list):
            for updated_signal in signals:
                _update_signal_in_state(updated_signal)
        elif data:
            _update_signal_in_state(data)
    elif msg_type == "price_update":
        st.session_state.market_data = data
    elif msg_type == "trend_update":
        pair = data.get('pair') if data else None
        if pair:
            st.session_state.trends[pair] = data
    # Can add more message types here

def _update_signal_in_state(updated_signal):
        found = False
        for i, signal in enumerate(st.session_state.signals):
            if signal.get("id") == updated_signal.get("id"):
                st.session_state.signals[i] = updated_signal
                found = True
                break
        if not found:
            st.session_state.signals.append(updated_signal)
        st.session_state.signals.sort(key=lambda x: x.get('generated_time', 0), reverse=True)
        st.session_state.active_signals = [s for s in st.session_state.signals if s.get('status') == 'ACTIVE']

async def handle_license_validation(license_code: str):
    """Validate license and set session state."""
    if not license_code:
        st.error("Please enter a license code.")
        return

    st.session_state.license_error = ""
    try:
        result = await validate_license(license_code)
        if result and result.get("valid"):
            st.session_state.license_verified = True
            st.session_state.license_code = license_code
            st.session_state.user_email = result.get("user_email", "")
            # Store the real JWT from the backend
            st.session_state.auth_token = result.get("access_token")
            st.success("License validated successfully!")
        else:
            st.session_state.license_verified = False
            error_detail = result.get("detail", "Invalid or expired license.")
            st.session_state.license_error = error_detail
    except Exception as e:
        logger.error(f"An exception occurred during license validation: {e}")
        st.session_state.license_verified = False
        st.session_state.license_error = "An error occurred. Please try again later."
    
        st.rerun()

def show_education_page():
    st.header("📚 Trading Education Center")
    # Fetch dynamic content
    data = fetch_education_content()
    if data and "articles" in data:
        for article in data["articles"]:
            st.subheader(f"{article['title']} ({article['category']})")
            st.markdown(article["content"])
            if article.get("video_url"):
                st.video(article["video_url"])
            if article.get("quiz"):
                st.markdown("**Quiz:**")
                for idx, q in enumerate(article["quiz"]):
                    st.write(q["question"])
                    st.radio(
                        label="Choose one:",
                        options=q["options"],
                        key=f"quiz_{article['id']}_{idx}"
                    )
            st.markdown("---")
    else:
        st.info("No dynamic education content available. Showing static content.")
        # ... (fallback to your current static content) ...
        # (Paste your previous static education content here if desired)

# Language support
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ms": "Bahasa Melayu", 
    "id": "Bahasa Indonesia",
    "th": "ไทย"
}

# Simple translations (in a real app, use a proper i18n library)
TRANSLATIONS = {
    "en": {
        "dashboard": "Dashboard",
        "signals": "Signals", 
        "market_data": "Market Data",
        "ai_chart_analyzer": "AI Chart Analyzer",
        "news_filter": "News Filter",
        "trading_assistant": "Trading Assistant",
        "education": "Education",
        "licenses": "Licenses",
        "profile": "Profile",
        "admin": "Admin",
        "trading_pair": "Trading Pair",
        "timeframe": "Timeframe",
        "analyze": "Analyze",
        "send": "Send",
        "fetch_news": "Fetch News",
        "logout": "Logout",
        "scalping": "Scalping",
        "intraday": "Intraday",
        "swing": "Swing",
        "upload_chart": "Upload Chart",
        "take_photo": "Take Photo",
        "trading_style": "Trading Style",
        "demo_mode_banner": "🚧 Demo Mode Active! Some data may be simulated or unavailable.",
        "dismiss_demo_banner": "Dismiss Demo Banner",
        "mobile_mode": "📱 Mobile Mode",
        "enable_mobile_mode": "Enable Mobile Mode (vertical layout)",
        "all_signals": "All Signals",
        "market_data": "Market Data",
        "licenses": "Licenses",
        "profile": "Profile",
        "admin_tools": "Admin Tools",
        "validate_license": "Validate License",
        "validate_license_code": "Validate License Code",
        "active_licenses": "Active Licenses:",
        "email": "Email:",
        "create_license": "Create License, View All Users, etc. (implement as needed)",
        "type_message": "Type your message...",
        "ask_anything": "Ask any trading question or request a summary of signals.",
        "model": "Model",
        "waiting_ai": "Waiting for AI response...",
        "powered_by": "Powered by Groq Llama/Mixtral/Gemma. For educational use only.",
        "ai_chart_analyzer": "AI Chart Analyzer",
        "get_ai_insights": "Get AI-powered insights for any pair and timeframe, or upload a chart image for instant analysis.",
        "analyze_chart": "Analyze Chart",
        "upload_chart_image": "Or upload a chart image (PNG, JPG)",
        "analyzing_chart": "Analyzing chart...",
        "ai_chart_error": "AI Chart Analyzer error:",
        "no_summary": "No summary available.",
        "trend": "Trend:",
        "support": "Support:",
        "resistance": "Resistance:",
        "patterns": "Patterns:",
        "last_analyzed": "Last analyzed:",
        "failed_analyze_chart": "Failed to analyze chart:",
        "uploaded_chart": "Uploaded Chart",
        "analyzing_uploaded_chart": "Analyzing uploaded chart image...",
        "no_analysis_result": "No analysis result.",
        "auth_failed": "Authentication failed. Please log in again.",
        "ai_chart_caption": "AI Chart Analyzer uses technical indicators, pattern recognition, and image analysis for educational purposes.",
        "news_filter": "News Filter & Headlines",
        "see_latest_news": "See the latest news and major events for your selected pair.",
        "fetch_news": "Fetch News",
        "fetching_news": "Fetching news...",
        "news_error": "News error:",
        "major_event": "⚠️ Major economic event detected in news headlines! Trade with caution.",
        "no_news": "No news data returned.",
        "failed_fetch_news": "Failed to fetch news:",
        "news_caption": "News is for informational purposes only. Always verify with official sources.",
        "trading_dashboard": "Trading Dashboard",
        "signal_dashboard": "Signal Dashboard",
        "active_signals": "Active Signals",
        "completed_signals": "Completed Signals",
        "no_active_signals": "No active signals at the moment.",
        "no_signals_selected_pair": "No signals available for the selected pair.",
        "loading_market_data": "Loading market data...",
        "price_and_signals": "Price and Signals",
        "could_not_create_chart": "Could not create price chart. Please try again.",
        "historical_data_last_updated": "Historical data last updated:",
        "historical_data_unavailable": "Historical data unavailable. Please try again later or check your data provider/API key.",
        "historical_data_source": "Historical data source:",
        "no_historical_data": "No historical data available.",
        "enter_license_key": "Enter Your License Key",
        "license_key": "License Key",
        "verify_license": "Verify License",
        "please_enter_license": "Please enter a license key.",
        "license_validated": "License validated successfully!",
        "invalid_or_expired_license": "Invalid or expired license.",
        "an_error_occurred": "An error occurred. Please try again later.",
        "trading_education_center": "📚 Trading Education Center",
        "no_dynamic_education": "No dynamic education content available. Showing static content.",
        "quiz": "Quiz:",
        "choose_one": "Choose one:",
        "controls": "Controls",
        "market_selection": "Market Selection",
        "connected_with_license": "Connected with license:",
        "force_close_signal": "Force Close Signal",
        "show_advanced_options": "Show advanced options for",
        "a_signal_still_open": "A signal is still open for this pair/timeframe. New signals will not be generated until it is fully closed (TP2, SL, or manual close).",
        "signal_force_closed": "Signal force-closed. New signals can now be generated.",
        "failed_force_close": "Failed to force close signal.",
        "no_signals": "No signals available for the selected pair.",
        "data_unavailable": "Data unavailable. Please try again later or check your data provider/API key.",
        "source": "Source:",
        "last_updated": "Last updated:",
        "current_price": "Current Price",
        "change_24h": "24h Change",
        "license_validated_success": "License validated successfully!",
        "backtest": "Backtest",
        "run_backtest": "Run Backtest",
        "running_backtest": "Running backtest...",
        "backtest_complete": "Backtest complete!",
        "total_pnl": "Total P&L",
        "win_rate": "Win Rate",
        "num_trades": "Number of Trades",
        "max_drawdown": "Max Drawdown",
        "equity_curve": "Equity Curve",
        "trade_list": "Trade List",
        "backtest_failed": "Backtest failed",
        "preferences": "Preferences",
        "notif_signals": "Enable Signal Notifications",
        "notif_news": "Enable News Notifications",
        "favorite_pairs": "Favorite Trading Pairs",
        "save_preferences": "Save Preferences",
        "preferences_saved": "Preferences saved!",
        "preferences_save_failed": "Failed to save preferences."
    },
    "license_validated_success": "License validated successfully!"
}

# Add translation keys for backtest UI (place at top-level, outside any function)
if "backtest" not in TRANSLATIONS["en"]:
    TRANSLATIONS["en"].update({
        "backtest": "Backtest",
        "run_backtest": "Run Backtest",
        "running_backtest": "Running backtest...",
        "backtest_complete": "Backtest complete!",
        "total_pnl": "Total P&L",
        "win_rate": "Win Rate",
        "num_trades": "Number of Trades",
        "max_drawdown": "Max Drawdown",
        "equity_curve": "Equity Curve",
        "trade_list": "Trade List",
        "backtest_failed": "Backtest failed"
    })

for lang in ["ms", "id", "th"]:
    if lang not in TRANSLATIONS:
        TRANSLATIONS[lang] = {}
    for k in ["backtest", "run_backtest", "running_backtest", "backtest_complete", "total_pnl", "win_rate", "num_trades", "max_drawdown", "equity_curve", "trade_list", "backtest_failed"]:
        if k not in TRANSLATIONS[lang]:
            TRANSLATIONS[lang][k] = TRANSLATIONS["en"][k]

def get_text(key: str, language: str = "en") -> str:
    """Get translated text for a given key and language."""
    if language in TRANSLATIONS and key in TRANSLATIONS[language]:
        return TRANSLATIONS[language][key]
    return TRANSLATIONS["en"].get(key, key)  # Fallback to English

# --- Education API Placeholder ---
# If you want dynamic education content, implement backend endpoints and fetch here.
def fetch_education_content():
    headers = {"Authorization": f"Bearer {st.session_state['auth_token']}"} if st.session_state.get("auth_token") else {}
    try:
        data = asyncio.run(make_api_request(f"{Config.API_BASE_URL}/education", headers=headers))
        if data and data.get("error"):
            st.error(f"Education error: {data['error']}")
        return data or {}
    except Exception as e:
        st.error(f"Failed to fetch education content: {e}")
        return {}

# --- DEMO MODE BANNER ---
def show_demo_mode_banner(data_sources=None):
    """Show a prominent banner if in demo/mock mode or data unavailable."""
    show_banner = False
    reason = []
    if Config.ENVIRONMENT != "production":
        show_banner = True
        reason.append(f"Environment: {Config.ENVIRONMENT}")
    if data_sources:
        for src in data_sources:
            if src in ("mock", "unavailable"):
                show_banner = True
                reason.append(f"Data source: {src}")
    if show_banner:
        if "hide_demo_banner" not in st.session_state:
            st.session_state["hide_demo_banner"] = False
        if not st.session_state["hide_demo_banner"]:
            with st.container():
                st.warning(
                    f"🚧 Demo Mode Active! Some data may be simulated or unavailable. {' | '.join(reason)}",
                    icon="⚠️"
                )
                if st.button("Dismiss Demo Banner", key="dismiss_demo_banner"):
                    st.session_state["hide_demo_banner"] = True

def show_backtest_page():
    import plotly.graph_objects as go
    st.header(get_text("backtest", st.session_state.get("language", "en")))
    # Form for backtest parameters
    with st.form("backtest_form"):
        pair = st.selectbox(get_text("trading_pair", st.session_state.get("language", "en")), Config.SUPPORTED_PAIRS)
        timeframe = st.selectbox(get_text("timeframe", st.session_state.get("language", "en")), Config.SUPPORTED_TIMEFRAMES)
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        submitted = st.form_submit_button(get_text("run_backtest", st.session_state.get("language", "en")))
    if submitted:
        with st.spinner(get_text("running_backtest", st.session_state.get("language", "en"))):
            try:
                url = f"{Config.API_BASE_URL}/backtest"
                headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"} if st.session_state.get("auth_token") else {}
                payload = {
                    "pair": pair,
                    "timeframe": timeframe,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "strategy": "default"
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(get_text("backtest_complete", st.session_state.get("language", "en")))
                    st.write(f"**{get_text('total_pnl', st.session_state.get('language', 'en'))}:** {result['total_pnl']:.2f}")
                    st.write(f"**{get_text('win_rate', st.session_state.get('language', 'en'))}:** {result['win_rate']*100:.2f}%")
                    st.write(f"**{get_text('num_trades', st.session_state.get('language', 'en'))}:** {result['num_trades']}")
                    st.write(f"**{get_text('max_drawdown', st.session_state.get('language', 'en'))}:** {result['max_drawdown']:.2f}")
                    # Equity curve chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=result['equity_curve'], mode='lines', name='Equity'))
                    fig.update_layout(title=get_text('equity_curve', st.session_state.get('language', 'en')), xaxis_title='Trade', yaxis_title='Equity')
                    st.plotly_chart(fig, use_container_width=True)
                    # Trades table
                    st.subheader(get_text('trade_list', st.session_state.get('language', 'en')))
                    st.dataframe(result['trades'])
                else:
                    st.error(f"{get_text('backtest_failed', st.session_state.get('language', 'en'))}: {resp.text}")
            except Exception as e:
                st.error(f"{get_text('backtest_failed', st.session_state.get('language', 'en'))}: {e}")

if __name__ == "__main__":
    main()