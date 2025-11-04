import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stock Bot Monitor",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 Telegram Stock Bot Monitor")
    
    st.markdown("""
    ## Bot Status Dashboard
    
    This dashboard monitors your Telegram Stock Analysis Bot.
    The bot runs independently and provides:
    
    - 📊 Technical analysis via Twelve Data API
    - 📰 News sentiment via Alpha Vantage  
    - 🎯 Trade recommendations
    - 💰 Risk management levels
    """)
    
    # Configuration section
    st.subheader("⚙️ Configuration Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        telegram_status = "✅ Set" if os.getenv('TELEGRAM_BOT_TOKEN') else "❌ Missing"
        st.metric("Telegram Bot Token", telegram_status)
    
    with col2:
        twelve_data_status = "✅ Set" if os.getenv('TWELVE_DATA_API_KEY') else "❌ Missing"
        st.metric("Twelve Data API", twelve_data_status)
    
    with col3:
        alpha_vantage_status = "✅ Set" if os.getenv('ALPHA_VANTAGE_API_KEY') else "❌ Missing"
        st.metric("Alpha Vantage API", alpha_vantage_status)
    
    # Quick test section
    st.subheader("🧪 API Test")
    
    test_col1, test_col2 = st.columns([1, 2])
    
    with test_col1:
        test_symbol = st.text_input("Test Symbol", "AAPL")
        if st.button("Test APIs"):
            test_apis(test_symbol)
    
    # Usage instructions
    st.subheader("📖 Usage Instructions")
    
    st.markdown("""
    ### How to Use the Telegram Bot:
    
    1. **Start the bot separately** (not in Streamlit):
       ```bash
       python telegram_bot.py
       ```
    
    2. **Find your bot on Telegram** and send:
       - `/start` - Initialize bot
       - `AAPL` or any stock symbol - Get analysis
       - `/help` - Show help
    
    3. **The bot will respond with:**
       - Current price and technical indicators
       - Market sentiment analysis  
       - Trade recommendation (BUY/SELL/HOLD)
       - Entry, Stop Loss, and Target prices
    """)
    
    # Example output
    with st.expander("📊 Example Bot Output"):
        st.markdown("""
        ```
        📊 ANALYSIS REPORT: AAPL
        💰 Price: $182.50

        🎯 TRADE RECOMMENDATION
        • Action: BUY (High)
        • Entry: $182.50
        • Stop Loss: $178.85
        • Target: $188.89

        📈 TECHNICALS
        • RSI: 45.2
        • Trend: Bullish

        📰 SENTIMENT
        • Score: 0.34 (Positive)
        • Assessment: Bullish sentiment...
        ```
        """)
    
    # Deployment instructions
    st.subheader("🚀 Deployment")
    
    st.markdown("""
    ### For Production Deployment:
    
    **Option 1: Cloud Server**
    - Deploy `telegram_bot.py` on a cloud server (AWS EC2, DigitalOcean, etc.)
    - Use `nohup python telegram_bot.py &` to run in background
    
    **Option 2: Railway/Render**
    - Create account on Railway.app or Render.com
    - Upload your bot code
    - Set environment variables
    - Deploy as a background worker
    
    **Option 3: Replit**
    - Create new Repl with `telegram_bot.py`
    - Set secrets for API keys
    - Run as always-on bot
    """)

def test_apis(symbol):
    """Test the APIs with a sample symbol"""
    try:
        # Test Twelve Data API
        twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
        if twelve_data_key:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': '15min',
                'apikey': twelve_data_key,
                'format': 'JSON'
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                st.success(f"✅ Twelve Data API: Working (Found {symbol})")
            else:
                st.error(f"❌ Twelve Data API: Error {response.status_code}")
        else:
            st.warning("⚠️ Twelve Data API key not set")
        
        # Test Alpha Vantage API
        alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_vantage_key:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': alpha_vantage_key
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                st.success(f"✅ Alpha Vantage API: Working (Found {symbol})")
            else:
                st.error(f"❌ Alpha Vantage API: Error {response.status_code}")
        else:
            st.warning("⚠️ Alpha Vantage API key not set")
            
    except Exception as e:
        st.error(f"❌ API Test Failed: {e}")

if __name__ == '__main__':
    main()
