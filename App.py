import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import sys
import subprocess

# Check and install missing packages
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        st.warning(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
required_packages = ['ta', 'textblob']
for package in required_packages:
    install_package(package)

# Now import the packages
try:
    import ta
    from textblob import TextBlob
except ImportError as e:
    st.error(f"Failed to import required packages: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        
    def safe_float_format(self, value, default="N/A"):
        """Safely format float values with error handling"""
        try:
            if value is None or pd.isna(value):
                return default
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return default
            
    def safe_get(self, data, key, default=None):
        """Safely get value from dictionary or series"""
        try:
            if isinstance(data, dict):
                return data.get(key, default)
            elif hasattr(data, 'get'):
                return data.get(key, default)
            else:
                return getattr(data, key, default)
        except:
            return default
    
    def get_stock_data(self, symbol: str, interval: str, periods: int = 100):
        """Fetch stock data using yfinance"""
        try:
            # Map intervals to yfinance format
            interval_map = {
                '1min': '1m',
                '15min': '15m',
                '1h': '1h',
                '1day': '1d'
            }
            
            # Calculate period based on interval
            if interval == '1m':
                period = '7d'
            elif interval == '15m':
                period = '60d'
            else:  # 1h or higher
                period = '60d'
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval_map.get(interval, '15m'))
            
            if len(data) < periods:
                data = ticker.history(period='max', interval=interval_map.get(interval, '15m'))
            
            if len(data) == 0:
                st.error(f"No data found for symbol {symbol}")
                return None
            
            data = data.tail(periods).reset_index()
            data.rename(columns={
                'Date': 'datetime',
                'Close': 'close',
                'High': 'high',
                'Low': 'low',
                'Open': 'open',
                'Volume': 'volume'
            }, inplace=True)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators with error handling"""
        if df is None or len(df) < 5:
            return None
            
        try:
            # Make a copy to avoid modifying original
            df = df.copy()
            
            # Basic calculations that should always work
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            
            # Fill NaN values with forward fill
            df = df.ffill()
            
            # RSI with safe calculation
            if len(df) >= 14:
                try:
                    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
                except:
                    df['rsi'] = 50
            else:
                df['rsi'] = 50
                
            # MACD with safe calculation
            if len(df) >= 26:
                try:
                    macd = ta.trend.MACD(df['close'])
                    df['macd'] = macd.macd()
                    df['macd_signal'] = macd.macd_signal()
                    df['macd_histogram'] = macd.macd_diff()
                except:
                    df['macd'] = 0
                    df['macd_signal'] = 0
                    df['macd_histogram'] = 0
            else:
                df['macd'] = 0
                df['macd_signal'] = 0
                df['macd_histogram'] = 0
            
            # Bollinger Bands with safe calculation
            if len(df) >= 20:
                try:
                    bollinger = ta.volatility.BollingerBands(df['close'])
                    df['bb_upper'] = bollinger.bollinger_hband()
                    df['bb_lower'] = bollinger.bollinger_lband()
                    df['bb_middle'] = bollinger.bollinger_mavg()
                except:
                    df['bb_upper'] = df['close']
                    df['bb_lower'] = df['close']
                    df['bb_middle'] = df['close']
            else:
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']
                df['bb_middle'] = df['close']
            
            # Moving Averages with safe calculation
            try:
                df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=min(20, len(df))).sma_indicator()
            except:
                df['sma_20'] = df['close']
                
            try:
                df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=min(12, len(df))).ema_indicator()
            except:
                df['ema_12'] = df['close']
                
            try:
                df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=min(26, len(df))).ema_indicator()
            except:
                df['ema_26'] = df['close']
            
            # Volume indicators
            try:
                df['volume_sma'] = ta.trend.SMAIndicator(df['volume'], window=min(20, len(df))).sma_indicator()
            except:
                df['volume_sma'] = df['volume']
            
            return df
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_candlestick_patterns(self, df):
        """Analyze candlestick patterns with error handling"""
        if df is None or len(df) < 3:
            return ["Insufficient data for pattern analysis"]
            
        try:
            recent = df.iloc[-3:]
            patterns = []
            
            # Bullish patterns
            if self.is_hammer(recent.iloc[-1]):
                patterns.append("Hammer (Bullish)")
            if self.is_bullish_engulfing(recent.iloc[-2], recent.iloc[-1]):
                patterns.append("Bullish Engulfing")
            if self.is_morning_star(recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]):
                patterns.append("Morning Star")
                
            # Bearish patterns
            if self.is_shooting_star(recent.iloc[-1]):
                patterns.append("Shooting Star (Bearish)")
            if self.is_bearish_engulfing(recent.iloc[-2], recent.iloc[-1]):
                patterns.append("Bearish Engulfing")
            if self.is_evening_star(recent.iloc[-3], recent.iloc[-2], recent.iloc[-1]):
                patterns.append("Evening Star")
                
            return patterns if patterns else ["No significant patterns"]
        except Exception as e:
            return [f"Pattern analysis error: {str(e)}"]
    
    def is_hammer(self, candle):
        try:
            body = abs(candle['close'] - candle['open'])
            if body == 0:  # Avoid division by zero
                return False
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            return lower_wick >= 2 * body and upper_wick <= body * 0.5
        except:
            return False
    
    def is_shooting_star(self, candle):
        try:
            body = abs(candle['close'] - candle['open'])
            if body == 0:
                return False
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            return upper_wick >= 2 * body and lower_wick <= body * 0.5
        except:
            return False
    
    def is_bullish_engulfing(self, prev_candle, curr_candle):
        try:
            return (prev_candle['close'] < prev_candle['open'] and
                    curr_candle['close'] > curr_candle['open'] and
                    curr_candle['open'] < prev_candle['close'] and
                    curr_candle['close'] > prev_candle['open'])
        except:
            return False
    
    def is_bearish_engulfing(self, prev_candle, curr_candle):
        try:
            return (prev_candle['close'] > prev_candle['open'] and
                    curr_candle['close'] < curr_candle['open'] and
                    curr_candle['open'] > prev_candle['close'] and
                    curr_candle['close'] < prev_candle['open'])
        except:
            return False
    
    def is_morning_star(self, candle1, candle2, candle3):
        try:
            # Simplified morning star pattern
            return (candle1['close'] < candle1['open'] and  # First red candle
                    abs(candle2['close'] - candle2['open']) < (candle1['high'] - candle1['low']) * 0.3 and  # Small body
                    candle3['close'] > candle3['open'] and  # Green candle
                    candle3['close'] > candle1['close'])    # Closes above first candle
        except:
            return False
    
    def is_evening_star(self, candle1, candle2, candle3):
        try:
            # Simplified evening star pattern
            return (candle1['close'] > candle1['open'] and  # First green candle
                    abs(candle2['close'] - candle2['open']) < (candle1['high'] - candle1['low']) * 0.3 and  # Small body
                    candle3['close'] < candle3['open'] and  # Red candle
                    candle3['close'] < candle1['close'])    # Closes below first candle
        except:
            return False
    
    def get_market_news_sentiment(self, symbol: str):
        """Fetch and analyze market news sentiment"""
        try:
            # Using Yahoo Finance news as primary source
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return self.get_fallback_sentiment(symbol)
            
            headlines = [item.get('title', '') for item in news[:10] if 'title' in item]
            return self.analyze_text_sentiment(headlines, symbol)
            
        except Exception as e:
            st.error(f"News sentiment error: {e}")
            return self.get_fallback_sentiment(symbol)
    
    def get_fallback_sentiment(self, symbol: str):
        """Fallback sentiment analysis"""
        return {
            'sentiment_score': 0.0,
            'sentiment_category': 'Neutral',
            'rationale': 'Limited news data available',
            'major_events': []
        }
    
    def analyze_text_sentiment(self, headlines, symbol):
        """Analyze sentiment from headlines using TextBlob"""
        sentiments = []
        major_events = []
        
        for headline in headlines:
            try:
                blob = TextBlob(headline)
                sentiment_score = blob.sentiment.polarity
                sentiments.append(sentiment_score)
                
                if abs(sentiment_score) > 0.3:
                    event_type = "Positive" if sentiment_score > 0 else "Negative"
                    major_events.append({
                        'headline': headline[:100] + "..." if len(headline) > 100 else headline,
                        'impact': 'High' if abs(sentiment_score) > 0.5 else 'Medium',
                        'type': event_type
                    })
            except:
                continue
        
        if not sentiments:
            avg_sentiment = 0.0
        else:
            avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Categorize sentiment
        if avg_sentiment > 0.1:
            category = "Positive"
        elif avg_sentiment < -0.1:
            category = "Negative"
        else:
            category = "Neutral"
        
        rationale = self.generate_sentiment_rationale(avg_sentiment, category, major_events)
        
        return {
            'sentiment_score': round(avg_sentiment, 3),
            'sentiment_category': category,
            'rationale': rationale,
            'major_events': major_events[:3]
        }
    
    def generate_sentiment_rationale(self, score, category, major_events):
        """Generate professional sentiment rationale"""
        if category == "Positive":
            base = f"Strong positive sentiment (Score: {score}) driven by favorable market conditions."
        elif category == "Negative":
            base = f"Negative sentiment pressure (Score: {score}) due to concerning developments."
        else:
            base = f"Neutral market sentiment (Score: {score}) with balanced news flow."
        
        if major_events:
            event_desc = " Key factors: " + "; ".join([
                f"{event['headline']} ({event['impact']} impact)"
                for event in major_events
            ])
            base += event_desc
        
        # Add market reaction assessment
        if score > 0.3:
            base += " Expect immediate positive price reaction with low volatility."
        elif score < -0.3:
            base += " Anticipate negative price pressure with increased volatility."
        else:
            base += " Market likely to trade in range with moderate volatility."
        
        return base
    
    def generate_trade_recommendation(self, technical_data, sentiment_data):
        """Generate unified trade recommendation"""
        if technical_data is None:
            return {
                'recommendation': "HOLD",
                'confidence': "Low",
                'entry_price': "N/A",
                'stop_loss': "N/A",
                'target_price': "N/A",
                'final_score': 0,
                'technical_score': 0,
                'sentiment_influence': 'Neutral'
            }
            
        # Technical scoring
        tech_score = 0
        current_price = technical_data['current_price']
        
        # RSI analysis
        rsi = technical_data.get('rsi', 50)
        if rsi and rsi < 30:
            tech_score += 2  # Oversold - bullish
        elif rsi and rsi > 70:
            tech_score -= 2  # Overbought - bearish
        
        # MACD analysis
        macd_hist = technical_data.get('macd_histogram', 0)
        if macd_hist and macd_hist > 0:
            tech_score += 1
        elif macd_hist and macd_hist < 0:
            tech_score -= 1
        
        # Price vs MA
        sma_20 = technical_data.get('sma_20', current_price)
        if current_price > sma_20:
            tech_score += 1
        else:
            tech_score -= 1
        
        # Volume analysis
        volume_ratio = technical_data.get('volume_ratio', 1)
        if volume_ratio > 1.2:
            tech_score += 1  # High volume confirmation
        
        # Sentiment adjustment
        sentiment_score = sentiment_data['sentiment_score']
        sentiment_multiplier = 1 + abs(sentiment_score) * 2
        
        if sentiment_data['sentiment_category'] == 'Positive':
            final_score = tech_score * sentiment_multiplier
        elif sentiment_data['sentiment_category'] == 'Negative':
            final_score = tech_score * (1/sentiment_multiplier)
        else:
            final_score = tech_score
        
        # Generate recommendation
        if final_score >= 3:
            recommendation = "BUY"
            confidence = "High"
        elif final_score >= 1:
            recommendation = "BUY"
            confidence = "Medium"
        elif final_score <= -3:
            recommendation = "SELL"
            confidence = "High"
        elif final_score <= -1:
            recommendation = "SELL"
            confidence = "Medium"
        else:
            recommendation = "HOLD"
            confidence = "Neutral"
        
        # Calculate levels
        if recommendation != "HOLD":
            entry, stop_loss, target = self.calculate_trade_levels(
                recommendation, current_price, technical_data
            )
        else:
            entry = stop_loss = target = "N/A"
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'entry_price': entry,
            'stop_loss': stop_loss,
            'target_price': target,
            'final_score': final_score,
            'technical_score': tech_score,
            'sentiment_influence': sentiment_data['sentiment_category']
        }
    
    def calculate_trade_levels(self, recommendation, current_price, tech_data):
        """Calculate trade entry, stop loss, and target prices"""
        try:
            atr = self.calculate_atr(tech_data['highs'], tech_data['lows'], tech_data['closes'])
            
            if recommendation == "BUY":
                entry = current_price
                stop_loss = current_price - (atr * 1.5)
                target = current_price + (atr * 2.5)
            else:  # SELL
                entry = current_price
                stop_loss = current_price + (atr * 1.5)
                target = current_price - (atr * 2.5)
            
            return round(entry, 2), round(stop_loss, 2), round(target, 2)
        except:
            # Fallback calculation if ATR fails
            volatility = current_price * 0.02  # 2% volatility assumption
            if recommendation == "BUY":
                return round(current_price, 2), round(current_price - volatility * 2, 2), round(current_price + volatility * 3, 2)
            else:
                return round(current_price, 2), round(current_price + volatility * 2, 2), round(current_price - volatility * 3, 2)
    
    def calculate_atr(self, highs, lows, closes, period=14):
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return (max(highs) - min(lows)) * 0.02  # Fallback
        
        try:
            true_ranges = []
            for i in range(1, len(highs)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            return sum(true_ranges[-period:]) / period
        except:
            return (max(highs) - min(lows)) * 0.02

def main():
    st.title("🤖 Stock Analysis Bot")
    st.markdown("""
    Comprehensive technical analysis and sentiment evaluation for stocks with AI-powered trade recommendations.
    """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    
    # Sidebar
    st.sidebar.header("Stock Selection")
    symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()
    
    analyze_button = st.sidebar.button("Analyze Stock", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Popular Symbols:**
    - AAPL (Apple)
    - TSLA (Tesla)
    - MSFT (Microsoft)
    - GOOGL (Google)
    - AMZN (Amazon)
    - NVDA (Nvidia)
    """)
    
    if analyze_button or symbol:
        with st.spinner(f"🔍 Analyzing {symbol}..."):
            
            # Fetch data for all timeframes
            timeframes = ['15min', '1h']
            timeframe_data = {}
            
            for tf in timeframes:
                data = st.session_state.analyzer.get_stock_data(symbol, tf, 100)
                if data is not None:
                    data = st.session_state.analyzer.calculate_technical_indicators(data)
                timeframe_data[tf] = data
            
            # Get sentiment analysis
            sentiment_data = st.session_state.analyzer.get_market_news_sentiment(symbol)
            
            # Use 15min data for primary analysis
            primary_data = timeframe_data['15min']
            if primary_data is None:
                st.error(f"❌ Unable to fetch data for {symbol}. Please check the symbol and try again.")
                return
            
            # Safely get current price
            try:
                current_price = float(primary_data.iloc[-1]['close'])
            except:
                st.error(f"❌ Error getting current price for {symbol}")
                return
            
            # Prepare technical data for recommendation
            tech_data = {
                'current_price': current_price,
                'rsi': st.session_state.analyzer.safe_get(primary_data.iloc[-1], 'rsi', 50),
                'macd_histogram': st.session_state.analyzer.safe_get(primary_data.iloc[-1], 'macd_histogram', 0),
                'sma_20': st.session_state.analyzer.safe_get(primary_data.iloc[-1], 'sma_20', current_price),
                'volume_ratio': st.session_state.analyzer.safe_get(primary_data.iloc[-1], 'volume', 1) / primary_data['volume'].mean() if 'volume' in primary_data else 1,
                'highs': primary_data['high'].tolist(),
                'lows': primary_data['low'].tolist(),
                'closes': primary_data['close'].tolist()
            }
            
            # Generate trade recommendation
            trade_rec = st.session_state.analyzer.generate_trade_recommendation(tech_data, sentiment_data)
            
            # Candlestick patterns
            patterns_15min = st.session_state.analyzer.analyze_candlestick_patterns(timeframe_data['15min'])
            patterns_1h = st.session_state.analyzer.analyze_candlestick_patterns(timeframe_data['1h'])
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                # Color code based on recommendation
                rec_color = "green" if trade_rec['recommendation'] == "BUY" else "red" if trade_rec['recommendation'] == "SELL" else "orange"
                st.markdown(f"<h2 style='color: {rec_color};'>{trade_rec['recommendation']}</h2>", unsafe_allow_html=True)
                st.caption(f"Confidence: {trade_rec['confidence']}")
            
            with col3:
                st.metric("Sentiment Score", f"{sentiment_data['sentiment_score']}", 
                         delta=f"{sentiment_data['sentiment_category']}")
            
            # Trade Recommendation Card
            st.subheader("🎯 Trade Recommendation")
            rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
            
            with rec_col1:
                st.info(f"**Entry Price**\n# ${trade_rec['entry_price']}")
            
            with rec_col2:
                st.error(f"**Stop Loss**\n# ${trade_rec['stop_loss']}")
            
            with rec_col3:
                st.success(f"**Target Price**\n# ${trade_rec['target_price']}")
            
            with rec_col4:
                risk_reward = "1:1.67" if trade_rec['recommendation'] != "HOLD" else "N/A"
                st.warning(f"**Risk/Reward**\n# {risk_reward}")
            
            # Technical Analysis
            st.subheader("📈 Technical Analysis")
            
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.write("**15-minute Chart:**")
                if primary_data is not None:
                    rsi_value = st.session_state.analyzer.safe_get(primary_data.iloc[-1], 'rsi', 'N/A')
                    rsi_comment = ''
                    
                    # Safely format RSI value
                    rsi_display = st.session_state.analyzer.safe_float_format(rsi_value, 'N/A')
                    if rsi_display != 'N/A':
                        rsi_num = float(rsi_value)
                        if rsi_num < 30:
                            rsi_comment = '(Oversold)'
                        elif rsi_num > 70:
                            rsi_comment = '(Overbought)'
                    
                    st.write(f"- RSI: {rsi_display} {rsi_comment}")
                    
                    # Safely format MACD
                    macd_value = st.session_state.analyzer.safe_get(primary_data.iloc[-1], 'macd', 0)
                    macd_display = st.session_state.analyzer.safe_float_format(macd_value, 'N/A')
                    st.write(f"- MACD: {macd_display}")
                    st.write(f"- Pattern: {', '.join(patterns_15min)}")
                
            with tech_col2:
                st.write("**1-hour Chart:**")
                if timeframe_data['1h'] is not None:
                    hourly_rsi = st.session_state.analyzer.safe_get(timeframe_data['1h'].iloc[-1], 'rsi', 'N/A')
                    hourly_rsi_display = st.session_state.analyzer.safe_float_format(hourly_rsi, 'N/A')
                    st.write(f"- RSI: {hourly_rsi_display}")
                    st.write(f"- Pattern: {', '.join(patterns_1h)}")
            
            # Market Sentiment
            st.subheader("📰 Market Sentiment")
            sentiment_col1, sentiment_col2 = st.columns([1, 2])
            
            with sentiment_col1:
                # Sentiment gauge
                score = sentiment_data['sentiment_score']
                if score > 0:
                    st.success(f"Sentiment: {sentiment_data['sentiment_category']}")
                elif score < 0:
                    st.error(f"Sentiment: {sentiment_data['sentiment_category']}")
                else:
                    st.warning(f"Sentiment: {sentiment_data['sentiment_category']}")
                
                st.write(f"Score: {score}")
            
            with sentiment_col2:
                st.write(sentiment_data['rationale'])
                
                if sentiment_data['major_events']:
                    st.write("**Major Events:**")
                    for event in sentiment_data['major_events']:
                        emoji = "📰" if event['impact'] == 'Medium' else "🚨"
                        st.write(f"{emoji} {event['headline']}")
            
            # Additional Information
            with st.expander("Detailed Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Technical Score Breakdown:**")
                    st.write(f"- Final Score: {trade_rec['final_score']:.1f}")
                    st.write(f"- Technical Score: {trade_rec['technical_score']}")
                    st.write(f"- Sentiment Influence: {trade_rec['sentiment_influence']}")
                
                with col2:
                    st.write("**Risk Disclaimer:**")
                    st.warning("""
                    This analysis is for educational purposes only. Always:
                    - Do your own research
                    - Understand the risks
                    - Use proper risk management
                    - Consider consulting a financial advisor
                    """)

if __name__ == '__main__':
    main()
