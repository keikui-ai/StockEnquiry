import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
TWELVE_DATA_API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", os.getenv("TWELVE_DATA_API_KEY", "demo"))
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", "demo"))

class StockAnalyzer:
    def __init__(self):
        self.twelve_data_url = "https://api.twelvedata.com"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.analysis_history = []
    
    def get_technical_data(self, symbol: str, interval: str, periods: int = 100):
        """Get technical data from Twelve Data API"""
        try:
            endpoint = f"{self.twelve_data_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': periods,
                'apikey': TWELVE_DATA_API_KEY,
                'format': 'JSON'
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            data = response.json()
            
            if 'values' not in data:
                st.error(f"API Error: {data.get('message', 'Unknown error')}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df = df.iloc[::-1].reset_index(drop=True)
            
            # Convert columns to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        if df is None or len(df) < 20:
            return df
            
        try:
            df = df.copy()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            return df.fillna(method='bfill')
            
        except Exception as e:
            st.error(f"Error calculating indicators: {e}")
            return df
    
    def get_news_sentiment(self, symbol: str):
        """Get news sentiment from Alpha Vantage"""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': ALPHA_VANTAGE_API_KEY,
                'limit': 50
            }
            
            response = requests.get(self.alpha_vantage_url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' not in data:
                st.warning("Using demo sentiment data (API limit may be reached)")
                return self.get_demo_sentiment(symbol)
            
            return self.analyze_sentiment_data(data['feed'], symbol)
            
        except Exception as e:
            st.error(f"Error fetching news sentiment: {e}")
            return self.get_demo_sentiment(symbol)
    
    def analyze_sentiment_data(self, news_feed, symbol):
        """Analyze sentiment from news feed"""
        try:
            sentiments = []
            major_events = []
            
            for item in news_feed[:20]:
                try:
                    ticker_sentiments = item.get('ticker_sentiment', [])
                    for ticker_sent in ticker_sentiments:
                        if ticker_sent.get('ticker') == symbol:
                            relevance_score = float(ticker_sent.get('relevance_score', '0'))
                            ticker_sentiment_score = float(ticker_sent.get('ticker_sentiment_score', '0'))
                            ticker_sentiment_label = ticker_sent.get('ticker_sentiment_label', 'Neutral')
                            
                            if relevance_score > 0.7:
                                # Convert to numerical score
                                if ticker_sentiment_label == 'Bullish':
                                    sentiment_score = 0.3 + min(ticker_sentiment_score, 0.7)
                                elif ticker_sentiment_label == 'Bearish':
                                    sentiment_score = -0.3 - min(abs(ticker_sentiment_score), 0.7)
                                elif ticker_sentiment_label == 'Neutral':
                                    sentiment_score = ticker_sentiment_score * 0.5
                                else:
                                    sentiment_score = 0
                                
                                sentiments.append(sentiment_score)
                                
                                # Track major events
                                if abs(sentiment_score) > 0.2:
                                    event_type = "Bullish" if sentiment_score > 0 else "Bearish"
                                    major_events.append({
                                        'headline': item.get('title', '')[:100] + "..." if len(item.get('title', '')) > 100 else item.get('title', ''),
                                        'impact': 'High' if abs(sentiment_score) > 0.4 else 'Medium',
                                        'sentiment_score': sentiment_score,
                                        'source': item.get('source', 'Unknown'),
                                        'time_published': item.get('time_published', '')
                                    })
                except Exception as e:
                    continue
            
            if not sentiments:
                return self.get_demo_sentiment(symbol)
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Categorize sentiment
            if avg_sentiment > 0.1:
                category = "Bullish"
                color = "🟢"
            elif avg_sentiment < -0.1:
                category = "Bearish"
                color = "🔴"
            else:
                category = "Neutral"
                color = "🟡"
            
            rationale = self.generate_sentiment_rationale(avg_sentiment, category, major_events)
            
            return {
                'sentiment_score': round(avg_sentiment, 3),
                'sentiment_category': category,
                'sentiment_color': color,
                'rationale': rationale,
                'major_events': sorted(major_events, key=lambda x: abs(x['sentiment_score']), reverse=True)[:5],
                'total_news_analyzed': len(sentiments)
            }
            
        except Exception as e:
            st.error(f"Error analyzing sentiment: {e}")
            return self.get_demo_sentiment(symbol)
    
    def get_demo_sentiment(self, symbol):
        """Demo sentiment data for testing"""
        demo_score = np.random.uniform(-0.5, 0.5)
        if demo_score > 0.1:
            category = "Bullish"
            color = "🟢"
        elif demo_score < -0.1:
            category = "Bearish"
            color = "🔴"
        else:
            category = "Neutral"
            color = "🟡"
        
        return {
            'sentiment_score': round(demo_score, 3),
            'sentiment_category': category,
            'sentiment_color': color,
            'rationale': f"Demo sentiment data for {symbol}. Real data requires Alpha Vantage API key.",
            'major_events': [],
            'total_news_analyzed': 0
        }
    
    def generate_sentiment_rationale(self, score, category, major_events):
        """Generate sentiment rationale"""
        if category == "Bullish":
            base = f"Positive market sentiment (Score: {score}). Favorable conditions detected."
        elif category == "Bearish":
            base = f"Negative sentiment pressure (Score: {score}). Caution advised."
        else:
            base = f"Neutral market sentiment (Score: {score}). Balanced outlook."
        
        if major_events:
            high_impact = [e for e in major_events if e['impact'] == 'High']
            if high_impact:
                base += f" {len(high_impact)} high-impact news events driving sentiment."
        
        return base
    
    def analyze_price_action(self, df_1min, df_15min, df_1h):
        """Analyze price action across timeframes"""
        analyses = []
        
        try:
            # 1-minute analysis
            if df_1min is not None and len(df_1min) >= 10:
                recent = df_1min.tail(10)
                price_change = ((recent.iloc[-1]['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']) * 100
                volatility = (recent['high'] - recent['low']).mean() / recent['close'].mean() * 100
                analyses.append(f"1min: {price_change:+.2f}% (Vol: {volatility:.1f}%)")
            
            # 15-minute analysis
            if df_15min is not None and len(df_15min) >= 10:
                recent = df_15min.tail(10)
                price_change = ((recent.iloc[-1]['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']) * 100
                volume_trend = "📈" if recent['volume'].iloc[-1] > recent['volume'].mean() else "📉"
                analyses.append(f"15min: {price_change:+.2f}% {volume_trend}")
            
            # 1-hour analysis
            if df_1h is not None and len(df_1h) >= 5:
                recent = df_1h.tail(5)
                price_change = ((recent.iloc[-1]['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']) * 100
                analyses.append(f"1h: {price_change:+.2f}%")
                
        except Exception as e:
            analyses.append("Price action analysis unavailable")
        
        return " | ".join(analyses) if analyses else "No trend data"
    
    def generate_trade_recommendation(self, technical_data, sentiment_data, current_price):
        """Generate trade recommendation"""
        try:
            # Technical scoring
            tech_score = 0
            
            # RSI analysis
            rsi = technical_data.get('rsi', 50)
            if rsi < 30:
                tech_score += 2  # Oversold - bullish
            elif rsi > 70:
                tech_score -= 2  # Overbought - bearish
            
            # MACD analysis
            macd_hist = technical_data.get('macd_histogram', 0)
            if macd_hist > 0:
                tech_score += 1
            else:
                tech_score -= 1
            
            # Price vs Moving Averages
            sma_20 = technical_data.get('sma_20', current_price)
            if current_price > sma_20:
                tech_score += 1
            else:
                tech_score -= 1
            
            # Volume confirmation
            volume_ratio = technical_data.get('volume_ratio', 1)
            if volume_ratio > 1.2:
                tech_score += 1
            
            # Sentiment adjustment
            sentiment_score = sentiment_data['sentiment_score']
            sentiment_weight = 2
            
            if sentiment_data['sentiment_category'] == 'Bullish':
                final_score = tech_score + (sentiment_score * sentiment_weight)
            elif sentiment_data['sentiment_category'] == 'Bearish':
                final_score = tech_score - (abs(sentiment_score) * sentiment_weight)
            else:
                final_score = tech_score
            
            # Generate recommendation
            if final_score >= 3:
                recommendation = "STRONG BUY"
                confidence = "High"
                color = "green"
                emoji = "🟢"
            elif final_score >= 1:
                recommendation = "BUY"
                confidence = "Medium"
                color = "lightgreen"
                emoji = "🟡"
            elif final_score <= -3:
                recommendation = "STRONG SELL"
                confidence = "High"
                color = "red"
                emoji = "🔴"
            elif final_score <= -1:
                recommendation = "SELL"
                confidence = "Medium"
                color = "lightcoral"
                emoji = "🟠"
            else:
                recommendation = "HOLD"
                confidence = "Neutral"
                color = "orange"
                emoji = "⚪"
            
            # Calculate trade levels
            if recommendation not in ["HOLD", "HOLD"]:
                entry, stop_loss, target = self.calculate_trade_levels(
                    recommendation, current_price, technical_data
                )
                risk_reward = round((target - entry) / (entry - stop_loss), 2) if entry > stop_loss else round((entry - target) / (stop_loss - entry), 2)
            else:
                entry = stop_loss = target = current_price
                risk_reward = "N/A"
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'color': color,
                'emoji': emoji,
                'entry_price': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'target_price': round(target, 2),
                'risk_reward': risk_reward,
                'final_score': round(final_score, 2),
                'technical_score': tech_score
            }
            
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
            return {
                'recommendation': "HOLD",
                'confidence': "Low",
                'color': "gray",
                'emoji': "⚪",
                'entry_price': "N/A",
                'stop_loss': "N/A",
                'target_price': "N/A",
                'risk_reward': "N/A",
                'final_score': 0,
                'technical_score': 0
            }
    
    def calculate_trade_levels(self, recommendation, current_price, tech_data):
        """Calculate trade levels"""
        try:
            # Use ATR-like calculation
            recent_high = tech_data.get('recent_high', current_price * 1.02)
            recent_low = tech_data.get('recent_low', current_price * 0.98)
            volatility = (recent_high - recent_low) / current_price
            
            risk_multiplier = 1 + min(volatility * 10, 2)
            
            if "BUY" in recommendation:
                entry = current_price
                stop_loss = current_price * (1 - 0.015 * risk_multiplier)  # 1.5% stop
                target = current_price * (1 + 0.025 * risk_multiplier)     # 2.5% target
            else:  # SELL
                entry = current_price
                stop_loss = current_price * (1 + 0.015 * risk_multiplier)  # 1.5% stop
                target = current_price * (1 - 0.025 * risk_multiplier)     # 2.5% target
            
            return entry, stop_loss, target
            
        except:
            # Fallback
            if "BUY" in recommendation:
                return current_price, current_price * 0.985, current_price * 1.025
            else:
                return current_price, current_price * 1.015, current_price * 0.975
    
    def create_candlestick_chart(self, df, title):
        """Create interactive candlestick chart"""
        if df is None or len(df) < 10:
            return None
            
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{title} - Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['datetime'], y=df['sma_20'], line=dict(color='orange', width=1), name='SMA 20'),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df['datetime'], y=df['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{title} Chart",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True
        )
        
        return fig

def main():
    st.title("🤖 Stock Analysis Pro")
    st.markdown("""
    Professional stock analysis with real-time technical indicators and market sentiment.
    Get AI-powered trade recommendations with entry, stop loss, and target prices.
    """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    
    # Sidebar
    st.sidebar.header("Stock Analysis")
    
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        analyze_btn = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)
    with col2:
        refresh_btn = st.button("🔄 Refresh", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Popular Symbols:**
    - AAPL (Apple)
    - TSLA (Tesla) 
    - MSFT (Microsoft)
    - GOOGL (Google)
    - AMZN (Amazon)
    - NVDA (Nvidia)
    - META (Meta)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if analyze_btn or refresh_btn or symbol:
        with st.spinner(f"🔍 Analyzing {symbol}..."):
            
            # Get technical data for all timeframes
            timeframes = ['1min', '15min', '1h']
            technical_data = {}
            
            for timeframe in timeframes:
                data = st.session_state.analyzer.get_technical_data(symbol, timeframe, 100)
                if data is not None:
                    data = st.session_state.analyzer.calculate_technical_indicators(data)
                technical_data[timeframe] = data
            
            # Get sentiment data
            sentiment_data = st.session_state.analyzer.get_news_sentiment(symbol)
            
            # Use 15min as primary for analysis
            primary_data = technical_data['15min']
            if primary_data is None or len(primary_data) == 0:
                st.error(f"❌ No data available for {symbol}. Please check the symbol and try again.")
                return
            
            current_price = primary_data.iloc[-1]['close']
            
            # Prepare technical summary
            tech_summary = {
                'current_price': current_price,
                'rsi': primary_data.iloc[-1].get('rsi', 50),
                'macd_histogram': primary_data.iloc[-1].get('macd_histogram', 0),
                'sma_20': primary_data.iloc[-1].get('sma_20', current_price),
                'volume_ratio': primary_data.iloc[-1].get('volume', 1) / primary_data['volume'].mean() if 'volume' in primary_data else 1,
                'recent_high': primary_data['high'].tail(20).max(),
                'recent_low': primary_data['low'].tail(20).min()
            }
            
            # Generate trade recommendation
            trade_rec = st.session_state.analyzer.generate_trade_recommendation(tech_summary, sentiment_data, current_price)
            
            # Price action analysis
            price_action = st.session_state.analyzer.analyze_price_action(
                technical_data.get('1min'),
                technical_data.get('15min'),
                technical_data.get('1h')
            )
            
            # Display results
            display_analysis_results(symbol, current_price, trade_rec, tech_summary, sentiment_data, price_action, technical_data)
            
            # Store in history
            st.session_state.analyzer.analysis_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'recommendation': trade_rec['recommendation'],
                'price': current_price
            })

def display_analysis_results(symbol, current_price, trade_rec, tech_summary, sentiment_data, price_action, technical_data):
    """Display the analysis results in Streamlit"""
    
    # Header with current price and recommendation
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Sentiment", f"{sentiment_data['sentiment_color']} {sentiment_data['sentiment_category']}")
    
    with col3:
        st.metric("RSI", f"{tech_summary['rsi']:.1f}")
    
    # Trade Recommendation Card
    st.markdown("---")
    
    rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)
    
    with rec_col1:
        st.markdown(f"<h2 style='color: {trade_rec['color']}; text-align: center;'>{trade_rec['emoji']} {trade_rec['recommendation']}</h2>", 
                   unsafe_allow_html=True)
        st.caption(f"Confidence: {trade_rec['confidence']}")
    
    with rec_col2:
        st.info(f"**Entry Price**\n# ${trade_rec['entry_price']}")
    
    with rec_col3:
        st.error(f"**Stop Loss**\n# ${trade_rec['stop_loss']}")
    
    with rec_col4:
        st.success(f"**Target Price**\n# ${trade_rec['target_price']}")
    
    # Technical Analysis
    st.markdown("---")
    st.subheader("📈 Technical Analysis")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.metric("RSI (14)", f"{tech_summary['rsi']:.1f}", 
                 delta="Oversold" if tech_summary['rsi'] < 30 else "Overbought" if tech_summary['rsi'] > 70 else "Neutral")
    
    with tech_col2:
        macd_status = "Bullish" if tech_summary['macd_histogram'] > 0 else "Bearish"
        st.metric("MACD", f"{tech_summary['macd_histogram']:.4f}", delta=macd_status)
    
    with tech_col3:
        trend_status = "Bullish" if current_price > tech_summary['sma_20'] else "Bearish"
        st.metric("Trend vs SMA20", trend_status)
    
    st.write(f"**Price Action:** {price_action}")
    
    # Market Sentiment
    st.markdown("---")
    st.subheader("📰 Market Sentiment")
    
    sent_col1, sent_col2 = st.columns([1, 2])
    
    with sent_col1:
        st.metric("Sentiment Score", f"{sentiment_data['sentiment_score']}")
        st.metric("News Analyzed", sentiment_data['total_news_analyzed'])
    
    with sent_col2:
        st.write(sentiment_data['rationale'])
        
        if sentiment_data['major_events']:
            st.write("**Major News Events:**")
            for event in sentiment_data['major_events']:
                emoji = "🚨" if event['impact'] == 'High' else "📰"
                sentiment_emoji = "🟢" if event['sentiment_score'] > 0 else "🔴"
                st.write(f"{emoji} {sentiment_emoji} {event['headline']}")
    
    # Charts
    st.markdown("---")
    st.subheader("📊 Price Charts")
    
    chart_tabs = st.tabs(["15-Minute Chart", "1-Hour Chart", "Technical Indicators"])
    
    with chart_tabs[0]:
        if technical_data['15min'] is not None:
            fig_15min = st.session_state.analyzer.create_candlestick_chart(technical_data['15min'], "15-Minute")
            if fig_15min:
                st.plotly_chart(fig_15min, use_container_width=True)
    
    with chart_tabs[1]:
        if technical_data['1h'] is not None:
            fig_1h = st.session_state.analyzer.create_candlestick_chart(technical_data['1h'], "1-Hour")
            if fig_1h:
                st.plotly_chart(fig_1h, use_container_width=True)
    
    with chart_tabs[2]:
        display_technical_indicators(technical_data['15min'])
    
    # Risk Disclaimer
    st.markdown("---")
    st.warning("""
    **Risk Disclaimer:** This analysis is for educational purposes only. 
    Always conduct your own research, understand the risks, and consider consulting 
    a financial advisor before making investment decisions.
    """)

def display_technical_indicators(df):
    """Display additional technical indicators"""
    if df is None or len(df) < 20:
        st.info("Insufficient data for detailed technical indicators")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Bollinger Band Position
        current_low = df.iloc[-1]['low']
        current_high = df.iloc[-1]['high']
        bb_upper = df.iloc[-1].get('bb_upper', current_high)
        bb_lower = df.iloc[-1].get('bb_lower', current_low)
        
        bb_position = (df.iloc[-1]['close'] - bb_lower) / (bb_upper - bb_lower) * 100
        st.metric("BB Position", f"{bb_position:.1f}%")
    
    with col2:
        # Volume Analysis
        volume_ratio = df.iloc[-1]['volume'] / df['volume'].tail(20).mean()
        st.metric("Volume vs Avg", f"{volume_ratio:.2f}x")
    
    with col3:
        # Price Change
        price_change = ((df.iloc[-1]['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close']) * 100
        st.metric("5-period Change", f"{price_change:+.2f}%")
    
    with col4:
        # Volatility
        volatility = (df['high'].tail(20) - df['low'].tail(20)).mean() / df['close'].tail(20).mean() * 100
        st.metric("Avg Volatility", f"{volatility:.2f}%")

if __name__ == '__main__':
    main()
