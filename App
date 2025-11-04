import os
import logging
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from datetime import datetime, timedelta
import ta
from transformers import pipeline
from textblob import TextBlob
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = "YOUR_BOT_TOKEN"
TWELVE_DATA_API_KEY = "YOUR_TWELVE_DATA_API_KEY"

# Conversation states
SELECTING_SYMBOL, ANALYZING = range(2)

class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    async def get_stock_data(self, symbol: str, interval: str, periods: int = 100):
        """Fetch stock data from Twelve Data API"""
        try:
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': periods,
                'apikey': TWELVE_DATA_API_KEY,
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'values' not in data:
                logger.error(f"API Error: {data}")
                return None
                
            df = pd.DataFrame(data['values'])
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['open'] = pd.to_numeric(df['open'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            # Fallback to yfinance
            return await self.get_yfinance_data(symbol, interval, periods)
    
    async def get_yfinance_data(self, symbol: str, interval: str, periods: int = 100):
        """Fallback to yfinance if Twelve Data fails"""
        try:
            # Map intervals to yfinance format
            interval_map = {
                '1min': '1m',
                '15min': '15m',
                '1h': '1h'
            }
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='60d', interval=interval_map.get(interval, '15m'))
            
            if len(data) < periods:
                data = ticker.history(period='max', interval=interval_map.get(interval, '15m'))
            
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
            logger.error(f"YFinance error: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        if df is None or len(df) < 20:
            return None
            
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume indicators
        df['volume_sma'] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
        
        return df
    
    def analyze_candlestick_patterns(self, df):
        """Analyze candlestick patterns"""
        if len(df) < 3:
            return "Insufficient data for pattern analysis"
            
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
    
    def is_hammer(self, candle):
        body = abs(candle['close'] - candle['open'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        return lower_wick >= 2 * body and upper_wick <= body * 0.5
    
    def is_shooting_star(self, candle):
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        return upper_wick >= 2 * body and lower_wick <= body * 0.5
    
    def is_bullish_engulfing(self, prev_candle, curr_candle):
        return (prev_candle['close'] < prev_candle['open'] and
                curr_candle['close'] > curr_candle['open'] and
                curr_candle['open'] < prev_candle['close'] and
                curr_candle['close'] > prev_candle['open'])
    
    def is_bearish_engulfing(self, prev_candle, curr_candle):
        return (prev_candle['close'] > prev_candle['open'] and
                curr_candle['close'] < curr_candle['open'] and
                curr_candle['open'] > prev_candle['close'] and
                curr_candle['close'] < prev_candle['open'])
    
    def is_morning_star(self, candle1, candle2, candle3):
        # Simplified morning star pattern
        return (candle1['close'] < candle1['open'] and  # First red candle
                abs(candle2['close'] - candle2['open']) < (candle1['high'] - candle1['low']) * 0.3 and  # Small body
                candle3['close'] > candle3['open'] and  # Green candle
                candle3['close'] > candle1['close'])    # Closes above first candle
    
    def is_evening_star(self, candle1, candle2, candle3):
        # Simplified evening star pattern
        return (candle1['close'] > candle1['open'] and  # First green candle
                abs(candle2['close'] - candle2['open']) < (candle1['high'] - candle1['low']) * 0.3 and  # Small body
                candle3['close'] < candle3['open'] and  # Red candle
                candle3['close'] < candle1['close'])    # Closes below first candle
    
    async def get_market_news_sentiment(self, symbol: str):
        """Fetch and analyze market news sentiment"""
        try:
            # Using Alpha Vantage for news (free tier available)
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': 'YOUR_ALPHA_VANTAGE_API_KEY',  # Get free key from alphavantage.co
                'limit': 10
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'feed' not in data:
                return await self.get_fallback_sentiment(symbol)
            
            return self.analyze_news_sentiment(data['feed'], symbol)
            
        except Exception as e:
            logger.error(f"News sentiment error: {e}")
            return await self.get_fallback_sentiment(symbol)
    
    async def get_fallback_sentiment(self, symbol: str):
        """Fallback sentiment analysis"""
        try:
            # Use Yahoo Finance news as fallback
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return {
                    'sentiment_score': 0.0,
                    'sentiment_category': 'Neutral',
                    'rationale': 'Limited news data available',
                    'major_events': []
                }
            
            headlines = [item['title'] for item in news[:5] if 'title' in item]
            return self.analyze_text_sentiment(headlines, symbol)
            
        except Exception as e:
            logger.error(f"Fallback sentiment error: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_category': 'Neutral',
                'rationale': 'Unable to fetch news data',
                'major_events': []
            }
    
    def analyze_news_sentiment(self, news_feed, symbol):
        """Analyze news sentiment using AI"""
        sentiments = []
        major_events = []
        
        for item in news_feed[:10]:  # Analyze top 10 news items
            title = item.get('title', '')
            summary = item.get('summary', '')
            text = f"{title}. {summary}"
            
            # Use transformer model for sentiment
            try:
                result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
                score = result['score']
                label = result['label']
                
                # Convert to numerical score
                if label == 'POSITIVE':
                    sentiment_score = score
                elif label == 'NEGATIVE':
                    sentiment_score = -score
                else:
                    sentiment_score = 0
                    
                sentiments.append(sentiment_score)
                
                # Track major events
                if abs(sentiment_score) > 0.7:
                    event_type = "Positive" if sentiment_score > 0 else "Negative"
                    major_events.append({
                        'headline': title,
                        'impact': 'High' if abs(sentiment_score) > 0.8 else 'Medium',
                        'type': event_type
                    })
                    
            except Exception as e:
                # Fallback to TextBlob
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                sentiments.append(sentiment_score)
        
        if not sentiments:
            return {
                'sentiment_score': 0.0,
                'sentiment_category': 'Neutral',
                'rationale': 'No analyzable news content',
                'major_events': []
            }
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Categorize sentiment
        if avg_sentiment > 0.1:
            category = "Positive"
        elif avg_sentiment < -0.1:
            category = "Negative"
        else:
            category = "Neutral"
        
        # Generate rationale
        rationale = self.generate_sentiment_rationale(avg_sentiment, category, major_events)
        
        return {
            'sentiment_score': round(avg_sentiment, 3),
            'sentiment_category': category,
            'rationale': rationale,
            'major_events': major_events[:3]  # Top 3 major events
        }
    
    def analyze_text_sentiment(self, headlines, symbol):
        """Analyze sentiment from headlines using TextBlob"""
        sentiments = []
        major_events = []
        
        for headline in headlines:
            blob = TextBlob(headline)
            sentiment_score = blob.sentiment.polarity
            sentiments.append(sentiment_score)
            
            if abs(sentiment_score) > 0.3:
                event_type = "Positive" if sentiment_score > 0 else "Negative"
                major_events.append({
                    'headline': headline,
                    'impact': 'High' if abs(sentiment_score) > 0.5 else 'Medium',
                    'type': event_type
                })
        
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
                f"{event['headline'][:50]}... ({event['impact']} impact)"
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
        # Technical scoring
        tech_score = 0
        current_price = technical_data['current_price']
        
        # RSI analysis
        rsi = technical_data['rsi']
        if rsi < 30:
            tech_score += 2  # Oversold - bullish
        elif rsi > 70:
            tech_score -= 2  # Overbought - bearish
        
        # MACD analysis
        macd_hist = technical_data['macd_histogram']
        if macd_hist > 0:
            tech_score += 1
        else:
            tech_score -= 1
        
        # Price vs MA
        if current_price > technical_data['sma_20']:
            tech_score += 1
        else:
            tech_score -= 1
        
        # Volume analysis
        if technical_data['volume_ratio'] > 1.2:
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
    
    def calculate_atr(self, highs, lows, closes, period=14):
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return (max(highs) - min(lows)) * 0.02  # Fallback
        
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        return sum(true_ranges[-period:]) / period

# Initialize analyzer
analyzer = StockAnalyzer()

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = """
🤖 *Stock Analysis Bot*

I provide comprehensive technical analysis and sentiment evaluation for stocks.

*Features:*
• Multi-timeframe analysis (1min, 15min, 1hour)
• Technical indicators (RSI, MACD, Bollinger Bands)
• Market news sentiment analysis
• AI-powered trade recommendations
• Risk management levels

*How to use:*
Send a stock symbol (e.g., AAPL, TSLA, MSFT) or use /analyze command.

*Example:* `AAPL` or `/analyze AAPL`
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')
    return SELECTING_SYMBOL

async def analyze_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze stock command handler"""
    if context.args:
        symbol = context.args[0].upper()
    else:
        await update.message.reply_text("Please provide a stock symbol. Example: /analyze AAPL")
        return SELECTING_SYMBOL
    
    await update.message.reply_text(f"🔍 Analyzing {symbol}... This may take a moment.")
    
    try:
        # Fetch data for all timeframes
        timeframes = ['1min', '15min', '1h']
        timeframe_data = {}
        
        for tf in timeframes:
            data = await analyzer.get_stock_data(symbol, tf, 100)
            if data is not None:
                data = analyzer.calculate_technical_indicators(data)
            timeframe_data[tf] = data
        
        # Get sentiment analysis
        sentiment_data = await analyzer.get_market_news_sentiment(symbol)
        
        # Generate analysis report
        report = await generate_analysis_report(symbol, timeframe_data, sentiment_data)
        
        await update.message.reply_text(report, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        await update.message.reply_text(f"❌ Error analyzing {symbol}. Please try again later.")
    
    return SELECTING_SYMBOL

async def handle_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stock symbol input"""
    symbol = update.message.text.upper()
    await analyze_stock(update, context)

async def generate_analysis_report(symbol, timeframe_data, sentiment_data):
    """Generate comprehensive analysis report"""
    
    # Use 15min data for primary analysis
    primary_data = timeframe_data['15min']
    if primary_data is None:
        return f"❌ Unable to fetch data for {symbol}. Please check the symbol and try again."
    
    current_price = primary_data.iloc[-1]['close']
    
    # Prepare technical data for recommendation
    tech_data = {
        'current_price': current_price,
        'rsi': primary_data.iloc[-1]['rsi'],
        'macd_histogram': primary_data.iloc[-1]['macd_histogram'],
        'sma_20': primary_data.iloc[-1]['sma_20'],
        'volume_ratio': primary_data.iloc[-1]['volume'] / primary_data['volume'].mean(),
        'highs': primary_data['high'].tolist(),
        'lows': primary_data['low'].tolist(),
        'closes': primary_data['close'].tolist()
    }
    
    # Generate trade recommendation
    trade_rec = analyzer.generate_trade_recommendation(tech_data, sentiment_data)
    
    # Candlestick patterns
    patterns_15min = analyzer.analyze_candlestick_patterns(timeframe_data['15min'])
    patterns_1h = analyzer.analyze_candlestick_patterns(timeframe_data['1h'])
    
    # Build report
    report = f"""
📊 *Analysis Report for {symbol}*
💰 Current Price: ${current_price:.2f}

*📈 TECHNICAL ANALYSIS*

*15-minute Chart:*
• RSI: {primary_data.iloc[-1]['rsi']:.1f} {'(Oversold)' if primary_data.iloc[-1]['rsi'] < 30 else '(Overbought)' if primary_data.iloc[-1]['rsi'] > 70 else ''}
• MACD: {primary_data.iloc[-1]['macd']:.3f}
• Pattern: {', '.join(patterns_15min)}

*1-hour Chart:*
• RSI: {timeframe_data['1h'].iloc[-1]['rsi']:.1f}
• Pattern: {', '.join(patterns_1h)}

*📰 MARKET SENTIMENT*
• Score: {sentiment_data['sentiment_score']} ({sentiment_data['sentiment_category']})
• Assessment: {sentiment_data['rationale']}

*🎯 TRADE RECOMMENDATION*
• *Action:* {trade_rec['recommendation']} ({trade_rec['confidence']} Confidence)
• Entry: ${trade_rec['entry_price']}
• Stop Loss: ${trade_rec['stop_loss']}
• Target: ${trade_rec['target_price']}

*Risk/Reward Ratio:* ~1:1.67

*Note:* This is AI-generated analysis. Always do your own research and manage risk appropriately.
    """
    
    return report

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the conversation"""
    await update.message.reply_text("Analysis cancelled.")
    return ConversationHandler.END

def main():
    """Start the bot"""
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start), CommandHandler('analyze', analyze_stock)],
        states={
            SELECTING_SYMBOL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_symbol)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    
    application.add_handler(conv_handler)
    
    # Start bot
    print("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()
