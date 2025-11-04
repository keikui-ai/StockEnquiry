import os
import logging
import pandas as pd
import numpy as np
import requests
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import asyncio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# API Configuration
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY', 'your_twelve_data_api_key')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'your_alpha_vantage_api_key')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token')

# Conversation states
SELECTING_SYMBOL = 1

class StockAnalysisBot:
    def __init__(self):
        self.twelve_data_url = "https://api.twelvedata.com"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
    
    async def get_technical_data(self, symbol: str, interval: str, periods: int = 100):
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
                logger.error(f"Twelve Data API Error: {data}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
            
            # Convert columns to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching technical data: {e}")
            return None
    
    async def get_news_sentiment(self, symbol: str):
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
                logger.error(f"Alpha Vantage API Error: {data}")
                return self.get_fallback_sentiment()
            
            return self.analyze_sentiment_data(data['feed'], symbol)
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return self.get_fallback_sentiment()
    
    def analyze_sentiment_data(self, news_feed, symbol):
        """Analyze sentiment from Alpha Vantage news feed"""
        try:
            sentiments = []
            major_events = []
            
            for item in news_feed[:20]:  # Analyze top 20 news items
                try:
                    # Get sentiment from Alpha Vantage if available
                    ticker_sentiments = item.get('ticker_sentiment', [])
                    for ticker_sent in ticker_sentiments:
                        if ticker_sent.get('ticker') == symbol:
                            relevance_score = float(ticker_sent.get('relevance_score', '0'))
                            ticker_sentiment_score = float(ticker_sent.get('ticker_sentiment_score', '0'))
                            ticker_sentiment_label = ticker_sent.get('ticker_sentiment_label', 'Neutral')
                            
                            if relevance_score > 0.7:  # Only consider highly relevant news
                                # Convert sentiment label to numerical score
                                if ticker_sentiment_label == 'Bullish':
                                    sentiment_score = 0.5 + (ticker_sentiment_score * 0.5)
                                elif ticker_sentiment_label == 'Bearish':
                                    sentiment_score = -0.5 - (abs(ticker_sentiment_score) * 0.5)
                                elif ticker_sentiment_label == 'Neutral':
                                    sentiment_score = ticker_sentiment_score
                                else:
                                    sentiment_score = 0
                                
                                sentiments.append(sentiment_score)
                                
                                # Track major events
                                if abs(sentiment_score) > 0.3:
                                    event_type = "Positive" if sentiment_score > 0 else "Negative"
                                    major_events.append({
                                        'headline': item.get('title', '')[:80] + "..." if len(item.get('title', '')) > 80 else item.get('title', ''),
                                        'impact': 'High' if abs(sentiment_score) > 0.6 else 'Medium',
                                        'sentiment_score': sentiment_score,
                                        'source': item.get('source', 'Unknown')
                                    })
                except Exception as e:
                    continue
            
            if not sentiments:
                return self.get_fallback_sentiment()
            
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
                'major_events': major_events[:3],  # Top 3 major events
                'total_news_analyzed': len(sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment data: {e}")
            return self.get_fallback_sentiment()
    
    def get_fallback_sentiment(self):
        """Fallback when sentiment analysis fails"""
        return {
            'sentiment_score': 0.0,
            'sentiment_category': 'Neutral',
            'rationale': 'Limited news data available. Market sentiment appears neutral.',
            'major_events': [],
            'total_news_analyzed': 0
        }
    
    def generate_sentiment_rationale(self, score, category, major_events):
        """Generate professional sentiment rationale"""
        if category == "Positive":
            base = f"Bullish sentiment detected (Score: {score}). Market outlook is favorable."
        elif category == "Negative":
            base = f"Bearish sentiment pressure (Score: {score}). Caution advised."
        else:
            base = f"Neutral market sentiment (Score: {score}). Balanced news flow."
        
        if major_events:
            high_impact_events = [e for e in major_events if e['impact'] == 'High']
            if high_impact_events:
                base += " Key drivers: " + "; ".join([e['headline'] for e in high_impact_events[:2]])
        
        # Add market reaction assessment
        if score > 0.3:
            base += " Expect positive momentum with moderate volatility."
        elif score < -0.3:
            base += " Anticipate selling pressure with elevated volatility."
        else:
            base += " Market likely to trade in consolidation pattern."
        
        return base
    
    def calculate_technical_indicators(self, df):
        """Calculate basic technical indicators"""
        if df is None or len(df) < 5:
            return None
            
        try:
            df = df.copy()
            
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=min(20, len(df))).mean()
            df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()
            
            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD calculation
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            return df.fillna(method='bfill')
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_price_action(self, df_1min, df_15min, df_1h):
        """Analyze price action across timeframes"""
        if any(df is None for df in [df_1min, df_15min, df_1h]):
            return "Insufficient data for complete analysis"
        
        try:
            analyses = []
            
            # 1-minute analysis
            if len(df_1min) >= 5:
                recent_1min = df_1min.tail(5)
                price_change_1min = ((recent_1min.iloc[-1]['close'] - recent_1min.iloc[0]['close']) / recent_1min.iloc[0]['close']) * 100
                analyses.append(f"1min: {price_change_1min:+.2f}%")
            
            # 15-minute analysis
            if len(df_15min) >= 10:
                recent_15min = df_15min.tail(10)
                price_change_15min = ((recent_15min.iloc[-1]['close'] - recent_15min.iloc[0]['close']) / recent_15min.iloc[0]['close']) * 100
                avg_volume_15min = recent_15min['volume'].mean()
                analyses.append(f"15min: {price_change_15min:+.2f}%")
            
            # 1-hour analysis
            if len(df_1h) >= 5:
                recent_1h = df_1h.tail(5)
                price_change_1h = ((recent_1h.iloc[-1]['close'] - recent_1h.iloc[0]['close']) / recent_1h.iloc[0]['close']) * 100
                analyses.append(f"1h: {price_change_1h:+.2f}%")
            
            return " | ".join(analyses) if analyses else "No trend data"
            
        except Exception as e:
            return f"Price action analysis error: {str(e)}"
    
    def generate_trade_recommendation(self, technical_data, sentiment_data, current_price):
        """Generate unified trade recommendation"""
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
            macd = technical_data.get('macd', 0)
            macd_signal = technical_data.get('macd_signal', 0)
            if macd > macd_signal:
                tech_score += 1  # Bullish crossover
            else:
                tech_score -= 1  # Bearish crossover
            
            # Price vs Moving Average
            sma_20 = technical_data.get('sma_20', current_price)
            if current_price > sma_20:
                tech_score += 1
            else:
                tech_score -= 1
            
            # Sentiment adjustment
            sentiment_score = sentiment_data['sentiment_score']
            sentiment_weight = 2  # Give sentiment significant weight
            
            if sentiment_data['sentiment_category'] == 'Positive':
                final_score = tech_score + (sentiment_score * sentiment_weight)
            elif sentiment_data['sentiment_category'] == 'Negative':
                final_score = tech_score - (abs(sentiment_score) * sentiment_weight)
            else:
                final_score = tech_score
            
            # Generate recommendation
            if final_score >= 3:
                recommendation = "BUY"
                confidence = "High"
                color = "🟢"
            elif final_score >= 1:
                recommendation = "BUY"
                confidence = "Medium"
                color = "🟡"
            elif final_score <= -3:
                recommendation = "SELL"
                confidence = "High"
                color = "🔴"
            elif final_score <= -1:
                recommendation = "SELL"
                confidence = "Medium"
                color = "🟠"
            else:
                recommendation = "HOLD"
                confidence = "Neutral"
                color = "⚪"
            
            # Calculate trade levels
            if recommendation != "HOLD":
                entry, stop_loss, target = self.calculate_trade_levels(
                    recommendation, current_price, technical_data
                )
            else:
                entry = stop_loss = target = current_price
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'color': color,
                'entry_price': round(entry, 2),
                'stop_loss': round(stop_loss, 2),
                'target_price': round(target, 2),
                'final_score': round(final_score, 2),
                'technical_score': tech_score
            }
            
        except Exception as e:
            logger.error(f"Error generating trade recommendation: {e}")
            return {
                'recommendation': "HOLD",
                'confidence': "Low",
                'color': "⚪",
                'entry_price': "N/A",
                'stop_loss': "N/A",
                'target_price': "N/A",
                'final_score': 0,
                'technical_score': 0
            }
    
    def calculate_trade_levels(self, recommendation, current_price, tech_data):
        """Calculate trade entry, stop loss, and target prices"""
        try:
            # Use volatility based on recent price range
            recent_high = tech_data.get('recent_high', current_price * 1.02)
            recent_low = tech_data.get('recent_low', current_price * 0.98)
            volatility = (recent_high - recent_low) / current_price
            
            # Adjust risk based on volatility
            risk_multiplier = 1 + min(volatility * 10, 2)  # Cap at 3x
            
            if recommendation == "BUY":
                entry = current_price
                stop_loss = current_price * (1 - 0.02 * risk_multiplier)  # 2% stop loss base
                target = current_price * (1 + 0.035 * risk_multiplier)    # 3.5% target base
            else:  # SELL
                entry = current_price
                stop_loss = current_price * (1 + 0.02 * risk_multiplier)  # 2% stop loss base
                target = current_price * (1 - 0.035 * risk_multiplier)    # 3.5% target base
            
            return entry, stop_loss, target
            
        except Exception as e:
            # Fallback calculation
            if recommendation == "BUY":
                return current_price, current_price * 0.98, current_price * 1.035
            else:
                return current_price, current_price * 1.02, current_price * 0.965

# Initialize bot
bot = StockAnalysisBot()

# Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = """
🤖 *Stock Analysis Bot*

I provide professional stock analysis using:
• 📊 Twelve Data API for technical charts (1min, 15min, 1h)
• 📰 Alpha Vantage for market news sentiment
• 🎯 AI-powered trade recommendations

*How to use:*
Simply send me a stock symbol!

*Examples:*
`AAPL` - Apple Inc.
`TSLA` - Tesla Inc.
`MSFT` - Microsoft
`GOOGL` - Google
`NVDA` - Nvidia

I'll analyze technicals, sentiment, and provide trade recommendations!
    """
    await update.message.reply_text(welcome_text, parse_mode='Markdown')
    return SELECTING_SYMBOL

async def handle_stock_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle stock symbol input"""
    symbol = update.message.text.upper().strip()
    
    # Validate symbol format (basic check)
    if len(symbol) < 1 or len(symbol) > 5:
        await update.message.reply_text("❌ Please enter a valid stock symbol (1-5 characters)")
        return SELECTING_SYMBOL
    
    await update.message.reply_text(f"🔍 Analyzing *{symbol}*...\n\nFetching data from:\n• Twelve Data (Technical Charts)\n• Alpha Vantage (News Sentiment)", parse_mode='Markdown')
    
    try:
        # Get technical data for all timeframes
        timeframes = ['1min', '15min', '1h']
        technical_data = {}
        
        for timeframe in timeframes:
            data = await bot.get_technical_data(symbol, timeframe, 100)
            if data is not None and len(data) > 0:
                technical_data[timeframe] = bot.calculate_technical_indicators(data)
            else:
                technical_data[timeframe] = None
        
        # Get news sentiment
        sentiment_data = await bot.get_news_sentiment(symbol)
        
        # Generate analysis report
        report = await generate_analysis_report(symbol, technical_data, sentiment_data)
        
        await update.message.reply_text(report, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        await update.message.reply_text(f"❌ Error analyzing {symbol}. Please check the symbol and try again.")
    
    return SELECTING_SYMBOL

async def generate_analysis_report(symbol, technical_data, sentiment_data):
    """Generate comprehensive analysis report for Telegram"""
    
    # Use 15min data as primary
    primary_data = technical_data.get('15min')
    if primary_data is None or len(primary_data) == 0:
        return f"❌ No data available for {symbol}. Please check the symbol exists."
    
    current_price = primary_data.iloc[-1]['close']
    
    # Prepare technical summary
    tech_summary = {
        'current_price': current_price,
        'rsi': primary_data.iloc[-1].get('rsi', 50),
        'macd': primary_data.iloc[-1].get('macd', 0),
        'sma_20': primary_data.iloc[-1].get('sma_20', current_price),
        'recent_high': primary_data['high'].tail(20).max(),
        'recent_low': primary_data['low'].tail(20).min()
    }
    
    # Generate trade recommendation
    trade_rec = bot.generate_trade_recommendation(tech_summary, sentiment_data, current_price)
    
    # Price action analysis
    price_action = bot.analyze_price_action(
        technical_data.get('1min'),
        technical_data.get('15min'), 
        technical_data.get('1h')
    )
    
    # Build the report
    report = f"""
📊 *STOCK ANALYSIS REPORT*
*Symbol:* {symbol}
*Current Price:* ${current_price:.2f}

{trade_rec['color']} *TRADE RECOMMENDATION*
*Action:* {trade_rec['recommendation']} ({trade_rec['confidence']} Confidence)
*Entry:* ${trade_rec['entry_price']}
*Stop Loss:* ${trade_rec['stop_loss']}
*Target:* ${trade_rec['target_price']}
*Risk/Reward:* ~1:1.75

📈 *TECHNICAL ANALYSIS*
*RSI (14):* {tech_summary['rsi']:.1f} {'(Oversold)' if tech_summary['rsi'] < 30 else '(Overbought)' if tech_summary['rsi'] > 70 else ''}
*MACD:* {tech_summary['macd']:.4f}
*Price vs SMA20:* {'Above' if current_price > tech_summary['sma_20'] else 'Below'}
*Price Action:* {price_action}

📰 *MARKET SENTIMENT*
*Score:* {sentiment_data['sentiment_score']} ({sentiment_data['sentiment_category']})
*News Analyzed:* {sentiment_data['total_news_analyzed']}
*Assessment:* {sentiment_data['rationale']}
"""

    # Add major events if available
    if sentiment_data['major_events']:
        report += "\n*🚨 MAJOR EVENTS:*\n"
        for event in sentiment_data['major_events']:
            report += f"• {event['headline']} ({event['impact']} Impact)\n"

    report += f"\n*Generated:* {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    report += "\n\n_⚠️ Disclaimer: This is AI-generated analysis. Always do your own research and consider consulting a financial advisor._"

    return report

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command handler"""
    help_text = """
*Available Commands:*
/start - Start the bot and see instructions
/analyze [SYMBOL] - Analyze a specific stock
/help - Show this help message

*Or simply send a stock symbol like:*
AAPL, TSLA, MSFT, GOOGL, NVDA, etc.

*Data Sources:*
• Technical Charts: Twelve Data API
• News Sentiment: Alpha Vantage API
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /analyze command with symbol"""
    if context.args:
        symbol = context.args[0].upper().strip()
        await handle_stock_symbol(update, context)
    else:
        await update.message.reply_text("Please provide a stock symbol. Example: /analyze AAPL")

def setup_telegram_bot():
    """Setup and return the Telegram bot application"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_stock_symbol))
    
    return application

def main():
    """Start the Telegram bot"""
    print("🤖 Starting Telegram Stock Analysis Bot...")
    
    # Check environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TWELVE_DATA_API_KEY', 'ALPHA_VANTAGE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the bot.")
        return
    
    application = setup_telegram_bot()
    
    print("✅ Bot is running...")
    print("📊 Data Sources: Twelve Data API + Alpha Vantage API")
    print("💬 Send stock symbols to the bot for analysis!")
    
    # Start polling
    application.run_polling()

if __name__ == '__main__':
    main()
