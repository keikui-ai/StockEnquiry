import os
import subprocess
import sys
from dotenv import load_dotenv

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['python-telegram-bot', 'pandas', 'numpy', 'requests', 'python-dotenv']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_environment():
    """Check if all required environment variables are set"""
    load_dotenv()
    
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TWELVE_DATA_API_KEY', 'ALPHA_VANTAGE_API_KEY']
    missing_vars = []
    
    print("\n🔍 Checking environment variables...")
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}")
        else:
            print(f"❌ {var}")
            missing_vars.append(var)
    
    return missing_vars

def main():
    print("🚀 Stock Analysis Bot Deployment Check")
    print("=" * 40)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    check_dependencies()
    
    # Check environment
    missing_vars = check_environment()
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease create a .env file with the following variables:")
        print("""
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TWELVE_DATA_API_KEY=your_twelve_data_api_key_here  
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
        """)
        return
    
    print("\n✅ All checks passed!")
    print("\n🎯 Starting Telegram Bot...")
    print("💬 Users can now send stock symbols to your bot for analysis!")
    
    # Import and run the bot
    from telegram_bot import main as run_bot
    run_bot()

if __name__ == '__main__':
    main()
