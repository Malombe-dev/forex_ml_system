#!/usr/bin/env python3
"""
Telegram Signal Pusher for Forex Predictor
Sends formatted trading signals to Telegram bot
"""

import json
import requests
import subprocess
from datetime import datetime
from typing import Dict, List

class TelegramSignalPusher:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, text: str, parse_mode: str = "Markdown"):
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.json()
        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return None
    
    def format_signal(self, signal: Dict) -> str:
        """Format signal as Telegram message"""
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        refined = "🎯" if signal.get('5min_refined') else ""
        cont = "🔄" if signal.get('trade_type') == 'continuation' else "✨"
        
        # Status indicator
        status = signal.get('entry_status', 'ready')
        if status == 'touched_wait_next':
            status_msg = "⏳ *WAIT FOR NEXT CANDLE*"
        elif status == 'pending':
            status_msg = "⚠️ *PRICE APPROACHING*"
        else:
            status_msg = "✅ *READY FOR ENTRY*"
        
        msg = f"""
{direction_emoji} *FOREX SIGNAL* {refined} {cont}

{status_msg}

*{signal['symbol']} • {signal['direction']}*
Probability: `{signal['probability']}%`
Type: `{signal.get('trade_type', 'fresh_zone').replace('_', ' ').title()}`

📍 *Entry:* `{signal['entry']}`
🛑 *Stop Loss:* `{signal['sl']}`
   Risk: `{signal['risk_pips']} pips`
🎯 *Take Profit:* `{signal['tp']}`
   Reward: `{signal['reward_pips']} pips`

📊 *Risk:Reward:* `{signal['risk_reward']}`
💪 *Strength:* `{signal['zone_strength']} ATR`
📈 *HTF Trend:* `{signal['htf_trend'].upper()}`
"""
        
        # Add continuation details
        if signal.get('trade_type') == 'continuation':
            cont_details = signal['continuation_details']
            msg += f"""
🔄 *Continuation Trade*
   {cont_details['original_type'].upper()} → {cont_details['flipped_to'].upper()}
   Break: {cont_details['break_bars_ago']} bars ago
   Pullback: {cont_details['pullback_pips']} pips
"""
        
        # Add 5min refinement details
        if signal.get('5min_refined'):
            msg += f"""
🎯 *5min Refinement*
   Score: {signal['5min_score']}/8
   Entry improved by: {signal.get('entry_shift_pips', 0)} pips
   SL improved by: {signal.get('sl_improvement_pips', 0)} pips
"""
        
        msg += f"\n⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT')}`"
        
        return msg
    
    def send_signals(self, signals: List[Dict]):
        """Send all signals to Telegram"""
        if not signals:
            # Send no-signal notification
            msg = f"""
⚠️ *NO TRADING SIGNALS*

No high-probability setups detected.

💡 Wait for next 15min candle close
⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT')}`
"""
            self.send_message(msg)
            return
        
        # Send header
        header = f"""
╔════════════════════════════╗
║   *FOREX TRADING SIGNALS*   ║
╚════════════════════════════╝

📊 Total: *{len(signals)} signal(s)*
⏰ `{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT')}`
"""
        self.send_message(header)
        
        # Send each signal
        for i, signal in enumerate(signals, 1):
            msg = self.format_signal(signal)
            self.send_message(msg)
            
            # Small delay between messages
            if i < len(signals):
                import time
                time.sleep(1)
    
    def send_alert(self, alert_type: str, message: str):
        """Send custom alert"""
        emoji_map = {
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        emoji = emoji_map.get(alert_type, 'ℹ️')
        msg = f"{emoji} *{alert_type.upper()}*\n\n{message}"
        self.send_message(msg)


# Shell script wrapper for backward compatibility
def send_via_shell(message: str):
    """Send message using existing tg.sh script"""
    try:
        subprocess.run(
            ["/home/malombejnr/Desktop/forex_ml_system/tg.sh", message],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except Exception as e:
        print(f"❌ Shell script failed: {e}")
        return False


if __name__ == "__main__":
    # Configuration
    TOKEN = "8584114284:AAGSa38ZQvFd2qUQSqqCpF-a22ZahzIh8FQ"
    CHAT_ID = "6523358903"
    
    # Test message
    pusher = TelegramSignalPusher(TOKEN, CHAT_ID)
    pusher.send_alert("info", "Forex Signal Pusher initialized and ready!")