#!/usr/bin/env python3
"""
SMART TELEGRAM SIGNAL MONITOR V2
- Sends ALL active signals to Telegram every time they're generated
- Tracks signal history in JSON (active vs invalid)
- Only sends signals with status='active'
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SmartTelegramMonitor:
    """Monitors a single JSON file and sends all ACTIVE signals"""
    
    def __init__(self, 
                 token: str, 
                 chat_id: str,
                 signals_file: str = "signals/all_signals.json",
                 check_interval: int = 10):
        """
        Args:
            token: Telegram bot token
            chat_id: Telegram chat ID
            signals_file: Single JSON file to monitor
            check_interval: Seconds between checks
        """
        self.token = token
        self.chat_id = chat_id
        self.signals_file = Path(signals_file)
        self.check_interval = check_interval
        self.base_url = f"https://api.telegram.org/bot{token}"
        
        # Track last processed timestamp to avoid re-sending same batch
        self.last_processed_timestamp = None
        
        # Ensure signals directory exists
        self.signals_file.parent.mkdir(exist_ok=True)
        
        # Track already sent signals to avoid duplicate messages
        self.sent_file = Path("signals/monitor_sent_ids.json")
        self.sent_ids = self._load_sent_ids()
        
        # Create empty file if it doesn't exist
        if not self.signals_file.exists():
            with open(self.signals_file, 'w') as f:
                json.dump([], f)
        
        print(f"\n{'='*80}")
        print(f"  SMART TELEGRAM MONITOR V2 - STARTED")
        print(f"{'='*80}")
        print(f"  Bot Token:       {token[:20]}...")
        print(f"  Chat ID:         {chat_id}")
        print(f"  Watching:        {self.signals_file.absolute()}")
        print(f"  Check Interval:  {check_interval} seconds")
        print(f"  Mode:            Send ALL active signals")
        print(f"{'='*80}\n")

    def _load_sent_ids(self) -> set:
        """Load sent signal IDs from disk."""
        if self.sent_file.exists():
            try:
                with open(self.sent_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                print(f"  ⚠️ Error loading sent IDs: {e}")
        return set()

    def _save_sent_ids(self):
        """Save sent signal IDs to disk."""
        try:
            self.sent_file.parent.mkdir(exist_ok=True)
            with open(self.sent_file, 'w') as f:
                json.dump(list(self.sent_ids), f, indent=2)
        except Exception as e:
            print(f"  ⚠️ Error saving sent IDs: {e}")
    
    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
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
            
            if response.status_code == 200:
                return True
            else:
                print(f"  ❌ Telegram API error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error sending to Telegram: {e}")
            return False
    
    def format_signal(self, signal: Dict) -> str:
        """Format signal as Telegram message"""
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        refined = "🎯" if signal.get('5min_refined') else ""
        cont = "🔄" if signal.get('trade_type') == 'continuation' else "✨"
        
        # Status
        status = signal.get('entry_status', 'ready')
        if status == 'touched_wait_next':
            status_msg = "⏳ *WAIT FOR NEXT CANDLE*"
        elif status == 'pending':
            status_msg = "⚠️ *PRICE APPROACHING*"
        else:
            status_msg = "✅ *READY FOR ENTRY*"
        
        msg = f"""{direction_emoji} *NEW FOREX SIGNAL* {refined} {cont}

{status_msg}

*{signal['symbol']} • {signal['direction']}*
Probability: `{signal['probability']}%`

📍 *Entry:* `{signal['entry']}`
🛑 *Stop Loss:* `{signal['sl']}`
   Risk: `{signal['risk_pips']} pips`
🎯 *Take Profit:* `{signal['tp']}`
   Reward: `{signal['reward_pips']} pips`

📊 *R:R:* `{signal['risk_reward']}`
💪 *Strength:* `{signal['zone_strength']} ATR`
📈 *HTF Trend:* `{signal['htf_trend'].upper()}`"""
        
        # Add continuation details
        if signal.get('trade_type') == 'continuation' and signal.get('continuation_details'):
            cont_details = signal['continuation_details']
            msg += f"""

🔄 *Continuation:*
   {cont_details['original_type'].upper()} → {cont_details['flipped_to'].upper()}
   Break: {cont_details['break_bars_ago']} bars ago"""
        
        # Add 5min refinement
        if signal.get('5min_refined'):
            msg += f"""

🎯 *5min Refined:*
   Score: {signal.get('5min_score', 0)}/8"""
        
        msg += f"\n\n⏰ `{signal.get('timestamp', datetime.now().isoformat())}`"
        
        return msg
    
    def load_all_signals(self) -> List[Dict]:
        """Load all signals from the JSON file"""
        try:
            with open(self.signals_file, 'r') as f:
                content = f.read().strip()
                
                if not content:
                    return []
                
                signals = json.loads(content)
                
                # Ensure it's a list
                if isinstance(signals, dict):
                    return [signals]
                
                return signals if isinstance(signals, list) else []
                
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON error in {self.signals_file.name}: {e}")
            return []
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"  ❌ Error loading signals: {e}")
            return []
    
    def get_active_signals(self, all_signals: List[Dict]) -> List[Dict]:
        """Filter to only active signals (not invalid)"""
        active_signals = []
        
        for signal in all_signals:
            status = signal.get('status', 'active')
            if status in ('active', 'pending', 'triggered'):
                active_signals.append(signal)
        
        return active_signals
    
    def get_latest_timestamp(self, signals: List[Dict]) -> Optional[str]:
        """Get the most recent timestamp from a list of signals"""
        if not signals:
            return None
        
        timestamps = [s.get('timestamp') for s in signals if s.get('timestamp')]
        if not timestamps:
            return None
        
        return max(timestamps)
    
    def monitor(self):
        """Main monitoring loop"""
        print("🔍 Monitoring for new signals...")
        print(f"📂 Watching: {self.signals_file.absolute()}")
        print(f"⏰ Checking every {self.check_interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        # Send startup notification
        startup_msg = f"""🚀 *FOREX SIGNAL MONITOR V2 STARTED*

Watching for new signals...

📂 File: `{self.signals_file.name}`
⏰ Interval: `{self.check_interval}s`
📤 Mode: Send ALL active signals

`{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT')}`"""
        self.send_message(startup_msg)
        
        iteration = 0
        last_file_mtime = None
        total_signals_sent = 0
        
        try:
            while True:
                iteration += 1
                
                # Check if file exists and has been modified
                if not self.signals_file.exists():
                    if iteration % 30 == 1:  # Print every 5 minutes
                        print(f"⏳ Waiting for {self.signals_file.name}...")
                    time.sleep(self.check_interval)
                    continue
                
                # Get file modification time
                current_mtime = self.signals_file.stat().st_mtime
                
                # Only check if file was modified
                if last_file_mtime is not None and current_mtime == last_file_mtime:
                    time.sleep(self.check_interval)
                    continue
                
                last_file_mtime = current_mtime
                
                print(f"\n{'='*80}")
                print(f"📝 FILE UPDATED - CHECK #{iteration}")
                print(f"{'='*80}")
                print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Load all signals
                all_signals = self.load_all_signals()
                print(f"  Total signals in file: {len(all_signals)}")
                
                # Get only ACTIVE signals
                active_signals = self.get_active_signals(all_signals)
                print(f"  Active signals: {len(active_signals)}")
                
                # Check for unsent active signals
                if active_signals:
                    unsent_signals = []
                    for s in active_signals:
                        sig_id = f"{s.get('symbol')}_{s.get('direction')}_{s.get('entry')}_{s.get('timestamp')}"
                        if sig_id not in self.sent_ids:
                            unsent_signals.append((sig_id, s))
                    
                    if unsent_signals:
                        print(f"  🆕 {len(unsent_signals)} new/unsent signal(s) detected!")
                        
                        # Send header message
                        header_msg = f"""📊 *NEW SIGNAL BATCH*

{len(unsent_signals)} new signal(s) detected

`{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT')}`"""
                        self.send_message(header_msg)
                        time.sleep(1)
                        
                        # Send each unsent signal
                        for i, (sig_id, signal) in enumerate(unsent_signals, 1):
                            msg = self.format_signal(signal)
                            success = self.send_message(msg)
                            
                            if success:
                                print(f"  ✅ Signal #{i}/{len(unsent_signals)} sent to Telegram")
                                self.sent_ids.add(sig_id)
                                self._save_sent_ids()
                                total_signals_sent += 1
                            else:
                                print(f"  ❌ Signal #{i}/{len(unsent_signals)} failed to send")
                            
                            # Delay between messages
                            if i < len(unsent_signals):
                                time.sleep(1.5)
                        
                        print(f"  📤 Complete: {len(unsent_signals)} new signal(s) sent")
                    else:
                        print(f"  ℹ️  All active signals already sent")
                else:
                    print(f"  ℹ️  No active signals in file")
                
                # Count invalid signals
                invalid_count = len(all_signals) - len(active_signals)
                if invalid_count > 0:
                    print(f"  📝 Archived signals: {invalid_count}")
                
                print(f"{'='*80}\n")
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print(f"  MONITOR STOPPED")
            print(f"{'='*80}")
            print(f"  Total checks:        {iteration}")
            print(f"  Signals sent:        {total_signals_sent}")
            print(f"{'='*80}\n")
            
            # Send shutdown notification
            shutdown_msg = f"""⏹️ *SIGNAL MONITOR STOPPED*

Total checks: `{iteration}`
Signals sent: `{total_signals_sent}`

`{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT')}`"""
            self.send_message(shutdown_msg)
        
        except Exception as e:
            print(f"\n❌ FATAL ERROR: {e}")
            error_msg = f"""❌ *MONITOR ERROR*

{str(e)}

`{datetime.now().strftime('%Y-%m-%d %H:%M:%S GMT')}`"""
            self.send_message(error_msg)


def main():
    """Run the monitor"""
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8584114284:AAGSa38ZQvFd2qUQSqqCpF-a22ZahzIh8FQ")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6523358903")
    
    # Single JSON file that contains ALL signals
    SIGNALS_FILE = "signals/all_signals.json"
    
    # Check interval (seconds)
    CHECK_INTERVAL = 10
    
    # ═══════════════════════════════════════════════════════════════════════
    
    # Test Telegram connection
    print("Testing Telegram connection...")
    test_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
    try:
        response = requests.get(test_url, timeout=5)
        if response.status_code == 200:
            bot_info = response.json()
            print(f"✅ Connected to bot: @{bot_info['result']['username']}\n")
        else:
            print(f"⚠️  Telegram issue: {response.status_code}\n")
    except Exception as e:
        print(f"❌ Cannot connect to Telegram: {e}\n")
        return
    
    # Create and run monitor
    monitor = SmartTelegramMonitor(
        token=TELEGRAM_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
        signals_file=SIGNALS_FILE,
        check_interval=CHECK_INTERVAL
    )
    
    monitor.monitor()


if __name__ == "__main__":
    main()