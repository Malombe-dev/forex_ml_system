#!/usr/bin/env python3
"""
Email Signal Pusher & Webhook Receiver for Forex ML System.
Receives TradingView alerts and sends formatted HTML signal emails to your inbox.
Works with Gmail, Outlook, SMTP, or transactional email providers.
"""

import os
import json
import smtplib
from http.server import HTTPServer, BaseHTTPRequestHandler
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path

def get_env():
    env = {}
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    env[k] = v
    return env

ENV = get_env()

# Email Configuration (Set these in your .env or environment)
SMTP_HOST = ENV.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(ENV.get("SMTP_PORT", "587"))
SMTP_USER = ENV.get("SMTP_USER", "") # your_email@gmail.com
SMTP_PASS = ENV.get("SMTP_PASS", "") # 16-character Google App Password
EMAIL_RECEIVER = ENV.get("EMAIL_RECEIVER", SMTP_USER) # Destination email

def send_signal_email(signal_data: dict) -> bool:
    """Send formatted HTML email with trade details"""
    if not SMTP_USER or not SMTP_PASS:
        print("⚠️ [Email Pusher] SMTP credentials not set in .env (SMTP_USER / SMTP_PASS).")
        print(f"   Signal payload received: {signal_data}")
        return False

    symbol = signal_data.get("symbol", "FOREX")
    action = signal_data.get("action", "SIGNAL").upper()
    entry = signal_data.get("entry", "N/A")
    sl = signal_data.get("sl", "N/A")
    tp1 = signal_data.get("tp", "N/A")
    tp2 = signal_data.get("tp2", "N/A")
    rr = signal_data.get("rr", "1.5")
    score = signal_data.get("score", "6/7")

    is_buy = "BUY" in action or "LONG" in action
    color_theme = "#00e676" if is_buy else "#ff1744"
    dir_emoji = "🟢 BUY" if is_buy else "🔴 SELL"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎯 TrendM30 High-WR Signal: {action} {symbol} @ {entry}"
    msg["From"] = f"Forex ML System <{SMTP_USER}>"
    msg["To"] = EMAIL_RECEIVER

    # Rich HTML Email Card
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
        .card {{ max-width: 520px; margin: 0 auto; background: #161b22; border-radius: 12px; border: 1px solid #30363d; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
        .header {{ background: {color_theme}; color: #000; padding: 18px 24px; font-size: 20px; font-weight: bold; display: flex; justify-content: space-between; }}
        .body {{ padding: 24px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 20px; }}
        .item {{ background: #21262d; padding: 12px 16px; border-radius: 8px; border-left: 3px solid #58a6ff; }}
        .item-sl {{ border-left-color: #f85149; }}
        .item-tp {{ border-left-color: #2ea043; }}
        .item-tp2 {{ border-left-color: #3fb950; }}
        .label {{ font-size: 11px; text-transform: uppercase; color: #8b949e; margin-bottom: 4px; }}
        .val {{ font-size: 17px; font-weight: 600; color: #f0f6fc; }}
        .footer {{ font-size: 12px; color: #8b949e; text-align: center; padding: 16px; border-top: 1px solid #30363d; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; background: #30363d; color: #fff; font-size: 12px; margin-top: 8px; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="header">
          <span>{dir_emoji} • {symbol}</span>
          <span style="font-size: 14px; background: rgba(0,0,0,0.15); padding: 2px 8px; border-radius: 4px;">M30 Retest</span>
        </div>
        <div class="body">
          <div style="margin-bottom: 16px;">
            <span class="badge">Confluence Score: {score}</span>
            <span class="badge" style="background: #238636;">4H Aligned</span>
            <span class="badge" style="background: #1f6feb;">EMA21 Retest</span>
          </div>
          <div class="grid">
            <div class="item">
              <div class="label">Entry (Limit / Touch)</div>
              <div class="val">{entry}</div>
            </div>
            <div class="item item-sl">
              <div class="label">Stop Loss (-1.0R)</div>
              <div class="val">{sl}</div>
            </div>
            <div class="item item-tp">
              <div class="label">Take Profit 1 ({rr}R)</div>
              <div class="val">{tp1}</div>
            </div>
            <div class="item item-tp2">
              <div class="label">Take Profit 2 (Runner)</div>
              <div class="val">{tp2 if tp2 != 'N/A' else '2.5R Runner'}</div>
            </div>
          </div>
          <div style="background: #0d1117; padding: 12px; border-radius: 6px; font-size: 12px; color: #8b949e;">
            💡 <b>Execution Tip:</b> When TP1 is hit, close 50% of the trade and move Stop Loss to Breakeven ({entry}) to make the remainder risk-free.
          </div>
        </div>
        <div class="footer">
          Generated by Forex ML TrendM30 High-WR System • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        </div>
      </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"  📧 [Email Sent] Successfully delivered signal alert for {symbol} {action} to {EMAIL_RECEIVER}")
        return True
    except Exception as e:
        print(f"  ❌ [Email Error] Failed to send email: {e}")
        return False

# ── Standalone Webhook Listener ─────────────────────────────────────────────
class TradingViewWebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len).decode("utf-8")
        print(f"\n📩 Received incoming TradingView alert:\n{post_data}")

        try:
            # Parse JSON or formatted text
            if post_data.strip().startswith("{"):
                payload = json.loads(post_data)
            else:
                # Text parse fallback
                payload = {"text": post_data, "action": "ALERT", "symbol": "FOREX"}
            
            send_signal_email(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f'{{"error":"{e}"}}'.encode("utf-8"))

def run_webhook_server(port: int = 5001):
    print("=" * 80)
    print(f"  TRADINGVIEW EMAIL WEBHOOK LISTENER ACTIVE ON PORT {port}")
    print(f"  Destination Email: {EMAIL_RECEIVER or 'Not configured (check .env)'}")
    print("=" * 80)
    server = HTTPServer(("0.0.0.0", port), TradingViewWebhookHandler)
    server.serve_forever()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing sample email alert...")
        send_signal_email({
            "symbol": "EUR/USD",
            "action": "BUY",
            "entry": 1.15350,
            "sl": 1.15050,
            "tp": 1.15800,
            "tp2": 1.16100,
            "rr": 1.5,
            "score": "6/7"
        })
    else:
        run_webhook_server(5001)
