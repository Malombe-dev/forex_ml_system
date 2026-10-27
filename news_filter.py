import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

class ForexFactoryNewsFilter:
    """
    Economic Calendar News Filter
    Fetches the public widget weekly economic calendar XML feed,
    caches it locally, and flags active high-impact news windows.
    """
    def __init__(self, cache_file="signals/ff_calendar.xml", cache_expiry_hours=6,
                 source_utc_offset_hours=0):
        self.cache_file = cache_file
        self.cache_expiry_hours = cache_expiry_hours
        # NFS FairEconomy is the standard, unblocked public widget mirror for Forex Factory
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        # IMPORTANT: this feed's timestamps are NOT guaranteed to be UTC — the
        # offset has been reported to vary. Verify against a known event
        # (e.g. NFP, FOMC) on forexfactory.com/calendar before trusting this at 0.
        # If the feed times are e.g. 4 hours ahead of UTC, set this to -4.
        self.source_utc_offset_hours = source_utc_offset_hours
        # Tracks whether the last load_events() call produced trustworthy data,
        # so callers can fail safe instead of assuming "no data" == "no news".
        self.last_load_reliable = False

    def _download_calendar(self) -> bool:
        logger.info("📡 Fetching economic calendar from mirror...")
        try:
            req = urllib.request.Request(
                self.url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                xml_data = response.read()

            # FF rate-limits this endpoint and returns an HTML "Request Denied"
            # page instead of XML when exceeded. Guard against silently caching that.
            stripped = xml_data.lstrip()
            if not stripped.startswith(b"<?xml") and b"<weeklyevents" not in stripped[:200]:
                logger.error(
                    "❌ Download returned non-XML payload (likely rate-limited or blocked) "
                    "— keeping existing cache."
                )
                return False

            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(xml_data)
            logger.info("💾 Economic calendar cached successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download economic calendar: {e}")
            return False

    def _should_update_cache(self) -> bool:
        if not os.path.exists(self.cache_file):
            return True
        mtime = os.path.getmtime(self.cache_file)
        last_modified = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return datetime.now(timezone.utc) - last_modified > timedelta(hours=self.cache_expiry_hours)

    def load_events(self) -> list:
        self.last_load_reliable = False

        if self._should_update_cache():
            success = self._download_calendar()
            if not success and os.path.exists(self.cache_file):
                age_hrs = (
                    datetime.now(timezone.utc)
                    - datetime.fromtimestamp(os.path.getmtime(self.cache_file), tz=timezone.utc)
                ).total_seconds() / 3600
                logger.warning(f"⚠️ Download failed — falling back to stale cache ({age_hrs:.1f}h old).")

        if not os.path.exists(self.cache_file):
            logger.warning("⚠️ Cache file not found and download failed. Using empty calendar.")
            return []

        try:
            tree = ET.parse(self.cache_file)
            root = tree.getroot()
            events = []
            
            for event_node in root.findall('event'):
                title = event_node.find('title').text if event_node.find('title') is not None else ""
                country = event_node.find('country').text if event_node.find('country') is not None else ""
                date_str = event_node.find('date').text if event_node.find('date') is not None else ""
                time_str = event_node.find('time').text if event_node.find('time') is not None else ""
                impact = event_node.find('impact').text if event_node.find('impact') is not None else ""
                
                # Exclude events without definite times
                if not date_str or not time_str or time_str.lower() in ("all day", "tentative"):
                    continue
                    
                try:
                    dt_str = f"{date_str} {time_str}"
                    # Feed timestamps are corrected by source_utc_offset_hours.
                    # Verify the correct offset against a known event before trusting this.
                    event_naive = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
                    event_utc = event_naive.replace(tzinfo=timezone.utc) - timedelta(
                        hours=self.source_utc_offset_hours
                    )

                    events.append({
                        'title': title,
                        'country': country.upper(),
                        'time_utc': event_utc,
                        'impact': impact.lower()
                    })
                except Exception:
                    continue

            self.last_load_reliable = True
            return events
        except Exception as e:
            logger.error(f"❌ Error parsing XML calendar: {e}")
            return []

    def is_news_active(self, symbol: str, window_minutes: int = 30) -> tuple:
        """
        Checks if there is a High Impact news event scheduled within window_minutes
        before or after the current time for any currency in the symbol (e.g. 'GBP/USD' -> 'GBP' and 'USD').
        
        Returns:
            (is_active, active_event_dict)
        """
        # Split symbol (e.g. "GBP/USD" -> ["GBP", "USD"])
        currencies = [c.upper() for c in symbol.replace("/", "_").split("_")]

        events = self.load_events()

        # Fail safe: if we couldn't confirm the calendar loaded correctly this
        # cycle, don't silently treat "unknown" as "no news" for a live system.
        if not self.last_load_reliable and not events:
            logger.warning(f"⚠️ Calendar data unreliable for {symbol} — blocking as a precaution.")
            return True, {'title': 'UNKNOWN (calendar load failed)', 'country': '', 'time_utc': None, 'impact': 'high'}

        now_utc = datetime.now(timezone.utc)

        for event in events:
            if event['impact'] == 'high' and event['country'] in currencies:
                time_diff = (now_utc - event['time_utc']).total_seconds() / 60.0
                # Check if we are inside the window (before or after)
                if abs(time_diff) <= window_minutes:
                    return True, event
                    
        return False, None

    # ------------------------------------------------------------------
    # Historical / replay support (the public feed only covers THIS week)
    # ------------------------------------------------------------------
    def load_calendar_csv(self, path: str) -> list:
        """Load a historical calendar export (UTC) for backtests.
        Columns: a time column (datetime_utc|time_utc|datetime|time|date), currency|country|ccy,
        optional impact (high/3/red) and title|event. Returns events in the same shape as load_events()."""
        import csv as _csv
        events = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            rd = _csv.DictReader(f)
            cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
            pick = lambda names: next((cols[n] for n in names if n in cols), None)
            tcol = pick(["datetime_utc", "time_utc", "datetime", "time", "date"])
            ccol = pick(["currency", "country", "ccy"]); icol = pick(["impact"]); ecol = pick(["title", "event", "name"])
            if tcol is None or ccol is None:
                raise ValueError(f"calendar CSV needs a time and a currency column, found {rd.fieldnames}")
            for r in rd:
                imp = (r.get(icol, "high") if icol else "high").strip().lower()
                imp = "high" if imp in ("high", "3", "red", "high impact") else imp
                try:
                    t = datetime.fromisoformat(r[tcol].strip().replace("Z", "+00:00"))
                except Exception:
                    continue
                t = t.astimezone(timezone.utc) if t.tzinfo else t.replace(tzinfo=timezone.utc)
                events.append({"title": r.get(ecol, "") if ecol else "", "country": r[ccol].strip().upper(),
                               "time_utc": t, "impact": imp})
        self.last_load_reliable = True
        return events

    @staticmethod
    def is_news_active_at(symbol: str, when_utc: datetime, events: list, window_minutes: int = 30) -> tuple:
        """Time-explicit version of is_news_active() for replay / audits. `events` from load_calendar_csv()."""
        currencies = [c.upper() for c in symbol.replace("/", "_").split("_")]
        if when_utc.tzinfo is None:
            when_utc = when_utc.replace(tzinfo=timezone.utc)
        for ev in events:
            if ev["impact"] == "high" and ev["country"] in currencies:
                if abs((when_utc - ev["time_utc"]).total_seconds()) / 60.0 <= window_minutes:
                    return True, ev
        return False, None
