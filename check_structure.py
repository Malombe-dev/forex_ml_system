import requests

KEYS = {
    'EUR/USD': 'ea3e4fce074e403f8cd4c83dd0035643',
    'GBP/USD': '2522b470eba249f88d2e29575035914e',
    'USD/JPY': '5bb48b4211144b92a4487ce7c63d1b7a',
    'XAU/USD': '63a513bb7eb848748d28d3deb64c1b59',
}

BASE = 'https://api.twelvedata.com'

def get_candles(symbol, interval, size, apikey):
    r = requests.get(f'{BASE}/time_series', params={
        'symbol': symbol, 'interval': interval,
        'outputsize': size, 'apikey': apikey
    })
    data = r.json()
    if 'values' not in data:
        print(f"  [API ERROR] {symbol} {interval}: {data.get('message', data)}")
        return []
    return data['values']

def get_price(symbol, apikey):
    r = requests.get(f'{BASE}/price', params={'symbol': symbol, 'apikey': apikey})
    data = r.json()
    return data.get('price', 'N/A')

def structure_label(vals):
    if len(vals) < 3:
        return 'INSUFFICIENT DATA'
    if vals[0] > vals[1] > vals[2]:
        return 'BULLISH ▲'
    elif vals[0] < vals[1] < vals[2]:
        return 'BEARISH ▼'
    else:
        return 'NEUTRAL ↔'

def check_pair(symbol):
    apikey = KEYS[symbol]
    sep = '=' * 55
    print(f'\n{sep}')
    print(f'  {symbol}')
    print(sep)

    price = get_price(symbol, apikey)
    print(f'  Live price : {price}')

    # ── 4H ──────────────────────────────────────────────
    h4 = get_candles(symbol, '4h', 10, apikey)
    if h4:
        print('\n  4H CANDLES (newest → oldest)')
        for c in h4:
            body = float(c['close']) - float(c['open'])
            icon = '🟢' if body >= 0 else '🔴'
            print(f"  {icon} {c['datetime']}  "
                  f"O={c['open']}  H={c['high']}  "
                  f"L={c['low']}  C={c['close']}")

        highs = [float(c['high'])  for c in h4]
        lows  = [float(c['low'])   for c in h4]
        closes= [float(c['close']) for c in h4]

        hh = structure_label(highs[:3])
        hl = structure_label(lows[:3])

        # Derive single bias
        if 'BULLISH' in hh and 'BULLISH' in hl:
            bias = '🟢 BULLISH'
        elif 'BEARISH' in hh and 'BEARISH' in hl:
            bias = '🔴 BEARISH'
        else:
            bias = '⚪ MIXED / RANGING'

        print(f'\n  4H Structure')
        print(f"    Highs  : {highs[:3]}  → {hh}")
        print(f"    Lows   : {lows[:3]}  → {hl}")
        print(f"    BIAS   : {bias}")

    # ── D1 ──────────────────────────────────────────────
    d1 = get_candles(symbol, '1day', 7, apikey)
    if d1:
        print('\n  D1 CANDLES (newest → oldest)')
        for c in d1:
            body = float(c['close']) - float(c['open'])
            icon = '🟢' if body >= 0 else '🔴'
            print(f"  {icon} {c['datetime']}  "
                  f"O={c['open']}  H={c['high']}  "
                  f"L={c['low']}  C={c['close']}")

        d_highs = [float(c['high']) for c in d1]
        d_lows  = [float(c['low'])  for c in d1]

        d_hh = structure_label(d_highs[:3])
        d_hl = structure_label(d_lows[:3])

        if 'BULLISH' in d_hh and 'BULLISH' in d_hl:
            d_bias = '🟢 BULLISH'
        elif 'BEARISH' in d_hh and 'BEARISH' in d_hl:
            d_bias = '🔴 BEARISH'
        else:
            d_bias = '⚪ MIXED / RANGING'

        print(f'\n  D1 Structure')
        print(f"    Highs  : {d_highs[:3]}  → {d_hh}")
        print(f"    Lows   : {d_lows[:3]}  → {d_hl}")
        print(f"    BIAS   : {d_bias}")

    print()

# ── Run all pairs ────────────────────────────────────────
print('\n' + '█' * 55)
print('  MULTI-PAIR STRUCTURE CHECK')
print('█' * 55)

for pair in KEYS:
    check_pair(pair)

print('█' * 55)
print('  END OF REPORT')
print('█' * 55 + '\n')
