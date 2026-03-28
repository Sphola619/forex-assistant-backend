from __future__ import annotations

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urlencode

import requests
from flask import Flask, jsonify, request

REQUEST_TIMEOUT_SECONDS = 8
MAX_WORKERS = 8
QUOTE_CACHE_TTL_SECONDS = 15 * 60
COMMENTARY_CACHE_TTL_SECONDS = 15 * 60
GEMINI_MODEL = "gemini-2.5-flash"


def load_env_file() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env_file()

app = Flask(__name__)
# Quote and commentary caches keep the UI responsive and reduce repeat upstream calls.
quote_cache: dict[str, dict[str, Any]] = {}
commentary_cache: dict[str, dict[str, Any]] = {}

# The frontend knows assets by slug/symbol, so the backend keeps a matching map
# for commentary generation and display metadata.
ASSET_METADATA = {
    "USDZAR.FOREX": {
        "symbol": "USDZAR",
        "name": "US Dollar / South African Rand",
        "category": "Forex",
    },
    "EURZAR.FOREX": {
        "symbol": "EURZAR",
        "name": "Euro / South African Rand",
        "category": "Forex",
    },
    "GBPZAR.FOREX": {
        "symbol": "GBPZAR",
        "name": "British Pound / South African Rand",
        "category": "Forex",
    },
    "USDCHF.FOREX": {
        "symbol": "USDCHF",
        "name": "US Dollar / Swiss Franc",
        "category": "Forex",
    },
    "EURUSD.FOREX": {
        "symbol": "EUR/USD",
        "name": "Euro / US Dollar",
        "category": "Forex",
    },
    "GBPUSD.FOREX": {
        "symbol": "GBP/USD",
        "name": "British Pound / US Dollar",
        "category": "Forex",
    },
    "USDJPY.FOREX": {
        "symbol": "USD/JPY",
        "name": "US Dollar / Japanese Yen",
        "category": "Forex",
    },
    "XAUUSD.FOREX": {
        "symbol": "XAU/USD",
        "name": "Gold / US Dollar",
        "category": "Precious Metal",
    },
    "XAGUSD.FOREX": {
        "symbol": "XAG/USD",
        "name": "Silver / US Dollar",
        "category": "Precious Metal",
    },
}

CURRENCY_DRIVERS = {
    "USD": {
        "central_bank": "Federal Reserve expectations",
        "macro": "US inflation, Treasury yields, and broad risk sentiment",
    },
    "EUR": {
        "central_bank": "ECB rate outlook",
        "macro": "Eurozone inflation, growth expectations, and policy tone",
    },
    "GBP": {
        "central_bank": "Bank of England policy expectations",
        "macro": "UK inflation, labor-market data, and BoE guidance",
    },
    "JPY": {
        "central_bank": "Bank of Japan policy normalization",
        "macro": "yield differentials and global risk appetite",
    },
    "CHF": {
        "central_bank": "Swiss National Bank policy path",
        "macro": "safe-haven flows and European risk sentiment",
    },
    "ZAR": {
        "central_bank": "South African Reserve Bank expectations",
        "macro": "South African rates, commodity sentiment, and EM risk appetite",
    },
}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def parse_quote(raw_quote: dict[str, Any]) -> dict[str, Any]:
    # Normalize both real-time and EOD responses into one frontend-friendly shape.
    try:
        price = float(raw_quote.get("close"))
    except (TypeError, ValueError):
        price = None

    try:
        change_percent = float(raw_quote.get("change_p"))
    except (TypeError, ValueError):
        change_percent = None

    if change_percent is None:
        try:
            open_price = float(raw_quote.get("open"))
            close_price = float(raw_quote.get("close"))
            if open_price != 0:
                change_percent = ((close_price - open_price) / open_price) * 100
        except (TypeError, ValueError):
            change_percent = None

    return {
        "code": raw_quote.get("code"),
        "timestamp": raw_quote.get("timestamp"),
        "gmtoffset": raw_quote.get("gmtoffset"),
        "price": price,
        "changePercent": change_percent,
    }


def should_prefer_open_close_change(symbol: str) -> bool:
    # ZAR crosses were showing tiny or unhelpful real-time `change_p` values, so
    # we fall back to session open/close math for a more useful daily change.
    return symbol.endswith("ZAR.FOREX")


def fetch_realtime_quote(symbol: str, api_token: str) -> dict[str, Any]:
    # Real-time quotes are preferred for the forex pairs in this dashboard.
    encoded_symbol = quote(symbol, safe="")
    query_string = urlencode(
        {
            "api_token": api_token,
            "fmt": "json",
        }
    )
    upstream_url = f"https://eodhd.com/api/real-time/{encoded_symbol}?{query_string}"

    upstream_response = requests.get(upstream_url, timeout=REQUEST_TIMEOUT_SECONDS)
    upstream_response.raise_for_status()
    raw_data = upstream_response.json()

    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError(f"No real-time data returned for {symbol}.")

    raw_data["code"] = symbol

    if should_prefer_open_close_change(symbol):
        raw_data["change_p"] = None

    return parse_quote(raw_data)


def fetch_eod_quote(symbol: str, api_token: str) -> dict[str, Any]:
    # Metals are more reliable from the EOD endpoint in this project.
    encoded_symbol = quote(symbol, safe="")
    query_string = urlencode(
        {
            "order": "d",
            "api_token": api_token,
            "fmt": "json",
        }
    )
    upstream_url = f"https://eodhd.com/api/eod/{encoded_symbol}?{query_string}"

    upstream_response = requests.get(upstream_url, timeout=REQUEST_TIMEOUT_SECONDS)
    upstream_response.raise_for_status()
    raw_data = upstream_response.json()

    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError(f"No EOD data returned for {symbol}.")

    latest_bar = raw_data[0]
    latest_bar["code"] = symbol
    return parse_quote(latest_bar)


def fetch_quote(symbol: str, api_token: str) -> dict[str, Any]:
    # Route each symbol to the source that has behaved most reliably.
    is_metal = symbol.startswith("XAU") or symbol.startswith("XAG") or symbol.startswith("XPT")

    if is_metal:
        return fetch_eod_quote(symbol, api_token)

    return fetch_realtime_quote(symbol, api_token)


def get_cached_quote(symbol: str) -> dict[str, Any] | None:
    cached_quote = quote_cache.get(symbol)
    if not cached_quote:
        return None

    cached_at = cached_quote.get("cachedAt")
    if not isinstance(cached_at, (int, float)):
        return None

    if (time.time() - cached_at) > QUOTE_CACHE_TTL_SECONDS:
        return None

    return {key: value for key, value in cached_quote.items() if key != "cachedAt"}


def get_cached_commentary(symbol: str) -> dict[str, Any] | None:
    cached_commentary = commentary_cache.get(symbol)
    if not cached_commentary:
        return None

    cached_at = cached_commentary.get("cachedAt")
    if not isinstance(cached_at, (int, float)):
        return None

    if (time.time() - cached_at) > COMMENTARY_CACHE_TTL_SECONDS:
        return None

    return {key: value for key, value in cached_commentary.items() if key != "cachedAt"}


def fetch_eod_bars(symbol: str, api_token: str, limit: int = 60) -> list[dict[str, Any]]:
    # Historical bars feed the lightweight technical snapshot used by Gemini.
    encoded_symbol = quote(symbol, safe="")
    query_string = urlencode(
        {
            "order": "d",
            "api_token": api_token,
            "fmt": "json",
        }
    )
    upstream_url = f"https://eodhd.com/api/eod/{encoded_symbol}?{query_string}"
    upstream_response = requests.get(upstream_url, timeout=REQUEST_TIMEOUT_SECONDS)
    upstream_response.raise_for_status()
    raw_data = upstream_response.json()

    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError(f"No EOD bar data returned for {symbol}.")

    return raw_data[:limit]


def calculate_ma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def calculate_rsi(values: list[float], period: int = 14) -> float | None:
    # RSI is calculated locally so the AI receives explicit indicator values
    # instead of inferring momentum from prose.
    if len(values) <= period:
        return None

    gains = []
    losses = []
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        gains.append(max(delta, 0))
        losses.append(abs(min(delta, 0)))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_technical_snapshot(symbol: str, api_token: str) -> dict[str, Any]:
    # This is the first-pass deterministic technical summary that feeds the
    # commentary prompt.
    raw_bars = fetch_eod_bars(symbol, api_token, limit=60)
    bars = list(reversed(raw_bars))

    closes = [float(bar["close"]) for bar in bars if bar.get("close") is not None]
    highs = [float(bar["high"]) for bar in bars[-10:] if bar.get("high") is not None]
    lows = [float(bar["low"]) for bar in bars[-10:] if bar.get("low") is not None]

    latest_bar = bars[-1]
    latest_open = float(latest_bar["open"]) if latest_bar.get("open") is not None else None
    latest_close = float(latest_bar["close"]) if latest_bar.get("close") is not None else None
    latest_high = float(latest_bar["high"]) if latest_bar.get("high") is not None else None
    latest_low = float(latest_bar["low"]) if latest_bar.get("low") is not None else None

    return {
        "date": latest_bar.get("date"),
        "dailyRange": {"low": latest_low, "high": latest_high},
        "sessionChangePercent": (
            ((latest_close - latest_open) / latest_open) * 100
            if latest_open not in (None, 0) and latest_close is not None
            else None
        ),
        "ma20": calculate_ma(closes, 20),
        "ma50": calculate_ma(closes, 50),
        "rsi14": calculate_rsi(closes, 14),
        "support": lows[:2] if len(lows) >= 2 else lows,
        "resistance": highs[-2:] if len(highs) >= 2 else highs,
    }


def build_macro_context(symbol: str) -> dict[str, Any]:
    # For the MVP, macro context is rule-based. This can later be upgraded with
    # live news and economic-calendar inputs.
    if symbol.startswith("XAU"):
        return {
            "macroThemes": [
                "gold is sensitive to US dollar direction",
                "real yield expectations often shape near-term demand",
                "geopolitical uncertainty can support safe-haven flows",
            ],
            "eventWatch": ["US CPI", "Federal Reserve commentary", "US Treasury yields"],
            "headlineContext": "Watch inflation expectations, yield moves, and risk sentiment.",
        }

    if symbol.startswith("XAG"):
        return {
            "macroThemes": [
                "silver reacts to both precious-metal sentiment and industrial demand",
                "US dollar direction remains important",
                "risk appetite and growth expectations can influence price swings",
            ],
            "eventWatch": ["US CPI", "Fed commentary", "global growth sentiment"],
            "headlineContext": "Watch dollar moves, real yields, and industrial-demand narratives.",
        }

    base = symbol[:3]
    quote_currency = symbol[3:6]
    base_context = CURRENCY_DRIVERS.get(base, {})
    quote_context = CURRENCY_DRIVERS.get(quote_currency, {})

    return {
        "macroThemes": [
            f"{base} is driven by {base_context.get('central_bank', 'policy expectations')}",
            f"{quote_currency} is driven by {quote_context.get('central_bank', 'policy expectations')}",
            "relative rate expectations and risk sentiment remain the main FX drivers",
        ],
        "eventWatch": [
            base_context.get("macro", f"{base} macro data"),
            quote_context.get("macro", f"{quote_currency} macro data"),
        ],
        "headlineContext": (
            f"Focus on {base} versus {quote_currency} rate expectations, inflation trends, "
            "and overall market risk appetite."
        ),
    }


def extract_json_text(raw_text: str) -> str:
    # Gemini sometimes wraps JSON in markdown fences, so we strip those first.
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def generate_commentary_with_gemini(symbol: str, quote: dict[str, Any], technical: dict[str, Any]) -> dict[str, Any]:
    # Gemini does the writing; the backend supplies structured market inputs so
    # the response stays grounded and renderable.
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
        raise ValueError("Missing GEMINI_API_KEY in backend/.env.")

    asset = ASSET_METADATA.get(symbol, {"symbol": symbol, "name": symbol, "category": "Market"})
    macro_context = build_macro_context(symbol)

    system_instruction = (
        "You are a disciplined market analyst. Use only the supplied market data. "
        "Do not invent headlines, levels, events, or technical conditions. "
        "Return valid JSON only. Keep the tone concise, professional, and readable. "
        "Do not give personalized financial advice."
    )

    user_payload = {
        "asset": asset,
        "quote": quote,
        "technical": technical,
        "fundamentalContext": macro_context,
        "task": (
            "Produce a short daily market commentary with a technical summary, "
            "fundamental summary, key drivers, risks, support/resistance levels, "
            "and an event watch list."
        ),
        "responseShape": {
            "asset": "string",
            "bias": "bullish|bearish|neutral",
            "confidence": "number from 0 to 1",
            "technical_summary": "string",
            "fundamental_summary": "string",
            "daily_commentary": "string",
            "key_drivers": ["string"],
            "risks": ["string"],
            "levels": {"support": ["string"], "resistance": ["string"]},
            "event_watch": ["string"],
            "time_horizon": "string",
            "disclaimer": "string",
        },
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={
            "x-goog-api-key": gemini_api_key,
            "Content-Type": "application/json",
        },
        json={
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [{"parts": [{"text": json.dumps(user_payload)}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=REQUEST_TIMEOUT_SECONDS + 10,
    )
    response.raise_for_status()
    payload = response.json()

    text_parts = []
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])

    if not text_parts:
        raise ValueError("Gemini returned no text content.")

    commentary = json.loads(extract_json_text("\n".join(text_parts)))
    commentary["generatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commentary["symbol"] = symbol
    commentary["technical"] = technical
    return commentary


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/quotes", methods=["GET", "OPTIONS"])
def quotes():
    if request.method == "OPTIONS":
        return ("", 204)

    api_token = os.getenv("EODHD_API_KEY")
    if not api_token or api_token == "YOUR_API_KEY":
        return jsonify(
            {"error": "Missing EODHD API key. Set EODHD_API_KEY in backend/.env."}
        ), 500

    symbols = request.args.get("symbols", "").strip()
    if not symbols:
        return jsonify({"error": "Missing symbols query parameter."}), 400

    requested_symbols = [item.strip() for item in symbols.split(",") if item.strip()]

    try:
        quotes_by_symbol = {}
        errors = {}
        # Fetch symbols in parallel so one slow upstream response does not stall
        # the entire dashboard.
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(requested_symbols) or 1)) as pool:
            futures = {
                pool.submit(fetch_quote, symbol, api_token): symbol
                for symbol in requested_symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    quote = future.result()
                    quotes_by_symbol[symbol] = quote
                    quote_cache[symbol] = {**quote, "cachedAt": time.time()}
                except requests.HTTPError as error:
                    details = error.response.text if error.response is not None else str(error)
                    status_code = (
                        error.response.status_code if error.response is not None else "unknown"
                    )
                    cached_quote = get_cached_quote(symbol)
                    if cached_quote is not None:
                        # Reuse the last known good quote when upstream data is
                        # temporarily flaky.
                        quotes_by_symbol[symbol] = cached_quote
                        errors[symbol] = f"Using cached quote after HTTP {status_code}."
                    else:
                        errors[symbol] = f"HTTP {status_code}: {details}"
                except requests.RequestException as error:
                    cached_quote = get_cached_quote(symbol)
                    if cached_quote is not None:
                        quotes_by_symbol[symbol] = cached_quote
                        errors[symbol] = "Using cached quote after upstream request failure."
                    else:
                        errors[symbol] = str(error)
                except Exception as error:
                    cached_quote = get_cached_quote(symbol)
                    if cached_quote is not None:
                        quotes_by_symbol[symbol] = cached_quote
                        errors[symbol] = "Using cached quote after parse failure."
                    else:
                        errors[symbol] = str(error)
    except Exception as error:
        return jsonify({"error": "Unexpected backend error.", "details": str(error)}), 500

    return jsonify({"quotes": quotes_by_symbol, "errors": errors})


@app.route("/api/ai/commentary", methods=["POST", "OPTIONS"])
def ai_commentary():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    symbol = str(payload.get("symbol", "")).strip()

    if not symbol:
        return jsonify({"error": "Missing symbol in request body."}), 400

    cached_commentary = get_cached_commentary(symbol)
    if cached_commentary is not None:
        return jsonify(cached_commentary)

    eodhd_api_key = os.getenv("EODHD_API_KEY")
    if not eodhd_api_key or eodhd_api_key == "YOUR_API_KEY":
        return jsonify({"error": "Missing EODHD_API_KEY in backend/.env."}), 500

    try:
        # Commentary is built from a fresh quote + technical snapshot and then
        # cached for 15 minutes to control cost and latency.
        quote = fetch_quote(symbol, eodhd_api_key)
        technical = build_technical_snapshot(symbol, eodhd_api_key)
        commentary = generate_commentary_with_gemini(symbol, quote, technical)
        commentary_cache[symbol] = {**commentary, "cachedAt": time.time()}
        return jsonify(commentary)
    except requests.HTTPError as error:
        details = error.response.text if error.response is not None else str(error)
        return jsonify({"error": "AI commentary request failed.", "details": details}), 502
    except requests.RequestException as error:
        return jsonify({"error": "Unable to reach an upstream service.", "details": str(error)}), 502
    except Exception as error:
        return jsonify({"error": "Unable to generate commentary.", "details": str(error)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "4000"))
    app.run(host="0.0.0.0", port=port, debug=True)
