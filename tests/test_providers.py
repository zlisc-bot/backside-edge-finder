"""Provider adapters against a local mock of each API (response shapes as documented)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

MS = 1760535000000  # 2025-10-15 13:30 UTC


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        base = f"http://127.0.0.1:{self.server.server_port}"
        # ---------------- polygon
        if p.startswith("/v2/aggs/grouped/"):
            assert q["adjusted"] == ["false"]
            return self._send({"status": "OK", "resultsCount": 2, "results": [
                {"T": "ABCD", "v": 9e6, "vw": 0.51, "o": 0.5, "c": 0.52, "h": 0.6, "l": 0.45, "t": MS, "n": 100},
                {"T": "WXYZ", "v": 1e5, "vw": 5.1, "o": 5, "c": 5.2, "h": 5.3, "l": 4.9, "t": MS, "n": 10}]})
        if p.startswith("/v2/aggs/ticker/ABCD/range/1/minute/"):
            if "cursor" not in q:
                return self._send({"results": [{"v": 1000, "vw": 0.5, "o": 0.5, "c": 0.5, "h": 0.51, "l": 0.49, "t": MS, "n": 3}],
                                   "next_url": f"{base}/v2/aggs/ticker/ABCD/range/1/minute/x/y?cursor=2"})
            return self._send({"results": [{"v": 2000, "o": 0.5, "c": 0.52, "h": 0.53, "l": 0.5, "t": MS + 60000, "n": 4}]})
        if p == "/v3/reference/splits":
            return self._send({"results": [{"ticker": "ABCD", "execution_date": "2025-10-15", "split_from": 10, "split_to": 1}]})
        if p == "/v3/reference/tickers":
            if q.get("type") == ["CS"]:
                return self._send({"results": [{"ticker": "ABCD", "type": "CS", "primary_exchange": "XNAS", "name": "Abcd Inc"}]})
            return self._send({"results": []})
        if p.startswith("/v3/reference/tickers/"):
            return self._send({"results": {"ticker": p.rsplit("/", 1)[1], "type": "WARRANT", "primary_exchange": "XNAS"}})
        # ---------------- alpaca
        if p == "/v2/assets":
            if q["status"] == ["active"]:
                return self._send([{"symbol": "ABCD", "exchange": "NASDAQ", "status": "active", "name": "Abcd Inc Common Stock",
                                    "shortable": True, "easy_to_borrow": False},
                                   {"symbol": "ABCDW", "exchange": "NASDAQ", "status": "active", "name": "Abcd Inc Warrant",
                                    "shortable": False, "easy_to_borrow": False},
                                   {"symbol": "OTCX", "exchange": "OTC", "status": "active", "name": "Otc Co",
                                    "shortable": False, "easy_to_borrow": False}])
            return self._send([])
        if p == "/v2/stocks/bars":
            assert q["adjustment"] == ["raw"] and q["feed"] == ["sip"]
            if q["timeframe"] == ["1Day"]:
                return self._send({"bars": {"ABCD": [{"t": "2025-10-15T04:00:00Z", "o": 0.5, "h": 0.6, "l": 0.45, "c": 0.52,
                                                      "v": 9000000, "n": 100, "vw": 0.51}]}, "next_page_token": None})
            if "page_token" not in q:
                return self._send({"bars": {"ABCD": [{"t": "2025-10-15T13:30:00Z", "o": 0.5, "h": 0.51, "l": 0.49, "c": 0.5, "v": 1000, "n": 3, "vw": 0.5}]},
                                   "next_page_token": "p2"})
            return self._send({"bars": {"ABCD": [{"t": "2025-10-15T13:31:00Z", "o": 0.5, "h": 0.53, "l": 0.5, "c": 0.52, "v": 2000, "n": 4, "vw": 0.51}]},
                               "next_page_token": None})
        if p == "/v1/corporate-actions":
            return self._send({"corporate_actions": {"reverse_splits": [{"symbol": "ABCD", "old_rate": 10, "new_rate": 1, "ex_date": "2025-10-15"}]},
                               "next_page_token": None})
        self._send({"error": p}, 404)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_polygon(server, monkeypatch):
    from bse.providers.polygon import PolygonProvider
    pv = PolygonProvider(api_key="k", base_url=server, rate_per_min=0)
    d = pv.daily_bars(["2025-10-15"])
    assert list(d["ticker"]) == ["ABCD", "WXYZ"] and d["l"].iloc[0] == 0.45
    frames = list(pv.minute_bars([("ABCD", "2025-10-15")], {"2025-10-15": "2025-10-14"}))
    bars = pd.concat(frames)
    assert len(bars) == 2 and bars["vw"].notna().all()          # pagination + vw fallback
    sp = pv.splits("2025-10-01", "2025-10-31")
    assert sp["ratio"].iloc[0] == 10.0                         # 1-for-10 reverse split
    ty = pv.security_types(["ABCD", "ABCDW"], ["2025-10-15"])
    assert dict(zip(ty["ticker"], ty["type"])) == {"ABCD": "CS", "ABCDW": "WARRANT"}


def test_alpaca(server, monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    monkeypatch.setenv("APCA_DATA_URL", server)
    monkeypatch.setenv("APCA_TRADING_URL", server)
    monkeypatch.setenv("APCA_RATE_PER_MIN", "0")
    from bse.providers.alpaca import AlpacaProvider
    pv = AlpacaProvider()
    assert set(pv.assets()["symbol"]) == {"ABCD", "ABCDW"}     # OTC dropped
    d = pv.daily_bars(["2025-10-15"])
    assert d["date"].iloc[0] == "2025-10-15" and d["v"].iloc[0] == 9e6
    bars = pd.concat(list(pv.minute_bars([("ABCD", "2025-10-15")], {"2025-10-15": "2025-10-14"})))
    assert len(bars) == 2 and bars["ts"].iloc[0] == MS
    assert pv.splits("2025-10-01", "2025-10-31")["ratio"].iloc[0] == 10.0
    ty = pv.security_types(["ABCD", "ABCDW"], ["2025-10-15"])
    assert dict(zip(ty["ticker"], ty["type"])) == {"ABCD": "CS", "ABCDW": "OTHER"}


def test_rate_limiter_backs_off_gently():
    from bse.providers.base import RateLimiter
    rl = RateLimiter(0)
    assert rl.slow_down() >= 60 and rl.per_min == 4.8      # unlimited start -> just under 5/min
    rl.slow_down()
    assert abs(rl.per_min - 3.84) < 1e-9                    # a second 429 trims, it does not halve
    for _ in range(10):
        rl.slow_down()
    assert rl.per_min == 3.0                                # floor
    assert RateLimiter(5).slow_down(retry_after=90) == 90   # honours Retry-After


def test_api_key_is_redacted_from_logs(caplog):
    import logging
    from bse.providers.base import log, redact
    key = "SECRETKEY123abc"
    assert key not in redact(f"Max retries exceeded with url: /v2/aggs?adjusted=false&apiKey={key} (Caused by X)")
    with caplog.at_level(logging.WARNING, logger="bse.provider"):
        log.warning("GET %s failed (%s)", "/v2/aggs", Exception(f"url: /x?apiKey={key}&a=1"))
    assert key not in caplog.text and "apiKey=***" in caplog.text
