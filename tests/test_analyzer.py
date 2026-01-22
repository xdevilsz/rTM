from analyzer import TradeAnalyzer


def test_basic_statistics_with_single_fill():
    fills = [
        {
            "ts": 1_700_000_000,
            "side": "Buy",
            "qty": 2,
            "price": 100.0,
            "notional": 200.0,
            "fee": 0.5,
            "exec_pnl": 1.25,
            "symbol": "BTCUSDT",
        }
    ]
    stats = TradeAnalyzer(fills).get_basic_statistics()

    assert stats["total_trades"] == 1
    assert stats["buy_count"] == 1
    assert stats["sell_count"] == 0
    assert stats["total_volume"] == 200.0
    assert stats["total_fee"] == 0.5
    assert stats["net_pnl"] == 0.75
    assert stats["symbols"] == ["BTCUSDT"]


def test_basic_statistics_with_no_fills_returns_empty():
    stats = TradeAnalyzer([]).get_basic_statistics()

    assert stats == {}
