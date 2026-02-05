import numpy as np
import plotly.graph_objects as go
import talib as ta
import yfinance as yf
from plotly.subplots import make_subplots

from signal_alerts import check_and_alert_latest_signals

TICKER = "BTC-USD"
PERIOD = "1y"
INTERVAL = "1d" 
VOLUME_PROFILE_BINS = 30

def _to_series(data, column: str):
    series = data[column]
    if hasattr(series, "ndim") and series.ndim > 1:
        series = series.iloc[:, 0]
    return series.astype("float64")


def fetch_market_data(ticker: str = TICKER, period: str = PERIOD, interval: str = INTERVAL):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    return {
        "close": _to_series(data, "Close"),
        "high": _to_series(data, "High"),
        "low": _to_series(data, "Low"),
        "volume": _to_series(data, "Volume"),
    }


def compute_indicators(close, high, low, volume):
    return {
        "ma50": ta.MA(close.to_numpy(), timeperiod=50),
        "ma200": ta.MA(close.to_numpy(), timeperiod=200),
        "rsi": ta.RSI(close.to_numpy(), timeperiod=14),
        "atr": ta.ATR(high.to_numpy(), low.to_numpy(), close.to_numpy(), timeperiod=14),
        "obv": ta.OBV(close.to_numpy(), volume.to_numpy()),
    }


def build_signal_dataframe(close, indicators):
    df_indicators = close.to_frame(name="close")
    df_indicators["rsi"] = indicators["rsi"]
    df_indicators["atr"] = indicators["atr"]
    df_indicators["obv"] = indicators["obv"]
    df_indicators["ma50"] = indicators["ma50"]
    df_indicators["ma200"] = indicators["ma200"]
    return df_indicators


def build_volume_profile(close, low, high, volume, num_bins: int = VOLUME_PROFILE_BINS):
    price_min = float(low.min())
    price_max = float(high.max())
    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_index = np.digitize(close.to_numpy(), bin_edges) - 1
    bin_index = np.clip(bin_index, 0, num_bins - 1)
    volume_profile = np.zeros(num_bins, dtype="float64")
    for i, vol in enumerate(volume.to_numpy()):
        volume_profile[bin_index[i]] += vol
    return volume_profile, bin_centers


def build_figure(close, ma50, ma200, volume_profile, bin_centers):
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.04,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Prix BTC + MA50/MA200", "Volume Profile"),
    )

    fig.add_trace(go.Scatter(x=close.index, y=close.values, mode="lines", name="BTC Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=close.index, y=ma50, mode="lines", name="MA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=close.index, y=ma200, mode="lines", name="MA 200"), row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=volume_profile,
            y=bin_centers,
            orientation="h",
            name="Volume Profile",
            marker_color="#8fb9a8",
            opacity=0.6,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Bitcoin (1y): Prix + MA50/MA200 avec Volume Profile",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        hovermode="x",
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%d %b %Y", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=2)
    return fig


def main():
    market = fetch_market_data()
    indicators = compute_indicators(market["close"], market["high"], market["low"], market["volume"])
    df_indicators = build_signal_dataframe(market["close"], indicators)
    check_and_alert_latest_signals(df_indicators)

    volume_profile, bin_centers = build_volume_profile(
        market["close"], market["low"], market["high"], market["volume"]
    )
    fig = build_figure(
        market["close"], indicators["ma50"], indicators["ma200"], volume_profile, bin_centers
    )
    fig.show()


if __name__ == "__main__":
    main()
