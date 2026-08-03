import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_chart(data):

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[
            0.5,
            0.25,
            0.25
        ]
    )


    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="株価"
        ),
        row=1,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA25"],
            name="MA25"
        ),
        row=1,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA75"],
            name="MA75"
        ),
        row=1,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI"
        ),
        row=2,
        col=1
    )


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD"
        ),
        row=3,
        col=1
    )


    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False
    )


    return fig