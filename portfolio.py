import yfinance as yf
import pandas as pd


def calculate_portfolio(portfolio):

    result = []

    total_profit = 0


    for stock in portfolio:

        try:

            data = yf.Ticker(
                stock["code"] + ".T"
            ).history(
                period="5d"
            )


            if data.empty:
                continue


            current_price = (
                data["Close"]
                .iloc[-1]
            )


            profit = (
                current_price
                -
                stock["buy"]
            ) * stock["amount"]


            profit_rate = (
                (current_price - stock["buy"])
                /
                stock["buy"]
                *
                100
            )


            total_profit += profit


            result.append(
                {
                    "銘柄":
                        stock["name"],

                    "コード":
                        stock["code"],

                    "株数":
                        stock["amount"],

                    "現在値":
                        round(
                            current_price,
                            2
                        ),

                    "損益":
                        round(
                            profit
                        ),

                    "損益率":
                        round(
                            profit_rate,
                            2
                        )
                }
            )


        except:

            pass


    return (
        pd.DataFrame(result),
        total_profit
    )
