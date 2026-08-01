import pandas as pd
import yfinance as yf


def create_ranking(model, codes):

    ranking = []


    for code in codes:

        try:

            data = yf.Ticker(
                code + ".T"
            ).history(
                period="1y"
            )


            if len(data) < 75:
                continue


            data["MA25"] = (
                data["Close"]
                .rolling(25)
                .mean()
            )


            data["MA75"] = (
                data["Close"]
                .rolling(75)
                .mean()
            )


            delta = data["Close"].diff()


            gain = delta.clip(
                lower=0
            )


            loss = -delta.clip(
                upper=0
            )


            rs = (
                gain.rolling(14).mean()
                /
                loss.rolling(14).mean()
            )


            data["RSI"] = (
                100 -
                100/(1+rs)
            )


            ema12 = (
                data["Close"]
                .ewm(span=12)
                .mean()
            )


            ema26 = (
                data["Close"]
                .ewm(span=26)
                .mean()
            )


            data["MACD"] = (
                ema12 - ema26
            )


            latest = data.iloc[-1]


            input_data = pd.DataFrame(
                [[
                    latest["MA25"],
                    latest["MA75"],
                    latest["RSI"],
                    latest["MACD"],
                    latest["Volume"]
                ]],
                columns=[
                    "MA25",
                    "MA75",
                    "RSI",
                    "MACD",
                    "Volume"
                ]
            )


            probability = (
                model.predict_proba(
                    input_data
                )[0][1]
            )


            # テクニカルスコア

            score = 0


            if latest["Close"] > latest["MA25"]:
                score += 20


            if latest["MA25"] > latest["MA75"]:
                score += 20


            if latest["MACD"] > 0:
                score += 20


            if 40 <= latest["RSI"] <= 70:
                score += 20


            if probability >= 0.6:
                score += 20



            total_score = int(
                score
            )


            if total_score >= 80:

                signal = "🔴 BUY"


            elif total_score >= 60:

                signal = "🟡 HOLD"


            else:

                signal = "🟢 WAIT"



            ranking.append(
                {
                    "コード": code,

                    "現在値":
                        round(
                            latest["Close"],
                            1
                        ),

                    "AI上昇確率(%)":
                        round(
                            probability * 100,
                            1
                        ),

                    "総合スコア":
                        total_score,

                    "判断":
                        signal
                }
            )


        except Exception:

            pass



    if ranking:


        result = pd.DataFrame(
            ranking
        )


        result = result.sort_values(
            [
                "総合スコア",
                "AI上昇確率(%)"
            ],
            ascending=False
        )


        return result



    return pd.DataFrame()
