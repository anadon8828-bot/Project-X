import joblib
import pandas as pd
import os


class AIEngine:

    def __init__(self):

        if not os.path.exists("model.pkl"):
            raise FileNotFoundError("model.pkl がありません")

        self.model = joblib.load("model.pkl")


    def predict(self, latest):

        features = pd.DataFrame(
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

        probability = self.model.predict_proba(features)[0][1]

        if probability >= 0.65:
            signal = "BUY"

        elif probability >= 0.50:
            signal = "WAIT"

        else:
            signal = "SELL"

        return probability, signal