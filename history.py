import pandas as pd
import os
from datetime import datetime


FILE = "prediction_history.csv"


def save_prediction(
    code,
    price,
    probability,
    score,
    signal
):

    new_data = pd.DataFrame(
        [
            {
                "date":
                    datetime.now().strftime("%Y-%m-%d"),

                "code":
                    code,

                "price":
                    price,

                "AI確率":
                    round(probability*100,1),

                "score":
                    score,

                "signal":
                    signal,

                "result":
                    ""
            }
        ]
    )


    if os.path.exists(FILE):

        old = pd.read_csv(FILE)

        df = pd.concat(
            [
                old,
                new_data
            ],
            ignore_index=True
        )

    else:

        df = new_data


    df.to_csv(
        FILE,
        index=False,
        encoding="utf-8-sig"
    )