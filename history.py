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

def show_history():

        import streamlit as st

        st.divider()

        st.subheader(
            "🏆 AI実績分析"
        )

        try:

            if os.path.exists(FILE):

                history_df = pd.read_csv(FILE)

                total = len(history_df)

                st.metric(
                    "AI予測回数",
                    f"{total}回"
                )


                if "result" in history_df.columns:

                    win = len(
                        history_df[
                            history_df["result"] == "WIN"
                        ]
                    )

                    lose = len(
                        history_df[
                            history_df["result"] == "LOSE"
                        ]
                    )


                    finished = win + lose


                    if finished > 0:

                        rate = (
                            win /
                            finished *
                            100
                        )


                        c1,c2,c3 = st.columns(3)

                        c1.metric(
                            "勝率",
                            f"{rate:.1f}%"
                        )

                        c2.metric(
                            "勝ち",
                            f"{win}回"
                        )

                        c3.metric(
                            "負け",
                            f"{lose}回"
                        )

                    else:

                        st.info(
                            "まだ結果判定待ちです"
                        )


                st.dataframe(
                    history_df,
                    width="stretch"
                )


            else:

                st.info(
                    "まだAI予測履歴がありません"
                )


        except:

            st.warning(
                "実績分析エラー"
            )