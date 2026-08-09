import streamlit as st
import pandas as pd
import os


FILE = "prediction_history.csv"


def show_simulation():

    st.divider()

    st.subheader(
        "💰 Project X 仮想運用"
    )

    try:

        start_money = 1000000

        if os.path.exists(FILE):

            sim = pd.read_csv(FILE)

            if "result" in sim.columns:

                win_count = len(
                    sim[
                        sim["result"] == "WIN"
                    ]
                )

                lose_count = len(
                    sim[
                        sim["result"] == "LOSE"
                    ]
                )

                total = (
                    win_count +
                    lose_count
                )

                if total > 0:

                    win_rate = (
                        win_count /
                        total
                    )

                    # 1回あたり仮想利益5%
                    # 負け3%

                    estimated = (
                        start_money *
                        (
                            1
                            +
                            (
                                win_rate * 0.05
                                -
                                (1 - win_rate) * 0.03
                            )
                            * total
                        )
                    )

                    profit = (
                        estimated -
                        start_money
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "開始資金",
                        "100万円"
                    )

                    c2.metric(
                        "評価額",
                        f"{estimated:,.0f}円"
                    )

                    c3.metric(
                        "損益",
                        f"{profit:+,.0f}円"
                    )

                else:

                    st.info(
                        "まだ検証データ不足"
                    )

            else:

                st.info(
                    "結果判定待ち"
                )

        else:

            st.info(
                "まだ検証データ不足"
            )

    except Exception as e:

        st.warning(
            f"シミュレーション計算不可: {e}"
        )