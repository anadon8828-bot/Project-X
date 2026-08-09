import streamlit as st
import pandas as pd
import os


FILE = "prediction_history.csv"


def show_stock_results():

    st.divider()

    st.subheader(
        "🧠 AI銘柄別成績"
    )

    try:

        if os.path.exists(FILE):

            df_result = pd.read_csv(FILE)

            if (
                "code" in df_result.columns
                and
                "result" in df_result.columns
            ):

                result_df = df_result[
                    df_result["result"].isin(
                        [
                            "WIN",
                            "LOSE"
                        ]
                    )
                ]

                if len(result_df) > 0:

                    stock_result = []

                    for stock_code in result_df["code"].unique():

                        stock_data = result_df[
                            result_df["code"] == stock_code
                        ]

                        wins = len(
                            stock_data[
                                stock_data["result"] == "WIN"
                            ]
                        )

                        total = len(stock_data)

                        rate = (
                            wins /
                            total *
                            100
                        )

                        stock_result.append(
                            {
                                "銘柄コード":
                                    stock_code,

                                "予測回数":
                                    total,

                                "勝率(%)":
                                    round(
                                        rate,
                                        1
                                    )
                            }
                        )

                    stock_df = pd.DataFrame(
                        stock_result
                    )

                    stock_df = stock_df.sort_values(
                        "勝率(%)",
                        ascending=False
                    )

                    st.dataframe(
                        stock_df,
                        width="stretch"
                    )

                else:

                    st.info(
                        "判定済みデータがありません"
                    )

            else:

                st.info(
                    "分析データ不足"
                )

        else:

            st.info(
                "履歴なし"
            )

    except Exception as e:

        st.warning(
            f"銘柄別分析エラー: {e}"
        )