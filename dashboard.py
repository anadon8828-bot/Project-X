import streamlit as st


def show_ai_judgment(signal, probability):

    st.divider()

    st.success(
        f"""
## 🚀 今日のAI判断

### {signal}

AI上昇確率：**{probability*100:.1f}%**
"""
    )
def show_dashboard(latest, probability, final_score):

    st.divider()

    st.subheader(
        "🚀 Project X ダッシュボード"
    )

    try:

        dash1, dash2, dash3, dash4 = st.columns(4)

        dash1.metric(
            "現在値",
            f"{latest['Close']:.2f}円"
        )

        dash2.metric(
            "AI上昇確率",
            f"{probability*100:.1f}%"
        )

        dash3.metric(
            "総合スコア",
            f"{final_score}/100"
        )


        if final_score >= 80:

            dash4.error(
                "🔴 BUY"
            )

        elif final_score >= 60:

            dash4.warning(
                "🟡 HOLD"
            )

        else:

            dash4.success(
                "🟢 WAIT"
            )

    except:

        st.info(
            "分析後に表示されます"
        )