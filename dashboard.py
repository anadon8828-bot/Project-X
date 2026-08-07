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

def show_ai_diagnosis(latest, probability, final_score):

    st.divider()

    st.subheader("🧠 AI診断")

    try:

        if final_score >= 80:
            grade = "★★★★★ 強気"
        elif final_score >= 60:
            grade = "★★★★ 期待"
        elif final_score >= 40:
            grade = "★★★ 中立"
        else:
            grade = "★★ 注意"

        st.success(grade)

        diagnosis = []

        if probability >= 0.65:
            diagnosis.append("✅ AI上昇確率が高い")
        else:
            diagnosis.append("⚠ AI上昇確率は低め")

        if latest["MACD"] > latest["Signal"]:
            diagnosis.append("✅ MACD買いシグナル")
        else:
            diagnosis.append("⚠ MACD弱い")

        if latest["Close"] > latest["MA25"]:
            diagnosis.append("✅ 短期トレンド上向き")
        else:
            diagnosis.append("⚠ 株価は移動平均以下")

        if latest["RSI"] < 30:
            diagnosis.append("🔥 売られすぎ反発期待")
        elif latest["RSI"] > 70:
            diagnosis.append("⚠ 過熱注意")

        for item in diagnosis:
            st.write(item)

    except:
        st.info("分析後に表示されます")