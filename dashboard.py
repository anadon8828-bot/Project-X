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