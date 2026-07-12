import yfinance as yf
import matplotlib.pyplot as plt

# 日本語フォント（Windows）
plt.rcParams["font.family"] = "Meiryo"

print("===================================")
print("      Project X 株価チャート")
print("===================================")

code = input("銘柄コードを入力してください（例: 7203）: ")

ticker = yf.Ticker(code + ".T")

# 過去1年分のデータ取得
data = ticker.history(period="1y")

if data.empty:
    print("銘柄が見つかりませんでした。")
else:
    # 移動平均線
    data["MA25"] = data["Close"].rolling(window=25).mean()
    data["MA75"] = data["Close"].rolling(window=75).mean()

    latest = data.iloc[-1]

    print("\n===== 現在の情報 =====")
    print(f"現在値 : {latest['Close']:.2f}円")
    print(f"始値   : {latest['Open']:.2f}円")
    print(f"高値   : {latest['High']:.2f}円")
    print(f"安値   : {latest['Low']:.2f}円")
    print(f"出来高 : {int(latest['Volume']):,}株")

    plt.figure(figsize=(14, 7))

    plt.plot(data.index, data["Close"], label="Close", linewidth=2)
    plt.plot(data.index, data["MA25"], label="MA25", linewidth=2)
    plt.plot(data.index, data["MA75"], label="MA75", linewidth=2)

    plt.title(f"{code} Stock Chart (1 Year)")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.legend()
    plt.grid(True)

    plt.show()