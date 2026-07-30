import joblib

# 元モデル読み込み
model = joblib.load("model.pkl")

# 圧縮保存
joblib.dump(
    model,
    "model_small.pkl",
    compress=9
)

print("圧縮保存完了")
