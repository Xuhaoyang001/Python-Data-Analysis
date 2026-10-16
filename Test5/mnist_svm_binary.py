import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==============================
# 1. 读取数据集
# ==============================
DATA_PATH = "train.csv"

print("=" * 60)
print("MNIST 手写数字二分类：SVM")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\n原始数据集大小：", df.shape)

# ==============================
# 2. 筛选数字 0 和 1
# ==============================
df = df[df["label"].isin([0, 1])].copy()

print("筛选后的数据集大小：", df.shape)
print("\n数字分布：")
print(df["label"].value_counts().sort_index())

# 分离特征和标签
X = df.drop(columns=["label"]).to_numpy(dtype="float32")
y = df["label"].to_numpy()

# 像素值归一化到 0~1
X = X / 255.0

print("\n特征维度：", X.shape[1])

# ==============================
# 3. 控制样本数量
# ==============================
MAX_SAMPLES = 6000

if MAX_SAMPLES is not None and len(y) > MAX_SAMPLES:
    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=MAX_SAMPLES,
        random_state=42,
        stratify=y
    )

print("实际使用样本数：", len(y))

# ==============================
# 4. 划分训练集和测试集
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

# ==============================
# 5. 定义模型
# ==============================
models = {
    "线性 SVM": LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=42
    ),

    "RBF 核 SVM": SVC(
        kernel="rbf",
        C=5.0,
        gamma="scale",
        cache_size=1024
    )
}

results = []

# ==============================
# 6. 训练和评估模型
# ==============================
for name, model in models.items():

    print("\n" + "=" * 60)
    print("当前模型：", name)
    print("=" * 60)

    # 记录训练时间
    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    train_time = time.perf_counter() - start_time

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算评价指标
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # 保存结果
    results.append({
        "模型": name,
        "准确率": accuracy,
        "精确率": precision,
        "召回率": recall,
        "F1值": f1,
        "训练时间(秒)": train_time
    })

    # 输出评价指标
    print(f"训练时间：{train_time:.3f} 秒")
    print(f"准确率：{accuracy:.4%}")
    print(f"精确率：{precision:.4%}")
    print(f"召回率：{recall:.4%}")
    print(f"F1 值：{f1:.4%}")

    print("\n分类报告：")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["数字 0", "数字 1"],
        zero_division=0
    ))

    # 输出混淆矩阵
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("混淆矩阵：")
    print(cm)

# ==============================
# 7. 汇总模型对比结果
# ==============================
result_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("SVM 模型性能对比")
print("=" * 80)

print(result_df.to_string(
    index=False,
    formatters={
        "准确率": "{:.4%}".format,
        "精确率": "{:.4%}".format,
        "召回率": "{:.4%}".format,
        "F1值": "{:.4%}".format,
        "训练时间(秒)": "{:.3f}".format
    }
))

