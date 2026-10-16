import os
import re
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ==================== 1. 文件路径 ====================

DATA_PATH = "data_set_ALL_AML_train.csv"
LABEL_PATH = "actual.csv"


# ==================== 2. 读取数据 ====================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到文件：{DATA_PATH}")

    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"找不到文件：{LABEL_PATH}")

    # 读取基因表达矩阵
    expression_df = pd.read_csv(DATA_PATH)

    # 读取样本标签
    label_df = pd.read_csv(LABEL_PATH)

    # 清理列名两侧的空格
    expression_df.columns = expression_df.columns.astype(str).str.strip()
    label_df.columns = label_df.columns.astype(str).str.strip()

    # 检查标签文件格式
    required_columns = {"patient", "cancer"}

    if not required_columns.issubset(set(label_df.columns)):
        raise ValueError(
            "actual.csv 必须包含 patient 和 cancer 两列。\n"
            f"当前列名：{label_df.columns.tolist()}"
        )

    # 样本编号转换为整数
    label_df["patient"] = pd.to_numeric(
        label_df["patient"], errors="coerce"
    )

    label_df = label_df.dropna(subset=["patient", "cancer"])
    label_df["patient"] = label_df["patient"].astype(int)
    label_df["cancer"] = label_df["cancer"].astype(str).str.strip().str.upper()

    # 仅保留 ALL 和 AML 两类
    label_df = label_df[
        label_df["cancer"].isin(["ALL", "AML"])
    ]

    # 建立“样本编号 -> 癌症类型”的映射
    label_map = dict(
        zip(label_df["patient"], label_df["cancer"])
    )

    # 找出样本列：
    # 列名为数字的才是样本，例如 1、2、3、27、34 等。
    # call、call.1 等列不会被选中。
    sample_columns = [
        col for col in expression_df.columns
        if re.fullmatch(r"\d+", col)
    ]

    if not sample_columns:
        raise ValueError(
            "没有找到数字形式的样本列，请检查 CSV 表头。"
        )

    print("=" * 60)
    print("白血病基因表达数据集")
    print("=" * 60)

    print("原始数据大小：", expression_df.shape)
    print("识别到的样本数量：", len(sample_columns))
    print("标签文件中的样本数量：", len(label_map))

    # 检查表达矩阵与标签是否匹配
    matched_columns = [
        col for col in sample_columns
        if int(col) in label_map
    ]

    unmatched_columns = [
        col for col in sample_columns
        if int(col) not in label_map
    ]

    if unmatched_columns:
        print("没有找到标签的样本：", unmatched_columns)

    if not matched_columns:
        raise ValueError(
            "表达矩阵中的样本编号与 actual.csv 无法匹配。"
        )

    # 只保留有标签的样本
    sample_columns = matched_columns

    # 提取基因表达数值
    # 原始数据：每行一个基因，每列一个样本
    gene_expression = expression_df[sample_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # 删除完全没有数值的基因
    valid_rows = gene_expression.notna().any(axis=1)
    gene_expression = gene_expression.loc[valid_rows]

    # 转置：每行一个样本，每列一个基因
    X_df = gene_expression.T

    # 转换样本编号
    X_df.index = X_df.index.astype(int)

    # 按样本编号获取标签，确保顺序严格一致
    y_series = pd.Series(
        X_df.index.map(label_map),
        index=X_df.index,
        name="cancer"
    )

    # 删除标签缺失的样本
    valid_samples = y_series.notna()
    X_df = X_df.loc[valid_samples]
    y_series = y_series.loc[valid_samples]

    # 转换为 NumPy 数组
    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy()

    print("最终样本数量：", X.shape[0])
    print("基因特征数量：", X.shape[1])
    print("类别分布：")
    print(y_series.value_counts().to_string())

    print("\n样本编号与标签示例：")
    print(y_series.head(10).to_string())

    return X, y


# ==================== 3. 建立模型 ====================

def build_models():
    models = {
        "线性 SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", C=1.0))
        ]),

        "KNN (k=3)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3))
        ]),

        "决策树": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                random_state=42
            ))
        ])
    }

    return models


# ==================== 4. 留一交叉验证 ====================

def evaluate_models(X, y):
    loo = LeaveOneOut()
    models = build_models()

    # 固定类别顺序，保证混淆矩阵一致
    labels = ["ALL", "AML"]

    print("\n" + "=" * 60)
    print("留一交叉验证（LOOCV）")
    print("=" * 60)

    results = []

    for name, model in models.items():
        print(f"\n正在评估：{name}")

        start_time = time.time()

        # 每次留出一个样本测试，其余样本训练
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=loo,
            n_jobs=1
        )

        elapsed_time = time.time() - start_time

        accuracy = accuracy_score(y, y_pred)

        precision = precision_score(
            y, y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y, y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y, y_pred,
            average="macro",
            zero_division=0
        )

        cm = confusion_matrix(
            y, y_pred,
            labels=labels
        )

        results.append({
            "算法": name,
            "准确率": accuracy,
            "宏平均精确率": precision,
            "宏平均召回率": recall,
            "宏平均F1": f1,
            "耗时(秒)": elapsed_time
        })

        print(f"交叉验证折数：{len(y)}")
        print(f"准确率：{accuracy:.4f}")
        print(f"宏平均精确率：{precision:.4f}")
        print(f"宏平均召回率：{recall:.4f}")
        print(f"宏平均 F1：{f1:.4f}")
        print(f"运行时间：{elapsed_time:.4f} 秒")

        print("\n分类报告：")
        print(classification_report(
            y,
            y_pred,
            labels=labels,
            zero_division=0
        ))

        print("混淆矩阵（行=真实类别，列=预测类别）：")
        print("类别顺序：", labels)
        print(cm)

    # ==================== 5. 汇总结果 ====================

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 75)
    print("三种算法的 LOOCV 结果对比")
    print("=" * 75)

    print(result_df.to_string(
        index=False,
        formatters={
            "准确率": "{:.4f}".format,
            "宏平均精确率": "{:.4f}".format,
            "宏平均召回率": "{:.4f}".format,
            "宏平均F1": "{:.4f}".format,
            "耗时(秒)": "{:.4f}".format
        }
    ))

    best_index = result_df["准确率"].idxmax()
    best_model = result_df.loc[best_index]

    print("\n准确率最高的算法：", best_model["算法"])
    print(f"LOOCV 准确率：{best_model['准确率']:.4f}")


# ==================== 6. 主程序 ====================

def main():
    try:
        X, y = load_data()

        if len(np.unique(y)) < 2:
            raise ValueError(
                "数据中不足两个类别，无法执行二分类。"
            )

        # LOOCV 中每次训练集少一个样本
        if len(y) < 4:
            raise ValueError(
                "有效样本数量过少，无法可靠地比较模型。"
            )

        evaluate_models(X, y)

    except Exception as e:
        print("\n程序运行出错：")
        print(e)


if __name__ == "__main__":
    main()