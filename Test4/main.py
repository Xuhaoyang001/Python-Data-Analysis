import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. 读取数据
# ============================================================

file_path = "netflix_titles.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("Netflix 数据集基本信息")
print("=" * 60)

print("数据集大小：", df.shape)

print("\n前5行数据：")
print(df.head())

print("\n数据类型：")
print(df.dtypes)

print("\n缺失值统计：")
print(df.isnull().sum())


# ============================================================
# 2. 只保留 Movie
# ============================================================

df = df[df["type"] == "Movie"].copy()

print("\n" + "=" * 60)
print("只保留 Movie 后")
print("=" * 60)

print("数据量：", len(df))


# ============================================================
# 3. 提取电影时长
# ============================================================

# duration 原始格式：
# 90 min
# 120 min
# 95 min

df["duration_min"] = (
    df["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)

# 删除没有电影时长的数据
df = df.dropna(subset=["duration_min"])

print("\n电影时长统计：")
print(df["duration_min"].describe())


# ============================================================
# 4. 提取 date_added 中的年份
# ============================================================

df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)

df["year_added"] = df["date_added"].dt.year


# ============================================================
# 5. 特征工程
# ============================================================

# -----------------------------
# 演员数量
# -----------------------------

def count_items(x):
    if pd.isna(x):
        return 0

    return len(str(x).split(","))


df["cast_count"] = df["cast"].apply(count_items)


# -----------------------------
# 国家数量
# -----------------------------

df["country_count"] = df["country"].apply(count_items)


# -----------------------------
# 类型数量
# -----------------------------

df["genre_count"] = df["listed_in"].apply(count_items)


# -----------------------------
# 是否有导演
# -----------------------------

df["has_director"] = df["director"].notna().astype(int)


# ============================================================
# 6. 删除不需要的变量
# ============================================================

# 我们最终使用：
#
# release_year
# year_added
# cast_count
# country_count
# genre_count
# has_director
# rating
#
# 预测：
#
# duration_min

features = [
    "release_year",
    "year_added",
    "cast_count",
    "country_count",
    "genre_count",
    "has_director",
    "rating"
]

target = "duration_min"

data = df[features + [target]].copy()


# ============================================================
# 7. 缺失值处理
# ============================================================

# year_added 缺失
data["year_added"] = data["year_added"].fillna(
    data["year_added"].median()
)

# rating 缺失
data["rating"] = data["rating"].fillna("Unknown")


print("\n" + "=" * 60)
print("处理后的数据")
print("=" * 60)

print(data.head())

print("\n处理后的缺失值：")
print(data.isnull().sum())


# ============================================================
# 8. 定义 X 和 y
# ============================================================

X = data[features]

y = data[target]


# ============================================================
# 9. 数值变量和分类变量
# ============================================================

numeric_features = [
    "release_year",
    "year_added",
    "cast_count",
    "country_count",
    "genre_count",
    "has_director"
]

categorical_features = [
    "rating"
]


# ============================================================
# 10. 训练集和测试集
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)


# ============================================================
# 11. 数据预处理
# ============================================================

# 数值变量：
# StandardScaler -> Z-score 标准化
#
# 分类变量：
# One-Hot Encoding

from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop=None
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 12. Elastic Net
# ============================================================

elastic_net = ElasticNetCV(
    alphas=np.logspace(-3, 2, 50),

    l1_ratio=[
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ],

    cv=5,

    max_iter=10000,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 13. 建立 Pipeline
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "elasticnet",
            elastic_net
        )
    ]
)


# ============================================================
# 14. 训练模型
# ============================================================

print("\n" + "=" * 60)
print("开始训练 Elastic Net")
print("=" * 60)

model.fit(X_train, y_train)

print("模型训练完成！")


# ============================================================
# 15. 获取最佳参数
# ============================================================

best_alpha = model.named_steps[
    "elasticnet"
].alpha_

best_l1_ratio = model.named_steps[
    "elasticnet"
].l1_ratio_

print("\n" + "=" * 60)
print("Elastic Net 最优参数")
print("=" * 60)

print("最佳 alpha：", best_alpha)

print("最佳 l1_ratio：", best_l1_ratio)


# ============================================================
# 16. 模型预测
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 17. 模型评价
# ============================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("Elastic Net 模型评价")
print("=" * 60)

print("MSE  =", mse)

print("RMSE =", rmse)

print("MAE  =", mae)

print("R²   =", r2)


# ============================================================
# 18. 获取特征名称
# ============================================================

preprocessor_fitted = model.named_steps[
    "preprocessor"
]

feature_names = preprocessor_fitted.get_feature_names_out()


# ============================================================
# 19. 获取 Elastic Net 系数
# ============================================================

coefficients = model.named_steps[
    "elasticnet"
].coef_


coef_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients
    }
)


# 去掉 Pipeline 自动添加的前缀
coef_df["Feature"] = (
    coef_df["Feature"]
    .str.replace(
        "num__",
        "",
        regex=False
    )
    .str.replace(
        "cat__rating_",
        "rating_",
        regex=False
    )
)


# 按绝对值排序
coef_df["AbsCoefficient"] = (
    coef_df["Coefficient"].abs()
)

coef_df = coef_df.sort_values(
    "AbsCoefficient",
    ascending=False
)


# ============================================================
# 20. 输出全部系数
# ============================================================

print("\n" + "=" * 60)
print("Elastic Net 特征系数")
print("=" * 60)

print(
    coef_df[
        [
            "Feature",
            "Coefficient"
        ]
    ].to_string(index=False)
)


# ============================================================
# 21. 输出重要特征
# ============================================================

print("\n" + "=" * 60)
print("影响最大的特征")
print("=" * 60)

print(
    coef_df[
        [
            "Feature",
            "Coefficient"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 22. 被压缩为 0 的变量
# ============================================================

zero_coef = coef_df[
    coef_df["Coefficient"] == 0
]

print("\n" + "=" * 60)
print("被 Elastic Net 压缩为 0 的变量")
print("=" * 60)

if len(zero_coef) == 0:

    print("没有变量被压缩为 0")

else:

    print(
        zero_coef[
            [
                "Feature",
                "Coefficient"
            ]
        ].to_string(index=False)
    )

print("\n被压缩为 0 的变量数量：", len(zero_coef))


# ============================================================
# 23. 绘制特征系数图
# ============================================================

top_n = 15

plot_df = coef_df.head(top_n).copy()

plot_df = plot_df.sort_values(
    "Coefficient"
)


plt.figure(figsize=(10, 7))

plt.barh(
    plot_df["Feature"],
    plot_df["Coefficient"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel("Elastic Net Coefficient")

plt.ylabel("Feature")

plt.title(
    "Top 15 Elastic Net Feature Coefficients"
)

plt.tight_layout()

plt.show()


# ============================================================
# 24. 实际值与预测值
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# 理想预测线
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value]
)

plt.xlabel("Actual Duration")

plt.ylabel("Predicted Duration")

plt.title(
    "Actual vs Predicted Movie Duration"
)

plt.tight_layout()

plt.show()


# ============================================================
# 25. 保存系数结果
# ============================================================

coef_df[
    [
        "Feature",
        "Coefficient"
    ]
].to_csv(
    "elastic_net_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n系数结果已经保存为：")
print("elastic_net_coefficients.csv")


# ============================================================
# 26. 最终结果总结
# ============================================================

print("\n" + "=" * 60)
print("Elastic Net 最终结果")
print("=" * 60)

print(f"最佳 alpha     : {best_alpha:.6f}")

print(f"最佳 l1_ratio  : {best_l1_ratio:.2f}")

print(f"MSE            : {mse:.4f}")

print(f"RMSE           : {rmse:.4f}")

print(f"MAE            : {mae:.4f}")

print(f"R²             : {r2:.4f}")

print(
    f"零系数特征数量 : {len(zero_coef)}"
)

print("=" * 60)