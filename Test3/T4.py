import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error


# =========================
# 1. 读取 Hitters 数据
# =========================

# 如果你已经下载了 Hitters.csv，修改成自己的路径
data = pd.read_csv("Hitters.csv")

print("数据前5行：")
print(data.head())

print("\n数据维度：", data.shape)


# =========================
# 2. 数据预处理
# =========================

# Salary 是预测目标
# 删除 Salary 缺失的数据
data = data.dropna(subset=["Salary"])

# 将分类变量转换为哑变量
data = pd.get_dummies(data, drop_first=True)

# 分离特征和目标变量
X = data.drop("Salary", axis=1)
y = data["Salary"]

# 保证全部为数值型
X = X.astype(float)
y = y.astype(float)


# =========================
# 3. 划分训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 4. Z-score 标准化
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 5. 设置 lambda(alpha) 搜索范围
# =========================

alphas = np.logspace(-4, 4, 200)


# =========================
# 6. RidgeCV
# =========================

ridge = RidgeCV(
    alphas=alphas,
    cv=10
)

ridge.fit(X_train_scaled, y_train)

ridge_pred = ridge.predict(X_test_scaled)

ridge_rmse = np.sqrt(
    mean_squared_error(y_test, ridge_pred)
)

ridge_nonzero = np.sum(
    np.abs(ridge.coef_) > 1e-6
)


# =========================
# 7. LassoCV
# =========================

lasso = LassoCV(
    alphas=alphas,
    cv=10,
    max_iter=100000,
    random_state=42
)

lasso.fit(X_train_scaled, y_train)

lasso_pred = lasso.predict(X_test_scaled)

lasso_rmse = np.sqrt(
    mean_squared_error(y_test, lasso_pred)
)

lasso_nonzero = np.sum(
    np.abs(lasso.coef_) > 1e-6
)


# =========================
# 8. Elastic Net CV
# =========================

l1_ratios = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    0.95,
    0.99
]

elastic = ElasticNetCV(
    alphas=alphas,
    l1_ratio=l1_ratios,
    cv=10,
    max_iter=100000,
    random_state=42
)

elastic.fit(X_train_scaled, y_train)

elastic_pred = elastic.predict(X_test_scaled)

elastic_rmse = np.sqrt(
    mean_squared_error(y_test, elastic_pred)
)

elastic_nonzero = np.sum(
    np.abs(elastic.coef_) > 1e-6
)


# =========================
# 9. 输出模型结果
# =========================

print("\n================ 模型结果 ================")

print("\nRidge:")
print("最优 λ =", ridge.alpha_)
print("测试集 RMSE =", ridge_rmse)
print("非零变量数 =", ridge_nonzero)

print("\nLasso:")
print("最优 λ =", lasso.alpha_)
print("测试集 RMSE =", lasso_rmse)
print("非零变量数 =", lasso_nonzero)

print("\nElastic Net:")
print("最优 λ =", elastic.alpha_)
print("最优 l1_ratio =", elastic.l1_ratio_)
print("测试集 RMSE =", elastic_rmse)
print("非零变量数 =", elastic_nonzero)


# =========================
# 10. 构造 Ridge 系数路径
# =========================

ridge_path = []

for alpha in alphas:
    model = RidgeCV(
        alphas=[alpha],
        cv=10
    )

    model.fit(X_train_scaled, y_train)

    ridge_path.append(model.coef_)

ridge_path = np.array(ridge_path)


# =========================
# 11. 构造 Lasso 系数路径
# =========================

from sklearn.linear_model import lasso_path

_, lasso_path_coef, _ = lasso_path(
    X_train_scaled,
    y_train,
    alphas=alphas,
    max_iter=100000
)


# =========================
# 12. 构造 Elastic Net 系数路径
# =========================

from sklearn.linear_model import enet_path

_, enet_path_coef, _ = enet_path(
    X_train_scaled,
    y_train,
    l1_ratio=elastic.l1_ratio_,
    alphas=alphas,
    max_iter=100000
)


# =========================
# 13. Ridge 路径图
# =========================

plt.figure(figsize=(10, 6))

for i in range(ridge_path.shape[1]):
    plt.plot(
        alphas,
        ridge_path[:, i],
        linewidth=1
    )

plt.axvline(
    ridge.alpha_,
    linestyle="--",
    label="Selected lambda"
)

plt.xscale("log")

plt.xlabel("lambda (alpha)")
plt.ylabel("Coefficients")

plt.title("Ridge Coefficient Paths")

plt.legend()

plt.grid(True)

plt.show()


# =========================
# 14. Lasso 路径图
# =========================

plt.figure(figsize=(10, 6))

for i in range(lasso_path_coef.shape[0]):
    plt.plot(
        alphas,
        lasso_path_coef[i],
        linewidth=1
    )

plt.axvline(
    lasso.alpha_,
    linestyle="--",
    label="Selected lambda"
)

plt.xscale("log")

plt.xlabel("lambda (alpha)")
plt.ylabel("Coefficients")

plt.title("Lasso Coefficient Paths")

plt.legend()

plt.grid(True)

plt.show()


# =========================
# 15. Elastic Net 路径图
# =========================

plt.figure(figsize=(10, 6))

for i in range(enet_path_coef.shape[0]):
    plt.plot(
        alphas,
        enet_path_coef[i],
        linewidth=1
    )

plt.axvline(
    elastic.alpha_,
    linestyle="--",
    label="Selected lambda"
)

plt.xscale("log")

plt.xlabel("lambda (alpha)")

plt.ylabel("Coefficients")

plt.title(
    "Elastic Net Coefficient Paths"
    + f" (l1_ratio={elastic.l1_ratio_:.2f})"
)

plt.legend()

plt.grid(True)

plt.show()


# =========================
# 16. 汇总结果
# =========================

result = pd.DataFrame({
    "Model": [
        "Ridge",
        "Lasso",
        "Elastic Net"
    ],

    "Best Lambda": [
        ridge.alpha_,
        lasso.alpha_,
        elastic.alpha_
    ],

    "RMSE": [
        ridge_rmse,
        lasso_rmse,
        elastic_rmse
    ],

    "Non-zero Variables": [
        ridge_nonzero,
        lasso_nonzero,
        elastic_nonzero
    ]
})

print("\n================ 最终比较 ================")
print(result.to_string(index=False))