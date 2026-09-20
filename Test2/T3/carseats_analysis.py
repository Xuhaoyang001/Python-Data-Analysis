import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==========================================
# 1. 读取数据
# ==========================================

data = pd.read_csv("Carseats.csv")

print("数据前5行：")
print(data.head())

print("\n数据基本信息：")
print(data.info())


# ==========================================
# 2. 设置 ShelveLoc 分类变量
# ==========================================

data["ShelveLoc"] = pd.Categorical(
    data["ShelveLoc"],
    categories=["Bad", "Medium", "Good"]
)


# ==========================================
# 3. 建立多元线性回归模型
# ==========================================

model = smf.ols(
    "Sales ~ Price + Income + Advertising + C(ShelveLoc, Treatment(reference='Bad'))",
    data=data
).fit()


# ==========================================
# 4. 输出模型拟合报告
# ==========================================

print("\n================ 模型拟合报告 ================\n")
print(model.summary())


# ==========================================
# 5. 输出回归系数
# ==========================================

print("\n================ 回归系数 ================\n")
print(model.params)


# ==========================================
# 6. 输出 ShelveLoc[Good] 系数
# ==========================================

good_name = "C(ShelveLoc, Treatment(reference='Bad'))[T.Good]"

print("\n================ ShelveLoc[Good] ================\n")

if good_name in model.params.index:
    print("ShelveLoc[Good] 系数：", model.params[good_name])
    print("P值：", model.pvalues[good_name])
else:
    print("没有找到 ShelveLoc[Good] 系数，请检查数据中的 ShelveLoc 分类。")


# ==========================================
# 7. 计算 VIF
# ==========================================

X = pd.get_dummies(
    data[["Price", "Income", "Advertising", "ShelveLoc"]],
    columns=["ShelveLoc"],
    drop_first=True
)

# 确保数据为数值类型
X = X.astype(float)

# 添加常数项
X.insert(0, "Intercept", 1.0)


vif = pd.DataFrame()

vif["Variable"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]


print("\n================ VIF 检验 ================\n")
print(vif)


# ==========================================
# 8. 输出 R² 和调整后的 R²
# ==========================================

print("\n================ 模型评价 ================\n")

print("R² =", model.rsquared)

print("调整后的 R² =", model.rsquared_adj)