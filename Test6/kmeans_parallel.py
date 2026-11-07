#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于多线程 K-means++ 的 Kaggle 信用卡客户聚类分析
=====================================================

数据集：Kaggle - Credit Card Dataset for Clustering (CC GENERAL)
        8950 位信用卡持卡人，17 个数值型行为特征。

本脚本完成：
1. 数据读取、缺失值处理、特征选择、对数变换与 Z-score 标准化；
2. 用肘部法则 + 轮廓系数选择聚类数 k；
3. 手写 K-means++ 初始化与 Lloyd 迭代，其中“分配步”用
   concurrent.futures.ThreadPoolExecutor 多线程分块并行实现；
4. 与 scikit-learn KMeans 的结果进行一致性对比；
5. 对分配步做不同线程数、不同数据规模的性能基准，并做
   “纯 Python 循环 + 多线程”的 GIL 对照实验；
6. 输出聚类画像表、k 选择表、基准结果 CSV 与全部图表。

运行方式：
    python kmeans_parallel.py

依赖：numpy, pandas, scikit-learn, matplotlib
"""

import os

# ------------------------------------------------------------------
# 必须在 import numpy 之前限制 BLAS 线程数。
# 否则 NumPy 内部 BLAS 线程会与我们的 ThreadPoolExecutor 叠加，
# 产生“线程超额订阅”，干扰并行加速比的测量。
# ------------------------------------------------------------------
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

import json
import math
import platform
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==================================================================
# 0. 全局配置
# ==================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "CC GENERAL.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42

# 只保留 12 个互不冗余、业务含义清晰的特征：
#   - ONEOFF_PURCHASES / INSTALLMENTS_PURCHASES 是 PURCHASES 的两个分项；
#   - ONEOFF_PURCHASES_FREQUENCY / PURCHASES_INSTALLMENTS_FREQUENCY
#     是 PURCHASES_FREQUENCY 的两个分项；
#   - CASH_ADVANCE_TRX 与 CASH_ADVANCE 高度相关。
# 保留分项会让“消费能力”在欧氏距离中被重复计权，因此这里做去冗余。
FEATURES = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE",
    "CASH_ADVANCE_FREQUENCY",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

# 金额/次数类变量右偏严重，先做 log1p 再标准化。
LOG_COLS = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
]

# 图表中英文特征名的中文别名，便于报告阅读。
FEATURE_CN = {
    "BALANCE": "账户余额",
    "BALANCE_FREQUENCY": "余额更新频率",
    "PURCHASES": "消费金额",
    "PURCHASES_FREQUENCY": "消费频率",
    "CASH_ADVANCE": "预借现金",
    "CASH_ADVANCE_FREQUENCY": "预借现金频率",
    "PURCHASES_TRX": "消费笔数",
    "CREDIT_LIMIT": "信用额度",
    "PAYMENTS": "还款金额",
    "MINIMUM_PAYMENTS": "最低还款额",
    "PRC_FULL_PAYMENT": "全额还款比例",
    "TENURE": "持卡月数",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


# ==================================================================
# 1. 数据读取与预处理
# ==================================================================

def load_and_preprocess():
    """读取数据，完成缺失值填补、特征选择、log1p 与标准化。

    返回
    ----
    df       : 原始 DataFrame（已删除 CUST_ID，缺失值已填补）
    X_raw    : 12 个特征的原始量纲矩阵（用于业务画像）
    X_log    : log1p 变换后的矩阵
    X        : Z-score 标准化后的矩阵（用于聚类）
    scaler   : 训练好的 StandardScaler
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"未找到数据文件：{DATA_PATH}\n"
            "请从 Kaggle 下载 Credit Card Dataset for Clustering，"
            "并将 CC GENERAL.csv 放到脚本同目录。"
        )

    df = pd.read_csv(DATA_PATH)
    n_raw, d_raw = df.shape
    missing = df.isnull().sum()
    n_missing = int(missing.sum())

    # CUST_ID 是标识列，不参与聚类
    df = df.drop(columns=["CUST_ID"]).copy()

    # 缺失值：金额类变量右偏，用中位数填补比均值更稳健
    for col in ("CREDIT_LIMIT", "MINIMUM_PAYMENTS"):
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X_raw = df[FEATURES].copy()

    # 右偏变量 log1p：压缩极端值，使欧氏距离不被少数巨值支配
    X_log = X_raw.copy()
    X_log[LOG_COLS] = np.log1p(X_log[LOG_COLS])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_log.values)

    print("=" * 78)
    print("一、数据概况")
    print("=" * 78)
    print(f"原始数据规模：{n_raw} 行 × {d_raw} 列")
    print(f"缺失值总数：{n_missing}（CREDIT_LIMIT 缺 {int(missing['CREDIT_LIMIT'])}，"
          f"MINIMUM_PAYMENTS 缺 {int(missing['MINIMUM_PAYMENTS'])}）")
    print(f"建模特征数：{len(FEATURES)}")
    print("缺失值已用中位数填补；金额/次数类变量已做 log1p；"
          "全部特征已做 Z-score 标准化。")

    return df, X_raw, X_log, X, scaler


# ==================================================================
# 2. 手写 K-means++（多线程分配步）
# ==================================================================

class ParallelKMeansPlusPlus:
    """K-means++ 初始化 + Lloyd 迭代，分配步支持多线程分块并行。

    参数
    ----
    n_clusters   : 簇数 k
    max_iter     : 最大迭代次数
    tol          : 质心最大位移阈值（收敛条件）
    n_threads    : 分配步线程数；1 表示串行
    chunk_factor : 每线程切分的块数（块数 = n_threads * chunk_factor），
                   块数越多负载均衡越好，但调度开销也越大
    n_init       : 用不同随机种子重复 K-means++ 的次数，取 inertia 最小者
    """

    def __init__(self, n_clusters=5, max_iter=100, tol=1e-4,
                 n_threads=8, chunk_factor=4, n_init=3,
                 random_state=RANDOM_STATE, verbose=False):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_threads = max(1, int(n_threads))
        self.chunk_factor = max(1, int(chunk_factor))
        self.n_init = max(1, int(n_init))
        self.random_state = int(random_state)
        self.verbose = verbose

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.inertia_history_ = []
        self.assignment_time_ = 0.0
        self.init_time_ = 0.0

    # ------------------------------------------------------------------
    # 2.1 K-means++ 初始化
    # ------------------------------------------------------------------
    def _kmeans_plusplus(self, X, rng):
        """按 D(x)^2 概率采样，依次选出 k 个初始质心。

        第 1 个质心均匀随机；此后每个样本被选为下一个质心的概率
        正比于它到已选质心集合的最短平方距离。
        """
        n, d = X.shape
        k = self.n_clusters
        centers = np.empty((k, d), dtype=X.dtype)

        first = int(rng.integers(n))
        centers[0] = X[first]
        closest = ((X - centers[0]) ** 2).sum(axis=1)

        for j in range(1, k):
            total = float(closest.sum())
            if total <= 0.0:
                # 所有点都与已选质心重合，退化为随机选择
                idx = int(rng.integers(n))
            else:
                idx = int(rng.choice(n, p=closest / total))
            centers[j] = X[idx]
            new_dist = ((X - centers[j]) ** 2).sum(axis=1)
            closest = np.minimum(closest, new_dist)

        return centers

    # ------------------------------------------------------------------
    # 2.2 分配步：串行 / 多线程分块并行
    # ------------------------------------------------------------------
    @staticmethod
    def _nearest_labels(chunk, centers, center_sq):
        """计算一个数据块中每个点到最近质心的编号。

        使用 ||x||^2 - 2x c^T + ||c||^2 的展开式，避免显式三维
        广播 (n, k, d) 带来的大内存分配；矩阵乘法在 NumPy 的 C
        层执行，会释放 GIL，因此多个线程可以真正并行。
        """
        part = np.einsum("ij,ij->i", chunk, chunk)[:, None]
        dist2 = part - 2.0 * (chunk @ centers.T) + center_sq[None, :]
        return np.argmin(dist2, axis=1)

    def _assign_sequential(self, X, centers):
        center_sq = np.einsum("ij,ij->i", centers, centers)
        return self._nearest_labels(X, centers, center_sq)

    def _assign_parallel(self, X, centers, executor=None):
        """把样本按行切成若干块，用线程池并发计算最近质心。"""
        if self.n_threads <= 1:
            return self._assign_sequential(X, centers)

        center_sq = np.einsum("ij,ij->i", centers, centers)
        n_chunks = min(len(X), self.n_threads * self.chunk_factor)
        chunks = [c for c in np.array_split(X, n_chunks) if len(c) > 0]

        def worker(chunk):
            return self._nearest_labels(chunk, centers, center_sq)

        if executor is None:
            with ThreadPoolExecutor(max_workers=self.n_threads) as ex:
                parts = list(ex.map(worker, chunks))
        else:
            parts = list(executor.map(worker, chunks))

        return np.concatenate(parts)

    def assign(self, X, centers=None, n_threads=None):
        """对外暴露的分配接口，便于单独做性能基准。"""
        if centers is None:
            centers = self.cluster_centers_
        if n_threads is not None:
            self.n_threads = max(1, int(n_threads))
        return self._assign_parallel(X, centers)

    # ------------------------------------------------------------------
    # 2.3 质心更新
    # ------------------------------------------------------------------
    @staticmethod
    def _update_centers(X, labels, old_centers):
        k = old_centers.shape[0]
        new_centers = old_centers.copy()

        for j in range(k):
            members = labels == j
            if np.any(members):
                new_centers[j] = X[members].mean(axis=0)
            else:
                # 空簇处理：把离自己质心最远的点设为该簇新质心
                d2 = ((X - old_centers[labels]) ** 2).sum(axis=1)
                new_centers[j] = X[int(np.argmax(d2))]

        return new_centers

    # ------------------------------------------------------------------
    # 2.4 单次 K-means++ 运行
    # ------------------------------------------------------------------
    def _single_run(self, X, seed):
        rng = np.random.default_rng(seed)

        t0 = time.perf_counter()
        centers = self._kmeans_plusplus(X, rng)
        init_time = time.perf_counter() - t0

        labels = None
        history = []
        assignment_time = 0.0
        n_iter = 0

        executor = (ThreadPoolExecutor(max_workers=self.n_threads)
                    if self.n_threads > 1 else None)
        try:
            for it in range(self.max_iter):
                t_a = time.perf_counter()
                labels = self._assign_parallel(X, centers, executor)
                assignment_time += time.perf_counter() - t_a

                new_centers = self._update_centers(X, labels, centers)
                inertia = float(((X - centers[labels]) ** 2).sum())
                history.append(inertia)

                shift = float(np.max(np.abs(new_centers - centers)))
                n_iter = it + 1
                centers = new_centers

                if self.verbose:
                    print(f"    iter {n_iter:3d}  inertia={inertia:.4f}  "
                          f"shift={shift:.3e}")
                if shift <= self.tol:
                    break
        finally:
            if executor is not None:
                executor.shutdown()

        # 用最终质心重新分配一次，保证 labels 与 centers 严格对应
        labels = self._assign_sequential(X, centers)
        inertia = float(((X - centers[labels]) ** 2).sum())

        return {
            "centers": centers,
            "labels": labels,
            "inertia": inertia,
            "n_iter": n_iter,
            "history": history,
            "assignment_time": assignment_time,
            "init_time": init_time,
        }

    # ------------------------------------------------------------------
    # 2.5 完整训练（多次初始化取最优）
    # ------------------------------------------------------------------
    def fit(self, X):
        runs = []
        for i in range(self.n_init):
            run = self._single_run(X, self.random_state + i)
            runs.append(run)
            if self.verbose:
                print(f"  init {i + 1}/{self.n_init}: "
                      f"inertia={run['inertia']:.6f}, "
                      f"iter={run['n_iter']}, "
                      f"assign={run['assignment_time']:.4f}s")

        best = min(runs, key=lambda r: r["inertia"])
        self.cluster_centers_ = best["centers"]
        self.labels_ = best["labels"]
        self.inertia_ = best["inertia"]
        self.n_iter_ = best["n_iter"]
        self.inertia_history_ = best["history"]
        self.assignment_time_ = sum(r["assignment_time"] for r in runs)
        self.init_time_ = sum(r["init_time"] for r in runs)
        return self

    def predict(self, X):
        return self._assign_parallel(X, self.cluster_centers_)


# ==================================================================
# 3. 选择 k：肘部法则 + 轮廓系数
# ==================================================================

def select_k(X, k_range, n_threads, n_init=3):
    rows = []
    for k in k_range:
        t0 = time.perf_counter()
        model = ParallelKMeansPlusPlus(
            n_clusters=k, n_threads=n_threads, n_init=n_init,
            random_state=RANDOM_STATE,
        ).fit(X)
        fit_time = time.perf_counter() - t0

        sample_size = min(3000, len(X))
        sil = silhouette_score(
            X, model.labels_, sample_size=sample_size,
            random_state=RANDOM_STATE,
        )
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": sil,
            "n_iter": model.n_iter_,
            "fit_time_s": fit_time,
        })
        print(f"  k={k:2d}  inertia={model.inertia_:10.2f}  "
              f"silhouette={sil:0.4f}  iter={model.n_iter_}  "
              f"time={fit_time:0.2f}s")

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULT_DIR, "k_selection.csv"),
                  index=False, encoding="utf-8-sig")
    return result


# ==================================================================
# 4. 聚类画像与一致性对比
# ==================================================================

def build_profile(X_raw, labels):
    profile = X_raw.copy()
    profile["cluster"] = labels

    sizes = profile["cluster"].value_counts().sort_index()
    means = profile.groupby("cluster")[FEATURES].mean()
    medians = profile.groupby("cluster")[FEATURES].median()
    overall = X_raw[FEATURES].mean()

    out = means.copy()
    out.insert(0, "size", sizes.values)
    out.insert(1, "share_pct", (sizes.values / len(profile) * 100).round(2))
    out.to_csv(os.path.join(RESULT_DIR, "cluster_profile.csv"),
               encoding="utf-8-sig")

    medians.to_csv(os.path.join(RESULT_DIR, "cluster_profile_median.csv"),
                   encoding="utf-8-sig")

    # 相对总体均值的指数：>1 表示高于平均，<1 表示低于平均
    relative = means.divide(overall, axis=1)
    relative.to_csv(os.path.join(RESULT_DIR, "cluster_relative_index.csv"),
                    encoding="utf-8-sig")

    return out, means, medians, overall, relative


# ==================================================================
# 5. 性能基准
# ==================================================================

def benchmark_assignment(model, X, sizes, thread_list, repeats):
    """对分配步做规模 × 线程数的基准测试。

    - 基准数据由真实标准化样本有放回重采样并加入极小扰动得到，
      只用于性能测量，不参与聚类分析。
    - 返回 DataFrame：size, threads, time_s, speedup, efficiency。
    """
    rows = []
    seq_times = {}

    for n in sizes:
        rng = np.random.default_rng(2024 + n)
        idx = rng.integers(0, len(X), size=n)
        Xb = X[idx] + rng.normal(0.0, 0.02, size=(n, X.shape[1]))

        centers = model.cluster_centers_

        # 串行基线（同时作为 threads=1 的结果）
        model._assign_sequential(Xb, centers)  # 预热
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model._assign_sequential(Xb, centers)
            times.append(time.perf_counter() - t0)
        seq_t = float(np.median(times))
        seq_times[n] = seq_t
        rows.append({
            "size": n, "threads": 1, "time_s": seq_t,
            "speedup": 1.0, "efficiency": 1.0,
        })
        print(f"  n={n:6d}  threads= 1  time={seq_t:8.4f}s  "
              f"speedup=1.00  efficiency=1.00")

        for nt in thread_list:
            if nt == 1:
                continue

            model.n_threads = nt
            # 预热
            model._assign_parallel(Xb, centers)
            times = []
            with ThreadPoolExecutor(max_workers=nt) as ex:
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    model._assign_parallel(Xb, centers, executor=ex)
                    times.append(time.perf_counter() - t0)
            t = float(np.median(times))
            speedup = seq_t / t if t > 0 else 0.0
            efficiency = speedup / nt
            rows.append({
                "size": n, "threads": nt, "time_s": t,
                "speedup": speedup, "efficiency": efficiency,
            })
            print(f"  n={n:6d}  threads={nt:2d}  time={t:8.4f}s  "
                  f"speedup={speedup:4.2f}  efficiency={efficiency:4.2f}")

        model.n_threads = 1

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULT_DIR, "benchmark_results.csv"),
                  index=False, encoding="utf-8-sig")
    return result


def benchmark_gil(X, model, n_points=30000, repeats=3):
    """纯 Python 双层循环分配步的 GIL 对照实验。

    NumPy 版本在 C 层释放 GIL，多线程能并行；
    纯 Python 循环始终持有 GIL，多线程只会增加切换开销。
    该实验用于解释“为什么本实现用 NumPy 分块而不是 Python 循环”。
    """
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(X), size=n_points)
    Xb = X[idx]
    centers = model.cluster_centers_
    k = centers.shape[0]

    def python_assign_chunk(chunk):
        labels = np.empty(len(chunk), dtype=np.int64)
        for i in range(len(chunk)):
            x = chunk[i]
            best_j = 0
            best_d = math.inf
            for j in range(k):
                c = centers[j]
                d = 0.0
                for a in range(len(x)):
                    diff = x[a] - c[a]
                    d += diff * diff
                if d < best_d:
                    best_d = d
                    best_j = j
            labels[i] = best_j
        return labels

    def run_parallel(nt):
        chunks = [c for c in np.array_split(Xb, nt * 4) if len(c) > 0]
        with ThreadPoolExecutor(max_workers=nt) as ex:
            parts = list(ex.map(python_assign_chunk, chunks))
        return np.concatenate(parts)

    rows = []
    for nt in (1, 8):
        run_parallel(nt)  # 预热
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_parallel(nt)
            times.append(time.perf_counter() - t0)
        t = float(np.median(times))
        rows.append({"mode": "纯 Python 循环", "threads": nt,
                     "time_s": t, "speedup": 1.0 if nt == 1 else np.nan})
        print(f"  纯 Python 循环  threads={nt:2d}  time={t:8.4f}s")

    # NumPy 版本在同样本上的对照
    model.n_threads = 1
    model._assign_parallel(Xb, centers)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model._assign_parallel(Xb, centers)
        times.append(time.perf_counter() - t0)
    seq_t = float(np.median(times))

    model.n_threads = 8
    model._assign_parallel(Xb, centers)
    times = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in range(repeats):
            t0 = time.perf_counter()
            model._assign_parallel(Xb, centers, executor=ex)
            times.append(time.perf_counter() - t0)
    par_t = float(np.median(times))

    rows.extend([
        {"mode": "NumPy 分块", "threads": 1, "time_s": seq_t, "speedup": 1.0},
        {"mode": "NumPy 分块", "threads": 8, "time_s": par_t,
         "speedup": seq_t / par_t},
    ])
    print(f"  NumPy 分块      threads= 1  time={seq_t:8.4f}s")
    print(f"  NumPy 分块      threads= 8  time={par_t:8.4f}s  "
          f"speedup={seq_t / par_t:4.2f}")

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(RESULT_DIR, "gil_benchmark.csv"),
                  index=False, encoding="utf-8-sig")
    return result


# ==================================================================
# 6. 可视化
# ==================================================================

def plot_feature_distributions(X_log, features):
    """log1p 变换后的关键特征分布。"""
    show = ["BALANCE", "PURCHASES", "CASH_ADVANCE",
            "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), show):
        ax.hist(X_log[col], bins=45, color="#4C78A8", alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{FEATURE_CN[col]}（log1p 后）", fontsize=11)
        ax.set_xlabel("log1p(金额)")
        ax.set_ylabel("客户数")
        ax.grid(alpha=0.25, linestyle="--")
    fig.suptitle("图 1  主要金额类特征的 log1p 分布（n = 8950）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_distributions.png"))
    plt.close(fig)


def plot_correlation(X_log, features):
    corr = X_log[features].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels([FEATURE_CN[f] for f in features], rotation=45,
                       ha="right", fontsize=9)
    ax.set_yticklabels([FEATURE_CN[f] for f in features], fontsize=9)
    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    ax.set_title("图 2  建模特征相关系数矩阵（log1p 后）",
                 fontsize=13, fontweight="bold", pad=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_correlation.png"))
    plt.close(fig)


def plot_k_selection(k_result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(k_result["k"], k_result["inertia"], marker="o",
                 color="#E45756", linewidth=2)
    axes[0].set_title("(a) 肘部法则：簇内平方和 SSE", fontsize=11)
    axes[0].set_xlabel("聚类数 k")
    axes[0].set_ylabel("inertia (SSE)")
    axes[0].set_xticks(k_result["k"])
    axes[0].grid(alpha=0.3, linestyle="--")

    axes[1].plot(k_result["k"], k_result["silhouette"], marker="s",
                 color="#4C78A8", linewidth=2)
    best = k_result.loc[k_result["silhouette"].idxmax()]
    axes[1].axvline(best["k"], color="#F58518", linestyle=":",
                    linewidth=1.5)
    axes[1].scatter([best["k"]], [best["silhouette"]], s=90,
                    color="#F58518", zorder=5)
    axes[1].annotate(f"最优 k={int(best['k'])}\n轮廓系数={best['silhouette']:.3f}",
                     xy=(best["k"], best["silhouette"]),
                     xytext=(10, -32), textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="#F58518",
                                     lw=1.0))
    axes[1].set_title("(b) 轮廓系数（样本量 3000）", fontsize=11)
    axes[1].set_xlabel("聚类数 k")
    axes[1].set_ylabel("silhouette")
    axes[1].set_xticks(k_result["k"])
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("图 3  聚类数 k 的选择", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_k_selection.png"))
    plt.close(fig)


def plot_pca_clusters(X, model, segment_names):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(model.cluster_centers_)

    fig, ax = plt.subplots(figsize=(10, 7.2))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
               "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
    for j in range(model.n_clusters):
        mask = model.labels_ == j
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=9, alpha=0.45,
                   color=palette[j % len(palette)],
                   label=f"C{j} {segment_names.get(j, '')}")
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=260, marker="X",
               color="black", edgecolor="white", linewidth=1.2, zorder=10)
    for j, (x, y) in enumerate(centers_pca):
        ax.annotate(f"C{j}", (x, y), xytext=(7, 7),
                    textcoords="offset points", fontsize=10,
                    fontweight="bold")

    ax.set_xlabel(f"主成分 1（解释方差 {pca.explained_variance_ratio_[0] * 100:.1f}%）")
    ax.set_ylabel(f"主成分 2（解释方差 {pca.explained_variance_ratio_[1] * 100:.1f}%）")
    ax.set_title("图 4  客户聚类的 PCA 二维投影", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, markerscale=1.6,
              framealpha=0.9)
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_pca_clusters.png"))
    plt.close(fig)
    return pca


def plot_cluster_profile(centers, segment_names):
    """标准化质心热力图：展示每个簇在各特征上的相对高低。"""
    n_clusters, n_features = centers.shape
    fig, ax = plt.subplots(figsize=(11, 6.2))
    norm = TwoSlopeNorm(vmin=min(-2.5, centers.min()),
                        vcenter=0.0,
                        vmax=max(2.5, centers.max()))
    im = ax.imshow(centers.T, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"C{j}\n{segment_names.get(j, '')}"
                        for j in range(n_clusters)], fontsize=9)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([FEATURE_CN[f] for f in FEATURES], fontsize=9)
    for i in range(n_features):
        for j in range(n_clusters):
            ax.text(j, i, f"{centers[j, i]:.2f}", ha="center",
                    va="center", fontsize=8,
                    color="white" if abs(centers[j, i]) > 1.4 else "black")
    ax.set_title("图 5  各簇质心的标准化特征画像（Z-score）",
                 fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Z-score")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_cluster_profile.png"))
    plt.close(fig)


def plot_speedup(bench):
    sizes = sorted(bench["size"].unique())
    threads = sorted(bench["threads"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    markers = ["o", "s", "^", "D", "v"]
    for i, n in enumerate(sizes):
        sub = bench[bench["size"] == n].sort_values("threads")
        axes[0].plot(sub["threads"], sub["time_s"], marker=markers[i % 5],
                     linewidth=1.8, label=f"n={n:,}")
    axes[0].set_xlabel("线程数")
    axes[0].set_ylabel("分配步耗时（秒）")
    axes[0].set_title("(a) 不同数据规模下的分配步耗时", fontsize=11)
    axes[0].set_yscale("log")
    axes[0].set_xticks(threads)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, linestyle="--", which="both")

    for i, n in enumerate(sizes):
        sub = bench[bench["size"] == n].sort_values("threads")
        axes[1].plot(sub["threads"], sub["speedup"], marker=markers[i % 5],
                     linewidth=1.8, label=f"n={n:,}")
    axes[1].plot(threads, threads, color="gray", linestyle=":",
                 linewidth=1.2, label="线性加速理想线")
    axes[1].set_xlabel("线程数")
    axes[1].set_ylabel("加速比（相对串行）")
    axes[1].set_title("(b) 加速比与线性理想线", fontsize=11)
    axes[1].set_xticks(threads)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("图 6  多线程分配步的性能基准（AMD Ryzen 7 7840H，8 核 16 线程）",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_speedup.png"))
    plt.close(fig)


def plot_gil(gil_result):
    modes = ["纯 Python 循环", "NumPy 分块"]
    x = np.arange(len(modes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, nt, color in [(-width / 2, 1, "#9E9E9E"),
                              (width / 2, 8, "#4C78A8")]:
        vals = []
        for m in modes:
            sub = gil_result[(gil_result["mode"] == m) &
                             (gil_result["threads"] == nt)]
            vals.append(float(sub["time_s"].iloc[0]))
        ax.bar(x + offset, vals, width, label=f"{nt} 线程", color=color)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v * 1.02, f"{v:.3f}s", ha="center",
                    va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("分配步耗时（秒）")
    ax.set_title(f"图 7  GIL 对照：纯 Python 循环 vs NumPy 分块（n=30000）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_gil.png"))
    plt.close(fig)


def plot_convergence(model):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(range(1, len(model.inertia_history_) + 1),
            model.inertia_history_, marker="o", color="#54A24B",
            linewidth=2)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("inertia (SSE)")
    ax.set_title("图 8  Lloyd 迭代的收敛曲线", fontsize=12,
                 fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig8_convergence.png"))
    plt.close(fig)


# ==================================================================
# 7. 主流程
# ==================================================================

def main():
    t_start = time.perf_counter()

    print("=" * 78)
    print("基于多线程 K-means++ 的 Kaggle 信用卡客户聚类分析")
    print("=" * 78)
    print(f"Python      : {platform.python_version()}")
    print(f"操作系统    : {platform.platform()}")
    print(f"CPU 逻辑核  : {os.cpu_count()}")
    print(f"NumPy       : {np.__version__}")
    print(f"pandas      : {pd.__version__}")
    print(f"随机种子    : {RANDOM_STATE}")

    df, X_raw, X_log, X, scaler = load_and_preprocess()

    # ------------------------------------------------------------------
    # 7.1 选择 k
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("二、选择聚类数 k")
    print("=" * 78)
    k_range = list(range(2, 9))
    k_result = select_k(X, k_range, n_threads=8, n_init=3)
    best_k = int(k_result.loc[k_result["silhouette"].idxmax(), "k"])

    # 肘部位置通常也落在轮廓系数峰值附近；若不一致，以轮廓系数为准，
    # 并在报告中讨论两者的差异。
    print(f"\n轮廓系数最优 k = {best_k}")

    # ------------------------------------------------------------------
    # 7.2 最终聚类
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"三、最终 K-means++ 聚类（k={best_k}，8 线程，n_init=10）")
    print("=" * 78)
    model = ParallelKMeansPlusPlus(
        n_clusters=best_k, n_threads=8, n_init=10,
        random_state=RANDOM_STATE, verbose=False,
    )
    t0 = time.perf_counter()
    model.fit(X)
    fit_time = time.perf_counter() - t0
    print(f"inertia            : {model.inertia_:.4f}")
    print(f"迭代次数           : {model.n_iter_}")
    print(f"总拟合耗时         : {fit_time:.3f}s")
    print(f"其中分配步累计耗时 : {model.assignment_time_:.3f}s")
    print(f"K-means++ 初始化   : {model.init_time_:.3f}s")

    # 与 sklearn 对比
    sk = KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                max_iter=300, tol=1e-4, random_state=RANDOM_STATE)
    sk.fit(X)
    ari = adjusted_rand_score(model.labels_, sk.labels_)
    print(f"sklearn inertia    : {sk.inertia_:.4f}")
    print(f"与 sklearn 的 ARI  : {ari:.4f}")

    # ------------------------------------------------------------------
    # 7.3 客户画像
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("四、客户群体画像（原始量纲均值）")
    print("=" * 78)
    profile, means, medians, overall, relative = build_profile(
        X_raw, model.labels_)

    show_cols = ["size", "share_pct", "BALANCE", "PURCHASES",
                 "CASH_ADVANCE", "PURCHASES_FREQUENCY",
                 "CASH_ADVANCE_FREQUENCY", "CREDIT_LIMIT",
                 "PAYMENTS", "PRC_FULL_PAYMENT"]
    print(profile[show_cols].round(3).to_string())

    print("\n相对总体均值的指数（1.0 = 平均水平）：")
    print(relative[["BALANCE", "PURCHASES", "CASH_ADVANCE",
                    "CREDIT_LIMIT", "PAYMENTS"]].round(3).to_string())

    # ------------------------------------------------------------------
    # 7.4 性能基准
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("五、分配步多线程性能基准")
    print("=" * 78)
    bench = benchmark_assignment(
        model, X,
        sizes=[len(X), 50000, 200000, 500000],
        thread_list=[1, 2, 4, 8, 16],
        repeats=5,
    )

    print("\nGIL 对照实验：")
    gil = benchmark_gil(X, model, n_points=30000, repeats=3)

    # ------------------------------------------------------------------
    # 7.5 图表
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("六、生成图表")
    print("=" * 78)

    # 群体名称依据 cluster_profile.csv 的原始量纲均值人工命名：
    #   C0 高余额、高消费、高预借、高额度，几乎不全额还款；
    #   C1 几乎不消费、以预借现金为主；
    #   C2 高频消费、低余额、全额还款比例最高；
    #   C3 大众客群，消费中等、携带余额；
    #   C4 余额与消费都很低、活跃度低；
    #   C5 信用额度低、持卡时间短、预借现金相对偏高。
    segment_names = {
        0: "高额度双高循环型",
        1: "纯预借现金依赖型",
        2: "高频消费全额还款型",
        3: "大众循环消费型",
        4: "低活跃低余额型",
        5: "低额度短期预借型",
    }

    plot_feature_distributions(X_log, FEATURES)
    plot_correlation(X_log, FEATURES)
    plot_k_selection(k_result)
    pca = plot_pca_clusters(X, model, segment_names)
    plot_cluster_profile(model.cluster_centers_, segment_names)
    plot_speedup(bench)
    plot_gil(gil)
    plot_convergence(model)

    print(f"图表已保存到：{FIG_DIR}")

    # ------------------------------------------------------------------
    # 7.6 汇总 JSON
    # ------------------------------------------------------------------
    summary = {
        "dataset": "Kaggle Credit Card Dataset for Clustering (CC GENERAL)",
        "n_samples": int(len(X)),
        "n_features": len(FEATURES),
        "best_k": best_k,
        "silhouette_best": float(
            k_result.loc[k_result["silhouette"].idxmax(), "silhouette"]),
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "fit_time_s": float(fit_time),
        "assignment_time_s": float(model.assignment_time_),
        "sklearn_inertia": float(sk.inertia_),
        "ari_vs_sklearn": float(ari),
        "pca_explained_variance": [float(v) for v in
                                   pca.explained_variance_ratio_],
        "cluster_sizes": [int((model.labels_ == j).sum())
                          for j in range(best_k)],
        "segment_names": {str(k): v for k, v in segment_names.items()},
        "benchmark_rows": bench.to_dict(orient="records"),
        "gil_rows": gil.to_dict(orient="records"),
        "total_runtime_s": float(time.perf_counter() - t_start),
    }
    with open(os.path.join(RESULT_DIR, "run_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"全部完成，总耗时 {summary['total_runtime_s']:.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
