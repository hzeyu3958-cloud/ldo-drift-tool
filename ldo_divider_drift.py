# -*- coding: utf-8 -*-
"""
LDO / DCDC 可调反馈分压网络 —— 输出电压漂移分析 + 电阻组合推荐 + sklearn 流程演示
=================================================================================

针对最常见的可调稳压器反馈分压拓扑：

        Vout
         |
        [R1]            R1 = 上分压电阻 (OUT -> FB)
         |
    FB o---- Vref       稳压器把 FB 节点钳位在 Vref
         |
        [R2]            R2 = 下分压电阻 (FB -> GND)
         |
        GND

理想关系（忽略偏置电流）：
    Vout = Vref * (1 + R1/R2)

含 FB 偏置电流 I_adj（电流流入 FB 引脚，标准 TI 模型）：
    Vout = Vref * (1 + R1/R2) + I_adj * R1

本工具把下面这些漂移源全部建进模型：
  1) 电阻初始容差      d1, d2      (±%)
  2) 电阻温漂 TCR      tcr1, tcr2  (±ppm/°C)，随温度改变阻值
  3) 基准 Vref 温漂/老化 vref_tc (ppm/°C), aging (±%)
  4) FB 偏置电流 I_adj (nA)，流过 R1 产生附加误差（高阻值分压尤其明显）

三大功能：
  A. 最坏情况漂移分析：Corner 极值法 + RSS 统计估计 + 蒙特卡洛分布
  B. 反推推荐：给定精度目标，在电阻"等级目录"里搜出满足要求且成本最低的组合
  C. sklearn 流程：用物理模型造数据训练回归模型，用特征重要性看"谁主导漂移"

设计成"没有显卡、只有 CPU、内存 4GB"也能跑：纯 numpy + pandas + scikit-learn，
若检测不到 sklearn/pandas，会自动跳过对应模块而不是崩溃。

用法：
    python ldo_divider_drift.py
    （输出目录可用环境变量 OUT_DIR 指定，默认 ./out）
"""

import os
import sys
import math
import itertools
import numpy as np

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能出图
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

OUT_DIR = os.environ.get("OUT_DIR", "./out")
os.makedirs(OUT_DIR, exist_ok=True)


def out_path(name):
    return os.path.join(OUT_DIR, name)


# ----------------------------------------------------------------------------
# 1. 物理模型
# ----------------------------------------------------------------------------
def resistor_at_temp(R_nom, tol_dev, tcr_ppm, T):
    """单个电阻在温度 T(°C) 下的阻值。
    tol_dev: 容差偏差(小数, 如 +0.01 表示偏 +1%)
    tcr_ppm: 温度系数(ppm/°C, 可正可负)
    一阶线性 TCR 模型: R(T) = R_nom*(1+tol_dev)*(1 + tcr_ppm*1e-6*(T-25))
    """
    return R_nom * (1.0 + tol_dev) * (1.0 + tcr_ppm * 1e-6 * (T - 25.0))


def vout_model(T, d1, d2, tcr1, tcr2, vref_tc_ppm, aging_frac, iadj_A,
               R1_nom, R2_nom, Vref_nom):
    """完整输出电压模型(标量或 numpy 数组温度均可)。返回 Vout(伏)。"""
    R1 = resistor_at_temp(R1_nom, d1, tcr1, T)
    R2 = resistor_at_temp(R2_nom, d2, tcr2, T)
    Vref = Vref_nom * (1.0 + vref_tc_ppm * 1e-6 * (T - 25.0) + aging_frac)
    return Vref * (1.0 + R1 / R2) + iadj_A * R1


# ----------------------------------------------------------------------------
# 2. 最坏情况 Corner 分析
# ----------------------------------------------------------------------------
def worst_case_corner(cfg, n_temp=200):
    """在温度范围内，对所有不确定量取极值组合，扫描出真正的 Vout 上/下限。
    返回 dict: vmax, vmin, T_at_max, T_at_min, drift_pct_max, drift_pct_min
    """
    temps = np.linspace(cfg["Tmin"], cfg["Tmax"], n_temp)
    tol1, tol2 = cfg["tol1"], cfg["tol2"]
    tcr1s, tcr2s = cfg["tcr1_ppm"], cfg["tcr2_ppm"]
    vtc = cfg["vref_tc_ppm"]
    ag = cfg["aging"]
    iadj = cfg["iadj_A"]

    # 每个不确定量的两个极端(容差、TCR 符号、Vref 温漂符号、老化符号、I_adj 符号)
    opts_d1 = [-tol1, +tol1]
    opts_d2 = [-tol2, +tol2]
    opts_t1 = [-tcr1s, +tcr1s]
    opts_t2 = [-tcr2s, +tcr2s]
    opts_vt = [-vtc, +vtc]
    opts_ag = [-ag, +ag]
    opts_ij = [-abs(iadj), +abs(iadj)]

    vnom = vout_model(25.0, 0, 0, 0, 0, 0, 0, 0,
                      cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])

    vmax = -np.inf; vmin = np.inf; Tmax_ = None; Tmin_ = None
    for d1, d2, t1, t2, vt, a, ij in itertools.product(
            opts_d1, opts_d2, opts_t1, opts_t2, opts_vt, opts_ag, opts_ij):
        vals = vout_model(temps, d1, d2, t1, t2, vt, a, ij,
                          cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
        i_max = int(np.argmax(vals)); i_min = int(np.argmin(vals))
        if vals[i_max] > vmax:
            vmax = vals[i_max]; Tmax_ = temps[i_max]
        if vals[i_min] < vmin:
            vmin = vals[i_min]; Tmin_ = temps[i_min]

    return {
        "Vnom": vnom, "Vmax": vmax, "Vmin": vmin,
        "T_at_max": Tmax_, "T_at_min": Tmin_,
        "drift_pct_max": (vmax - vnom) / cfg["Vout_target"] * 100.0,
        "drift_pct_min": (vmin - vnom) / cfg["Vout_target"] * 100.0,
        "worst_abs_drift_pct": max(abs((vmax - vnom) / cfg["Vout_target"] * 100.0),
                                   abs((vmin - vnom) / cfg["Vout_target"] * 100.0)),
    }


def rss_estimate(cfg):
    """RSS 统计估计：对每个不确定量单独施加 ±边界，取其它为标称，
    求各 ΔVout 后按平方和开根号。是比 Corner 更"现实"的漂移量级估计。
    返回 RSS 漂移(±%)。
    """
    T_hi_worst = cfg["Tmax"] if cfg["Tmax"] - 25 > 25 - cfg["Tmin"] else cfg["Tmin"]
    # 对每个量，用能产生最大效应的温度
    base = vout_model(25.0, 0, 0, 0, 0, 0, 0, 0,
                      cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions = []

    # 电阻容差：影响分压比
    v = vout_model(25.0, cfg["tol1"], 0, 0, 0, 0, 0, 0,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions.append(abs(v - base))
    v = vout_model(25.0, 0, cfg["tol2"], 0, 0, 0, 0, 0,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions.append(abs(v - base))
    # TCR：在极端温度下评估
    v = vout_model(T_hi_worst, 0, 0, cfg["tcr1_ppm"], 0, 0, 0, 0,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions.append(abs(v - base))
    v = vout_model(T_hi_worst, 0, 0, 0, cfg["tcr2_ppm"], 0, 0, 0,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions.append(abs(v - base))
    # Vref 温漂
    v = vout_model(T_hi_worst, 0, 0, 0, 0, cfg["vref_tc_ppm"], 0, 0,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions.append(abs(v - base))
    # 老化
    v = vout_model(25.0, 0, 0, 0, 0, 0, cfg["aging"], 0,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions.append(abs(v - base))
    # I_adj
    v = vout_model(25.0, 0, 0, 0, 0, 0, 0, abs(cfg["iadj_A"]),
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    contributions.append(abs(v - base))

    rss_V = math.sqrt(sum(c * c for c in contributions))
    return {"rss_pct": rss_V / cfg["Vout_target"] * 100.0,
            "per_source_pct": [c / cfg["Vout_target"] * 100.0 for c in contributions]}


# ----------------------------------------------------------------------------
# 3. 蒙特卡洛
# ----------------------------------------------------------------------------
def monte_carlo(cfg, n=20000, seed=42):
    """在容差/TCR/Vref/I_adj 的允许范围内按分布抽样，温度也随机，
    得到 Vout 漂移的经验分布。返回 drift_pct 数组 + 统计量。
    """
    rng = np.random.default_rng(seed)
    d1 = rng.uniform(-cfg["tol1"], cfg["tol1"], n)
    d2 = rng.uniform(-cfg["tol2"], cfg["tol2"], n)
    t1 = rng.uniform(-cfg["tcr1_ppm"], cfg["tcr1_ppm"], n)
    t2 = rng.uniform(-cfg["tcr2_ppm"], cfg["tcr2_ppm"], n)
    vt = rng.uniform(-cfg["vref_tc_ppm"], cfg["vref_tc_ppm"], n)
    ag = rng.uniform(-cfg["aging"], cfg["aging"], n)
    ij = rng.uniform(-abs(cfg["iadj_A"]), abs(cfg["iadj_A"]), n)
    T = rng.uniform(cfg["Tmin"], cfg["Tmax"], n)

    v = vout_model(T, d1, d2, t1, t2, vt, ag, ij,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    drift_pct = (v - cfg["Vnom"]) / cfg["Vout_target"] * 100.0

    spec = cfg["spec_pct"]
    within = float(np.mean(np.abs(drift_pct) <= spec)) * 100.0
    stats = {
        "mean": float(np.mean(drift_pct)),
        "std": float(np.std(drift_pct)),
        "min": float(np.min(drift_pct)),
        "max": float(np.max(drift_pct)),
        "p01": float(np.percentile(drift_pct, 0.1)),
        "p999": float(np.percentile(drift_pct, 99.9)),
        "within_spec_pct": within,
    }
    return drift_pct, stats


# ----------------------------------------------------------------------------
# 4. 反推推荐电阻组合
# ----------------------------------------------------------------------------
# 电阻"等级"目录：(容差小数, TCR ppm, 相对成本系数)  —— 成本随精度上升而上升
RESISTOR_GRADES = [
    {"name": "1%/200ppm 通用",        "tol": 0.01,  "tcr": 200.0, "cost": 1.0},
    {"name": "1%/100ppm",             "tol": 0.01,  "tcr": 100.0, "cost": 1.2},
    {"name": "0.5%/100ppm",           "tol": 0.005, "tcr": 100.0, "cost": 1.5},
    {"name": "0.1%/50ppm",            "tol": 0.001, "tcr": 50.0,  "cost": 2.5},
    {"name": "0.1%/25ppm",            "tol": 0.001, "tcr": 25.0,  "cost": 3.5},
    {"name": "0.05%/15ppm",           "tol": 0.0005,"tcr": 15.0,  "cost": 5.5},
    {"name": "0.05%/5ppm(匹配)",      "tol": 0.0005,"tcr": 5.0,   "cost": 8.0},
]

# 分压绝对阻值候选(下电阻 R2)，影响 I_adj 误差与功耗
R2_CANDIDATES_OHM = [4.7e3, 10e3, 20e3, 100e3]


def design_ratio(cfg, R2_ohm):
    """给定 R2，按目标 Vout/Vref 反算理想 R1，并取到 E96 系列。"""
    ratio_ideal = cfg["Vout_target"] / cfg["Vref_nom"] - 1.0
    R1_ideal = ratio_ideal * R2_ohm
    R1_e96 = e96(R1_ideal)
    return R1_e96, R2_ohm, ratio_ideal


E96 = [100,102,105,107,109,113,115,118,121,124,127,130,133,137,140,143,147,150,154,158,
       162,165,169,174,178,182,187,191,196,200,205,210,215,221,226,232,237,243,249,255,
       261,267,274,280,287,294,301,309,316,324,332,340,348,357,365,374,383,392,402,412,
       422,432,442,453,464,475,487,499,511,523,536,549,562,576,590,604,619,634,649,665,
       681,698,715,732,750,768,787,806,825,845,866,887,909,931,953,976]


def e96(x):
    """把任意阻值就近取到 E96 系列。E96 列表存 100..976 代表尾数 1.00..9.76。"""
    if x <= 0:
        return x
    exp = math.floor(math.log10(x))
    mant = x / (10 ** exp)          # 归一化到 [1,10)
    # 找最接近的尾数(以 1.00..9.76 表示)
    best = min(E96, key=lambda m: abs(m / 100.0 - mant))
    val = best / 100.0 * (10 ** exp)
    # 若就近进位到 10.0，则升一个数量级
    if best / 100.0 >= 10.0:
        val = 1.0 * (10 ** (exp + 1))
    return val


def recommend(cfg, verbose=True):
    """在 R2 候选 × 电阻等级(上下电阻可不同) 的组合里，
    用 Corner 最坏情况漂移筛出满足 spec 的方案，按成本排序。
    """
    results = []
    for R2 in R2_CANDIDATES_OHM:
        R1, R2v, ratio = design_ratio(cfg, R2)
        for g1 in RESISTOR_GRADES:
            for g2 in RESISTOR_GRADES:
                sub = dict(cfg)
                sub["R1_nom"] = R1
                sub["R2_nom"] = R2v
                sub["Vnom"] = vout_model(25.0, 0, 0, 0, 0, 0, 0, 0, R1, R2v, cfg["Vref_nom"])
                sub["tol1"] = g1["tol"]; sub["tol2"] = g2["tol"]
                sub["tcr1_ppm"] = g1["tcr"]; sub["tcr2_ppm"] = g2["tcr"]
                wc = worst_case_corner(sub, n_temp=80)
                # I_adj 误差项(单独看绝对阻值影响)
                iadj_err_pct = abs(cfg["iadj_A"]) * R1 / cfg["Vout_target"] * 100.0
                cost = g1["cost"] + g2["cost"]
                meets = wc["worst_abs_drift_pct"] <= cfg["spec_pct"]
                results.append({
                    "R1_ohm": R1, "R2_ohm": R2v,
                    "grade1": g1["name"], "grade2": g2["name"],
                    "worst_drift_pct": wc["worst_abs_drift_pct"],
                    "iadj_err_pct": iadj_err_pct,
                    "cost": cost, "meets_spec": bool(meets),
                })
    df = pd.DataFrame(results) if HAVE_PANDAS else results
    if HAVE_PANDAS:
        ok = df[df["meets_spec"]].sort_values("cost").reset_index(drop=True)
        best = df.sort_values("worst_drift_pct").reset_index(drop=True)
        floor_pct = float(df["worst_drift_pct"].min())
    else:
        ok = sorted([r for r in results if r["meets_spec"]], key=lambda r: r["cost"])
        best = sorted(results, key=lambda r: r["worst_drift_pct"])
        floor_pct = best[0]["worst_drift_pct"] if best else float("nan")

    if verbose:
        print("\n" + "=" * 78)
        print(f"[反推推荐] 目标精度 ±{cfg['spec_pct']:.2f}% (全温区 {cfg['Tmin']:.0f}~{cfg['Tmax']:.0f}°C 最坏情况)")
        print("=" * 78)
        print(f"  电阻选择可达到的漂移下限(换更好的电阻也压不下去的部分): ±{floor_pct:.3f}%")
        print(f"    → 这部分主要来自 Vref 温漂(±{cfg['vref_tc_ppm']:.0f}ppm/°C)与老化(±{cfg['aging']*100:.2f}%)，")
        print(f"      是电阻等级无法消除的。要更严的 spec，得换更稳的基准或放宽目标。")
        if HAVE_PANDAS and len(ok) > 0:
            print(f"\n  [OK] 满足 ±{cfg['spec_pct']:.2f}% 的方案(按成本从低到高，前 8):")
            show = ok.head(8).copy()
            show["meets_spec"] = show["meets_spec"].map({True: "PASS", False: "FAIL"})
            show["worst_drift_pct"] = show["worst_drift_pct"].round(3)
            show["iadj_err_pct"] = show["iadj_err_pct"].round(4)
            show = show.rename(columns={"R1_ohm": "R1(Ω)", "R2_ohm": "R2(Ω)",
                                        "grade1": "上电阻等级", "grade2": "下电阻等级",
                                        "worst_drift_pct": "最坏漂移%", "iadj_err_pct": "I_adj误差%",
                                        "cost": "成本", "meets_spec": "判定"})
            print(show.to_string(index=False))
        else:
            print(f"\n  [NO] 当前候选目录内没有方案能满足 ±{cfg['spec_pct']:.2f}%。")
            print(f"     最接近目标的方案(按最坏漂移从优到差，前 8):")
            show = best.head(8).copy()
            if HAVE_PANDAS:
                show["meets_spec"] = show["meets_spec"].map({True: "PASS", False: "FAIL"})
                show["worst_drift_pct"] = show["worst_drift_pct"].round(3)
                show["iadj_err_pct"] = show["iadj_err_pct"].round(4)
                show = show.rename(columns={"R1_ohm": "R1(Ω)", "R2_ohm": "R2(Ω)",
                                            "grade1": "上电阻等级", "grade2": "下电阻等级",
                                            "worst_drift_pct": "最坏漂移%", "iadj_err_pct": "I_adj误差%",
                                            "cost": "成本", "meets_spec": "判定"})
                print(show.to_string(index=False))
            else:
                for r in show:
                    print("   ", r)
            print(f"     建议: 把 spec 放宽到 ≥±{math.ceil(floor_pct*100)/100:.2f}%，或改用更低 TC/老化的基准。")
    return ok


# ----------------------------------------------------------------------------
# 5. sklearn 流程：造数据 -> 训练 -> 特征重要性
# ----------------------------------------------------------------------------
def build_ml_demo(cfg, n=6000, seed=7):
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_absolute_error
    except Exception as e:
        print("\n[sklearn] 未安装或导入失败，跳过 ML 演示：", e)
        return None

    rng = np.random.default_rng(seed)
    # 每个样本 = 一组"实际偏差"，标签 = 该偏差造成的 |漂移%|
    d1 = rng.uniform(-cfg["tol1"], cfg["tol1"], n)
    d2 = rng.uniform(-cfg["tol2"], cfg["tol2"], n)
    t1 = rng.uniform(-cfg["tcr1_ppm"], cfg["tcr1_ppm"], n)
    t2 = rng.uniform(-cfg["tcr2_ppm"], cfg["tcr2_ppm"], n)
    vt = rng.uniform(-cfg["vref_tc_ppm"], cfg["vref_tc_ppm"], n)
    ag = rng.uniform(-cfg["aging"], cfg["aging"], n) * 1e6  # 转 ppm 便于看
    ij = rng.uniform(-abs(cfg["iadj_A"]), abs(cfg["iadj_A"]), n) * 1e9  # nA
    T = rng.uniform(cfg["Tmin"], cfg["Tmax"], n)

    X = np.column_stack([
        d1 * 100, d2 * 100,          # 上/下电阻容差偏差(%)
        t1, t2,                      # 上/下电阻 TCR(ppm)
        vt,                          # Vref 温漂(ppm/°C)
        ag,                          # 老化(ppm)
        ij,                          # I_adj(nA)
        T,                           # 温度(°C)
    ])
    feat_names = ["R1容差%", "R2容差%", "R1_TCR", "R2_TCR",
                  "Vref温漂ppm", "老化ppm", "I_adj(nA)", "温度°C"]
    # 图里用英文名(容器/本机 DejaVu 无中文字体，避免豆腐块)
    feat_names_en = ["R1 tol %", "R2 tol %", "R1 TCR", "R2 TCR",
                     "Vref TC ppm", "aging ppm", "I_adj nA", "Temp C"]

    v = vout_model(T, d1, d2, t1, t2, vt,
                   ag * 1e-6, ij * 1e-9,
                   cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    y = np.abs((v - cfg["Vnom"]) / cfg["Vout_target"] * 100.0)  # |漂移%|

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                               random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    r2_rf = r2_score(yte, rf.predict(Xte))

    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                   random_state=0)
    gb.fit(Xtr, ytr)
    r2_gb = r2_score(yte, gb.predict(Xte))

    mae = mean_absolute_error(yte, rf.predict(Xte))
    imp = rf.feature_importances_

    print("\n" + "=" * 78)
    print("[sklearn 流程] 用物理模型造数据 -> 训练回归模型预测 |漂移%|")
    print("=" * 78)
    print(f"  样本数={n}  特征={feat_names}")
    print(f"  RandomForest 测试集 R² = {r2_rf:.3f}   MAE = {mae:.4f}%")
    print(f"  GradBoosting 测试集 R² = {r2_gb:.3f}")
    print("  特征重要性(谁主导输出电压漂移):")
    order = np.argsort(imp)[::-1]
    for i in order:
        bar = "#" * int(imp[i] * 50)
        print(f"    {feat_names[i]:<12} {imp[i]*100:5.1f}%  {bar}")

    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ypos = np.arange(len(feat_names))
        ax.barh(ypos, imp[order], color="#2b6cb0")
        ax.set_yticks(ypos); ax.set_yticklabels([feat_names_en[i] for i in order])
        ax.set_xlabel("Feature importance (RandomForest)")
        ax.set_title("Which parameter dominates Vout drift?")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_path("feature_importance.png"), dpi=130)
        plt.close(fig)
        print("  图已保存: feature_importance.png")
    return {"r2_rf": r2_rf, "r2_gb": r2_gb, "importances": dict(zip(feat_names, imp))}


# ----------------------------------------------------------------------------
# 出图：漂移随温度曲线 + 蒙特卡洛直方图
# ----------------------------------------------------------------------------
def plot_results(cfg):
    if not HAVE_MPL:
        print("\n[matplotlib] 不可用，跳过出图。")
        return
    temps = np.linspace(cfg["Tmin"], cfg["Tmax"], 200)
    # 标称曲线
    vnom_curve = vout_model(temps, 0, 0, 0, 0, 0, 0, 0,
                            cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
    # 最坏情况上下包络(逐温度取极值组合)
    up = np.full_like(temps, -np.inf); lo = np.full_like(temps, np.inf)
    for d1, d2, t1, t2, vt, a, ij in itertools.product(
            [-cfg["tol1"], cfg["tol1"]], [-cfg["tol2"], cfg["tol2"]],
            [-cfg["tcr1_ppm"], cfg["tcr1_ppm"]], [-cfg["tcr2_ppm"], cfg["tcr2_ppm"]],
            [-cfg["vref_tc_ppm"], cfg["vref_tc_ppm"]], [-cfg["aging"], cfg["aging"]],
            [-abs(cfg["iadj_A"]), abs(cfg["iadj_A"])]):
        vv = vout_model(temps, d1, d2, t1, t2, vt, a, ij,
                        cfg["R1_nom"], cfg["R2_nom"], cfg["Vref_nom"])
        up = np.maximum(up, vv); lo = np.minimum(lo, vv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    drift_up = (up - cfg["Vnom"]) / cfg["Vout_target"] * 100
    drift_lo = (lo - cfg["Vnom"]) / cfg["Vout_target"] * 100
    ax.fill_between(temps, drift_lo, drift_up, alpha=0.2, color="#e53e3e",
                    label="worst-case envelope")
    ax.plot(temps, (vnom_curve - cfg["Vnom"]) / cfg["Vout_target"] * 100,
            color="#2b6cb0", lw=2, label="nominal")
    ax.axhline(cfg["spec_pct"], ls="--", color="gray", lw=1)
    ax.axhline(-cfg["spec_pct"], ls="--", color="gray", lw=1,
               label=f"spec ±{cfg['spec_pct']}%")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Vout drift (%)")
    ax.set_title("Worst-case Vout drift vs temperature")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    drift, mc_stats = monte_carlo(cfg, n=30000)
    ax.hist(drift, bins=80, color="#38a169", alpha=0.8)
    ax.axvline(cfg["spec_pct"], ls="--", color="red")
    ax.axvline(-cfg["spec_pct"], ls="--", color="red")
    ax.set_xlabel("Vout drift (%)"); ax.set_ylabel("count")
    ax.set_title(f"Monte-Carlo distribution (within spec {mc_stats['within_spec_pct']:.1f}%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path("drift_analysis.png"), dpi=130)
    plt.close(fig)
    print("\n[出图] 已保存: drift_analysis.png, feature_importance.png(若已装sklearn)")
    return mc_stats


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def make_default_cfg():
    """一个典型 buck/LDO 可调输出例子：Vref=0.8V，目标 3.3V，工业温度 -40~125°C。"""
    Vref_nom = 0.8
    Vout_target = 3.3
    R2_nom = 10e3
    ratio = Vout_target / Vref_nom - 1.0        # = 3.125
    R1_nom = e96(ratio * R2_nom)                 # 取到 E96
    Vnom = vout_model(25.0, 0, 0, 0, 0, 0, 0, 0, R1_nom, R2_nom, Vref_nom)
    cfg = {
        "Vref_nom": Vref_nom, "Vout_target": Vout_target,
        "R1_nom": R1_nom, "R2_nom": R2_nom, "Vnom": Vnom,
        "Tmin": -40.0, "Tmax": 125.0,
        # 默认用"便宜"的 1%/100ppm 电阻，方便看它为什么不达标
        "tol1": 0.01, "tol2": 0.01,
        "tcr1_ppm": 100.0, "tcr2_ppm": 100.0,
        "vref_tc_ppm": 50.0,     # 基准温漂 ppm/°C
        "aging": 0.005,          # 老化 ±0.5%
        "iadj_A": 100e-9,        # FB 偏置电流 100nA
        "spec_pct": 1.0,         # 全温区最坏情况精度目标 ±1%
    }
    return cfg


def report(cfg):
    print("=" * 78)
    print("LDO/DCDC 反馈分压网络 —— 输出电压漂移工具")
    print("=" * 78)
    print(f"  拓扑: Vout = Vref*(1+R1/R2) + I_adj*R1")
    print(f"  Vref(标称) = {cfg['Vref_nom']:.3f} V   目标 Vout = {cfg['Vout_target']:.3f} V")
    print(f"  R1(上) = {cfg['R1_nom']/1e3:.2f} kΩ   R2(下) = {cfg['R2_nom']/1e3:.2f} kΩ")
    print(f"  设计标称 Vout = {cfg['Vnom']:.4f} V (设定点偏差 {(cfg['Vnom']-cfg['Vout_target'])/cfg['Vout_target']*100:+.2f}%)")
    print(f"  温度范围 = {cfg['Tmin']:.0f} ~ {cfg['Tmax']:.0f} °C")
    print(f"  默认电阻: 容差 ±{cfg['tol1']*100:.2f}% / TCR {cfg['tcr1_ppm']:.0f}ppm (上、下相同)")
    print(f"  Vref 温漂 {cfg['vref_tc_ppm']:.0f}ppm/°C, 老化 ±{cfg['aging']*100:.2f}%, I_adj {cfg['iadj_A']*1e9:.0f}nA")
    print(f"  精度目标: 全温区最坏情况 |漂移| ≤ ±{cfg['spec_pct']:.2f}%")


def main():
    cfg = make_default_cfg()
    report(cfg)

    wc = worst_case_corner(cfg)
    print("\n" + "=" * 78)
    print("[A. 最坏情况漂移分析]")
    print("=" * 78)
    print(f"  Corner 极值法:  Vout ∈ [{wc['Vmin']:.4f}, {wc['Vmax']:.4f}] V")
    print(f"    → 漂移范围 [{wc['drift_pct_min']:+.3f}%, {wc['drift_pct_max']:+.3f}%]  "
          f"最坏 |漂移| = {wc['worst_abs_drift_pct']:.3f}%")
    print(f"    → 最高点在 {wc['T_at_max']:.0f}°C, 最低点在 {wc['T_at_min']:.0f}°C")
    rss = rss_estimate(cfg)
    labels = ["R1容差", "R2容差", "R1_TCR", "R2_TCR", "Vref温漂", "老化", "I_adj"]
    print(f"  RSS 统计估计:   ±{rss['rss_pct']:.3f}%  (各源贡献 ±%)")
    for lb, c in zip(labels, rss["per_source_pct"]):
        print(f"      {lb:<10} {c:6.3f}")
    verdict = "[PASS] 满足" if wc["worst_abs_drift_pct"] <= cfg["spec_pct"] else "[FAIL] 不满足"
    print(f"  对照目标 ±{cfg['spec_pct']:.2f}%:  {verdict}")

    plot_results(cfg)

    recommend(cfg)
    cfg_loose = dict(cfg); cfg_loose["spec_pct"] = 2.0
    recommend(cfg_loose)
    build_ml_demo(cfg)

    print("\n完成。图表与脚本在:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
