# LDO/DCDC 反馈分压网络 · 输出电压漂移工具

一个纯 CPU 的小工具，针对可调稳压器最常见的分压反馈拓扑：

```
    Vout
     |
    [R1]                 R1 = 上分压电阻 (OUT -> FB)
     |
 FB o---- Vref           稳压器把 FB 钳位在 Vref
     |
    [R2]                 R2 = 下分压电阻 (FB -> GND)
     |
    GND

    Vout = Vref * (1 + R1/R2) + I_adj * R1
```

把四类漂移源全建进模型：电阻初始容差、电阻温漂 TCR、基准 Vref 温漂/老化、FB 偏置电流 I_adj。

## 三大功能

- **A. 最坏情况漂移分析**：Corner 极值法给出全温区 Vout 上下限；RSS 统计估计给出更现实的量级并拆解各源贡献；蒙特卡洛给出漂移分布与"落在 spec 内的概率"。
- **B. 反推推荐电阻组合**：给一个精度目标，在电阻"等级目录"（容差×TCR×成本）和分压绝对阻值里搜索，按成本排序给出满足要求的方案；并明确告诉你"电阻再好的那部分也压不下去的漂移下限"来自 Vref/老化。
- **C. sklearn 流程**：用物理模型造数据训练 RandomForest/GradBoosting 回归，用特征重要性回答"到底谁主导漂移"。

## 在 HyperAI free-cpu 容器里跑

1. 新建部署，算力选 **Free-CPU**，模式选 **PyTorch**（CPU 镜像即可，本工具用不到 GPU）。
2. 进容器终端，装依赖（镜像一般已带 numpy/matplotlib，缺啥补啥）：
   ```bash
   pip install -r requirements.txt
   # 国内网络更快：
   # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn pandas
   ```
3. 把 `ldo_divider_drift.py` 和 `requirements.txt` 上传到容器（部署文件上传，或 `git clone` 你自己的仓库）。
4. 运行：
   ```bash
   OUT_DIR=./out python ldo_divider_drift.py
   ```
   控制台打印完整分析报告，图表输出到 `./out/drift_analysis.png` 与 `./out/feature_importance.png`。

> 免费档 2 核 / 4GB 内存完全够：蒙特卡洛默认 3 万样本、ML 默认 6000 样本，几秒钟跑完。想加大规模改 `monte_carlo(n=...)` / `build_ml_demo(n=...)` 即可。

## 改造成你自己的电路

编辑脚本顶部 `make_default_cfg()` 里的参数就行：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `Vref_nom` | 基准电压 | 0.8 V |
| `Vout_target` | 目标输出 | 3.3 V |
| `R2_nom` | 下分压电阻标称 | 10 kΩ |
| `Tmin` / `Tmax` | 工作温度范围 | -40 / 125 °C |
| `tol1` / `tol2` | 上下电阻容差 | ±1% |
| `tcr1_ppm` / `tcr2_ppm` | 上下电阻温漂 | 100 ppm/°C |
| `vref_tc_ppm` | Vref 温漂 | 50 ppm/°C |
| `aging` | Vref 老化 | ±0.5% |
| `iadj_A` | FB 偏置电流 | 100 nA |
| `spec_pct` | 全温区精度目标 | ±1% |

想加新的电阻档位，改 `RESISTOR_GRADES`；想换分压绝对阻值候选，改 `R2_CANDIDATES_OHM`。

## 默认例子跑出来的结论

- 用便宜的 1%/100ppm 电阻，全温区最坏漂移约 **±4.2%**，远达不到 ±1%。
- 特征重要性：**下电阻容差 > 上电阻容差 > 老化** 主导漂移——分压比对下电阻 R2 的容差最敏感。
- 换更好的电阻，漂移下限也只能压到约 **±1.2%**，因为 Vref 温漂+老化是电阻选择消不掉的。要 ±1% 得换更稳的基准；把目标放宽到 ±2% 时，最省钱的达标方案是 **0.1%/50ppm 电阻对**。

## 文件

- `ldo_divider_drift.py` — 主脚本（物理模型 + 三种分析 + 推荐 + sklearn）
- `requirements.txt` — 依赖
- `out/` — 运行后生成的图表（示例图已附）
- `run_report.txt` — 一次完整运行的示例输出
