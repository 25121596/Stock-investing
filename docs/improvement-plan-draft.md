# 改进执行方向（初版计划草稿 · 供我继续细化）

任务：user (m00002) 要求自主评审进度 → 查文献 → 自主计划改进 → 汇报执行方向。本文件是我的计划工作底稿。

## 证据基线（已有，全部带随机对照）
- base_lgb10：2022–25 +28.33%（138t, 47.8%w, DD -13.07%）；2026 +3.36%
- breadth_lgb10：2022–25 +31.36%（55t, 58.2%w, DD -5.33%）；2026 +6.72% — 最强但交易少
- weekly_lowvol：+15.66%/+4.00% — 处于随机对照范围内，证据弱
- 交集双门槛 (p_win≥.55 AND pred≥.005)：2022-25 +2.23%（22t）→ 过严，仅作为 meta-labeling 雏形参考
- breadth_ensemble_avg：2022-25 +35.12% / 2026 +1.24% → 样本外衰减严重

## 文献信号（来自 arXiv 检索，非指令）
- 1806.01743 (Fu et al. 2018, ML stock selection China): 244 特征 + GA 筛选出 114 特征，性能几乎不变 → 特征冗余普遍，堆特征≠提升
- 2401.06139 (Stockformer 2024): 小波变换+多任务自注意力，多任务同时预测收益+方向
- 2312.05756 (PSO-BPNN + MGHMM 2023): 择时(picking)+选股(timing)融合，CSI300 状态模型
- 2012.06289 (ADD 2020): 解耦 excess/market 信息
- 2408.08483 (triple-I weights 2024): 趋势信号加权增强，多种子稳健性研究
- meta-labeling arXiv 检索命中的多为 label correction（CV/教育），金融 meta-labeling 主线仍是 López de Prado《Advances in Financial ML》第3章（书，非论文）；triple-barrier 检索 0 命中
- 结论：学术文献直接套用有限，我的 20 特征虽窄但"特征窄"未必是瓶颈；瓶颈更可能在标签设计（固定 horizon vs 事件驱动）与样本外稳定性

## 候选改进方向（待定稿）
1. Meta-labeling 二级模型：一级 = 现有 h10 LGB 产生信号；二级 = 逻辑回归学"信号是否值得执行"，输入含 lgb 分数、广度、波动、turnover 等 → 输出执行概率，调阈值控胜率。低成本：复用 predictions-h10.csv (29.8MB, 含 p_win/pred_net_return) + features.csv。⚠ 交集策略证明朴素双门槛太严，二级模型必须学权重而非硬门槛。
2. Triple-barrier 标签：替代固定 10d horizon，改用 上/下障碍+时间障碍 事件标签，与 3% stop 自然对齐。可用 existing Replay 引擎逐事件重放。
3. 换手率特征 (`turn` 已在 donor CSV 但不在 20 FEATURES)：turn_mean20、turn_delta、turn_vs_prev5 → 最便宜的消融实验。
4. 波动率目标仓位（Moreira-Muir）：对 h10 信号按 ret_vol20 反比缩放单笔金额，可能降 DD 而少损收益。
5. Purged K-Fold + embargo 验证协议升级（防泄漏），用于所有新实验。
6. 概率校准（isotonic/Platt）用于 logistic p_win，使 55% 阈值有真实概率含义。
7. 长假期周五问题：weekly policy 的 week_last_day_mask 升级（新实验，不动冻结线）。

## 纪律约束（必须保持）
- 2020–2026 全部历史已看过 → 不再对旧数据做阈值/参数搜索；新实验只能用新结果目录 + 预注册协议
- 冻结模型 hash 校验、无自动实盘、无券商、paper-forward 每日封存照常
- 永远不把 +28%/+31% 当未来预期陈述
- paper_forward.py:240 硬编码周五 vs exploration_features.week_last_day_mask 的不一致是未来策略升级项，不是热修复

## 下一步（写作顺序）
1. 定稿实验优先级（本轮候选 1→3→2→4→5→6→7）
2. 每个实验写预注册协议（假设、数据、判定标准、成功/失败条件）
preregistration 草稿见 improvement-plan.md
