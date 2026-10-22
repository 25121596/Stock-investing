# 股票模型改进计划（执行方向）v1.0

日期：2026-10-16 起草 · 状态：待用户过目后按序执行
依据：README.md / exploration-assessment.md / midterm-assessment.md / ml-assessment.md / ensemble 对照实验 / arXiv 文献检索（1806.01743, 2401.06139, 2312.05756, 2012.06289, 2408.08483 等）

---

## 0. 一句话执行方向

**"先加证据、再换标签、后加仓位"**：在不推翻现有冻结线的前提下，按 P0→P3 优先级跑四组预注册实验，每组都有成功/失败判定标准和随机对照；paper-forward 13 账户前瞻照常运行，不受影响。

## 1. 为什么是这个方向（诊断 → 处方）

| 诊断（已有证据） | 处方（实验） |
|---|---|
| 胜率 36–58% 波动大，+28%/+31% 集中在少数年份，交易数少（25–138 笔） | Meta-labeling 二级模型学"执行/不执行"，而不是手工加门槛 |
| 交集双门槛 (p_win≥.55 ∧ pred≥.005) 只有 22 笔交易 +2.23% → 硬门槛过严 | 二级模型学连续权重，输出可调执行概率 |
| 标签 = 固定 10d horizon + 3% stop，持有期分布中位数 5d | Triple-barrier 事件标签与止损/止盈结构天然对齐 |
| 20 特征全是价格/成交量衍生，`turn`（换手率）已下载未使用 | 最便宜的消融：+turnover 特征组 |
| weekly_lowvol 处于随机对照范围内（证据弱） | 保持观察，不新增仓位权重 |
| breadth_ensemble_avg 样本外 +35%→+1.2% 衰减 | 所有新实验必须报 2026 out-of-sample + 随机对照 |

## 2. 实验队列（预注册，按优先级）

### P0 — turnover 特征消融（最便宜，先跑）
- **假设**：加入 turn_mean20 / turn_delta5 / turn_vs_prev5（`turn` 已在 donor CSV 列中）能提升 h10 LGB 的 rank IC 与策略净收益
- **数据**：复用 data/forward-2026-10-13 + run_midterm.py 的 --features-from 缓存机制（需扩展缓存校验以覆盖新特征列）
- **判定**：新结果目录 results/turnover-feat-v1；rank IC 平均提升 ≥0.005 且 2022–25 与 2026 同向改善 → 采纳进候选特征池（不自动进冻结线）
- **失败处理**：无改善或样本外恶化 → 记录负结果，确认"特征窄不是瓶颈"（与 1806.01743 的 GA 特征筛选结论一致）

### P1 — Meta-labeling 二级执行模型（核心赌注）
- **假设**：一级 = 现有 h10 LGB 出信号；二级 = 逻辑回归（C 调小防过拟合）在信号样本上学 P(该信号净收益>0)，输入：pred_net_return、p_win、广度比、ret_vol20、turn 特征、市场 mret 系列 → 阈值可调控胜率
- **关键教训**：朴素交集双门槛 22 笔 +2.23% 证明"两个好指标 AND"不是元模型；必须让二级模型在年度 walk-forward 下自己学权重
- **判定**：results/meta-label-v1；相比 base_lgb10，2022–25 净收益不降超 20% 且 2026 胜率 ≥45%（当前 36%）→ 继续；否则回退
- **参考**：López de Prado《AFML》第3章思路（arXiv 无直接金融 meta-labeling 论文，triple-barrier 检索 0 命中，属书斋方法）

### P2 — Triple-barrier 事件标签重标
- **假设**：固定 10d horizon + 3% stop 改为 上障碍(+2σ·√h) / 下障碍(-2σ·√h) / 时间障碍(10d)，标签更贴近真实退出分布
- **实现**：midterm_labels.relabel_panel 已有 Replay2026 逐事件框架，扩展 barrier 事件类型
- **判定**：results/triple-barrier-v1；同样随机对照 + 2026 OOS；标签分布 audit（label_reason 分布变化）
- **风险**：σ 用 ret_vol20 估计，波动聚集期障碍宽度不稳 → 先做标签分布诊断再进策略回测

### P3 — 波动率目标仓位（Moreira-Muir）
- **假设**：对 h10 信号单笔金额按 1/ret_vol20 缩放（受单笔 5000 上限、3 只上限约束），DD 降、收益近似
- **判定**：results/vol-target-v1；DD 改善 ≥2pp 且 2022–25 净收益降幅 ≤1/3 → 采纳为仓位层选项

### 观察项（不占实验名额）
- weekly_lowvol：paper-forward 里继续观察，20 日流程检查 / 60 日阶段复盘
- paper_forward.py:240 硬编码周五 vs week_last_day_mask：属冻结策略，一致性已确认非 bug；假期周的"最后交易日"变体留给未来策略升级实验
- monitor_drift.py（PSI + Rank IC）已有，接入 P0/P1 实验产物做日常监控

## 3. 验证协议（所有实验共用，防泄漏铁律）
1. 年度 walk-forward，training_mask 保证 train.label_end.max() < test 起点
2. 2022–2025 为主要评估窗，2026（至 10-13）为 out-of-sample 只读验证
3. 每个策略版本配 3 个随机种子对照（2026101401/02/03 模式沿用）
4. 费用模型不变：佣金万2.5 min5元 + 过户 + 印花 + 滑点 0.05%/边（压力测试 ×2）
5. 新实验 → 新结果目录，sources 快照 + started.json 哈希，沿 run_midterm.py 惯例
6. **不再对 2020–2026 已看历史做阈值/参数搜索**：本计划所有判定标准都是预注册的，跑完即定论，不迭代调参

## 4. 纪律（不变）
- 不自动实盘、不接券商、paper-forward 每日封存（hash 链）
- 永不把历史回测 +28%/+31% 陈述为未来预期
- 冻结模型不动：改进全部走"新结果目录 + 新冻结线"，旧冻结线保持可复现

## 5. 时间安排（工作日节奏，均为业余可完成的粒度）
- 第 1 晚：P0 turnover 特征消融跑通 + 报告
- 第 2–3 晚：P1 meta-labeling 二级模型 + 报告
- 第 1 个周末：P2 triple-barrier 标签诊断 + 策略回测
- 第 2 周：P3 vol-target；视结果合并采纳，冻结新线（若通过）
- 每晚 paper-forward 封存照常（15:30 信号日 → 次日 09:15 前封存）

## 6. 成功的总标准
- 主线：2026 OOS 胜率 ≥45% 且净收益 >0，DD 不高于现 breadth_lgb10 的 5.33%
- 过程：四组实验无论成败都有预注册结论归档，负结果同样有价值（证伪方向）
