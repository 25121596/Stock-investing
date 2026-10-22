# A股小资金交易模型：面向V0.9原型的论文调研

调研结论：优先把“是否接受交易”作为可单独检验的决策，再研究逐持仓退出；同时把随机种子、多次试验和未来验证纳入评估。这个优先级是结合本地实验作出的工程判断，不是论文给本策略的盈利保证。

2026-10-21定向调研，核对13篇原始研究/正式论文来源，包含2025与2026研究。不是系统综述或完整精读清单；每篇明确本次阅读范围，未复现论文实验，也未新训本地模型。

## 本地问题与证据边界

此前动作拆分中，2026截至10月13日仅过滤收益16.43%，V0.8为15.73%；加入提前退出后的完整RL为7.81%。仅过滤仅9笔平仓，原三种子在2023—2025过滤结果仍明显分歧。这些事实来自本地冻结账本，不来自下列论文。

[本地动作拆分报告](D:/Benefits/stock-research/results/v09-rl-ablation-20261021-v2/report.html)；当前采用版本仍V0.8。

## 优先阅读顺序

P01（过滤任务）→P02（二层样本生成）→P04/P05（种子和统计）→P06/P07（研究选优）→P03（退出建模）。其余论文作为理论或工程补充。

## 逐篇核对

<a id="p01"></a>

### P01 · Trading via Selective Classification（2021）

Nestoras Chalkidis; Rahul Savani。ACM ICAIF 2021。 DOI：10.1145/3490354.3494379。

- 来源：[Trading via Selective Classification](https://arxiv.org/abs/2110.14914)；[公开正文](https://arxiv.org/html/2110.14914v2)
- 阅读范围：作者公开正文：第3节选择分类、第4节滚动验证、第5节回测及表4；未复现实验。
- 原研究：分类器可对不确定输入弃权，弃权时不持仓。作者在商品期货上用滚动训练、验证和测试比较选择性分类与普通分类，并加入滑点情景。
- 我方可借鉴（工程外推）：把接受/跳过既有机会作为独立任务；同时报告接受率、交易数、扣费净收益和风险，先建立逻辑回归等简单对照。
- 适用边界：实验为五种金属期货、30分钟数据，允许多空；不是A股日线、T+1、小额账户或独立二层过滤的直接实证。表4仍有大量亏损配置，不能概括成过滤必然盈利。
- 代码：本次未核实作者官方代码。

<a id="p02"></a>

### P02 · Stacked Generalization（1992）

David H. Wolpert。Neural Networks 5(2), 241–259。 DOI：10.1016/S0893-6080(05)80023-1。

- 来源：[Stacked Generalization](https://www.sciencedirect.com/science/article/pii/S0893608005800231)
- 阅读范围：出版社元数据与可检索摘要；未读取付费全文，直接打开页面时遇403。
- 原研究：二层学习使用第一层模型对未参与其拟合的数据所作的预测，作为新的学习输入。
- 我方可借鉴（工程外推）：二层过滤所用V0.8/V2分数应按时间前向生成；对应标签结束时间必须早于拟合边界。按事件持有窗口排除跨边界标签。
- 适用边界：这是一般集成学习论文，不是金融meta-labeling实证。时间前向、事件窗口清除是我方结合金融时序作出的工程外推，不能说原论文证明了A股二层策略。
- 代码：不以任何未核实复现库作为原作者代码。

<a id="p03"></a>

### P03 · Deep Optimal Stopping（2019）

Sebastian Becker; Patrick Cheridito; Arnulf Jentzen。Journal of Machine Learning Research 20(74), 1–25。

- 来源：[Deep Optimal Stopping](https://www.jmlr.org/papers/v20/18-232.html)；[公开正文](https://jmlr.csail.mit.edu/papers/volume20/18-232/18-232.pdf)
- 阅读范围：JMLR元数据、摘要、正文引言与最优停止建模段；未复现数值实验。
- 原研究：从模拟路径学习停止规则；最优停止的核心是比较现在兑现的收益与继续等待的价值。实验包括百慕大期权等合约及数学停止问题。
- 我方可借鉴（工程外推）：若重做退出模块，按单笔持仓记录年龄、剩余期限、浮盈亏、波动、可卖状态和净退出成本，分别估计持有与卖出的价值。
- 适用边界：不是股票选股盈利论文，也不是当前表格Q算法的验证。依赖可模拟路径的假设；个股收益过程与成交失败须另行处理，数据期末不应为套论文而虚构清仓。
- 代码：本次未核实作者代码。

<a id="p04"></a>

### P04 · Unstable Gains: Multiplicity-Aware Evaluation of Financial Deep Reinforcement Learning（2026）

Przemysław Grądzki。The Journal of Finance and Data Science 12, 100205；2026-09-23上线。 DOI：10.1016/j.jfds.2026.100205。

- 来源：[Unstable Gains: Multiplicity-Aware Evaluation of Financial Deep Reinforcement Learning](https://www.sciengine.com/doi/10.1016/j.jfds.2026.100205)；[公开正文](https://doi.org/10.1016/j.jfds.2026.100205)
- 阅读范围：正式页面元数据与摘要、出版社索引正文第4—5节的实验及统计说明；未逐项核查附录和代码。DOI直链本轮访问不稳定。
- 原研究：固定配置下用20个种子研究美股交易及加密资产配置；种子间结果和配置会变化，一些比较经多重比较修正后不再显著。
- 我方可借鉴（工程外推）：记录全部种子和淘汰结果，报告策略分歧及结果分布，不选最佳种子。将训练随机性与不同市场历史的不确定性分开。
- 适用边界：其统计分析以同一历史路径为条件，不覆盖未来结构变化；作者也限定DSR的近似性及比较范围。20种子是该文设计，不是可靠性通用门槛，也不等于20段独立行情。
- 代码：本次未逐项核实该文代码或复现实验。

<a id="p05"></a>

### P05 · Deep Reinforcement Learning at the Edge of the Statistical Precipice（2021）

Rishabh Agarwal; Max Schwarzer; Pablo Samuel Castro; Aaron Courville; Marc G. Bellemare。NeurIPS 2021。

- 来源：[Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)；[公开正文](https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
- 阅读范围：官方论文摘要、引言与少运行次数/统计评估段；未复现多任务基准。
- 原研究：少数训练运行的均值、排名可能忽视较大不确定性；提出性能分布、区间估计和较稳健聚合指标，配套rliable。
- 我方可借鉴（工程外推）：三个种子先完整展示，报告范围和近空仓比例。以后新增种子须预定数量、全部保留；同时另报按时间段的经济结果。
- 适用边界：原实验主要是游戏/控制多任务基准。不能直接把其跨任务bootstrap用于单一金融市场，也不能凭三个种子生成看似可靠的显著性结论。
- 代码：https://github.com/google-research/rliable

<a id="p06"></a>

### P06 · The Probability of Backtest Overfitting（2015作者修订稿；后刊JCF）

David H. Bailey; Jonathan M. Borwein; Marcos López de Prado; Qiji Jim Zhu。Journal of Computational Finance；本次方法引用2015作者修订稿。 DOI：10.21314/JCF.2016.322。

- 来源：[The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- 阅读范围：作者稿摘要、介绍与CSCV/PBO方法及限制相关段；出版社核对题目作者和DOI。
- 原研究：把多次回测后的选优作为统计问题，用组合对称交叉验证分析样本内胜出者的样本外排序退化。
- 我方可借鉴（工程外推）：先登记所有模型、参数、奖励、种子和失败试验，保存统一日期的收益矩阵；在候选集完整时增加选优敏感性诊断。
- 适用边界：PBO不是未来亏损概率或盈利概率，也不能修复错误成交和行情。互补分块允许未来块参与拟合，不等同真实前向部署；时序验证和未来封存仍需单独保留。
- 代码：本次未核实原作者代码。

<a id="p07"></a>

### P07 · The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality（2014）

David H. Bailey; Marcos López de Prado。The Journal of Portfolio Management 40(5), 94–107。 DOI：10.3905/jpm.2014.40.5.094。

- 来源：[The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- 阅读范围：作者原稿摘要、选择偏差和多重检验说明及DSR方法；未对本项目实际计算DSR。
- 原研究：对多次尝试后报告的高Sharpe进行选择偏差及非正态性修正。需要关注试验次数、样本长度和收益分布。
- 我方可借鉴（工程外推）：保留原始指标，同时登记完整研究搜索范围；满足数据与假设条件后再报告修正指标，不能只统计最后一次成功实验。
- 适用边界：有效独立试验数和时序依赖不能随意设置。DSR不是通过一次计算就获得的实盘盈利认证，且不能把同一路径的种子当独立市场样本。
- 代码：本次未核实原作者代码。

<a id="p08"></a>

### P08 · Optimal Trading with Linear Costs（2012公开稿）

Joachim de Lataillade; Cyril Deremble; Marc Potters; Jean-Philippe Bouchaud。本次按arXiv:1203.5957作者公开稿引用。 DOI：10.48550/arXiv.1203.5957。

- 来源：[Optimal Trading with Linear Costs](https://arxiv.org/abs/1203.5957)；[公开正文](https://arxiv.org/pdf/1203.5957)
- 阅读范围：arXiv作者元数据与摘要；未逐节推导解析阈值。
- 原研究：在带线性成本和持仓上限的模型下，交易阈值随成本变化；与二次风险惩罚情形下的不交易带建立联系。
- 我方可借鉴（工程外推）：为是否成交加入成本门槛或迟滞对照，衡量信号优势是否足以覆盖切换仓位的成本。
- 适用边界：原模型有多空切换及特定预测过程假设；小额A股的最低佣金、整手和不能即时卖出不满足简单线性成本模型，阈值需重新验证。
- 代码：本次未核实原作者代码。

<a id="p09"></a>

### P09 · Learning to Trade via Direct Reinforcement（2001）

John Moody; Matthew Saffell。IEEE Transactions on Neural Networks 12(4), 875–889。 DOI：10.1109/72.935097。

- 来源：[Learning to Trade via Direct Reinforcement](https://doi.org/10.1109/72.935097)；[公开正文](https://people.idsia.ch/~juergen/rnnaissance2003talks/MoodySaffellTNN01.pdf)
- 阅读范围：原论文可检索摘要与元数据；PDF文本编码异常，本次不宣称逐页精读。
- 原研究：直接优化含成本的风险调整收益，使用recurrent reinforcement learning；与估计价值函数的Q-learning路径不同。
- 我方可借鉴（工程外推）：把真实净值和成本作为训练目标基础，区分预测准确率与账户效用。我们现有奖励已从扣费净值计算，不能再无理由重复扣成本。
- 适用边界：其外汇/资产配置演示不能验证本项目A股策略；不能由文中某些结果推断所有策略梯度方法都优于Q-learning。
- 代码：本次未核实原作者代码。

<a id="p10"></a>

### P10 · FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning（2022）

Xiao-Yang Liu et al.。NeurIPS 2022 Datasets and Benchmarks。 DOI：10.52202/068431-0134。

- 来源：[FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0bf54b80686d2c4dc0808c2e98d430f7-Abstract-Datasets_and_Benchmarks.html)；[公开正文](https://proceedings.nips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf)
- 阅读范围：官方摘要、元数据与作者仓库说明；不是对每个环境实现的源码认证。
- 原研究：把动态数据处理、市场环境、示例和基准组织成金融RL研究基础设施，强调低信噪比、幸存偏差与回测过拟合。
- 我方可借鉴（工程外推）：借鉴统一环境接口和数据血缘；把新算法接入现有可审核撮合账本，以相同成交规则比较。
- 适用边界：框架覆盖市场和算法多，不等于其默认模拟自动满足本项目T+1、最低佣金、限价及公司行动规则；示例收益不是可迁移盈利证据。
- 代码：https://github.com/AI4Finance-Foundation/FinRL-Meta

<a id="p11"></a>

### P11 · TradeMaster: A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning（2023）

Shuo Sun et al.。NeurIPS 2023 Datasets and Benchmarks。 DOI：10.52202/075280-2576。

- 来源：[TradeMaster: A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8f6f7f2ba4137124ac976286eacb611-Abstract.html)；[公开正文](https://proceedings.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf)
- 阅读范围：官方摘要、论文引言及任务分层说明；未部署或复现该平台。
- 原研究：将数据、特征、市场环境、投资决策和评估等环节组织为可比较、可复现的交易研究平台。
- 我方可借鉴（工程外推）：分开评价选股、是否交易、仓位和执行；每项改造使用统一输入、成本及基线，避免不同任务结果混榜。
- 适用边界：执行某个既定订单、组合配置和产生买入方向是不同任务；平台覆盖更多任务不等于解决小资金A股收益问题。
- 代码：https://github.com/TradeMaster-NTU/TradeMaster

<a id="p12"></a>

### P12 · Robust Reinforcement Learning in Finance: Modeling Market Impact with Elliptic Uncertainty Sets（2025）

Shaocong Ma; Heng Huang。NeurIPS 2025 Main Conference。 DOI：10.52202/085713-1404。

- 来源：[Robust Reinforcement Learning in Finance: Modeling Market Impact with Elliptic Uncertainty Sets](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3c129892b4f9c8326aba665425a470c5-Abstract-Conference.html)；[公开正文](https://papers.neurips.cc/paper_files/paper/2025/file/3c129892b4f9c8326aba665425a470c5-Paper-Conference.pdf)
- 阅读范围：官方元数据和摘要；未逐式核查椭圆不确定集合的数学推导或复现实验。
- 原研究：研究历史环境中动作不影响价格、实际交易却有市场冲击的错配，用带方向性的椭圆不确定集合进行鲁棒策略评估。
- 我方可借鉴（工程外推）：作为以后流动性、成交量占比和不利成交情景的研究参考；先保留费用、延迟和拒单压力测试。
- 适用边界：大额成交冲击与3万元账户的主要问题不同；本项目当前瓶颈是过滤信息量、退出状态和统计证据，不应把该文当作必须立即增加复杂网络的依据。
- 代码：本次未核实官方代码。

<a id="p13"></a>

### P13 · On the Foundations of Noise-free Selective Classification（2010）

Ran El-Yaniv; Yair Wiener。Journal of Machine Learning Research 11(53), 1605–1641。

- 来源：[On the Foundations of Noise-free Selective Classification](https://jmlr.org/papers/v11/el-yaniv10a.html)；[公开正文](https://jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf)
- 阅读范围：JMLR正式页面元数据和摘要；未逐一核对理论证明。
- 原研究：研究允许拒绝分类时，覆盖率与所接受样本的分类风险如何权衡。
- 我方可借鉴（工程外推）：过滤器的评价需配合接受机会比例，避免靠近乎空仓形成好看的条件胜率。若把分数解释为概率，须另做校准；接受阈值本身仍须前向验证。
- 适用边界：noise-free理论不等于金融噪声市场的风险保证；分类错误率也不等于扣费收益或账户回撤。
- 代码：本次未核实官方代码。

## 面向本项目的改造顺序（尚未实施）

### 第一步：给仅过滤建立独立的简单对照

先比较V0.8、固定简单过滤、相同粒度的监督过滤、现有仅过滤RL。初版可用逻辑回归，再决定是否需要更复杂模型。当前RL是对一批待买单统一决策；若改为逐股过滤，须明确这是动作粒度变化，避免把粒度收益误归给算法。 相关：[P01](#p01)、[P02](#p02)、[P13](#p13)。

### 第二步：用可执行净收益建标签，防止二层偷看

训练特征只含决策时已知信息；上游分数按时间前向产生，不能用第一层在自身训练样本上的拟合分数。训练标签须在拟合时已完成，阈值调优或校准只用前向验证段。标签用次日可成交、完整持仓退出、费用及公司行动后的结果，记录无法成交；个股候选标签只是代理，逐股过滤也属于新政策，最终仍回到同一现金/持仓账本。 相关：[P01](#p01)、[P02](#p02)、[P09](#p09)。

### 第三步：承认现有模型已经有门控

V0.8已有V2收益预测阈值门控和V0.8排序。新增过滤若只复用同一批特征，未必增加信息。V0.8的排序分数也不是校准胜率，不能把某个分数解释为百分之几的盈利概率。新增门控必须证明相对现有门控有额外价值。 相关：[P02](#p02)、[P13](#p13)。

### 第四步：评价赚钱能力与接受率，控制研究选优

先定义机会单位：当前是每日一批待买单，逐股过滤则是候选股票；接受率只能在相同粒度下比较，分母是原策略本可排单的机会，实际成交率另列。同时报告净收益、回撤、费用、暴露和交易数。预先限定模型/阈值/种子，登记全部尝试，区分减少暴露与筛选价值。增补种子只评估优化随机性，不增加独立市场年份。 相关：[P01](#p01)、[P04](#p04)、[P05](#p05)、[P06](#p06)、[P07](#p07)。

### 第五步：退出模块作为另一个有限期决策课题

研究逐持仓的继续/卖出净价值，加入剩余期限和交易约束；基线止损仍作可审核对照。学习“卖得更频繁”不等于学习“卖得更好”。策略到期退出与数据样本期末要分开；无法成交时不按理想卖出结算。 相关：[P03](#p03)、[P08](#p08)、[P09](#p09)。

### 第六步：多年度只用于开发，独立证据留给未来封存

若用户另行授权重训，可预注册2022—2024训练、2025开发验证，并将已看过的2026回放标记为开发诊断。按标签完成日清理边界，不能随机拆股票/日期造成信息串入。最终性能由冻结以后另立的未来影子记录检验；本轮文献调研不启动训练。 相关：[P02](#p02)、[P04](#p04)、[P06](#p06)、[P10](#p10)。

## 不应从文献推出的结论

- 概率、分类正确率、交易胜率与成本后收益是不同指标。即使60%交易各赚1%、40%各亏2%，未计费期望收益仍为−0.2%。
- 选择分类论文的风险界限依赖其统计条件，不能移植成A股未来盈利/回撤保证。公开论文中未找到直接验证本地固定300股、3万元、当前交易制度与账本条件下稳定胜率超过60%的证据；这不是断言全世界不存在此类论文。
- Meta-labeling是有用的方法概念，但《Advances in Financial Machine Learning》是书，本调研不把它写成独立论文实证，也不把选择分类和二层meta模型说成完全同一方法。
- 金融执行/大单清算论文的收益对象常是执行成本，而非股票方向收益；P12因此列为补充。
- 现有奖励已使用扣费净值；额外成本惩罚会改变目标。强迫空仓惩罚也会改变风险偏好，不能替代盈利证据。
- 三个种子多数投票可以是预定策略，但不能据此认定三组都稳定。研究试验登记和真实未来封存比把旧历史改名盲测更关键。

## 调研保存

原始书目信息见papers.json；网页阅读版为report.html。本轮只新增笔记和报告，未安装交易框架、未重训、未修改DAY1或RQAlpha账户。
