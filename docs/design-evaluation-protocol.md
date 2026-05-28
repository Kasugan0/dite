# V2 评估协议设计

本文档定义 DITE 聚类重构 V2 的评估协议。

目标不是立刻跑出一个分数，而是先规定：

- V2 到底该如何被判断为“更好”
- 哪些数据集负责什么角色
- 哪些指标必须一起看
- 什么结果足以否决一个方案

## 核心原则

1. 不再只看最终簇数
2. 不再只看噪音数
3. 不再只靠人工肉眼抽样
4. 同时看过程、内部结构、外部约束
5. 一个方案如果在某一类数据集上明显破坏关键约束，应直接否决

## 评估分层

V2 的评估至少分 3 层：

### 1. 过程层

回答：

- 系统做了什么

关注：

- 候选边数量
- 强连接组件数量
- 进入边界裁决层的请求数
- LLM 判定数
- 降级路径次数
- `filename_dominant` 比例
- `extraction_failed` 比例

### 2. 内部聚类层

回答：

- 在不看人工标签时，结构本身是否比以前更合理

关注：

- DBCV 或同类密度友好指标
- 簇内稳定性
- 簇间分离度
- 中间聚类草案与最终结果的变化幅度

### 3. 外部约束层

回答：

- 系统是否满足人工知道的关键关系

关注：

- `must_link_recall`
- `must_not_link_violation`
- `fragmentation_per_cluster_id`
- `boundary_assignment_accuracy`

## 数据集角色

延续现有分层，但职责更明确。

### regression

用途：

- 防止代码重构后出现明显退步

不能决定：

- 默认策略晋升

### representative

用途：

- 决定主路径行为是否值得晋升

必须回答：

- 真实混合目录里是否还在严重碎簇
- 元数据和 LLM 介入后是否真的改善

### adversarial

用途：

- 否决看起来平均更好、但边界更烂的方案

必须覆盖：

- 文件名很像但内容不同
- 内容接近但标题不同
- 弱提取样本
- 多语言
- 模板相近但主题不同

## 必报指标

### A. 过程指标

- `num_files_total`
- `num_files_extraction_failed`
- `num_files_short_text`
- `num_files_filename_dominant`
- `candidate_edges_total`
- `candidate_components_total`
- `cluster_drafts_total`
- `adjudication_requests_total`
- `adjudication_requests_by_type`
- `adjudication_by_model`

### B. 聚类结构指标

- `final_num_clusters`
- `num_noise`
- `cluster_size_distribution`
- `fragmentation_score`
- `density_validation_score`

### C. 外部约束指标

- `must_link_recall`
- `must_not_link_precision`
- `must_not_link_violations`
- `cluster_id_fragmentation`
- `cluster_id_purity`

### D. V2 专项指标

- `filename_bias_rate`
- `llm_merge_accept_rate`
- `llm_merge_reversal_rate`
- `constraint_block_count`
- `manual_review_queue_size`

## 评估规则

### 一个方案不能因为这些现象就被误判为好

- 最终簇数下降
- 噪音数下降
- 小簇数量下降

原因：

- 这些结果可能来自粗暴合并
- 也可能来自文件名偏置增强

### 一个方案必须同时满足这些要求，才有资格继续推进

1. 代表性集上碎簇问题显著缓解
2. 对抗集上 `must_not_link` 违规不恶化
3. 文件名偏置率不上升，或上升有清晰解释
4. LLM 介入次数与收益成正比，而不是无差别膨胀

## 否决条件

出现以下任一情况，应直接否决某轮方案：

- 对抗集 `must_not_link` 违规明显上升
- 代表性集虽然簇数变少，但 `cluster_id_fragmentation` 没改善
- 文件名主导样本大规模影响最终结构
- LLM 裁决层成为主路径，无法在预算内稳定运行

## 结果呈现要求

每轮实验输出至少包含：

1. 总体摘要
2. 过程指标
3. 结构指标
4. 外部约束指标
5. 变化最大的文件或簇
6. 典型成功案例
7. 典型失败案例

如果只给一个总分，评估就会重新退化成拍脑袋。

## 与现有文档的关系

- `docs/validation.md`
  - 继续定义验证集的样本组织方式
- `docs/experiments.md`
  - 继续定义当前实验执行流程
- `docs/design-evaluation-protocol.md`
  - 专门定义 V2 应该怎样被评价

## 下一步

评估协议定下后，后续需要补：

1. 新版 `manifest` 规范
2. 新版实验输出 schema
3. 晋升默认策略的明确门槛
