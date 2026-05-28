# V2 数据结构设计

本文档定义 DITE 聚类重构 V2 需要的核心数据结构。

它不讨论具体代码实现，也不规定类名必须与本文完全一致。它的目标，是先把系统内部真正要流动的数据讲清楚，避免后续实现重新滑回“一个文件只有一段文本和一个 embedding”的老路。

## 设计目标

V2 数据结构必须满足这些要求：

1. 一个文件必须能承载多种信号，而不是只承载正文文本
2. 每种信号都要能追溯来源和质量
3. 聚类、边界判定、簇表示、评估都要共享同一套对象语义
4. LLM、规则和 embedding 产生的中间结论必须显式落在结构里
5. 弱提取、短文本、文件名主导等异常状态必须可观测

## 1. DocumentFeatures

`DocumentFeatures` 是 V2 的核心文件级对象。

它表达的是：

- 一个文件在进入聚类系统后，正式拥有的全部结构化特征

它不等于：

- 原始文件
- 单次提取结果
- 单个 embedding

### 必要字段

#### 标识与路径

- `file_id`
  - 当前运行内稳定唯一 ID
- `path`
  - 原始相对路径或绝对路径
- `name`
  - 文件名
- `stem`
  - 去扩展名文件名
- `extension`
  - 文件扩展名
- `parent_path_tokens`
  - 路径分词结果

#### 内容与文本

- `content_text`
  - 当前用于语义理解的主文本
- `content_excerpt`
  - 截断后的短内容，用于报告和 LLM 输入
- `title_candidates`
  - 从正文、文件名、页标题中提取出的标题候选列表
- `language`
  - 主语言
- `token_count_estimate`
  - 粗略 token 数

#### 元数据与语义压缩

- `summary`
  - 文件级一句话摘要
- `keywords`
  - 关键词列表
- `entities`
  - 命名实体列表
- `topic`
  - 主题短语
- `domain`
  - 领域标签
- `document_type`
  - 例如讲义、论文、报告、配置说明、教程、课件

#### 结构与布局

- `layout_hints`
  - 结构提示对象
- `page_count`
  - 页数或近似页数
- `has_table`
  - 是否含明显表格
- `has_image_heavy_layout`
  - 是否明显偏图像/扫描
- `template_signals`
  - 模板化强度或模板提示

#### 表示层特征

- `content_embedding`
  - 正文语义向量
- `metadata_features`
  - 文件名、路径、扩展名、标题等元数据特征
- `entity_features`
  - 实体、术语或对象级特征
- `layout_features`
  - 布局或模板级特征

#### 质量与来源

- `quality_flags`
  - 质量状态对象
- `selected_source`
  - 最终内容来源，例如 text/docling/vlm/cache
- `extraction_trace`
  - 提取过程摘要

### quality_flags

`quality_flags` 至少应包含：

- `extraction_failed`
- `short_text`
- `filename_dominant`
- `ocr_noisy`
- `layout_sparse`
- `language_uncertain`
- `low_confidence_analysis`

这些标记不只是日志信息，而是后续聚类和裁决层的正式输入。

## 2. CandidateEdge

`CandidateEdge` 表达两个文件或两个对象之间的候选关联关系。

它存在的目的，是把“哪些东西值得拿去进一步比较”正式化。

### 必要字段

- `source_id`
- `target_id`
- `edge_type`
  - 例如 `content_similarity`、`filename_similarity`、`entity_overlap`、`near_duplicate`
- `score`
  - 标准化分数
- `evidence`
  - 解释这条边从何而来
- `hard_constraint`
  - 可选，`must_link` 或 `must_not_link`
- `quality_guard`
  - 是否受某类质量问题影响

### 设计要求

- 任何会影响合并或召回的强关系，都应尽量先表现为 `CandidateEdge`
- 不允许把大量中间关系藏在“某个函数内部算过但没留下痕迹”

## 3. CandidateComponent

`CandidateComponent` 表达强连接召回层得到的初始连通块或近重复团体。

它的目标不是成为最终簇，而是：

- 表达系统已经非常确信的一组局部强关系

### 必要字段

- `component_id`
- `member_file_ids`
- `component_type`
  - 例如 `near_duplicate_group`、`strong_semantic_group`
- `formation_evidence`
- `confidence`

## 4. ClusterDraft

`ClusterDraft` 表达主题聚类阶段得到的中间簇。

它是：

- 聚类器当前给出的主题结构草案

它不是：

- 最终用户可见簇

### 必要字段

- `draft_cluster_id`
- `member_file_ids`
- `origin`
  - 例如 `hdbscan`、`graph_community`、`manual_seed`
- `centroid_features`
  - 内容/元数据/布局多视图中心
- `noise_members`
  - 可选，暂未稳定归属的成员
- `merge_candidates`
  - 候选目标簇列表

## 5. ClusterRepresentation

`ClusterRepresentation` 是最终面向解释和命名的簇表示对象。

V2 里不再只输出 `name`，而是至少输出：

- `name`
- `summary`
- `keywords`
- `topic`
- `domain`
- `representative_files`
- `evidence_summary`

### 必要字段

- `cluster_id`
- `name`
- `summary`
- `keywords`
- `topic`
- `domain`
- `document_type_profile`
- `representative_file_ids`
- `evidence_summary`

### 设计要求

- `ClusterRepresentation` 必须能脱离原始长文，独立支撑人工复核
- LLM 如参与生成，必须保留输入证据摘要

## 6. AdjudicationRequest

`AdjudicationRequest` 表达一个需要更贵决策层处理的模糊案例。

### 类型

- 文件归属裁决
- 小簇是否合并
- 两个簇是否只是被错误拆开
- 同主题但异模板是否应保持分离

### 必要字段

- `request_id`
- `request_type`
- `subjects`
  - 文件或簇 ID
- `candidate_targets`
- `evidence_bundle`
- `trigger_reason`
  - 为什么进入裁决层

## 7. AdjudicationDecision

`AdjudicationDecision` 表达边界判定层给出的结构化结论。

### 必要字段

- `request_id`
- `decision`
  - 例如 `merge`、`reject_merge`、`assign_to_cluster`、`keep_noise`
- `confidence`
- `reason`
- `supporting_evidence`
- `model_used`
  - 规则 / cross-encoder / LLM
- `fallback_used`

### 设计要求

- 不允许只返回自由文本
- 所有 LLM 判定都必须落成结构化决策对象

## 8. EvaluationRecord

`EvaluationRecord` 不是运行时核心对象，但它必须和上面所有对象共享语义。

它至少要能引用：

- 文件级特征
- 候选边
- 中间簇
- 最终簇表示
- 裁决结果

这样评估才不会退化成只看最后标签。

## 9. 对象之间的关系

最小关系图如下：

```text
raw file
  -> DocumentFeatures
  -> CandidateEdge
  -> CandidateComponent
  -> ClusterDraft
  -> AdjudicationRequest / AdjudicationDecision
  -> ClusterRepresentation
  -> EvaluationRecord
```

这里最关键的约束是：

- `DocumentFeatures` 是一切上层对象的根
- `CandidateEdge` 和 `ClusterDraft` 必须可回溯到文件级证据
- `ClusterRepresentation` 必须可回溯到中间簇和代表文件
- `AdjudicationDecision` 必须可回溯到触发它的请求和证据

## 10. 设计边界

本文档明确不决定这些内容：

- 具体 Python dataclass 或 pydantic 模型怎么写
- 哪些字段何时缓存到 SQLite
- 哪些字段会暴露到最终 CLI JSON
- 哪些特征由 LLM 生成、哪些由规则生成

这些属于下一层设计或实现问题。

## 11. 下一步

有了这份数据结构设计后，后续必须补两件事：

1. 流水线契约
2. 评估协议

否则这些对象仍然会停留在纸面上，无法约束真正的系统行为。
