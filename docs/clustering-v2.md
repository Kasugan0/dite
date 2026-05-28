# 聚类重构 V2

本文档描述 DITE 下一阶段的聚类重构方案。

它不是当前实现说明。当前真实行为仍以 `src/` 和 `docs/clustering.md` 为准。

这次重构明确允许破坏当前聚类输出、默认参数、内部数据结构和部分配置语义。目标不是保持旧行为，而是把系统从“单一 embedding 驱动的碎簇流水线”重构成“多信号、两阶段、可解释”的文档聚簇系统。

## 为什么要重构

当前主路径大致是：

1. 提取最终文本
2. 生成单个 embedding
3. 直接在高维向量上运行 HDBSCAN
4. 对噪音点做 k-NN 修复
5. 对小簇做一轮受限合并
6. 用 LLM 命名

这条路径已经暴露出几个结构性问题：

- 它把“同主题”“近重复”“模板相似”“文件名提示”“格式差异”都压进一个向量里解决。
- 文件名信号当前会直接混入 embedding 输入，甚至在弱提取时退化成“只看文件名”。
- HDBSCAN 被直接用于高维 embedding 空间，而它本身并不擅长承担全部语义决策。
- k-NN 修噪音和小簇合并都只是后补丁，不是主解法。
- LLM 当前只在最后做命名，参与得太晚，无法帮助边界判定和簇间重审。

结果就是：

- 近重复版本经常能聚到一起
- 同主题但不同标题、不同格式、不同来源的文件容易被拆碎
- 用户看到的簇名看似合理，但底层分组并不可靠

## 重构目标

这次重构的目标不是“把 HDBSCAN 参数调好一点”，而是：

1. 把文件从“单向量对象”升级成“多视图文档对象”
2. 把“近重复识别”和“主题聚类”拆开
3. 把文件名、路径、扩展名、结构提示变成显式元数据，而不是偷偷混入正文
4. 让 LLM 参与保守判定和簇表示，而不只是命名
5. 建立能支撑回归测试和人工复核的观测与评估体系

## 设计原则

### 1. 多信号优先，不再单押 embedding

embedding 仍然保留，但只作为重要信号之一。

聚类决策至少要显式考虑这些输入：

- 正文内容
- 文件名
- 路径上下文
- 扩展名和格式
- 标题候选
- LLM 摘要、关键词、主题、领域
- 提取质量和文本长度
- 版式或模板提示

### 2. 先召回，再精判

便宜模型负责粗筛，贵模型负责少量边界判断。

翻译成 DITE 的结构就是：

- embedding 负责召回候选边和候选簇
- 结构化元数据负责补强与约束
- LLM 或 cross-encoder 只处理少量模糊样本

### 3. 元数据必须显式

文件名和路径不是脏信号，它们是独立信号。

问题不在于“用了文件名”，而在于当前做法把文件名伪装成正文的一部分，导致评估和调参都被污染。

新方案里，文件名、路径、扩展名都必须单独建模、单独观测、单独权衡。

### 4. LLM 只做高价值判断

LLM 不直接负责全量分簇。

它主要用于：

- 生成稳定的文档摘要、关键词、主题和领域标签
- 生成簇级表示
- 对小簇合并、边界文件归属做保守判定
- 解释为什么合并或不合并

### 5. 先能解释，再谈最优

新的聚类系统必须能回答这些问题：

- 某两个文件为什么被拉近
- 某个小簇为什么被合并
- 某个边界文件为什么留在噪音区
- 文件名信号有没有压过正文信号
- LLM 在哪里改变了最终结果

## 目标架构

### 阶段 0：输入净化

先解决最会污染实验的输入问题。

规则：

- 提取失败文件不再默认混入主内容聚类
- 极短文本必须显式打标
- 文件名 fallback 不再伪装成正文 embedding
- 失败提取、短文本、文件名主导都要进入报告和评估

产物：

- `DocumentFeatures`
- `quality_flags`

### 阶段 1：文档特征构建

每个文件不再只产出一段最终文本和一个向量，而是产出结构化特征。

建议的内部结构：

```text
DocumentFeatures
  content_text
  title_candidates
  file_name_tokens
  path_tokens
  extension
  language
  summary
  keywords
  topic
  domain
  layout_hints
  content_embedding
  metadata_features
  quality_flags
```

这里的 `summary / keywords / topic / domain / layout_hints` 优先复用现有 `analyzer` 能力，而不是另造一套系统。

当前状态补充：

- `DocumentFeatures` 已经落地为正式运行时对象。
- analyzer 增强入口已经存在，并且通过 `feature_extraction.analysis_enabled` 受控开启。
- 默认仍关闭，失败回退到轻规则特征，不会中止主流程。

### 阶段 2：强连接召回

先识别“几乎肯定应该在一起”的文件对或小团体。

这一层主要处理：

- 同文档不同格式
- 重命名副本
- 近重复版本
- 强相似 must-link 对

可用信号：

- 正文 embedding 高相似边
- 文件名规范化后高相似边
- 标题候选高相似边
- 重复哈希或近重复文本
- `community_detection` / `paraphrase_mining` 风格的高置信边

这一层输出的不是最终簇，而是：

- 强连接边
- 初始组件
- 近重复族群

当前状态补充：

- `CandidateEdge` 和 `CandidateComponent` 已落地。
- `CandidateComponent` 已拆成 `near_duplicate_group` 和 `strong_semantic_group`。
- 只有 near-duplicate 族群会自动预绑定；强语义组件只保留作证据。

### 阶段 3：主题聚类

对剩余文件或初始组件做主题层面的分组。

默认实验主线：

- 内容 embedding
- 显式元数据特征
- 可选降维
- 聚类或图社区发现

短期建议保留 HDBSCAN 路线，但不再直接在原始高维 embedding 上硬跑。

建议至少实验：

- `PCA/UMAP -> HDBSCAN`
- 基于加权图的社区发现

这里的关键不是具体算法名字，而是：

- 不再让一个高维单向量承担全部分组责任
- 不再把“近重复”和“同主题”混成同一种关系

当前状态补充：

- `ClusterDraft` 已落地为正式中间对象。
- `topic_clustering` 已成为正式默认配置来源。
- `PipelineOptions` 中的相关字段现在只作为内部实验覆盖层。

### 阶段 4：边界判定

把昂贵模型留给最值钱的判断。

适用对象：

- 小簇之间是否应合并
- 边界文件是否应并入候选簇
- 多个候选簇是否只是被错误拆开

这里可以按成本分层：

1. 便宜规则和相似度阈值
2. cross-encoder 风格 pairwise 精判
3. LLM JSON 判定

LLM 输入不直接喂整篇长文，而是喂：

- 代表文件摘要
- 关键词
- 标题候选
- 文件名模式
- 扩展名分布
- 路径上下文
- 结构化簇表示

LLM 输出必须结构化，例如：

```json
{
  "should_merge": true,
  "confidence": 0.87,
  "shared_topic": "植物病害识别",
  "reason": "两个簇都围绕番茄叶片病害图像识别，差异主要来自模型名称和论文写法",
  "conflict_signals": [
    "一个簇更偏综述，另一个簇更偏具体 YOLO 实现"
  ]
}
```

当前状态补充：

- 规则裁决路径已正式接入运行时。
- `cluster_adjudication.enable_llm_judging` 已支持可选 LLM adjudication。
- 默认关闭，失败回退到规则结果，不影响主流程稳定性。

### 阶段 5：簇表示与命名

命名不再只是最后贴个标题，而是要产出完整簇表示：

- `name`
- `summary`
- `keywords`
- `topic`
- `domain`
- `evidence`

这一步可以继续使用 LLM，但它的职责从“命名器”升级成“簇表示器”。

当前状态补充：

- `cluster_representation.mode` 已支持 `deterministic` 和 `llm_enhanced` 两条路径。
- 默认仍是 deterministic。
- `llm_enhanced` 失败会回退到 deterministic 聚合表示。

## LLM 的角色

### 应该做什么

- 为文件生成稳定摘要和关键词
- 为簇生成主题表示
- 对少量候选合并做保守裁决
- 输出理由，供调试和人工复核

### 不应该做什么

- 直接对全部文件做全量分类式分簇
- 直接替代第一阶段召回
- 在没有结构化约束的情况下输出自由文本决策

### Prompt 设计要求

本次重构不会引入一个沉重的 prompt 管理子系统，但需要最小化版本化：

- `document_analysis_prompt_v1`
- `cluster_merge_judge_prompt_v1`
- `cluster_representation_prompt_v1`

每个 prompt 都必须固定：

- 输入字段
- 截断策略
- JSON schema
- 失败回退路径

## 配置重构方向

当前 `clustering` 配置主要围绕 HDBSCAN 和后处理参数设计。

V2 预计会新增或重构这些配置域：

- `feature_extraction`
  - 是否启用摘要、关键词、主题抽取
  - 每种特征的 token 预算
- `metadata_signals`
  - 文件名、路径、扩展名、标题候选的权重策略
- `candidate_generation`
  - 强连接阈值
  - 近重复识别开关
- `topic_clustering`
  - 降维策略
  - 聚类算法
- `cluster_adjudication`
  - 小簇合并策略
  - 是否启用 LLM 判定
  - cross-encoder / LLM 的预算阈值
- `cluster_representation`
  - 命名、摘要、关键词生成策略

这意味着当前很多聚类配置项会被保留、降级、重命名或删除。

## 可观测性与评估

V2 不再只看：

- 总簇数
- 噪音数
- 合并数

至少要补这些指标：

- `must_link_recall`
- `must_not_link_violation`
- `fragmentation_per_cluster_id`
- `filename_bias_rate`
- `extraction_failure_impact`
- `llm_merge_accept_rate`
- `llm_merge_reversal_rate`
- `boundary_assignment_accuracy`

同时，报告里应该能看到：

- 每个簇的形成路径
- 哪些边来自正文相似
- 哪些边来自元数据
- 哪些合并是 LLM 决定的

## 分阶段实施计划

### Phase 1：止血

目标：

- 去掉最明显的输入污染
- 建立新特征结构的骨架

优先事项：

- 禁止提取失败文件伪装成正文 embedding
- 把 `filename_dominant`、`short_text`、`extraction_failed` 写入报告
- 建立 `DocumentFeatures` 结构

### Phase 2：特征化

目标：

- 让元数据和语义表示进入正式流水线

优先事项：

- 复用 `analyzer` 生成 `summary / keywords / topic / domain`
- 显式建模文件名、路径、扩展名
- 为后续边界判定准备簇级表示

### Phase 3：两阶段聚簇

目标：

- 把近重复识别和主题聚类拆开

优先事项：

- 新增强连接召回层
- 引入降维实验或图聚类实验
- 降低 HDBSCAN 在系统中的唯一决策权

### Phase 4：LLM 保守判定

目标：

- 只让 LLM 处理高价值、不稳定、难规则化的边界情况

优先事项：

- 小簇合并裁决
- 边界文件归属裁决
- 簇表示生成

### Phase 5：默认策略晋升

目标：

- 用代表性验证集和对抗集决定哪些 V2 行为成为默认值

优先事项：

- 固定评估口径
- 迁移旧配置
- 清理已废弃旧路径

## 非目标

这次重构不以这些内容为主线：

- 新建 prompt 管理产品
- 引入向量数据库
- 增量在线学习
- 图片重命名
- HTML 报告系统
- 把 LLM 变成全量主聚类器

## 外部参考

这些资料直接影响了 V2 方案的边界判断：

- HDBSCAN FAQ
  - https://hdbscan.readthedocs.io/en/latest/faq.html
- BERTopic algorithm
  - https://maartengr.github.io/BERTopic/algorithm/algorithm.html
- BERTopic LLM representation docs
  - https://github.com/maartengr/bertopic/blob/master/docs/getting_started/representation/llm.md
- BERTopic quickstart
  - https://github.com/maartengr/bertopic/blob/master/docs/getting_started/quickstart/quickstart.md
- Sentence Transformers retrieve & rerank
  - https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- Sentence Transformers package reference
  - https://www.sbert.net/docs/package_reference/sentence_transformer/index.html
- `ref/2506.12116v3.pdf`
  - *Unsupervised Document and Template Clustering using Multimodal Embeddings*

如果要看更系统的论文整理和阅读顺序，请读 `docs/related-work.md`。

## 与现有文档的关系

- `docs/clustering.md`
  - 说明当前实现和当前问题
- `docs/clustering-v2.md`
  - 说明下一阶段重构方案
- `docs/design-data-model.md`
  - 定义 V2 核心对象
- `docs/design-pipeline-contract.md`
  - 定义 V2 分层流水线契约
- `docs/design-evaluation-protocol.md`
  - 定义 V2 应如何被评估
- `docs/refactor-plan-v2.md`
  - 定义 V2 的实施顺序和模块重组计划
- `docs/validation.md`
  - 定义验证集和评估要求
- `docs/experiments.md`
  - 保留当前实验流程；V2 稳定后再重写
