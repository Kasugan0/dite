# 聚类 V2 大重构实施计划

本文档描述 DITE 聚类 V2 的实施计划。

状态说明：

- 本文档的大部分章节记录的是这次大重构启动前后的问题分析、迁移目标和阶段规划。
- 当前代码库的实际目录结构已经落地为：
  - `src/dite/app`
  - `src/dite/doc`
  - `src/dite/io`
  - `src/dite/flow`
  - `src/dite/cluster`
  - `src/dite/report`
  - `src/dite/util`
  - `src/dite/cache`
- 因此，文中出现的 `core / extractors / features / adjudication / reporting / clustering / utils` 等路径，应视为“改造前的问题定位和目标草图”，不是当前目录事实。

它不是当前实现说明，也不是抽象愿景文档。它的目标，是把这次大重构真正分解成可以执行的阶段、模块调整和落地顺序。

如果你只想看为什么要重构，读 `docs/clustering-v2.md`。

如果你只想看对象和边界设计，读：

- `docs/design-data-model.md`
- `docs/design-pipeline-contract.md`
- `docs/design-evaluation-protocol.md`

## 1. 这次重构到底在改什么

这次重构不是“再调一轮 HDBSCAN 参数”，而是三件事一起做：

1. 重建数据结构
2. 重建流水线阶段边界
3. 重建代码组织结构

当前系统最大的问题，不是某个函数写得丑，而是下面这件事：

- 一个文件几乎只被表示成“最终文本 + embedding”
- 一个主聚类器被迫承担主题理解、近重复识别、噪音修补、边界判定等多种职责
- 一些已经过胖的文件继续承载越来越多逻辑

如果只改算法，不改结构，系统会继续膨胀。

如果只拆文件，不改数据对象，系统只会从“大块垃圾”变成“分散垃圾”。

## 2. 当前结构问题

### 2.1 已经过胖的文件

这些文件必须拆：

- `src/dite/core/clusterer.py`
  - 当前同时负责：
    - 聚类
    - 噪音修复
    - 小簇合并
    - 簇命名
    - 命名后合并
- `src/dite/core/pipeline.py`
  - 当前同时负责：
    - 顶层编排
    - 提取工作项执行
    - 哈希折叠
    - 缓存回填
    - 向量化
    - 聚类接线
- `src/dite/cli.py`
  - 当前除了 CLI 入口，还负责：
    - 报告构建
    - 输出格式化
    - 提取详情表
    - 多种标签和说明文本拼装
- `src/dite/extractors/router.py`
  - 当前同时负责：
    - 类型识别
    - 提取器路由
    - PDF/VLM 选择
    - 最终内容决策

### 2.2 已经过碎的模块

这些地方需要收拢或重新分边界：

- `src/dite/core/extractor.py`
  - 体量太小，没有明显独立价值
- 部分过薄的 `extractors/*`
  - 如果未来仍然只是轻转发器，应考虑合并回更强边界模块

### 2.3 结构层面的问题

当前真正的问题不是“文件多”，而是职责边界不稳定：

- 聚类逻辑和簇表示耦合
- 编排逻辑和特征构建耦合
- 提取路由和内容决策耦合
- CLI 和报告展示耦合

这会直接拖慢 V2 的每一步实现。

## 3. 重构总原则

### 3.1 先按职责拆，不按名词拆

不要因为出现了 `LLM`、`metadata`、`features` 这些新概念，就机械增加目录数量。

应该优先围绕这些职责拆：

- 输入净化
- 文件特征构建
- 强连接召回
- 主题聚类
- 边界裁决
- 簇表示
- 报告输出

### 3.2 先建新对象，再迁旧逻辑

顺序必须是：

1. 定义对象
2. 定义阶段契约
3. 迁移逻辑
4. 删除旧入口

不要一边改行为一边随手搬函数。

### 3.3 保持一段时间的双轨期

虽然这次允许破坏用户空间，但工程上仍然不该一刀切。

建议在中期保留：

- 旧主线还能运行
- 新 V2 主线逐步接管

直到评估协议能够证明 V2 更好。

## 4. 目标代码组织

### 4.1 `src/dite/core/`

目标定位：

- 只保留真正的核心编排与少量稳定基础能力

建议保留：

- `pipeline.py`
- `scanner.py`
- `organizer.py`

建议迁出：

- 特征构建
- 聚类策略
- 簇表示
- 裁决逻辑

### 4.2 新增 `src/dite/features/`

职责：

- 文件级多视图特征构建

建议模块：

- `models.py`
- `builder.py`
- `content.py`
- `metadata.py`
- `layout.py`
- `analysis.py`

目标：

- 让 `DocumentFeatures` 成为正式一等公民

### 4.3 新增 `src/dite/clustering/`

职责：

- 候选边、主题聚类、中间簇、簇表示

建议模块：

- `models.py`
- `candidate_edges.py`
- `candidate_components.py`
- `topic_clustering.py`
- `noise.py`
- `merge.py`
- `representation.py`
- `naming.py`
- `service.py`

目标：

- 彻底拆散今天的 `core/clusterer.py`

### 4.4 新增 `src/dite/adjudication/`

职责：

- 模糊案例裁决

建议模块：

- `models.py`
- `triggers.py`
- `rules.py`
- `llm.py`
- `service.py`

目标：

- 给 LLM 和其他昂贵判定器一个明确位置

### 4.5 `src/dite/extractors/`

目标定位：

- 保留提取层，但把“提取”和“内容决策”拆开

建议保留：

- `router.py`
- `pdf_policy.py`
- `pdf_vlm.py`
- `pdf_finalize.py`
- `docling.py`
- `text.py`
- `markitdown.py`
- `vlm.py`

建议进一步拆分：

- `routing.py`
  - 类型识别与提取器选择
- `resolution.py`
  - 最终内容选择与 VLM 回退裁决

### 4.6 新增 `src/dite/reporting/`

职责：

- JSON 报告、终端报告、提取详情表

建议模块：

- `models.py`
- `json_report.py`
- `terminal_report.py`

目标：

- 从 `cli.py` 迁出展示与报告拼装逻辑

## 5. 分阶段实施计划

## Phase 0：设计冻结

目标：

- 在动代码前把 V2 的基础设计冻结

已完成：

- `docs/clustering-v2.md`
- `docs/design-data-model.md`
- `docs/design-pipeline-contract.md`
- `docs/design-evaluation-protocol.md`
- `docs/literature-insights.md`

仍需补：

- 新版 `manifest` 规范
- 实验输出 schema

完成标准：

- 对象、阶段、评估三件事有明确文档约束

## Phase 1：对象先行

目标：

- 先把新对象引入代码库

建议新增：

- `features/models.py`
- `clustering/models.py`
- `adjudication/models.py`

要求：

- 只定义对象与最小构造逻辑
- 暂不引入复杂行为

完成标准：

- 后续迁移逻辑时，不再需要继续临时造匿名 dict

## Phase 2：结构止损

目标：

- 先瘦身最胖的文件，但尽量不改行为

建议顺序：

1. 从 `cli.py` 迁出 `reporting`
2. 从 `clusterer.py` 迁出 `naming` 和 `representation`
3. 从 `pipeline.py` 迁出 hash / vectorize 辅助逻辑
4. 从 `router.py` 拆出 routing 和 resolution

完成标准：

- 巨型文件开始回落到更合理的职责边界
- 旧行为基本保持

## Phase 3：文件特征层落地

目标：

- 正式引入 `DocumentFeatures`

实施内容：

- 构造文件名、路径、扩展名特征
- 引入摘要、关键词、主题、领域特征
- 引入 `quality_flags`

关键要求：

- 文件名不再隐式混入正文语义
- 弱提取和短文本显式化

完成标准：

- V2 不再只有“原文 + embedding”

## Phase 4：强连接召回层

目标：

- 把近重复和强相似关系前移

实施内容：

- `CandidateEdge`
- `CandidateComponent`
- near-duplicate 检测
- 强相似内容边
- 文件名/标题/实体强一致边

完成标准：

- 一部分“几乎肯定在一起”的关系，不再依赖主题聚类器事后猜出

## Phase 5：主题聚类层替换

目标：

- 逐步替换旧 `cluster_documents()` 主路径

实施内容：

- 引入 `ClusterDraft`
- 正式实验：
  - `PCA/UMAP -> HDBSCAN`
  - 图社区发现

要求：

- 主题聚类只负责结构草案
- 不负责包办所有边界问题

完成标准：

- 新主路径能独立产出可诊断的中间簇结构

## Phase 6：边界裁决层

目标：

- 让模糊案例有正式去处

实施内容：

- 设计 ambiguity trigger
- 引入规则裁决
- 引入 LLM 裁决
- 引入 `AdjudicationDecision`

要求：

- 绝不允许全量文件送入 LLM
- 所有裁决必须结构化、可追溯

完成标准：

- 小簇合并和边界文件归属不再继续依赖临时补丁

## Phase 7：评估协议与实验工具升级

目标：

- 让 V2 有资格晋升默认主线

实施内容：

- 新版 manifest
- 新版实验输出 schema
- 新版评估脚本
- 纳入：
  - `must_link_recall`
  - `must_not_link_violations`
  - `cluster_id_fragmentation`
  - `filename_bias_rate`
  - `density_validation_score`
  - `llm_merge_accept_rate`

完成标准：

- 方案优劣不再靠肉眼和簇数猜

## 6. 哪些文件必须拆，哪些不该乱拆

### 必拆

- `src/dite/core/clusterer.py`
- `src/dite/core/pipeline.py`
- `src/dite/cli.py`
- `src/dite/extractors/router.py`

### 可以基本保持

- `src/dite/core/scanner.py`
- `src/dite/core/organizer.py`

### 暂不急着拆

- `src/dite/core/embedder.py`

原因：

- 它虽然不小，但当前职责相对单一
- 更适合等 V2 特征层稳定后再决定是否迁入 `features/`

### 建议收拢

- `src/dite/core/extractor.py`
- 仅作薄包装的过小模块

## 7. 风险控制

### 风险 1：文件拆了，但行为边界没真变

表现：

- 只是把一个大文件拆成多个小文件
- 逻辑耦合仍然原样保留

避免方式：

- 先按职责建对象和契约，再迁行为

### 风险 2：LLM 重新变成万能补锅器

表现：

- 新系统虽然分层，但模糊案例全部滚进 LLM

避免方式：

- 明确 ambiguity trigger
- 限制预算
- 强制结构化裁决

### 风险 3：拆分过细，阅读成本上升

表现：

- 目录漂亮了，但跳转更多、语义更散

避免方式：

- 只在职责边界明显时拆模块
- 不为“看起来分层”而分层

## 8. 建议提交顺序

建议按原子提交推进：

1. `docs(v2): define refactor plan and core design docs`
2. `refactor(reporting): extract report builders from cli`
3. `refactor(clustering): split naming and representation from clusterer`
4. `refactor(extraction): separate routing and resolution concerns`
5. `feat(features): introduce DocumentFeatures and quality flags`
6. `feat(clustering): add candidate edge and component generation`
7. `feat(clustering): introduce topic clustering draft pipeline`
8. `feat(adjudication): add boundary adjudication layer`
9. `test(validation): upgrade manifests and evaluation outputs`

## 9. 当前建议

下一步不要直接开始大写代码。

最合理的顺序是：

1. 先补新版 `manifest` 规范
2. 再补实验输出 schema
3. 然后进入 Phase 1 的对象落地

如果跳过这两步，后面的实现很容易重新变成“先写代码，再想怎么验证”的旧路径。
