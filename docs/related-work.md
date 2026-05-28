# 相关工作

本文档整理 DITE 当前聚类重构最值得参考的论文和工程资料。

它不是完整综述，也不是按发表年份堆参考文献。它只保留对下面这些决策真正有帮助的材料：

- HDBSCAN 还能不能继续当主聚类器
- embedding-only 为什么会碎簇
- 文档版式、视觉和文本是否应该一起建模
- LLM 应该放在主聚类、边界判定还是簇表示层
- 文件名、实体、路径等元数据应该怎么作为 side information 使用
- 新评估体系应该参考什么

如果你只想看未来方案，读 `docs/clustering-v2.md`。

## 使用方式

阅读这些资料时，不要只问“这篇论文效果好不好”，而要问：

1. 它解决的是哪一层问题
2. 它的输入假设和 DITE 是否一致
3. 它需要的成本、标签和基础设施我们有没有
4. 它对 DITE 最小可落地的启发是什么

## A. 聚类基础与评估

### 1. HDBSCAN 主论文

- Campello, Moulavi, Zimek, Sander
- *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection*
- 2015
- https://researchonline.jcu.edu.au/47065/

为什么重要：

- 这是当前 DITE 主聚类器的理论来源。
- 不读它，就不知道 HDBSCAN 真正擅长什么、又不擅长什么。

对 DITE 的启发：

- HDBSCAN 适合处理密度不均、形状不规则的簇。
- 但它不是“任何高维 embedding 都能直接拿来跑”的通用魔法。
- 如果 DITE 继续保留 HDBSCAN，就必须明确它是在什么空间上跑、承担哪一层任务。

### 2. HDBSCAN FAQ

- 官方 FAQ
- https://hdbscan.readthedocs.io/en/latest/faq.html

为什么重要：

- 这里有最直接的工程边界说明，尤其是高维空间退化问题。

对 DITE 的启发：

- 官方明确提醒高维数据上密度聚类会变难。
- 这直接支持 DITE V2 中“不要再让原始高维 embedding 单独承担全部聚类决策”的判断。

### 3. DBCV

- Moulavi, Jaskowiak, Campello, Zimek, Sander
- *Density-Based Clustering Validation*
- 2014
- https://epubs.siam.org/doi/10.1137/1.9781611973440.96

为什么重要：

- 当前 DITE 的评估指标太偏报告统计，不够像真正的聚类验证。

对 DITE 的启发：

- 如果系统继续保留密度聚类路线，就该补充密度聚类专用评估指标。
- 不能只看总簇数、噪音数和小簇合并数。

### 4. PLM 表示为何不天然适合聚类

- Zhang et al.
- *Topic Discovery via Latent Space Clustering of Pretrained Language Model Representations*
- 2022
- https://arxiv.org/abs/2202.04582

为什么重要：

- 它直接对应 DITE 当前“拿现成 embedding 就直接聚类”的假设。

对 DITE 的启发：

- 预训练语言模型的表示不一定天然 cluster-friendly。
- 需要考虑降维、重表示、簇级表示或图结构补强。

## B. 文本表示与主题表示

### 5. SBERT

- Reimers, Gurevych
- *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- 2019
- https://arxiv.org/abs/1908.10084

为什么重要：

- 这是今天大多数文本 embedding 聚类、检索、相似度系统的共同起点之一。

对 DITE 的启发：

- bi-encoder 适合做召回、近邻和候选边生成。
- 它适合作为“第一阶段粗筛”，不适合作为唯一最终裁决。

### 6. BERTopic

- Grootendorst
- *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*
- 2022
- https://arxiv.org/abs/2203.05794

为什么重要：

- 这不是 DITE 要直接照搬的工具，而是一个很好的结构参考。

对 DITE 的启发：

- 它把 `embedding -> dimensionality reduction -> clustering -> representation` 明确拆开了。
- 这与 DITE V2 的方向高度一致。
- 最重要的不是它的 topic name，而是它承认“聚类”和“主题表示”是两层不同问题。

### 7. BERTopic 官方算法与 LLM 表示资料

- Algorithm
- https://maartengr.github.io/BERTopic/algorithm/algorithm.html
- LLM representation docs
- https://github.com/maartengr/bertopic/blob/master/docs/getting_started/representation/llm.md

为什么重要：

- 工程上比论文更直接，特别适合决定 LLM 应该放在哪一层。

对 DITE 的启发：

- LLM 更适合作为 `representation model`，也就是簇表示器和标签生成器。
- 如果要让 LLM 参与更多，也应该优先放在小簇重审和边界判定，而不是替代第一阶段聚类。

## C. 文档多模态表示

### 8. LayoutLMv3

- Huang et al.
- *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking*
- 2022
- https://arxiv.org/abs/2204.08387

为什么重要：

- 这是文档理解领域非常强的文本+布局+视觉基线。

对 DITE 的启发：

- 文档聚簇不一定只能看正文文本。
- 对扫描 PDF、课件、表单、模板化材料，布局本身就是语义信号。

### 9. DocFormer

- Appalaraju et al.
- *DocFormer: End-to-End Transformer for Document Understanding*
- 2021
- https://arxiv.org/abs/2106.11539

为什么重要：

- 代表另一条文档多模态建模路线。

对 DITE 的启发：

- 如果未来要做布局特征或模板提示，应该把它作为独立视图，而不是硬塞进正文 embedding。

### 10. Donut

- Kim et al.
- *OCR-free Document Understanding Transformer*
- 2021
- https://arxiv.org/abs/2111.15664

为什么重要：

- 它提醒我们：OCR 不是唯一入口，尤其是弱提取和扫描文档。

对 DITE 的启发：

- 对弱 OCR 文档，内容提取质量本身就是聚类质量的一部分。
- “提取失败后退化成文件名 embedding”不是长期方案。

### 11. DITE 仓库内已收录的直接相关论文

- Sampaio, Maxcici
- *Unsupervised Document and Template Clustering using Multimodal Embeddings*
- 2025
- https://arxiv.org/abs/2506.12116
- 本地副本：`ref/2506.12116v3.pdf`

为什么重要：

- 这是目前最贴近 DITE 任务的参考之一。
- 它同时讨论了 document-level clustering 和 template-level clustering。

对 DITE 的启发：

- 多模态表示对于模板发现非常重要。
- 文本信号在分布偏移和 OCR 噪声下仍然关键。
- “一种向量打天下”并不现实，模态之间有 trade-off。

## D. 元数据、实体与 side information

### 12. Named Entity 特征做文档聚类

- Li et al.
- *Semantic Document Clustering on Named Entity Features*
- 2018
- https://arxiv.org/abs/1807.07777

为什么重要：

- 它直接回答“实体是否能成为独立聚类视图”。

对 DITE 的启发：

- 标题、实体、人名、机构名、术语可以独立建模。
- 这支持 V2 中把 `keywords / entities / domain` 从正文里拆出来单独使用。

### 13. NER + LLM embedding + 图聚类

- *Graph-Convolutional Networks: Named Entity Recognition and Large Language Model Embedding in Document Clustering*
- 2024
- https://arxiv.org/abs/2412.14867

为什么重要：

- 这篇很新，也很像 DITE 想走的组合路线。

对 DITE 的启发：

- 实体图和正文 embedding 可以是互补视图。
- 如果 V2 后期要走图聚类路线，这篇值得回来看。

### 14. Pairwise constraints 与 constrained clustering

- Śmieja, Struski, Figueiredo
- *A Classification-Based Approach to Semi-Supervised Clustering with Pairwise Constraints*
- 2020
- https://arxiv.org/abs/2001.06720

补充综述：

- *Semi-Supervised Constrained Clustering: An In-Depth Overview, Ranked Taxonomy and Future Research Directions*
- 2023
- https://arxiv.org/abs/2303.00522

为什么重要：

- DITE 已经在 `validation` 文档里引入了 `must_link` / `must_not_link` 思路。
- 这条线能给未来评估和弱监督晋升提供方法论。

对 DITE 的启发：

- 代表性验证集不应该只用于人工 eyeballing。
- 后续可以把 must-link / must-not-link 变成真实的评估约束，甚至用于主动改进边界判定层。

## E. LLM 参与主题建模与聚类

### 15. TopicGPT

- Wang et al.
- *TopicGPT: A Prompt-based Topic Modeling Framework*
- 2023 / NAACL 2024
- https://arxiv.org/abs/2311.01449

为什么重要：

- 它展示了 LLM 怎样提升 topic representation 和可解释性。

对 DITE 的启发：

- LLM 很适合做“簇表示器”和“主题解释器”。
- 但不意味着它适合全量替代聚类器。

### 16. PromptTopic

- Törnberg
- *Prompting Large Language Models for Topic Modeling*
- 2023
- https://arxiv.org/abs/2312.09693

为什么重要：

- 这是 “LLM 直接做 topic modeling” 的代表性路线。

对 DITE 的启发：

- 它值得看，但更适合作为对照组和警惕材料。
- 对 DITE 这种本地文件工具来说，全量 prompt 驱动的成本和稳定性压力太大。

### 17. LITA

- *LITA: An Efficient LLM-assisted Iterative Topic Augmentation Framework*
- 2024
- https://arxiv.org/abs/2412.12459

为什么重要：

- 这条路线最像 DITE 该学的：不是让 LLM 全权处理，而是只处理少量高价值、模糊、边界不清的样本。

对 DITE 的启发：

- 很适合参考到 V2 的“边界判定层”。
- 也支持“便宜模型召回，贵模型保守裁决”的分层设计。

### 18. LLM 直接参与聚类的能力边界

- Zhang et al.
- *Large Language Models Enable Few-Shot Clustering*
- 2024
- https://aclweb.org/anthology/2024.tacl-1.18.pdf

为什么重要：

- 这条线回答的是：“LLM 能不能真的直接做聚类？”

对 DITE 的启发：

- 能，但代价和约束都明显更高。
- 更适合 few-shot 或带示例的场景，不适合作为 DITE 的默认全量主路径。

## F. 需要带着怀疑去读的材料

### 19. 模态混合的风险

- *When Text and Images Don't Mix: Bias-Capturing in Multimodal Anomaly Detection*
- 2024
- https://arxiv.org/abs/2407.17083

为什么重要：

- 虽然任务不是文档聚类，但它提醒了一个很现实的问题：多模态信号并不天然协同，可能存在偏置和互相污染。

对 DITE 的启发：

- 文件名、正文、视觉、版式不应无脑拼接成同一表示。
- 更稳妥的路线是多视图、显式融合、可观测加权。

## DITE 当前最该读什么

如果目标是推进 `docs/clustering-v2.md`，建议阅读顺序如下：

### 第一组：必须先读

1. HDBSCAN 主论文
2. HDBSCAN FAQ
3. DBCV
4. SBERT
5. BERTopic

为什么：

- 这组决定你对“当前 embedding-only 主路径到底哪里错了”的理解是否扎实。

### 第二组：决定 V2 架构边界

6. LayoutLMv3
7. `Unsupervised Document and Template Clustering using Multimodal Embeddings`
8. Topic Discovery via Latent Space Clustering of PLM Representations

为什么：

- 这组决定你是否要把布局和视觉作为正式信号，以及为什么不能再直接拿高维 embedding 硬聚。

### 第三组：决定 LLM 放哪一层

9. BERTopic LLM representation docs
10. TopicGPT
11. LITA
12. Large Language Models Enable Few-Shot Clustering

为什么：

- 这组决定 LLM 应该做表示、边界判定，还是全量分簇。

### 第四组：决定元数据和弱监督怎么进来

13. Semantic Document Clustering on Named Entity Features
14. GCN + NER + LLM embedding in document clustering
15. Pairwise constraints 两篇

为什么：

- 这组决定 V2 后续如何把 `entities / keywords / must-link / must-not-link` 变成正式能力。

## 当前建议

如果只给 DITE 当前阶段挑 8 篇最值得读的，建议是：

1. HDBSCAN 主论文
2. DBCV
3. SBERT
4. BERTopic
5. LayoutLMv3
6. `Unsupervised Document and Template Clustering using Multimodal Embeddings`
7. LITA
8. Semantic Document Clustering on Named Entity Features

这 8 篇足够支撑 V2 的第一轮实现决策。
