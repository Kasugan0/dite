# 配置

本文档描述当前代码里实际使用的配置模型。

## 配置来源

DITE 始终从下面这个路径加载配置：

`~/.config/dite/config.yaml`

如果文件不存在，DITE 会自动创建默认配置文件。

当前不支持这些配置方式：

- 工作区本地配置文件
- 按命令临时覆盖配置
- CLI `--config` 选项

## 环境变量展开

YAML 中的字符串值支持环境变量展开。

当前实际可靠支持的形式是：

- `${VAR}`

代码里对 `$VAR` 也有过意图上的兼容说明，但当前真正会被展开的是 `${VAR}`。

## 顶层配置段

当前顶层配置包含这些部分：

- `api`
- `models`
- `request_profiles`
- `clustering`
- `processing`
- `cache`
- `formats`
- `i18n`

## `api`

字段包括：

- `base_url`
- `api_key`
- `connect_timeout_sec`
- `read_timeout_sec`
- `write_timeout_sec`
- `pool_timeout_sec`
- `max_retries`
- `max_connections`
- `max_keepalive_connections`
- `keepalive_expiry_sec`

当前校验规则：

- `api.max_keepalive_connections` 必须小于等于 `api.max_connections`

## `models`

字段包括：

- `embedding`
- `vlm`
- `llm`

当前默认值指向 Qwen3 系列模型，但代码本质上把它们当作可配置的 OpenAI-compatible 模型标识符处理。

## `request_profiles`

当前实际实现的请求配置只有：

- `cluster_naming`

`cluster_naming` 下的字段包括：

- `max_tokens`
- `reasoning_mode`
- `thinking_budget`

`reasoning_mode` 当前可接受的值：

- `default`
- `off`
- `on`

当前代码还会把 YAML 中的布尔值做归一化：

- `false` 变成 `off`
- `true` 变成 `on`

## `clustering`

字段包括：

- `min_cluster_size`
- `min_samples`
- `cluster_selection_epsilon`
- `cluster_selection_method`
- `knn_k`
- `knn_distance_threshold`

这些参数控制 HDBSCAN 聚类，以及可选的 k-NN 噪音修复。

当前默认值是：

- `min_cluster_size=2`
- `min_samples=1`
- `cluster_selection_epsilon=0.0`
- `cluster_selection_method="eom"`
- `knn_k=3`
- `knn_distance_threshold=None`

这些默认值目前对“小簇成立”相对宽松，但 `eom` 仍然比 `leaf` 更保守。参数含义、已知问题和计划中的收紧方向，见 `docs/clustering.md`。

## `processing`

字段包括：

- `text_truncate_limit`
- `vlm_fallback_threshold`
- `docling_pdf_timeout_sec`
- `docling_device`
- `extract_workers`
- `docling_pdf_workers`
- `cluster_naming_workers`
- `vlm_api_workers`
- `vlm_pages_per_document`

当前行为要点：

- `text_truncate_limit` 是最终内容截断上限，供 `extract_content` 使用。
- `vlm_fallback_threshold` 是判断 PDF 提取结果是否过弱的有效长度阈值。
- `extract_workers` 限制规范样本文件的并行提取数。
- `docling_pdf_workers` 限制 Docling PDF 子进程的并发量。
- `cluster_naming_workers` 限制簇命名请求并发量。
- `vlm_api_workers` 限制全局 VLM 请求并发量。
- `vlm_pages_per_document` 限制单个 PDF 内部的页级并发量，而不是总采样页数。

需要特别区分的一点：

- PDF 的 VLM 回退最多只采样前 10 页。
- `vlm_pages_per_document` 只控制这批采样页的并发度，不控制采样深度。

## `cache`

字段包括：

- `enabled`
- `directory`
- `max_size_gb`

当前行为要点：

- 缓存存放在配置目录下的 SQLite 数据库中。
- 缓存大小限制已经实现。
- 当数据库文件超过配置上限时，会删除较旧条目并执行 `VACUUM`。

## `formats`

字段包括：

- `documents`
- `images`

当前规范化规则：

- 文档扩展名会被统一转成小写。
- `.md` 和 `.markdown` 会保持同步，保证兼容性。

## `i18n`

字段包括：

- `locale`

当前支持的值：

- `zh-CN`
- `en`
- `en-US`
