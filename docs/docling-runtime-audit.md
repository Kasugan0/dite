# Docling 与提取链审计报告

本文档记录本轮对 `docs/test/rep` 扫描异常、Docling 日志泛滥、提取失败后续处理路径的实现审计结果。

说明：

- 本文档引用的源码路径对应审计发生时的仓库结构。
- 当前仓库已经完成目录重构，相关逻辑现在主要位于 `src/dite/app`、`src/dite/io`、`src/dite/flow`、`src/dite/doc` 和 `src/dite/util` 下。
- 因此，这里的旧路径引用应理解为历史审计定位，不是当前目录事实。

这不是修复方案文档，也不是参数调优文档。它只回答三件事：

- 现在到底哪里在出问题
- 问题是怎么沿着代码路径发生的
- 这些问题为什么会污染实验结论

## 范围

本轮审计只覆盖当前仓库中的下列路径和现象：

- `uv run dite scan docs/test/rep ...` 期间出现的大量 `Stage preprocess failed for run 1: ...`
- `rep` 扫描中出现的提取失败对后续 embedding 和聚类的影响
- CLI 报告中的 `extraction_failed` 统计口径
- DITE 当前对第三方日志，尤其是 Docling 日志的控制方式

本轮审计不包含：

- 任何代码修改
- 默认聚类参数晋升结论
- `0x92` 解码错误的最终根因定案

## 执行摘要

当前问题不是“聚类参数还需要微调”这么简单，而是运行时链路本身会污染实验。

结论按严重度排序如下：

1. 现有 Docling 日志抑制逻辑压不住 `ERROR` 级别日志，因此 Docling 预处理错误会直接刷到终端。
2. PDF 提取走的是子进程路径，子进程没有重新建立 DITE 的日志配置，因此父进程的日志设置不能保证对子进程生效。
3. 提取失败的文件不会被排除出 embedding，而是会退化成“只用文件名做 embedding”，这会直接污染聚类结果。
4. CLI 中的 `extraction_failed` 不是基于真实提取状态，而是基于文本长度的启发式判断，因此只能当作近似指标，不能当作真值。
5. `valid` 和 `adv` 目前仍可用于 smoke 和边界行为观察；`rep` 在当前状态下不能直接作为默认参数晋升依据。

## 观测到的现象

### 1. 当前实验输出

本轮已观察到的真实运行现象：

- `valid`
  - 找到 `60` 个文件
  - 聚类完成，发现 `21` 个簇，`8` 个未分类
  - 从生成的 JSON 报告看，没有文件被标记为 `extraction_failed`
- `adv`
  - 找到 `41` 个文件
  - 聚类完成，发现 `13` 个簇，`3` 个未分类
  - 从生成的 JSON 报告看，没有文件被标记为 `extraction_failed`
- `rep`
  - 运行过程中出现大量 `Stage preprocess failed for run 1: ...`
  - 终端输出夹带长段 PDF 对象内容，例如 `"/Type": "/Page"`、`"/Resources"`、`"/Font"` 等结构
  - 这说明问题已经不是“少数文件聚错”，而是提取阶段本身在持续报错

### 2. 日志表现

用户可见的直接症状是：

- CLI 正常输出被大量第三方日志淹没
- `tee` 保存的 `.log` 文件噪声很大
- 很难直接从 CLI 输出判断到底是“程序失败”还是“底层解析器失败”

这会直接降低实验可读性，也会拖慢失败文件定位。

## 关键发现

### 发现 1：Docling 的 `ERROR` 日志不会被当前抑制逻辑屏蔽

这是当前最确定、最直接的问题。

#### 证据

Docling 在 PDF 预处理阶段失败时，会直接记录 `error` 级别日志：

- `.venv/lib/python3.12/site-packages/docling/pipeline/standard_pdf_pipeline.py:390`

对应代码行为是：

- `Stage preprocess failed for run %d: %s`

DITE 当前对第三方日志的控制在这里：

- `src/dite/utils/logging.py:176-207`

这里会：

- 获取若干 `docling.*` logger
- `setLevel(level)`
- `propagate = False`
- `handlers.clear()`

另外，Docling 提取器内部还有一层所谓“抑制 warning”的逻辑：

- `src/dite/extractors/docling.py:104-119`

这里实际做的是：

- 把 `docling.*` logger 级别设为 `logging.ERROR`
- `propagate = False`

#### 为什么这会出问题

这套实现的方向本身就有缺陷。

如果一个 logger：

- 允许 `ERROR`
- 没有 handler
- 不向上层传播

那么 Python logging 的 `lastResort` 机制仍可能把 `ERROR` 输出到标准错误。

这和“静音”正好相反。它会让你把 `WARNING` 压住，但把最吵、最长的 `ERROR` 留下来。

#### 影响

- 终端出现大量 Docling 原始错误
- `tee` 出来的日志可读性显著变差
- 用户容易误判成“DITE 主流程崩了”
- 真实失败样本难以定位

### 发现 2：PDF 提取走子进程，父进程日志配置不能保证对子进程生效

这是导致日志压制进一步失效的第二层原因。

#### 证据

当前 PDF 主提取路径是：

- `src/dite/core/pipeline.py:200-214`

这里对 PDF 会调用：

- `extract_docling_pdf_in_subprocess(...)`

该函数定义在：

- `src/dite/extractors/docling.py:143-196`

它明确使用：

- `multiprocessing.get_context("spawn")`

子进程入口是：

- `src/dite/extractors/docling.py:122-139`

这个子进程入口里只做了：

- `set_locale(...)`
- 构造 `DoclingExtractor(...)`
- `extractor.extract(...)`

这里没有调用：

- `setup_logging(...)`

#### 为什么这很重要

父进程里做过的日志设置，并不天然等于子进程也继承了同样的状态。

特别是在 `spawn` 模式下，子进程更接近“重新启动一个 Python 解释器，再从头执行入口代码”。

这意味着：

- 父进程的 DITE logger 已经准备好
- 但子进程里的 Docling logger 可能仍然以自己的默认方式工作

#### 影响

- 即使父进程试图抑制第三方日志，子进程里仍可能把 Docling 错误直接打到终端
- 这解释了为什么 PDF 相关错误特别容易刷屏

### 发现 3：提取失败文件仍会进入 embedding，并退化成“只用文件名”

这是最危险的实验污染点。

#### 证据

embedding 输入构造逻辑在：

- `src/dite/core/embedder.py:51-71`

关键行为：

- 如果提取内容长度很短，且当前是 `with_filename` 模式
- 则直接返回 `File name: <文件名>`

向量化流程在：

- `src/dite/core/pipeline.py:748-823`

这里不会先过滤提取失败文件，而是把 `contents` 原样送进 `get_embeddings(...)`。

#### 为什么这很危险

一旦某个文件：

- 提取失败
- 或提取结果几乎为空

它仍然会继续参与 embedding，只是输入从“文档内容”变成“文件名”。

这会产生两个直接副作用：

1. 文件名相似的失败文件可能被强行拉近
2. 内容其实应该相关、但文件名不相似的失败文件可能被拉远

在 `rep` 这种真实世界混合集上，这种偏差会直接污染参数比较结果。

#### 影响

- 当前 `rep` 的聚类结果即使跑完，也不能简单视为“内容语义聚类结果”
- 它可能混入一部分“文件名聚类”

### 发现 4：CLI 的 `extraction_failed` 统计口径只是启发式

当前 CLI 报告里的“提取失败”不是严格真值。

#### 证据

CLI 报告构造逻辑在：

- `src/dite/cli.py:367-377`

当前判断是：

- `len(content.strip()) < 10` 就算 `extraction_failed`

#### 这意味着什么

这不是“提取器返回了失败状态”的直接映射，而只是一个近似规则。

它会带来两类误差：

- 假阳性：
  - 真正合法但很短的文档会被标成失败
- 假阴性：
  - 提取出一段长度超过 10、但明显是垃圾的内容，不会被标成失败

#### 影响

- `num_extraction_failed` 不能当成严格统计
- CLI 输出里 `(提取失败)` 标记只能作为提示，不能直接拿来做实验真值

### 发现 5：`rep` 当前不能直接作为默认参数晋升依据

这是综合前四条之后的结论。

#### 依据

`valid` 和 `adv` 目前至少满足一条底线：

- 当前生成的 JSON 报告中，没有文件被标成 `extraction_failed`

但 `rep` 已经出现：

- 大量 Docling 预处理错误日志
- 提取链稳定性不足
- 可能存在失败文件继续参与 embedding 的情况

#### 结论

在这些问题未隔离之前：

- `rep` 不能直接作为正式 A/B 的裁判
- `rep` 也不能直接拿来晋升默认参数

否则比较的就不只是“参数优劣”，还混入了“提取链是否稳定”和“文件名 fallback 是否介入”。

## 调用链梳理

当前与 PDF 提取和日志问题直接相关的主路径如下：

1. CLI 执行 `scan`
2. 进入 `PipelineService._extract_primary_result(...)`
3. PDF 文件转入 `PipelineService._extract_docling_pdf_primary_result(...)`
4. 调用 `extract_docling_pdf_in_subprocess(...)`
5. `spawn` 一个子进程执行 `_docling_pdf_extract_child(...)`
6. 子进程内构造 `DoclingExtractor(...)`
7. 调用 `DoclingExtractor.extract(...)`
8. 内部执行 `converter.convert(...)`
9. Docling PDF 预处理阶段报错
10. Docling 自己输出 `Stage preprocess failed for run 1: ...`

如果提取最终返回空内容或极短内容，后续路径是：

1. `PipelineService._vectorize(...)`
2. `get_embeddings(...)`
3. `_build_embedding_input(...)`
4. 回退到 `File name: <文件名>`
5. 该文件继续参与 embedding 和聚类

## 目前能确定的事

以下判断已经足够确定：

- `Stage preprocess failed for run 1: ...` 这类日志来自 Docling，不是 DITE 自己拼出来的业务日志
- 当前日志抑制实现不能有效压住 Docling 的 `ERROR`
- 子进程路径会削弱父进程日志配置的作用
- 提取失败文件仍可能继续进入 embedding
- CLI 的失败统计不是严格真值

## 目前还不能完全确定的事

以下问题本轮还没有完全钉死：

### 1. `0x92` 解码错误的最终来源

本轮只确认了两件事：

- 这条错误不在 DITE 自己的源码报错模板中
- 至少另一类高频错误 `page-dimensions` 已经明确来自 Docling

但 `'utf-8' codec can't decode byte 0x92 ...` 这一条，还没有最终定位到：

- 是 Docling 某一层触发
- 还是其他上游组件触发
- 还是某个具体文件的特殊内容引起

### 2. `rep` 中到底哪些文件触发了哪一类失败

当前只看到现象，不足以完成逐文件归因。

还需要结合：

- `scan-rep.log`
- `pdf-check-rep.log`

去做逐文件映射。

## 对实验结论的影响

本轮审计意味着：

- 现在可以继续用 `valid` 和 `adv` 观察聚类边界行为
- 但不能用当前 `rep` 结果直接下默认参数结论

原因很简单：

- 当前 `rep` 结果混入了提取阶段的不稳定性
- 而且提取失败文件仍可能继续进入聚类

这会把“聚类参数问题”和“提取链问题”搅在一起。

如果不先拆开，后面的 A/B 实验结论会失真。

## 建议的后续审计方向

本文件不包含修复方案，但可以明确下一步应该继续审计什么：

1. 逐文件定位 `rep` 中的失败样本
2. 区分：
   - Docling 预处理失败
   - Docling 超时
   - 真正的编码异常
   - 成功提取但内容极差
3. 检查失败样本是否继续参与了 embedding
4. 统计当前 `rep` 中有多少文件实际走了文件名 fallback
5. 在此基础上，再讨论默认参数实验是否恢复

## 关键代码位置

与本轮问题最相关的代码位置如下：

- 第三方日志配置
  - `src/dite/utils/logging.py:176-207`
- Docling 日志“抑制”逻辑
  - `src/dite/extractors/docling.py:104-119`
- Docling 子进程入口
  - `src/dite/extractors/docling.py:122-139`
- PDF 子进程提取包装
  - `src/dite/extractors/docling.py:143-196`
- PDF 主提取入口
  - `src/dite/core/pipeline.py:200-214`
- embedding 输入构造与文件名 fallback
  - `src/dite/core/embedder.py:51-71`
- 向量化主流程
  - `src/dite/core/pipeline.py:748-823`
- CLI 中 `extraction_failed` 的启发式判断
  - `src/dite/cli.py:367-377`
- Docling 报错位置
  - `.venv/lib/python3.12/site-packages/docling/pipeline/standard_pdf_pipeline.py:390`

## 最终结论

当前最应该警惕的不是“这个簇名字起得准不准”，而是：

- 底层提取器正在刷出大量未经控制的错误日志
- 提取失败文档仍在继续参与聚类

如果不先把这两件事搞清楚，后续所有基于 `rep` 的 A/B 结果都可能掺假。
