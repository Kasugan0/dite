"""聚类模块"""

import re
from dataclasses import dataclass
from pathlib import Path

import hdbscan
import numpy as np
from openai import APIError, OpenAI
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier

from dite.config import ClusteringConfig, Config
from dite.i18n import get_locale, t
from dite.utils.llm import (
    build_chat_completion_kwargs,
    format_api_error,
    should_retry_api_error,
)
from dite.utils.logging import get_logger

CLUSTER_NAME_CONTENT_LIMIT = 600
CLUSTER_NAME_EXCERPT_LIMIT = 180
CLUSTER_NAME_OUTPUT_LIMIT = 24
CLUSTER_NAME_MAX_RETRIES = 3
KNN_DYNAMIC_THRESHOLD_MULTIPLIER = 2.0
NON_MERGEABLE_CLUSTER_NAMES = {"", "未命名", "Unnamed"}
PLACEHOLDER_CLUSTER_NAMES = {
    "image",
    "cover",
    "scan",
    "scanned",
    "untitled",
    "unknown",
}
CLUSTER_NAME_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _normalize_cluster_name_text(text: str) -> str:
    """Normalize a raw cluster-name candidate into a single readable line."""
    cleaned = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    cleaned = cleaned.strip().strip("\"'`")
    cleaned = cleaned.lstrip("#*- ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _looks_like_author_line(text: str) -> bool:
    """Detect typical academic author lists so they are not used as titles."""
    if re.search(r"[\u4e00-\u9fff]", text):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", text)
    if not 4 <= len(words) <= 8:
        return False
    if any(word.casefold() in CLUSTER_NAME_STOPWORDS for word in words):
        return False

    has_separator = any(ch in text for ch in "*,&;")
    if not has_separator and len(words) < 6:
        return False

    return all(
        word[:1].isupper() and any(ch.islower() for ch in word[1:]) for word in words
    )


def _is_invalid_cluster_name(text: str) -> bool:
    """Return whether a cluster-name candidate is empty or low-information."""
    cleaned = _normalize_cluster_name_text(text)
    if not cleaned or cleaned in NON_MERGEABLE_CLUSTER_NAMES:
        return True
    if cleaned.startswith("第") and "页" in cleaned[:6]:
        return True
    if re.fullmatch(r"[\W_]+", cleaned):
        return True

    lowered = cleaned.casefold()
    if lowered in PLACEHOLDER_CLUSTER_NAMES:
        return True
    if re.fullmatch(r"(page|slide)\s*\d+", lowered):
        return True
    return bool(_looks_like_author_line(cleaned))


def _fallback_name_from_file_name(file_name: str) -> str:
    """Return a readable fallback name from the source file name."""
    stem = Path(file_name).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    cleaned = _normalize_cluster_name_text(stem)
    if _is_invalid_cluster_name(cleaned):
        return ""
    return cleaned[:CLUSTER_NAME_OUTPUT_LIMIT]


def _display_cluster_name(label: int, name: str) -> str:
    """Return a user-facing cluster name without changing cluster structure."""
    cleaned = _normalize_cluster_name_text(name)
    if _is_invalid_cluster_name(cleaned):
        return f"{t('cluster_unnamed_label')}-{label}"
    return cleaned[:CLUSTER_NAME_OUTPUT_LIMIT]


def _cluster_debug_token(index: int) -> str:
    """Return a stable A/B/C-style debug token for cluster ordering."""
    if index < 26:
        return chr(ord("A") + index)
    return _cluster_debug_token(index // 26 - 1) + chr(ord("A") + index % 26)


def _build_cluster_debug_labels(labels: np.ndarray) -> dict[int, str]:
    """Map internal numeric labels to stable debug-facing tokens."""
    cluster_labels = sorted(label for label in set(labels) if label != -1)
    return {
        label: _cluster_debug_token(index)
        for index, label in enumerate(cluster_labels)
    }


def _extract_title_like_line(content: str) -> str:
    """Pick a short, title-like line from extracted content."""
    for raw_line in content.splitlines():
        line = _normalize_cluster_name_text(raw_line)
        if len(line) < 4 or len(line) > 120:
            continue
        if _is_invalid_cluster_name(line):
            continue
        return line
    return ""


def _compact_sample_for_naming(file_name: str, content: str) -> str:
    """Build a compact naming summary for one representative sample."""
    title = _extract_title_like_line(content) or _fallback_name_from_file_name(file_name)
    if not title:
        title = Path(file_name).stem
    excerpt = re.sub(r"\s+", " ", content.strip())[:CLUSTER_NAME_EXCERPT_LIMIT]
    if get_locale() == "zh-CN":
        return (
            f"文件名: {file_name}\n"
            f"标题候选: {title}\n"
            f"内容片段: {excerpt}"
        )
    return (
        f"File name: {file_name}\n"
        f"Title candidate: {title}\n"
        f"Content excerpt: {excerpt}"
    )


def _heuristic_cluster_name(sample_contents: list[str], sample_names: list[str]) -> str:
    """Fallback to a conservative display label when LLM naming is unavailable."""
    for content in sample_contents:
        title = _extract_title_like_line(content)
        if title:
            return title[:CLUSTER_NAME_OUTPUT_LIMIT]

    for file_name in sample_names:
        fallback_name = _fallback_name_from_file_name(file_name)
        if fallback_name:
            return fallback_name

    return t("cluster_unnamed_label")


def _build_cluster_naming_prompt(compact_samples: list[str], sample_names: list[str], top_k: int) -> str:
    """Build a locale-aware prompt for cluster naming."""
    if get_locale() == "zh-CN":
        forbidden_terms = "「聚类」「分类」「类别」「类型」「封面」「扫描」「文档」「资料」「合集」「汇编」"
        if compact_samples:
            combined = "\n---\n".join(compact_samples[:top_k])
            return (
                f"以下是属于同一类别的代表文档信息：\n\n{combined}\n\n"
                f"请用2-4个中文词为这个类别命名。\n"
                f"要求：\n"
                f"1. 名称应描述文档的【主题内容】，而非文档的形式或载体\n"
                f"2. 禁止使用以下词语：{forbidden_terms}\n"
                f"3. 优先参考标题候选和文件名中的主题词\n"
                f"4. 直接输出名称，不要解释"
            )

        names = "\n".join(sample_names[:top_k])
        return (
            f"以下文件属于同一类别，请根据文件名推测类别：\n\n{names}\n\n"
            f"请用2-4个中文词为这个类别命名。\n"
            f"要求：\n"
            f"1. 名称应描述文档的【主题内容】，而非文档的形式或载体\n"
            f"2. 禁止使用以下词语：{forbidden_terms}\n"
            f"3. 直接输出名称，不要解释"
        )

    forbidden_terms = '"cluster", "category", "type", "cover", "scan", "document", "file", "collection", "archive"'
    if compact_samples:
        combined = "\n---\n".join(compact_samples[:top_k])
        return (
            f"Here are representative documents from the same category:\n\n{combined}\n\n"
            f"Name this category in 2-4 English words.\n"
            f"Requirements:\n"
            f"1. The name must describe the topical content, not the document form or medium\n"
            f"2. Do not use these terms: {forbidden_terms}\n"
            f"3. Prefer topic words from the title candidate and file name\n"
            f"4. Output only the name"
        )

    names = "\n".join(sample_names[:top_k])
    return (
        f"These files belong to the same category. Infer the category from their file names:\n\n{names}\n\n"
        f"Name this category in 2-4 English words.\n"
        f"Requirements:\n"
        f"1. The name must describe the topical content, not the document form or medium\n"
        f"2. Do not use these terms: {forbidden_terms}\n"
        f"3. Output only the name"
    )


@dataclass
class ClusterResult:
    """聚类结果"""

    labels: np.ndarray  # 每个文件的聚类标签，-1 表示噪音
    cluster_names: dict[int, str]  # 聚类标签到名称的映射
    noise_repaired: int = 0  # k-NN 修复的噪音点数量
    clusters_merged: int = 0  # 合并的簇数量

    @property
    def n_clusters(self) -> int:
        """簇数量（不含噪音）"""
        unique = set(self.labels)
        return len([lbl for lbl in unique if lbl != -1])

    @property
    def n_noise(self) -> int:
        """噪音点数量"""
        return int(np.sum(self.labels == -1))


def repair_noise_with_knn(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
    distance_threshold: float | None = None,
    item_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    使用 k-NN 修复 HDBSCAN 的噪音点（参考论文 §3.3.3）

    在标准 k-NN 修复基础上增加距离阈值保护：
    1. 在非噪音点上训练 k-NN
    2. 为噪音点预测标签并计算最近邻距离
    3. 仅当距离 <= 阈值时归入簇，否则保持噪音

    Args:
        embeddings: 向量矩阵 (N, D)
        labels: HDBSCAN 输出的标签数组
        k: k-NN 的邻居数（默认3，使用多数投票）
        distance_threshold: 余弦距离阈值。None 时使用动态阈值。

    Returns:
        修复后的标签数组和修复的噪音点数量
    """
    logger = get_logger()
    debug_labels = _build_cluster_debug_labels(labels)
    labels = labels.copy()
    core_mask = labels != -1

    # 如果没有噪音点或没有核心簇，直接返回
    if core_mask.all() or not core_mask.any():
        return labels, 0

    X_core, y_core = embeddings[core_mask], labels[core_mask]
    X_noise = embeddings[~core_mask]
    noise_indices = np.where(~core_mask)[0]

    # 确保 k 不超过核心点数量
    actual_k = min(k, len(X_core))
    if actual_k < 1:
        return labels, 0

    # 训练 k-NN（使用余弦距离）
    knn = KNeighborsClassifier(n_neighbors=actual_k, metric="cosine")
    knn.fit(X_core, y_core)

    # 预测噪音点标签并计算距离
    predictions = knn.predict(X_noise)
    distances, _ = knn.kneighbors(X_noise)

    # 动态阈值：核心点最近邻平均距离 * 2
    if distance_threshold is None:
        core_dist_matrix = cosine_distances(X_core, X_core)
        np.fill_diagonal(core_dist_matrix, np.inf)
        nearest_core_distance = np.min(core_dist_matrix, axis=1)
        mean_core_distance = float(np.mean(nearest_core_distance))
        distance_threshold = mean_core_distance * KNN_DYNAMIC_THRESHOLD_MULTIPLIER
        logger.debug(
            t(
                "debug_cluster_knn_dynamic_threshold",
                threshold=distance_threshold,
                mean_core_distance=mean_core_distance,
            )
        )
    else:
        logger.debug(
            t("debug_cluster_knn_fixed_threshold", threshold=distance_threshold)
        )

    repaired_count = 0
    for pred, dist, idx in zip(predictions, distances[:, 0], noise_indices, strict=False):
        item_name = item_names[idx] if item_names is not None else f"#{idx}"
        debug_label = debug_labels.get(int(pred), str(pred))
        if dist <= distance_threshold:
            labels[idx] = pred
            repaired_count += 1
            logger.debug(
                t(
                    "debug_cluster_knn_assignment",
                    index=idx,
                    name=item_name,
                    label=debug_label,
                    distance=float(dist),
                    threshold=distance_threshold,
                )
            )
        else:
            logger.debug(
                t(
                    "debug_cluster_knn_kept",
                    index=idx,
                    name=item_name,
                    label=debug_label,
                    distance=float(dist),
                    threshold=distance_threshold,
                )
            )

    kept_noise = len(noise_indices) - repaired_count
    logger.debug(
        t("debug_cluster_knn_summary", repaired=repaired_count, kept=kept_noise)
    )

    return labels, repaired_count


def cluster_documents(
    embeddings: np.ndarray,
    *,
    config: Config,
    repair_noise: bool = True,
    knn_k: int | None = None,
    knn_distance_threshold: float | None = None,
    clustering: ClusteringConfig | None = None,
    item_names: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """
    使用 HDBSCAN 对文档进行聚类，可选 k-NN 噪音修复

    Args:
        embeddings: 向量矩阵 (N, D)
        repair_noise: 是否使用 k-NN 修复噪音点
        knn_k: k-NN 的邻居数（None 时使用配置值）
        knn_distance_threshold: 距离阈值（None 时使用配置值）

    Returns:
        聚类标签数组和修复的噪音点数量
    """
    logger = get_logger()
    clustering_cfg = clustering or config.clustering
    effective_knn_k = knn_k if knn_k is not None else clustering_cfg.knn_k
    effective_distance_threshold = (
        knn_distance_threshold
        if knn_distance_threshold is not None
        else clustering_cfg.knn_distance_threshold
    )

    logger.debug(t("debug_cluster_hdbscan_header"))
    logger.debug(
        t(
            "debug_cluster_hdbscan_min_cluster_size",
            value=clustering_cfg.min_cluster_size,
        )
    )
    logger.debug(
        t("debug_cluster_hdbscan_min_samples", value=clustering_cfg.min_samples)
    )
    logger.debug(
        t(
            "debug_cluster_hdbscan_epsilon",
            value=clustering_cfg.cluster_selection_epsilon,
        )
    )
    logger.debug(
        t(
            "debug_cluster_hdbscan_method",
            value=clustering_cfg.cluster_selection_method,
        )
    )
    logger.debug(
        t(
            "debug_cluster_input_vectors",
            count=embeddings.shape[0],
            dimension=embeddings.shape[1],
        )
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=clustering_cfg.min_cluster_size,
        min_samples=clustering_cfg.min_samples,
        cluster_selection_epsilon=clustering_cfg.cluster_selection_epsilon,
        metric="euclidean",
        cluster_selection_method=clustering_cfg.cluster_selection_method,
    )
    labels = clusterer.fit_predict(embeddings)

    # 输出初始聚类结果
    unique_labels = set(labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
    n_noise = int(np.sum(labels == -1))
    logger.debug(
        t("debug_cluster_initial_result", clusters=n_clusters, noise=n_noise)
    )

    debug_labels = _build_cluster_debug_labels(labels)

    # 输出各簇大小（使用大写字母标识，避免与数字混淆）
    if n_clusters > 0:
        cluster_sizes = []
        for lbl in sorted(lbl for lbl in unique_labels if lbl != -1):
            size = int(np.sum(labels == lbl))
            cluster_sizes.append(f"{debug_labels[int(lbl)]}:{size}")
        logger.debug(t("debug_cluster_sizes", sizes=", ".join(cluster_sizes)))

    # k-NN 噪音修复
    repaired_count = 0
    if repair_noise:
        labels, repaired_count = repair_noise_with_knn(
            embeddings,
            labels,
            k=effective_knn_k,
            distance_threshold=effective_distance_threshold,
            item_names=item_names,
        )

    return labels, repaired_count


def generate_cluster_name(
    client: OpenAI,
    cluster_embeddings: np.ndarray | None,
    sample_contents: list[str],
    sample_names: list[str],
    *,
    config: Config,
    top_k: int = 5,
    llm_model: str | None = None,
) -> str:
    """
    使用 LLM 为簇生成名称（基于最具代表性的 Top-K 文件）

    Args:
        client: OpenAI 兼容客户端
        cluster_embeddings: 簇内所有文档的向量
        sample_contents: 簇内文档的内容样本
        sample_names: 簇内文档的文件名
        top_k: 选择最具代表性的 K 个文件

    Returns:
        簇名称
    """
    logger = get_logger()
    model = llm_model or config.models.llm
    request_profile = config.request_profiles.cluster_naming

    # 如果有向量，选择最具代表性的文件
    if cluster_embeddings is not None and len(cluster_embeddings) > 0:
        # 计算簇中心
        centroid = cluster_embeddings.mean(axis=0)

        # 计算每个文件到中心的距离
        distances = cosine_distances(
            cluster_embeddings, centroid.reshape(1, -1)
        ).flatten()

        # 选出最近的 K 个
        top_indices = np.argsort(distances)[:top_k]

        # 只使用最具代表性的文件
        sample_contents = [
            sample_contents[i] for i in top_indices if i < len(sample_contents)
        ]
        sample_names = [sample_names[i] for i in top_indices if i < len(sample_names)]

    compact_samples = []
    for content, file_name in zip(sample_contents, sample_names, strict=False):
        stripped = content.strip()
        if not stripped:
            continue
        compact_samples.append(
            _compact_sample_for_naming(
                file_name,
                stripped[:CLUSTER_NAME_CONTENT_LIMIT],
            )
        )

    prompt = _build_cluster_naming_prompt(compact_samples, sample_names, top_k)

    last_error = ""
    for attempt in range(CLUSTER_NAME_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                **build_chat_completion_kwargs(
                    client=client,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    profile=request_profile,
                )
            )
            choice = response.choices[0]
            message = choice.message
            raw_name = message.content or ""
            reasoning_content = getattr(message, "reasoning_content", None)
            if not raw_name.strip():
                logger.debug(
                    t(
                        "debug_cluster_name_empty_response",
                        model=response.model,
                        finish_reason=choice.finish_reason,
                        reasoning_chars=len(reasoning_content or ""),
                    )
                )
                fallback = _heuristic_cluster_name(sample_contents, sample_names)
                logger.debug(
                    t("debug_cluster_name_empty_fallback", fallback=fallback)
                )
                return fallback

            name = _normalize_cluster_name_text(raw_name.split("\n")[0])
            if _is_invalid_cluster_name(name):
                fallback = _heuristic_cluster_name(sample_contents, sample_names)
                logger.debug(
                    t("debug_cluster_name_invalid_fallback", fallback=fallback)
                )
                return fallback
            return name[:CLUSTER_NAME_OUTPUT_LIMIT]
        except APIError as exc:
            last_error = format_api_error(exc)
            is_last_attempt = attempt == CLUSTER_NAME_MAX_RETRIES - 1
            if not is_last_attempt and should_retry_api_error(exc):
                logger.debug(
                    t(
                        "debug_cluster_name_retry",
                        attempt=attempt + 1,
                        max_retries=CLUSTER_NAME_MAX_RETRIES,
                        error=last_error,
                    )
                )
                continue
        except Exception as exc:
            last_error = str(exc)
        break

    fallback = _heuristic_cluster_name(sample_contents, sample_names)
    logger.debug(
        t(
            "debug_cluster_name_failed_fallback",
            fallback=fallback,
            error=last_error,
        )
    )
    return fallback


def merge_clusters_by_name(
    labels: np.ndarray,
    cluster_names: dict[int, str],
) -> tuple[np.ndarray, dict[int, str], int]:
    """
    合并具有相同名称的簇

    当 LLM 为不同簇生成相同名称时，说明这些簇语义上应该合并。

    Args:
        labels: 聚类标签数组
        cluster_names: 标签到名称的映射

    Returns:
        (合并后的标签, 合并后的名称映射, 合并的簇数量)
    """
    logger = get_logger()
    debug_labels = _build_cluster_debug_labels(labels)

    # 找出重复的名称
    name_to_labels: dict[str, list[int]] = {}
    for label, name in cluster_names.items():
        if name not in name_to_labels:
            name_to_labels[name] = []
        name_to_labels[name].append(label)

    # 统计需要合并的簇
    merged_count = 0
    labels = labels.copy()
    new_cluster_names = {}

    for name, label_list in name_to_labels.items():
        if name.strip() in NON_MERGEABLE_CLUSTER_NAMES:
            for label in label_list:
                new_cluster_names[label] = name
            continue

        if len(label_list) > 1:
            # 需要合并：将所有标签统一为第一个
            target_label = label_list[0]
            for other_label in label_list[1:]:
                labels[labels == other_label] = target_label
                merged_count += 1
                logger.debug(
                    t(
                        "debug_cluster_merge",
                        source=debug_labels.get(other_label, str(other_label)),
                        target=debug_labels.get(target_label, str(target_label)),
                        name=name,
                    )
                )
            new_cluster_names[target_label] = name
        else:
            new_cluster_names[label_list[0]] = name

    return labels, new_cluster_names, merged_count


def generate_all_cluster_names(
    client: OpenAI,
    labels: np.ndarray,
    contents: list[str],
    files: list[Path],
    *,
    config: Config,
    embeddings: np.ndarray | None = None,
    merge_same_name: bool = True,
    llm_model: str | None = None,
) -> tuple[np.ndarray, dict[int, str], int]:
    """
    为所有簇生成名称，并可选合并同名簇

    Args:
        client: OpenAI 兼容客户端
        labels: 聚类标签数组
        contents: 文档内容列表
        files: 文件路径列表
        embeddings: 向量矩阵（用于选择代表性文件）
        merge_same_name: 是否合并同名簇

    Returns:
        (可能修改后的标签, 标签到名称的映射, 合并的簇数量)
    """
    unique_labels = set(labels)
    cluster_names = {}
    debug_labels = _build_cluster_debug_labels(labels)

    for label in unique_labels:
        if label == -1:
            continue

        # 获取该簇的索引
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]

        # 获取该簇的文件内容和文件名
        cluster_contents = [contents[i] for i in cluster_indices]
        cluster_file_names = [files[i].name for i in cluster_indices]

        # 获取该簇的向量（如果提供）
        cluster_embeddings = None
        if embeddings is not None:
            cluster_embeddings = embeddings[cluster_indices]

        name = generate_cluster_name(
            client,
            cluster_embeddings,
            cluster_contents,
            cluster_file_names,
            config=config,
            llm_model=llm_model,
        )
        cluster_names[label] = _display_cluster_name(label, name)
        logger = get_logger()
        logger.debug(
            t(
                "debug_cluster_name_result",
                label=debug_labels.get(label, str(label)),
                name=cluster_names[label],
                count=len(cluster_indices),
            )
        )

    # 合并同名簇
    merged_count = 0
    if merge_same_name:
        labels, cluster_names, merged_count = merge_clusters_by_name(
            labels, cluster_names
        )

    return labels, cluster_names, merged_count
