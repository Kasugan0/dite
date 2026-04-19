"""向量化模块"""

import numpy as np
from openai import OpenAI

from dite.config import Config
from dite.i18n import t
from dite.utils.logging import get_logger

EMBEDDING_INPUT_VERSION = "filename-smart-content-v1"


def get_embedding_cache_version(embedding_model: str) -> str:
    """Return the cache key for the current embedding input format."""
    return f"{embedding_model}|input={EMBEDDING_INPUT_VERSION}"


def _get_file_name(file_names: list[str] | None, index: int) -> str | None:
    if file_names is None or index >= len(file_names):
        return None
    return file_names[index]


def get_embeddings(
    client: OpenAI,
    texts: list[str],
    file_names: list[str] | None = None,
    embedding_model: str | None = None,
) -> np.ndarray:
    """
    批量获取文本的 Embedding 向量

    Args:
        client: OpenAI 兼容客户端
        texts: 文本列表
        file_names: 文件名列表（作为分类信号，并用于空文本回退）

    Returns:
        Embedding 向量矩阵 (N, D)
    """
    logger = get_logger()
    model = embedding_model or Config().models.embedding

    logger.debug(t("debug_vectorizing_documents", count=len(texts)))
    logger.debug(t("debug_vectorizing_model", model=model))

    valid_texts = []
    fallback_count = 0
    fallback_names: list[str] = []
    for i, text in enumerate(texts):
        name = _get_file_name(file_names, i)
        stripped = text.strip() if text else ""
        if len(stripped) > 10:
            if name:
                valid_texts.append(f"File name: {name}\n\nContent:\n{stripped}")
            else:
                valid_texts.append(stripped)
            continue

        fallback_name = name or f"file_{i}"
        valid_texts.append(f"File name: {fallback_name}")
        fallback_count += 1
        fallback_names.append(fallback_name)

    if fallback_count > 0:
        logger.debug(
            t(
                "debug_vector_fallback_names",
                count=fallback_count,
                names=", ".join(fallback_names[:5]),
            )
        )

    # 统计文本长度
    text_lengths = [len(t) for t in valid_texts]
    logger.debug(
        t(
            "debug_vector_text_stats",
            min_length=min(text_lengths),
            max_length=max(text_lengths),
            avg_length=sum(text_lengths) // len(text_lengths),
        )
    )

    response = client.embeddings.create(
        model=model,
        input=valid_texts,
    )

    embeddings = [item.embedding for item in response.data]
    result = np.array(embeddings)

    logger.debug(t("debug_vector_dimension", dimension=result.shape[1]))
    logger.debug(t("debug_vector_api_usage", tokens=response.usage.total_tokens))

    return result


class ContentTruncator:
    """
    内容截断器：保留首尾，中间采样

    用于将长文档截断到 Embedding 模型的 token 限制内。
    """

    EMBEDDING_MAX_TOKENS = 8192
    CONTENT_BUDGET = 7500  # tokens

    @staticmethod
    def truncate_smart(content: str, max_chars: int = 15000) -> str:
        """
        智能截断：保留首尾重要信息，中间采样

        策略：
        - 首部 60%：标题、摘要、引言通常在开头
        - 中部 20%：采样中间内容
        - 尾部 20%：结论、参考文献通常在结尾

        Args:
            content: 原始内容
            max_chars: 最大字符数

        Returns:
            截断后的内容
        """
        if len(content) <= max_chars:
            return content

        first_marker = "\n\n[... middle omitted ...]\n\n"
        second_marker = "\n\n[... omitted ...]\n\n"
        marker_len = len(first_marker) + len(second_marker)
        if max_chars <= marker_len + 3:
            return content[:max_chars]

        content_budget = max_chars - marker_len
        head_len = int(content_budget * 0.6)
        mid_len = int(content_budget * 0.2)
        tail_len = content_budget - head_len - mid_len

        head = content[:head_len]

        mid_start = max((len(content) - mid_len) // 2, 0)
        mid = content[mid_start : mid_start + mid_len]

        tail = content[-tail_len:]

        return f"{head}{first_marker}{mid}{second_marker}{tail}"
