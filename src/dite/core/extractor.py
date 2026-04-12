"""内容提取模块（封装）"""

from pathlib import Path

from openai import OpenAI

from dite.extractors import extract_content as _extract_content


def extract_content(
    file_path: Path,
    client: OpenAI | None = None,
    truncate_limit: int | None = None,
) -> str:
    """
    提取文件内容

    这是一个便捷的封装函数，实际调用 extractors.router.extract_content

    Args:
        file_path: 文件路径
        client: OpenAI 兼容客户端（VLM 需要）
        truncate_limit: 截断长度限制

    Returns:
        提取的内容（可能被截断）
    """
    return _extract_content(file_path, client, truncate_limit)
