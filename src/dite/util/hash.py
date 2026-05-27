"""文件哈希工具"""

import hashlib
from pathlib import Path


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    计算文件的 SHA256 哈希值（流式读取，内存友好）

    Args:
        file_path: 文件路径
        chunk_size: 每次读取的块大小

    Returns:
        64 字符的十六进制哈希字符串
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()
