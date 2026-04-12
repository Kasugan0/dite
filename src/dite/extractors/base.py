"""提取器基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractionResult:
    """内容提取结果"""

    content: str  # 提取的 Markdown 内容
    success: bool  # 是否成功
    extractor: str  # 使用的提取器名称
    error: str | None = None  # 错误信息


class BaseExtractor(ABC):
    """提取器抽象基类"""

    name: str = "base"

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """支持的文件扩展名"""
        ...

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractionResult:
        """
        提取文件内容

        Args:
            file_path: 文件路径

        Returns:
            ExtractionResult 对象
        """
        ...

    def can_handle(self, file_path: Path) -> bool:
        """判断是否能处理该文件"""
        return file_path.suffix.lower() in self.supported_extensions
