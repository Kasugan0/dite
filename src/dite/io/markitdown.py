"""MarkItDown 提取器 - 用于旧版 Office 格式"""

from pathlib import Path

from .base import BaseExtractor, ExtractionResult


class MarkItDownExtractor(BaseExtractor):
    """使用 MarkItDown 提取 DOC/PPT/XLS 等旧版 Office 格式"""

    name = "markitdown"

    @property
    def supported_extensions(self) -> set[str]:
        return {".doc", ".ppt", ".xls", ".rtf"}

    def __init__(self):
        self._md = None

    def _get_markitdown(self):
        """延迟加载 MarkItDown"""
        if self._md is None:
            from markitdown import MarkItDown

            self._md = MarkItDown()
        return self._md

    def extract(self, file_path: Path) -> ExtractionResult:
        """提取文档内容为 Markdown"""
        try:
            md = self._get_markitdown()
            result = md.convert(str(file_path))
            content = result.text_content

            return ExtractionResult(
                content=content,
                success=True,
                extractor=self.name,
            )
        except Exception as e:
            return ExtractionResult(
                content="",
                success=False,
                extractor=self.name,
                error=str(e),
            )
