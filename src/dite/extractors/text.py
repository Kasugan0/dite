"""纯文本提取器 - 用于 MD/TXT 等纯文本格式"""

from pathlib import Path

from dite.i18n import t

from .base import BaseExtractor, ExtractionResult


class TextExtractor(BaseExtractor):
    """直接读取纯文本文件"""

    name = "text"

    @property
    def supported_extensions(self) -> set[str]:
        return {".md", ".txt", ".markdown"}

    def extract(self, file_path: Path) -> ExtractionResult:
        """直接读取文本内容"""
        try:
            # 尝试多种编码
            encodings = ["utf-8", "gbk", "gb2312", "latin-1"]

            for encoding in encodings:
                try:
                    content = file_path.read_text(encoding=encoding)
                    return ExtractionResult(
                        content=content,
                        success=True,
                        extractor=self.name,
                    )
                except UnicodeDecodeError:
                    continue

            # 所有编码都失败
            return ExtractionResult(
                content="",
                success=False,
                extractor=self.name,
                error=t(
                    "error_text_decode_failed",
                    encodings=", ".join(encodings),
                ),
            )

        except Exception as e:
            return ExtractionResult(
                content="",
                success=False,
                extractor=self.name,
                error=str(e),
            )
