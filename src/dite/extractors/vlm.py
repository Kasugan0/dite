"""VLM 提取器 - 用于图片和扫描件"""

import base64
from pathlib import Path

from openai import OpenAI

from dite.config import Config
from dite.i18n import get_locale, set_locale, t

from .base import BaseExtractor, ExtractionResult


def _build_vlm_prompt() -> str:
    """Return a locale-aware prompt for image/document description."""
    if get_locale() == "zh-CN":
        return (
            "请详细描述这张图片的内容。"
            "如果是文档，请提取其标题、核心主题和关键词。"
            "直接输出描述，不要啰嗦。"
        )

    return (
        "Describe this image in detail. "
        "If it is a document, extract its title, core topic, and keywords. "
        "Return only the description."
    )


class VLMExtractor(BaseExtractor):
    """使用 VLM 描述图片或扫描件"""

    name = "vlm"

    # 图片格式到 MIME 类型的映射
    MIME_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    @property
    def supported_extensions(self) -> set[str]:
        return set(self.MIME_TYPES.keys())

    def __init__(self, config: Config, client: OpenAI | None = None):
        self._config = config
        self._client = client

    def set_client(self, client: OpenAI):
        """设置 OpenAI 客户端"""
        self._client = client

    def extract(self, file_path: Path) -> ExtractionResult:
        """使用 VLM 描述图片"""
        set_locale(self._config.i18n.locale)
        if self._client is None:
            return ExtractionResult(
                content="",
                success=False,
                extractor=self.name,
                error=t("error_vlm_client_not_initialized"),
            )

        suffix = file_path.suffix.lower()
        mime_type = self.MIME_TYPES.get(suffix, "image/jpeg")

        try:
            # 读取图片并转为 base64
            with open(file_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            response = self._client.chat.completions.create(
                model=self._config.models.vlm,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                            {
                                "type": "text",
                                "text": _build_vlm_prompt(),
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )

            content = response.choices[0].message.content or ""

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
