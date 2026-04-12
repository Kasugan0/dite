"""文件扫描模块"""

from pathlib import Path

from dite.config import Config
from dite.i18n import t
from dite.utils.logging import get_logger


def scan_files(
    folder: Path,
    extensions: set[str] | None = None,
    recursive: bool = True,
    exclude_paths: list[Path] | None = None,
) -> list[Path]:
    """
    扫描文件夹，返回支持的文件列表

    Args:
        folder: 要扫描的文件夹路径
        extensions: 要扫描的文件扩展名集合，默认使用配置
        recursive: 是否递归扫描子目录
        exclude_paths: 需要排除的目录路径列表

    Returns:
        按路径排序的文件列表
    """
    logger = get_logger()

    if extensions is None:
        extensions = Config().formats.all_extensions

    logger.debug(t("debug_scan_folder", folder=folder))
    logger.debug(t("debug_scan_recursive", recursive=recursive))
    logger.debug(t("debug_scan_extensions", extensions=", ".join(sorted(extensions))))

    excludes = [p.resolve() for p in (exclude_paths or [])]
    if excludes:
        logger.debug(
            t(
                "debug_scan_excluded_dirs",
                paths=", ".join(f"[path]{str(path)}[/path]" for path in excludes),
            )
        )

    files = []
    excluded_count = 0
    skipped_count = 0
    skipped_extensions: dict[str, int] = {}

    iterator = folder.rglob("*") if recursive else folder.glob("*")

    for file in iterator:
        if file.is_file():
            resolved_file = file.resolve()
            if any(resolved_file.is_relative_to(exclude) for exclude in excludes):
                excluded_count += 1
                continue

            if file.suffix.lower() in extensions:
                files.append(file)
            else:
                skipped_count += 1
                ext = (
                    file.suffix.lower()
                    if file.suffix
                    else t("debug_scan_no_extension_label")
                )
                skipped_extensions[ext] = skipped_extensions.get(ext, 0) + 1

    if skipped_count > 0:
        logger.debug(t("debug_scan_skipped_unsupported", count=skipped_count))
        if skipped_extensions:
            top_skipped = sorted(
                skipped_extensions.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for ext, count in top_skipped:
                logger.debug(
                    t(
                        "debug_scan_skipped_extension_count",
                        extension=ext,
                        count=count,
                    )
                )

    logger.debug(
        t(
            "debug_scan_summary",
            supported=len(files),
            excluded=excluded_count,
            skipped=skipped_count,
        )
    )

    return sorted(files)
