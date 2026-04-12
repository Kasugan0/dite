"""CLI 命令定义"""

import json
import logging
import warnings
from pathlib import Path
from typing import Annotated

# 抑制第三方库的警告（如 Pillow 的 WMF 警告）
warnings.filterwarnings("ignore", message=".*cannot be loaded by Pillow.*")
warnings.filterwarnings("ignore", message=".*cannot find loader.*")

# 在任何其他导入之前配置 logging
# 这必须在导入 docling 等库之前完成，否则它们会调用 basicConfig
# 先调用 basicConfig 占位，防止第三方库配置 root logger
logging.basicConfig(level=logging.WARNING, handlers=[logging.NullHandler()])

# 禁用 docling 的所有日志器（包括子模块）
_docling_logger_names = [
    "docling",
    "docling_core",
    "docling.pipeline",
    "docling.pipeline.standard_pdf_pipeline",
    "docling.backend",
    "docling.backend.mspowerpoint_backend",
    "docling.backend.msword_backend",
    "docling.datamodel",
    "docling.document_converter",
]
for _name in _docling_logger_names:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False
    _lg.addHandler(logging.NullHandler())

# 以下导入必须在 logging 配置之后，因为 docling 等库在导入时会尝试配置日志
import numpy as np  # noqa: E402
import typer  # noqa: E402
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI  # noqa: E402
from rich.table import Table  # noqa: E402

from dite import __version__  # noqa: E402
from dite.cache import FileCache  # noqa: E402
from dite.config import Config, load_config  # noqa: E402
from dite.core.clusterer import _build_cluster_debug_labels  # noqa: E402
from dite.core.organizer import OrganizePreview  # noqa: E402
from dite.core.pipeline import (  # noqa: E402
    PipelineOptions,
    PipelineResult,
    PipelineService,
)
from dite.core.scanner import scan_files  # noqa: E402
from dite.extractors.docling import (  # noqa: E402
    download_docling_pdf_models,
    get_docling_pdf_artifacts_path,
    has_docling_pdf_artifacts,
)
from dite.extractors.router import _compute_effective_content_length  # noqa: E402
from dite.i18n import set_locale, t  # noqa: E402
from dite.utils.llm import format_api_error  # noqa: E402
from dite.utils.logging import get_logger, setup_logging  # noqa: E402


def _initialize_cli_locale() -> None:
    """在 CLI 构建前按配置设置语言，确保 --help 也使用配置语言。"""
    cfg = load_config()
    set_locale(cfg.i18n.locale)


_initialize_cli_locale()

# 创建 CLI 应用
app = typer.Typer(
    name="dite",
    help=t("cli_description"),
    no_args_is_help=True,
)

# 缓存子命令
cache_app = typer.Typer(help=t("cli_help_cache_group"))
app.add_typer(cache_app, name="cache")

# 环境准备子命令
setup_app = typer.Typer(help=t("cli_help_setup_group"))
app.add_typer(setup_app, name="setup")


def version_callback(value: bool):
    """版本回调"""
    if value:
        logger = get_logger()
        logger.print(f"{t('version_prefix')} {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=t("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=t("cli_help_quiet")),
    ] = False,
    color: Annotated[
        bool,
        typer.Option(
            "--color",
            help=t("cli_help_color"),
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help=t("cli_help_version"),
        ),
    ] = False,
):
    """DITE - Multimodal document intelligent clustering tool"""
    # 设置日志
    setup_logging(verbose=verbose, quiet=quiet, force_color=color)

    # 加载配置
    cfg = load_config()
    ctx.obj = {"config": cfg}

    # 设置语言环境
    set_locale(cfg.i18n.locale)


def _get_app_config(ctx: typer.Context | None) -> Config:
    if ctx is None or not ctx.obj or "config" not in ctx.obj:
        return load_config()
    return ctx.obj["config"]


def _get_client(config: Config) -> OpenAI:
    """获取 OpenAI 兼容客户端"""
    return OpenAI(
        api_key=config.api.api_key,
        base_url=config.api.base_url,
    )


def _run_pipeline_or_exit(
    pipeline: PipelineService,
    folder: Path,
    options: PipelineOptions,
    cache: FileCache | None,
) -> PipelineResult:
    """Run pipeline and convert runtime/API failures to user-friendly CLI errors."""
    logger = get_logger()
    try:
        return pipeline.run(folder, options)
    except (APIConnectionError, APITimeoutError):
        logger.error(t("error_api_connection_failed"))
    except APIError as exc:
        logger.error(t("error_api_request_failed", error=format_api_error(exc)))
    except Exception as exc:
        logger.error(t("error_processing_failed", error=exc))

    if cache:
        cache.close()
    raise typer.Exit(1)


def _build_report(
    files: list[Path],
    contents: list[str],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    noise_repaired: int = 0,
    clusters_merged: int = 0,
) -> dict:
    """构建聚类报告"""
    clusters = {}
    debug_labels = _build_cluster_debug_labels(labels)
    noise = []
    failed_count = 0

    for file, content, label in zip(files, contents, labels):
        is_failed = len(content.strip()) < 10
        if is_failed:
            failed_count += 1

        entry = {
            "path": str(file),
            "name": file.name,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "extraction_failed": is_failed,
        }

        if label == -1:
            noise.append(entry)
        else:
            if label not in clusters:
                clusters[label] = {
                    "name": cluster_names.get(
                        label, t("cluster_default_name", label=label)
                    ),
                    "debug_label": debug_labels.get(label),
                    "files": [],
                }
            clusters[label]["files"].append(entry)

    return {
        "summary": {
            "total_files": len(files),
            "num_clusters": len(clusters),
            "num_noise": len(noise),
            "num_extraction_failed": failed_count,
            "noise_repaired": noise_repaired,
            "clusters_merged": clusters_merged,
        },
        "clusters": [clusters[label] for label in sorted(clusters)],
        "noise": noise,
    }


def _print_report(report: dict) -> None:
    """打印聚类报告"""
    logger = get_logger()
    logger.print(f"\n[bold green]{t('cluster_report_title')}[/bold green]\n")

    summary = report["summary"]
    logger.print(f"[bold]{t('cluster_total_files')}[/bold] {summary['total_files']}")
    logger.print(f"[bold]{t('cluster_num_clusters')}[/bold] {summary['num_clusters']}")
    logger.print(f"[bold]{t('cluster_num_noise')}[/bold] {summary['num_noise']}")

    if summary.get("noise_repaired", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_knn_label')}[/bold] "
            f"{t('cluster_knn_repair', count=summary['noise_repaired'])}"
        )

    if summary.get("clusters_merged", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_merge_label')}[/bold] "
            f"{t('cluster_merged', count=summary['clusters_merged'])}"
        )

    if summary.get("num_extraction_failed", 0) > 0:
        logger.warning(
            t("cluster_extraction_failed", count=summary["num_extraction_failed"])
        )

    for cluster in report["clusters"]:
        cluster_title = cluster["name"]
        if logger.verbose and cluster.get("debug_label"):
            cluster_title = f"{cluster['debug_label']} | {cluster_title}"
        logger.print(
            f"\n[bold cyan]【{cluster_title}】[/bold cyan] "
            f"({t('organize_files_count', count=len(cluster['files']))})"
        )
        table = Table(show_header=False, box=None, padding=(0, 2))
        for f in cluster["files"]:
            marker = (
                f"[dim]{t('cluster_report_extraction_failed_marker')}[/dim]"
                if f.get("extraction_failed")
                else ""
            )
            table.add_row("  -", f["name"], marker)
        logger.print_table(table)

    if report["noise"]:
        logger.print(
            f"\n[bold yellow]【{t('cluster_uncategorized')}】[/bold yellow] "
            f"({t('organize_files_count', count=len(report['noise']))})"
        )
        for f in report["noise"]:
            marker = (
                f" [dim]{t('cluster_report_extraction_failed_marker')}[/dim]"
                if f.get("extraction_failed")
                else ""
            )
            logger.print(f"  - {f['name']}{marker}")


@app.command(help=t("scan_description"))
def scan(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=t("cli_help_folder_scan")),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help=t("cli_help_output_report")),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=t("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=t("cli_help_quiet")),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=t("cli_help_disable_cache")),
    ] = False,
    no_knn_repair: Annotated[
        bool,
        typer.Option("--no-knn-repair", help=t("cli_help_disable_knn_repair")),
    ] = False,
):
    """
    Scan folder and perform cluster analysis.

    Scan documents in the specified folder, extract content, cluster using HDBSCAN,
    and generate cluster names using LLM.
    """
    # 如果子命令指定了日志级别，覆盖全局设置
    if verbose or quiet:
        setup_logging(verbose=verbose, quiet=quiet)

    logger = get_logger()
    config = _get_app_config(ctx)

    logger.print(f"[bold blue]{t('cli_title')}[/bold blue]\n")

    if not folder.exists():
        logger.error(t("scan_folder_not_found", folder=folder))
        raise typer.Exit(1)

    # 初始化
    client = _get_client(config)
    cache = (
        FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
        if not no_cache and config.cache.enabled
        else None
    )
    pipeline = PipelineService(client=client, config=config, cache=cache)
    result = _run_pipeline_or_exit(
        pipeline,
        folder,
        PipelineOptions(
            use_cache=cache is not None,
            use_embedding_cache=True,
            repair_noise=not no_knn_repair,
            merge_same_name=False,
            exclude_paths=[],
        ),
        cache,
    )

    if not result.files:
        if cache:
            cache.close()
        logger.error(t("scan_no_files"))
        raise typer.Exit(1)

    logger.status(t("scan_found_files", count=len(result.files)))

    cache_msg_parts = []
    if result.docling_cache_hits:
        cache_msg_parts.append(t("cache_docling_hit", count=result.docling_cache_hits))
    if result.vlm_cache_hits:
        cache_msg_parts.append(t("cache_vlm_hit", count=result.vlm_cache_hits))
    if result.vlm_fallback_count:
        cache_msg_parts.append(t("cache_vlm_fallback", count=result.vlm_fallback_count))
    if result.duplicate_count:
        cache_msg_parts.append(t("cache_duplicate", count=result.duplicate_count))
    cache_msg = f" ({', '.join(cache_msg_parts)})" if cache_msg_parts else ""
    logger.status(f"{t('scan_extraction_done')}{cache_msg}")

    if result.duplicate_groups and logger.verbose:
        logger.debug(t("debug_duplicate_groups"))
        for file_hash, file_list in result.duplicate_groups.items():
            logger.debug(t("debug_duplicate_group_hash", hash=file_hash[:12]))
            for f in file_list:
                logger.debug(t("debug_duplicate_group_file", name=Path(f).name))

    logger.status(t("scan_vectorizing_done", dim=result.embeddings.shape[1]))

    unique_labels = set(result.labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
    n_noise = int(np.sum(result.labels == -1))
    status_msg = t("scan_clustering_done", clusters=n_clusters, noise=n_noise)
    if result.noise_repaired > 0:
        status_msg += (
            f" [{t('scan_status_knn_suffix', count=result.noise_repaired)}]"
        )
    logger.status(status_msg)

    status_msg = t("scan_naming_done")
    if result.clusters_merged > 0:
        status_msg += (
            f" [{t('scan_status_merged_suffix', count=result.clusters_merged)}]"
        )
    logger.status(status_msg)

    # 关闭缓存
    if cache:
        cache.close()

    # 生成报告
    report = _build_report(
        result.files,
        result.contents,
        result.labels,
        result.cluster_names,
        result.noise_repaired,
        result.clusters_merged,
    )

    # 保存 JSON
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.success(t("scan_report_saved", path=output))

    # 打印报告
    _print_report(report)


@app.command("pdf-check", help=t("pdf_check_description"))
def pdf_check(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=t("cli_help_folder_pdf_check")),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=t("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=t("cli_help_quiet")),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=t("cli_help_disable_cache")),
    ] = False,
    cached_vlm_only: Annotated[
        bool,
        typer.Option("--cached-vlm-only", help=t("cli_help_cached_vlm_only")),
    ] = False,
):
    """Run PDF extraction checks without embedding, clustering, or naming."""
    if verbose or quiet:
        setup_logging(verbose=verbose, quiet=quiet)

    logger = get_logger()
    config = _get_app_config(ctx)

    logger.print(f"[bold blue]{t('cli_title')}[/bold blue]\n")

    if not folder.exists():
        logger.error(t("scan_folder_not_found", folder=folder))
        raise typer.Exit(1)

    files = scan_files(folder, extensions={".pdf"})
    if not files:
        logger.error(t("pdf_check_no_pdfs"))
        raise typer.Exit(1)

    client = _get_client(config)
    cache = (
        FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
        if not no_cache and config.cache.enabled
        else None
    )
    pipeline = PipelineService(client=client, config=config, cache=cache)

    try:
        result = pipeline.extract_files(
            files,
            PipelineOptions(
                use_cache=cache is not None,
                use_embedding_cache=False,
                repair_noise=False,
                merge_same_name=False,
                allow_vlm_api=not cached_vlm_only,
                exclude_paths=[],
            ),
        )
    except (APIConnectionError, APITimeoutError):
        logger.error(t("error_api_connection_failed"))
        if cache:
            cache.close()
        raise typer.Exit(1) from None
    except APIError as exc:
        logger.error(t("error_api_request_failed", error=format_api_error(exc)))
        if cache:
            cache.close()
        raise typer.Exit(1) from None
    except Exception as exc:
        logger.error(t("error_processing_failed", error=exc))
        if cache:
            cache.close()
        raise typer.Exit(1) from None

    if cache:
        cache.close()

    threshold = config.processing.vlm_fallback_threshold
    weak_files = [
        (file, _compute_effective_content_length(content))
        for file, content in zip(result.files, result.contents, strict=True)
        if _compute_effective_content_length(content) < threshold
    ]
    empty_count = sum(1 for _file, length in weak_files if length == 0)

    logger.status(t("pdf_check_found_pdfs", count=len(result.files)))
    logger.status(
        t(
            "pdf_check_done",
            doc_cache_hits=result.docling_cache_hits,
            vlm_cache_hits=result.vlm_cache_hits,
            vlm_fallback_calls=result.vlm_fallback_count,
            duplicates=result.duplicate_count,
            weak=len(weak_files),
            empty=empty_count,
        )
    )

    if weak_files:
        table = Table(title=t("pdf_check_weak_table_title"))
        table.add_column(t("pdf_check_table_file"), style="path")
        table.add_column(t("pdf_check_table_effective_length"), justify="right")
        for file, length in weak_files:
            table.add_row(str(file), str(length))
        logger.print_table(table)
        logger.error(t("pdf_check_failed", count=len(weak_files)))
        raise typer.Exit(1)

    logger.success(t("pdf_check_passed", count=len(result.files)))


@app.command(help=t("organize_description"))
def organize(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=t("cli_help_folder_organize")),
    ],
    target: Annotated[
        Path | None,
        typer.Option("--target", "-t", help=t("cli_help_target_folder")),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=t("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=t("cli_help_quiet")),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help=t("cli_help_preview_mode")),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help=t("cli_help_execute_move")),
    ] = False,
    output_script: Annotated[
        Path | None,
        typer.Option("--output-script", help=t("cli_help_output_script")),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=t("cli_help_disable_cache")),
    ] = False,
    no_knn_repair: Annotated[
        bool,
        typer.Option("--no-knn-repair", help=t("cli_help_disable_knn_repair")),
    ] = False,
):
    """
    Organize documents in folder.

    Move files to subfolders based on clustering results.
    """
    # 如果子命令指定了日志级别，覆盖全局设置
    if verbose or quiet:
        setup_logging(verbose=verbose, quiet=quiet)

    logger = get_logger()
    config = _get_app_config(ctx)

    # 参数验证
    if not dry_run and not execute and not output_script:
        logger.warning(t("organize_specify_mode"))
        logger.print(t("organize_mode_help"))
        raise typer.Exit(1)

    if not folder.exists():
        logger.error(t("organize_folder_not_found", folder=folder))
        raise typer.Exit(1)

    # 设置目标文件夹
    if target is None:
        target = folder / "organized"

    logger.print(f"[bold blue]{t('cli_title')}[/bold blue]\n")

    # 初始化
    client = _get_client(config)
    cache = (
        FileCache(
            cache_dir=config.cache.directory,
            max_size_gb=config.cache.max_size_gb,
        )
        if not no_cache and config.cache.enabled
        else None
    )
    pipeline = PipelineService(client=client, config=config, cache=cache)
    result = _run_pipeline_or_exit(
        pipeline,
        folder,
        PipelineOptions(
            use_cache=cache is not None,
            use_embedding_cache=True,
            repair_noise=not no_knn_repair,
            merge_same_name=False,
            exclude_paths=[target.resolve()],
        ),
        cache,
    )

    if not result.files:
        if cache:
            cache.close()
        logger.error(t("scan_no_files"))
        raise typer.Exit(1)

    if result.noise_repaired > 0:
        logger.debug(t("scan_status_knn_suffix", count=result.noise_repaired))
    if result.clusters_merged > 0:
        logger.debug(t("scan_status_merged_suffix", count=result.clusters_merged))

    # 关闭缓存
    if cache:
        cache.close()

    # 构建预览
    preview = OrganizePreview(
        source_folder=folder,
        target_folder=target,
    )

    # 按簇分组文件
    for label in set(result.labels):
        if label == -1:
            # 噪音文件
            noise_files = [
                f
                for f, lbl in zip(result.files, result.labels, strict=False)
                if lbl == -1
            ]
            preview.add_noise(noise_files)
        else:
            cluster_name = result.cluster_names.get(label, f"Cluster_{label}")
            cluster_files = [
                f
                for f, lbl in zip(result.files, result.labels, strict=False)
                if lbl == label
            ]
            preview.add_cluster(cluster_name, cluster_files)

    # 显示预览
    preview.display(logger.console)

    # 根据模式执行
    if output_script:
        preview.generate_script(output_script)
        logger.success(t("organize_script_generated", path=output_script))
        logger.print(t("organize_script_hint"))
    elif execute:
        logger.print("\n")
        confirm = typer.confirm(t("organize_confirm"))
        if confirm:
            success, failed = preview.execute(dry_run=False)
            logger.success(t("organize_done", success=success, failed=failed))
        else:
            logger.print(t("organize_cancelled"))
    else:
        # dry_run 模式，已经显示预览
        logger.print(f"\n[dim]{t('organize_dry_run_hint')}[/dim]")


@cache_app.command("clear", help=t("cli_help_cache_clear"))
def cache_clear(ctx: typer.Context):
    """Clear all cache."""
    logger = get_logger()
    config = _get_app_config(ctx)
    cache = FileCache(
        cache_dir=config.cache.directory,
        max_size_gb=config.cache.max_size_gb,
    )

    count = cache.clear()
    cache.close()

    logger.success(t("cache_cleared", count=count))


@cache_app.command("clear-vlm", help=t("cli_help_cache_clear_vlm"))
def cache_clear_vlm(ctx: typer.Context):
    """Clear VLM cache only."""
    logger = get_logger()
    config = _get_app_config(ctx)
    cache = FileCache(
        cache_dir=config.cache.directory,
        max_size_gb=config.cache.max_size_gb,
    )

    count = cache.clear_vlm_cache()
    cache.close()

    logger.success(t("cache_vlm_cleared", count=count))


@cache_app.command("status", help=t("cli_help_cache_status"))
def cache_status(ctx: typer.Context):
    """View cache status."""
    logger = get_logger()
    config = _get_app_config(ctx)
    cache = FileCache(
        cache_dir=config.cache.directory,
        max_size_gb=config.cache.max_size_gb,
    )

    stats = cache.get_stats()
    cache.close()

    logger.print(f"[bold]{t('cache_status_title')}[/bold]\n")
    logger.print(f"[bold]{t('cache_db_path')}[/bold] {stats['db_path']}")
    logger.print(f"[bold]{t('cache_total_entries')}[/bold] {stats['total_entries']}")
    logger.print(f"[bold]{t('cache_with_embedding')}[/bold] {stats['with_embedding']}")
    logger.print(f"[bold]{t('cache_with_vlm')}[/bold] {stats['with_vlm']}")
    logger.print(f"[bold]{t('cache_vlm_version')}[/bold] v{stats['vlm_cache_version']}")
    logger.print(f"[bold]{t('cache_unique_hashes')}[/bold] {stats['unique_hashes']}")
    logger.print(f"[bold]{t('cache_db_size')}[/bold] {stats['db_size_mb']:.2f} MB")


@setup_app.command("docling-pdf", help=t("cli_help_setup_docling_pdf"))
def setup_docling_pdf(
    force: Annotated[
        bool,
        typer.Option("--force", help=t("cli_help_setup_docling_pdf_force")),
    ] = False,
    progress: Annotated[
        bool,
        typer.Option("--progress", help=t("cli_help_setup_docling_pdf_progress")),
    ] = False,
):
    """Install local Docling PDF models required by DITE."""
    logger = get_logger()
    target_dir = get_docling_pdf_artifacts_path()
    logger.print(t("setup_docling_pdf_start", path=target_dir))

    try:
        download_docling_pdf_models(
            output_dir=target_dir,
            force=force,
            progress=progress,
        )
    except Exception as exc:
        logger.error(t("setup_docling_pdf_failed", error=exc))
        raise typer.Exit(1) from exc

    if not has_docling_pdf_artifacts(target_dir):
        logger.error(t("setup_docling_pdf_incomplete", path=target_dir))
        raise typer.Exit(1)

    logger.success(t("setup_docling_pdf_done", path=target_dir))


if __name__ == "__main__":
    app()
