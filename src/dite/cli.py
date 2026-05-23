"""CLI 命令定义"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import click
import numpy as np
import typer
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from rich.table import Table
from typer.core import TyperGroup

from dite import __version__
from dite.cache import FileCache
from dite.config import Config, load_config
from dite.core.clusterer import _build_cluster_debug_labels
from dite.core.embedder import get_embedding_cache_version
from dite.i18n import set_locale, t
from dite.utils.api_runtime import build_sync_openai_client
from dite.utils.llm import format_api_error
from dite.utils.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from dite.core.clusterer import ClusterMetrics
    from dite.core.organizer import OrganizePreview
    from dite.core.pipeline import (
        ExtractionFileReport,
        PipelineOptions,
        PipelineResult,
        PipelineService,
    )


_HELP_KEY_PREFIX = "__dite_help__:"
PipelineOptions = None
PipelineService = None
OrganizePreview = None
download_docling_pdf_models = None
get_docling_pdf_artifacts_path = None
has_docling_pdf_artifacts = None


def _help_key(key: str) -> str:
    return f"{_HELP_KEY_PREFIX}{key}"


def _translate_help_text(text: str | None) -> str | None:
    if text is None or not text.startswith(_HELP_KEY_PREFIX):
        return text
    return t(text.removeprefix(_HELP_KEY_PREFIX))


def _sync_command_help(command: click.Command) -> None:
    command.help = _translate_help_text(command.help)
    command.short_help = _translate_help_text(command.short_help)
    command.epilog = _translate_help_text(command.epilog)

    for param in command.params:
        if hasattr(param, "help"):
            param.help = _translate_help_text(param.help)

    if isinstance(command, click.Group):
        for subcommand in command.commands.values():
            _sync_command_help(subcommand)


class LocalizedTyperGroup(TyperGroup):
    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        cfg = load_config()
        set_locale(cfg.i18n.locale)
        _sync_command_help(self)
        return super().make_context(info_name, args, parent=parent, **extra)


def _install_cli_warning_filters() -> None:
    warnings.filterwarnings("ignore", message=".*cannot be loaded by Pillow.*")
    warnings.filterwarnings("ignore", message=".*cannot find loader.*")


def _get_pipeline_options_cls():
    global PipelineOptions
    if PipelineOptions is None:
        from dite.core.pipeline import PipelineOptions as _PipelineOptions

        PipelineOptions = _PipelineOptions
    return PipelineOptions


def _get_pipeline_service_cls():
    global PipelineService
    if PipelineService is None:
        from dite.core.pipeline import PipelineService as _PipelineService

        PipelineService = _PipelineService
    return PipelineService


def _get_organize_preview_cls():
    global OrganizePreview
    if OrganizePreview is None:
        from dite.core.organizer import OrganizePreview as _OrganizePreview

        OrganizePreview = _OrganizePreview
    return OrganizePreview


def _get_docling_helpers():
    global download_docling_pdf_models
    global get_docling_pdf_artifacts_path
    global has_docling_pdf_artifacts
    if (
        download_docling_pdf_models is None
        or get_docling_pdf_artifacts_path is None
        or has_docling_pdf_artifacts is None
    ):
        from dite.extractors.docling import (
            download_docling_pdf_models as _download_docling_pdf_models,
        )
        from dite.extractors.docling import (
            get_docling_pdf_artifacts_path as _get_docling_pdf_artifacts_path,
        )
        from dite.extractors.docling import (
            has_docling_pdf_artifacts as _has_docling_pdf_artifacts,
        )

        if download_docling_pdf_models is None:
            download_docling_pdf_models = _download_docling_pdf_models
        if get_docling_pdf_artifacts_path is None:
            get_docling_pdf_artifacts_path = _get_docling_pdf_artifacts_path
        if has_docling_pdf_artifacts is None:
            has_docling_pdf_artifacts = _has_docling_pdf_artifacts
    return (
        download_docling_pdf_models,
        get_docling_pdf_artifacts_path,
        has_docling_pdf_artifacts,
    )


# 创建 CLI 应用
app = typer.Typer(
    name="dite",
    cls=LocalizedTyperGroup,
    help=_help_key("cli_description"),
    no_args_is_help=True,
)

# 缓存子命令
cache_app = typer.Typer(
    cls=LocalizedTyperGroup,
    help=_help_key("cli_help_cache_group"),
)
app.add_typer(cache_app, name="cache")

# 环境准备子命令
setup_app = typer.Typer(
    cls=LocalizedTyperGroup,
    help=_help_key("cli_help_setup_group"),
)
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
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    color: Annotated[
        bool,
        typer.Option(
            "--color",
            help=_help_key("cli_help_color"),
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help=_help_key("cli_help_version"),
        ),
    ] = False,
):
    """DITE - Multimodal document intelligent clustering tool"""
    _install_cli_warning_filters()
    cfg = load_config()
    set_locale(cfg.i18n.locale)
    setup_logging(verbose=verbose, quiet=quiet, force_color=color)
    ctx.obj = {"config": cfg}


def _get_app_config(ctx: typer.Context | None) -> Config:
    if ctx is None or not ctx.obj or "config" not in ctx.obj:
        return load_config()
    return ctx.obj["config"]


def _get_client(config: Config) -> OpenAI:
    """获取 OpenAI 兼容客户端"""
    return build_sync_openai_client(config)


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


def _label_selected_source(source: str) -> str:
    labels = {
        "primary": t("extract_source_primary"),
        "vlm_cache": t("extract_source_vlm_cache"),
        "vlm_api": t("extract_source_vlm_api"),
    }
    return labels.get(source, source)


def _label_source_profile(profile: str | None) -> str:
    if profile is None:
        return "-"
    labels = {
        "native_text": t("extract_profile_native_text"),
        "weak_text": t("extract_profile_weak_text"),
        "scanned_image": t("extract_profile_scanned_image"),
        "mixed_pdf": t("extract_profile_mixed_pdf"),
        "parser_timeout_or_broken": t("extract_profile_parser_timeout_or_broken"),
    }
    return labels.get(profile, profile)


def _label_pdf_reason(reason: str | None) -> str:
    if not reason:
        return "-"
    labels = {
        "cached_vlm_available": t("pdf_check_reason_cached_vlm_available"),
        "effective_text_below_threshold": t(
            "pdf_check_reason_effective_text_below_threshold"
        ),
        "extractor_failed": t("pdf_check_reason_extractor_failed"),
        "glyph_noise_dominates": t("pdf_check_reason_glyph_noise_dominates"),
        "no_effective_text": t("pdf_check_reason_no_effective_text"),
        "parser_timeout_or_broken": t("extract_profile_parser_timeout_or_broken"),
        "text_with_glyph_noise": t("pdf_check_reason_text_with_glyph_noise"),
        "usable_text_layer": t("pdf_check_reason_usable_text_layer"),
        "vlm_api_allowed": t("pdf_check_reason_vlm_api_allowed"),
        "vlm_fallback_unavailable": t("pdf_check_reason_vlm_fallback_unavailable"),
    }
    return labels.get(reason, reason)


def _format_bool(value: bool) -> str:
    return t("label_yes") if value else t("label_no")


def _format_sample_limit(limit: int | None) -> str:
    return "-" if limit is None else str(limit)


def _format_fallback_vlm(report: ExtractionFileReport) -> str:
    return (
        f"{_format_bool(report.fallback_needed)}; "
        f"{report.vlm_api_page_calls}/{_format_sample_limit(report.sample_page_limit)}"
    )


def _select_problem_file_reports(
    file_reports: list[ExtractionFileReport],
) -> list[ExtractionFileReport]:
    problem_profiles = {
        "weak_text",
        "scanned_image",
        "parser_timeout_or_broken",
    }
    return [
        report
        for report in file_reports
        if (
            not report.primary_success
            or report.selected_source != "primary"
            or report.source_profile in problem_profiles
        )
    ]


def _print_extraction_detail_table(file_reports: list[ExtractionFileReport]) -> None:
    logger = get_logger()
    problem_reports = _select_problem_file_reports(file_reports)
    if not problem_reports:
        return
    table = Table(title=t("pdf_check_verbose_table_title"))
    table.add_column(t("pdf_check_table_file"), style="path")
    table.add_column(t("pdf_check_table_primary_extractor"))
    table.add_column(t("pdf_check_table_source_profile"))
    table.add_column(t("pdf_check_table_reason"))
    table.add_column(t("pdf_check_table_selected_source"))
    table.add_column(t("pdf_check_table_lengths"), justify="right")
    table.add_column(t("pdf_check_table_fallback_vlm"), justify="right")
    for report in problem_reports:
        table.add_row(
            report.file.name,
            report.primary_extractor,
            _label_source_profile(report.source_profile),
            _label_pdf_reason(report.primary_error or report.source_reason),
            _label_selected_source(report.selected_source),
            f"{report.source_effective_length}->{report.final_effective_length}",
            _format_fallback_vlm(report),
        )
    logger.print_table(table)


def _build_report(
    files: list[Path],
    contents: list[str],
    labels: np.ndarray,
    cluster_names: dict[int, str],
    cluster_metrics: ClusterMetrics | None = None,
) -> dict:
    """构建聚类报告"""
    from dite.core.clusterer import ClusterMetrics

    metrics = cluster_metrics or ClusterMetrics()
    clusters = {}
    debug_labels = _build_cluster_debug_labels(labels)
    noise = []
    failed_count = 0

    for file, content, label in zip(files, contents, labels, strict=True):
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
            "initial_num_clusters": metrics.initial_clusters,
            "initial_num_noise": metrics.initial_noise,
            "noise_repaired": metrics.noise_repaired,
            "small_clusters_merged": metrics.small_clusters_merged,
            "name_clusters_merged": metrics.name_clusters_merged,
            "total_clusters_merged": (
                metrics.small_clusters_merged + metrics.name_clusters_merged
            ),
            "small_cluster_merge_candidates": metrics.small_cluster_merge_candidates,
            "small_cluster_merge_skipped": metrics.small_cluster_merge_skipped,
        },
        "clusters": [clusters[label] for label in sorted(clusters)],
        "cluster_diagnostics": {
            "small_cluster_merge_max_similarity": (
                metrics.small_cluster_merge_max_similarity
            ),
            "small_cluster_merge_events": [
                event.__dict__.copy() for event in metrics.small_cluster_merge_events
            ],
            "small_cluster_skip_events": [
                event.__dict__.copy() for event in metrics.small_cluster_skip_events
            ],
        },
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

    if summary.get("small_clusters_merged", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_merge_label')}[/bold] "
            f"{t('cluster_small_merged', count=summary['small_clusters_merged'])}"
        )
    if summary.get("name_clusters_merged", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_merge_label')}[/bold] "
            f"{t('cluster_name_merged', count=summary['name_clusters_merged'])}"
        )
    if summary.get("total_clusters_merged", 0) > 0:
        logger.print(
            f"[bold]{t('cluster_merge_label')}[/bold] "
            f"{t('cluster_total_merged', count=summary['total_clusters_merged'])}"
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


@app.command(help=_help_key("scan_description"))
def scan(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_folder_scan")),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help=_help_key("cli_help_output_report")),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=_help_key("cli_help_disable_cache")),
    ] = False,
    no_knn_repair: Annotated[
        bool,
        typer.Option(
            "--no-knn-repair",
            help=_help_key("cli_help_disable_knn_repair"),
        ),
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

    pipeline_options_cls = _get_pipeline_options_cls()
    pipeline_service_cls = _get_pipeline_service_cls()

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
    pipeline = pipeline_service_cls(client=client, config=config, cache=cache)
    result = _run_pipeline_or_exit(
        pipeline,
        folder,
        pipeline_options_cls(
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

    extraction = result.extraction
    cache_msg_parts = []
    if extraction.doc_cache_hits:
        cache_msg_parts.append(t("cache_docling_hit", count=extraction.doc_cache_hits))
    if extraction.vlm_cache_hits:
        cache_msg_parts.append(t("cache_vlm_hit", count=extraction.vlm_cache_hits))
    if extraction.selected_vlm_files:
        cache_msg_parts.append(
            t("cache_vlm_fallback", count=extraction.selected_vlm_files)
        )
    if extraction.duplicate_count:
        cache_msg_parts.append(t("cache_duplicate", count=extraction.duplicate_count))
    cache_msg = f" ({', '.join(cache_msg_parts)})" if cache_msg_parts else ""
    logger.status(f"{t('scan_extraction_done')}{cache_msg}")

    if logger.verbose:
        logger.debug(
            t(
                "scan_extraction_verbose",
                primary_failures=extraction.primary_failures,
                source_fallback_needed=extraction.source_fallback_needed,
                vlm_api_page_calls=extraction.vlm_api_page_calls,
            )
        )

    if extraction.duplicate_groups and logger.verbose:
        logger.debug(t("debug_duplicate_groups"))
        for file_hash, file_list in extraction.duplicate_groups.items():
            logger.debug(t("debug_duplicate_group_hash", hash=file_hash[:12]))
            for f in file_list:
                logger.debug(t("debug_duplicate_group_file", name=Path(f).name))

    logger.status(t("scan_vectorizing_done", dim=result.embeddings.shape[1]))

    unique_labels = set(result.labels)
    n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
    n_noise = int(np.sum(result.labels == -1))
    status_msg = t("scan_clustering_done", clusters=n_clusters, noise=n_noise)
    if result.noise_repaired > 0:
        status_msg += f" [{t('scan_status_knn_suffix', count=result.noise_repaired)}]"
    logger.status(status_msg)

    status_msg = t("scan_naming_done")
    if result.cluster_metrics.small_clusters_merged > 0:
        small_merged = result.cluster_metrics.small_clusters_merged
        status_msg += (
            f" [{t('scan_status_small_merged_suffix', count=small_merged)}]"
        )
    if result.cluster_metrics.name_clusters_merged > 0:
        name_merged = result.cluster_metrics.name_clusters_merged
        status_msg += (
            f" [{t('scan_status_name_merged_suffix', count=name_merged)}]"
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
        result.cluster_metrics,
    )

    # 保存 JSON
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.success(t("scan_report_saved", path=output))

    # 打印报告
    _print_report(report)


@app.command("pdf-check", help=_help_key("pdf_check_description"))
def pdf_check(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_folder_pdf_check")),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=_help_key("cli_help_disable_cache")),
    ] = False,
    cached_vlm_only: Annotated[
        bool,
        typer.Option(
            "--cached-vlm-only",
            help=_help_key("cli_help_cached_vlm_only"),
        ),
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

    from dite.core.scanner import scan_files

    pipeline_options_cls = _get_pipeline_options_cls()
    pipeline_service_cls = _get_pipeline_service_cls()

    files = scan_files(folder, config=config, extensions={".pdf"})
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
    pipeline = pipeline_service_cls(client=client, config=config, cache=cache)

    try:
        result = pipeline.extract_files(
            files,
            pipeline_options_cls(
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

    extraction = result.extraction
    threshold = config.processing.vlm_fallback_threshold
    weak_files = [
        (report.file, report.final_effective_length)
        for report in result.file_reports
        if report.final_effective_length < threshold
    ]
    empty_count = sum(1 for _file, length in weak_files if length == 0)

    logger.status(t("pdf_check_found_pdfs", count=len(result.files)))
    logger.status(
        t(
            "pdf_check_done",
            doc_cache_hits=extraction.doc_cache_hits,
            vlm_cache_hits=extraction.vlm_cache_hits,
            primary_failures=extraction.primary_failures,
            source_fallback_needed=extraction.source_fallback_needed,
            selected_vlm_files=extraction.selected_vlm_files,
            vlm_api_page_calls=extraction.vlm_api_page_calls,
            duplicates=extraction.duplicate_count,
            weak=len(weak_files),
            empty=empty_count,
        )
    )
    logger.print(t("pdf_check_note"))

    if logger.verbose and _select_problem_file_reports(result.file_reports):
        _print_extraction_detail_table(result.file_reports)

    if weak_files:
        table = Table(title=t("pdf_check_weak_table_title"))
        table.add_column(t("pdf_check_table_file"), style="path")
        table.add_column(t("pdf_check_table_reason"))
        table.add_column(t("pdf_check_table_final_effective_length"), justify="right")
        weak_report_by_file = {report.file: report for report in result.file_reports}
        for file, length in weak_files:
            report = weak_report_by_file[file]
            table.add_row(
                file.name,
                _label_pdf_reason(report.primary_error or report.source_reason),
                str(length),
            )
        logger.print_table(table)
        logger.error(t("pdf_check_failed", count=len(weak_files)))
        raise typer.Exit(1)

    logger.success(t("pdf_check_passed", count=len(result.files)))


@app.command(help=_help_key("organize_description"))
def organize(
    ctx: typer.Context,
    folder: Annotated[
        Path,
        typer.Argument(help=_help_key("cli_help_folder_organize")),
    ],
    target: Annotated[
        Path | None,
        typer.Option("--target", "-t", help=_help_key("cli_help_target_folder")),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help=_help_key("cli_help_verbose")),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help=_help_key("cli_help_quiet")),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help=_help_key("cli_help_preview_mode")),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help=_help_key("cli_help_execute_move")),
    ] = False,
    output_script: Annotated[
        Path | None,
        typer.Option("--output-script", help=_help_key("cli_help_output_script")),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help=_help_key("cli_help_disable_cache")),
    ] = False,
    no_knn_repair: Annotated[
        bool,
        typer.Option(
            "--no-knn-repair",
            help=_help_key("cli_help_disable_knn_repair"),
        ),
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

    organize_preview_cls = _get_organize_preview_cls()
    pipeline_options_cls = _get_pipeline_options_cls()
    pipeline_service_cls = _get_pipeline_service_cls()

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
    pipeline = pipeline_service_cls(client=client, config=config, cache=cache)
    result = _run_pipeline_or_exit(
        pipeline,
        folder,
        pipeline_options_cls(
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
    if result.cluster_metrics.small_clusters_merged > 0:
        logger.debug(
            t(
                "scan_status_small_merged_suffix",
                count=result.cluster_metrics.small_clusters_merged,
            )
        )
    if result.cluster_metrics.name_clusters_merged > 0:
        logger.debug(
            t(
                "scan_status_name_merged_suffix",
                count=result.cluster_metrics.name_clusters_merged,
            )
        )

    # 关闭缓存
    if cache:
        cache.close()

    # 构建预览
    preview = organize_preview_cls(
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


@cache_app.command("clear", help=_help_key("cli_help_cache_clear"))
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


@cache_app.command("clear-vlm", help=_help_key("cli_help_cache_clear_vlm"))
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


@cache_app.command("status", help=_help_key("cli_help_cache_status"))
def cache_status(ctx: typer.Context):
    """View cache status."""
    logger = get_logger()
    config = _get_app_config(ctx)
    cache = FileCache(
        cache_dir=config.cache.directory,
        max_size_gb=config.cache.max_size_gb,
    )

    embedding_cache_version = get_embedding_cache_version(config.models.embedding)
    stats = cache.get_stats(required_embedding_version=embedding_cache_version)
    cache.close()

    logger.print(f"[bold]{t('cache_status_title')}[/bold]\n")
    logger.print(f"[bold]{t('cache_db_path')}[/bold] {stats['db_path']}")
    logger.print(f"[bold]{t('cache_total_entries')}[/bold] {stats['total_entries']}")
    logger.print(f"[bold]{t('cache_with_embedding')}[/bold] {stats['with_embedding']}")
    logger.print(
        f"[bold]{t('cache_current_embedding')}[/bold] {stats['current_embeddings']}"
    )
    logger.print(
        f"[bold]{t('cache_stale_embedding')}[/bold] {stats['stale_embeddings']}"
    )
    logger.print(
        f"[bold]{t('cache_embedding_version')}[/bold] "
        f"{stats['current_embedding_version']}"
    )
    logger.print(f"[bold]{t('cache_with_vlm')}[/bold] {stats['with_vlm']}")
    logger.print(f"[bold]{t('cache_vlm_version')}[/bold] v{stats['vlm_cache_version']}")
    logger.print(f"[bold]{t('cache_unique_hashes')}[/bold] {stats['unique_hashes']}")
    logger.print(f"[bold]{t('cache_db_size')}[/bold] {stats['db_size_mb']:.2f} MB")


@setup_app.command("docling-pdf", help=_help_key("cli_help_setup_docling_pdf"))
def setup_docling_pdf(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=_help_key("cli_help_setup_docling_pdf_force"),
        ),
    ] = False,
    progress: Annotated[
        bool,
        typer.Option(
            "--progress",
            help=_help_key("cli_help_setup_docling_pdf_progress"),
        ),
    ] = False,
):
    """Install local Docling PDF models required by DITE."""
    logger = get_logger()
    (
        download_docling_pdf_models_fn,
        get_docling_pdf_artifacts_path_fn,
        has_docling_pdf_artifacts_fn,
    ) = _get_docling_helpers()
    target_dir = get_docling_pdf_artifacts_path_fn()
    logger.print(t("setup_docling_pdf_start", path=target_dir))

    try:
        download_docling_pdf_models_fn(
            output_dir=target_dir,
            force=force,
            progress=progress,
        )
    except Exception as exc:
        logger.error(t("setup_docling_pdf_failed", error=exc))
        raise typer.Exit(1) from exc

    if not has_docling_pdf_artifacts_fn(target_dir):
        logger.error(t("setup_docling_pdf_incomplete", path=target_dir))
        raise typer.Exit(1)

    logger.success(t("setup_docling_pdf_done", path=target_dir))


if __name__ == "__main__":
    app()
