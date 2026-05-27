"""PDF check reporting helpers."""

from __future__ import annotations

from rich.table import Table

from dite.app.i18n import t
from dite.util.log import get_logger


def label_selected_source(source: str) -> str:
    labels = {
        "primary": t("extract_source_primary"),
        "vlm_cache": t("extract_source_vlm_cache"),
        "vlm_api": t("extract_source_vlm_api"),
    }
    return labels.get(source, source)


def label_source_profile(profile: str | None) -> str:
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


def label_pdf_reason(reason: str | None) -> str:
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


def format_bool(value: bool) -> str:
    return t("label_yes") if value else t("label_no")


def format_sample_limit(limit: int | None) -> str:
    return "-" if limit is None else str(limit)


def format_fallback_vlm(report) -> str:
    return (
        f"{format_bool(report.fallback_needed)}; "
        f"{report.vlm_api_page_calls}/{format_sample_limit(report.sample_page_limit)}"
    )


def select_problem_file_reports(file_reports) -> list:
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


def print_extraction_detail_table(file_reports) -> None:
    logger = get_logger()
    problem_reports = select_problem_file_reports(file_reports)
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
            label_source_profile(report.source_profile),
            label_pdf_reason(report.primary_error or report.source_reason),
            label_selected_source(report.selected_source),
            f"{report.source_effective_length}->{report.final_effective_length}",
            format_fallback_vlm(report),
        )
    logger.print_table(table)
