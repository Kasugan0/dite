"""CLI-facing report builders."""

from .cluster import build_report, print_report
from .move import apply_clustering_to_preview, finalize_organize_run
from .pdf import (
    format_bool,
    format_fallback_vlm,
    format_sample_limit,
    label_pdf_reason,
    label_selected_source,
    label_source_profile,
    print_extraction_detail_table,
    select_problem_file_reports,
)

__all__ = [
    "build_report",
    "print_report",
    "apply_clustering_to_preview",
    "finalize_organize_run",
    "format_bool",
    "format_fallback_vlm",
    "format_sample_limit",
    "label_pdf_reason",
    "label_selected_source",
    "label_source_profile",
    "print_extraction_detail_table",
    "select_problem_file_reports",
]
