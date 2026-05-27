"""Organize command helpers."""

from __future__ import annotations

from dite.app.i18n import t


def apply_clustering_to_preview(
    *,
    preview,
    files,
    labels,
    cluster_names,
) -> None:
    """Populate an organize preview from pipeline clustering output."""
    for label in set(labels):
        if label == -1:
            noise_files = [
                file
                for file, item_label in zip(files, labels, strict=False)
                if item_label == label
            ]
            preview.add_noise(noise_files)
            continue

        cluster_name = cluster_names.get(label, f"Cluster_{label}")
        cluster_files = [
            file
            for file, item_label in zip(files, labels, strict=False)
            if item_label == label
        ]
        preview.add_cluster(cluster_name, cluster_files)


def finalize_organize_run(
    *,
    preview,
    logger,
    output_script,
    execute,
) -> None:
    """Display and optionally execute an organize preview."""
    preview.display(logger.console)

    if output_script:
        preview.generate_script(output_script)
        logger.success(t("organize_script_generated", path=output_script))
        logger.print(t("organize_script_hint"))
        return

    if execute:
        logger.print("\n")
        confirm = __import__("typer").confirm(t("organize_confirm"))
        if confirm:
            success, failed = preview.execute(dry_run=False)
            logger.success(t("organize_done", success=success, failed=failed))
        else:
            logger.print(t("organize_cancelled"))
        return

    logger.print(f"\n[dim]{t('organize_dry_run_hint')}[/dim]")
