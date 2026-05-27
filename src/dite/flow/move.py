"""文件整理模块"""

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dite.app.i18n import t


@dataclass
class FileOperation:
    """文件操作"""

    type: Literal["move", "copy"]
    source: Path
    destination: Path
    status: Literal["pending", "success", "failed"] = "pending"
    error: str | None = None


@dataclass
class OrganizePreview:
    """整理预览"""

    source_folder: Path
    target_folder: Path
    operations: list[FileOperation] = field(default_factory=list)
    cluster_map: dict[str, list[Path]] = field(
        default_factory=dict
    )  # 簇名称 -> 文件列表
    noise_files: list[Path] = field(default_factory=list)  # 未分类的文件

    @staticmethod
    def _sanitize_cluster_name(cluster_name: str) -> str:
        """清洗簇名称，确保可安全用于目录名。"""
        name = cluster_name.strip()

        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "-")
        for i in range(32):
            name = name.replace(chr(i), "-")

        name = " ".join(name.split()).strip(" .")
        while "--" in name:
            name = name.replace("--", "-")

        if not name or name in {".", ".."}:
            name = "Cluster_Untitled"

        return name[:80]

    def _unique_destination(self, destination: Path) -> Path:
        """为目标路径分配唯一文件名，避免覆盖。"""
        existing_destinations = {op.destination for op in self.operations}
        candidate = destination
        counter = 1

        while candidate in existing_destinations or candidate.exists():
            candidate = destination.with_name(
                f"{destination.stem}_{counter}{destination.suffix}"
            )
            counter += 1

        return candidate

    def add_cluster(self, cluster_name: str, files: list[Path]) -> None:
        """添加一个簇的文件"""
        safe_cluster_name = self._sanitize_cluster_name(cluster_name)
        self.cluster_map.setdefault(safe_cluster_name, [])
        self.cluster_map[safe_cluster_name].extend(files)

        # 生成移动操作
        cluster_folder = self.target_folder / safe_cluster_name
        for file in files:
            destination = self._unique_destination(cluster_folder / file.name)
            self.operations.append(
                FileOperation(
                    type="move",
                    source=file,
                    destination=destination,
                )
            )

    def add_noise(self, files: list[Path]) -> None:
        """添加未分类的文件"""
        self.noise_files = files

    def display(self, console: Console) -> None:
        """显示预览"""
        total_files = len(self.operations) + len(self.noise_files)
        # 头部信息
        header = Panel(
            f"[bold]{t('organize_source_folder')}[/bold] {self.source_folder}\n"
            f"[bold]{t('organize_target_folder')}[/bold] {self.target_folder}\n"
            f"[bold]{t('organize_files_found')}[/bold] {total_files}\n"
            f"[bold]{t('organize_clusters_suggested')}[/bold] {len(self.cluster_map)}",
            title=t("organize_preview_title"),
            border_style="blue",
        )
        console.print(header)
        console.print()

        # 显示每个簇
        for cluster_name, files in self.cluster_map.items():
            console.print(
                f"[bold cyan]【{cluster_name}】[/bold cyan] "
                f"({t('organize_files_count', count=len(files))})"
            )

            table = Table(show_header=False, box=None, padding=(0, 2))
            for f in files[:5]:  # 最多显示 5 个
                table.add_row("  ├──", f.name)
            if len(files) > 5:
                table.add_row("  └──", t("organize_more_files", count=len(files) - 5))
            console.print(table)

            console.print(f"  [dim]{t('organize_move_to', name=cluster_name)}[/dim]")
            console.print()

        # 显示未分类文件
        if self.noise_files:
            console.print(
                f"[bold yellow]【{t('organize_uncategorized')}】[/bold yellow] "
                f"({t('organize_files_count', count=len(self.noise_files))})"
            )
            for f in self.noise_files[:3]:
                console.print(f"  • {f.name}")
            if len(self.noise_files) > 3:
                console.print(
                    f"  {t('organize_more_files', count=len(self.noise_files) - 3)}"
                )
            console.print(f"  [dim]{t('organize_keep_in_place')}[/dim]")

    def generate_script(self, output_path: Path) -> None:
        """
        生成可执行的 shell 脚本

        Args:
            output_path: 脚本输出路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "#!/bin/bash",
            f"# {t('organize_script_header')} - {timestamp}",
            f"# Source: {self.source_folder}",
            f"# Target: {self.target_folder}",
            "#",
            f"# {t('organize_script_usage')}",
            "#   1. Review this script",
            "#   2. chmod +x organize.sh",
            "#   3. ./organize.sh",
            "",
            "set -e  # Stop on error",
            "",
            f"# {t('organize_script_create_dirs')}",
        ]

        # 创建目录命令
        for cluster_name in self.cluster_map:
            dest_dir = self.target_folder / cluster_name
            lines.append(f'mkdir -p "{dest_dir}"')

        lines.append("")
        lines.append(f"# {t('organize_script_copy_files')}")

        # 复制命令
        for op in self.operations:
            lines.append(f'cp -p "{op.source}" "{op.destination}"')

        lines.append("")
        lines.append(f"# {t('organize_script_verify')}")
        lines.append("# (Uncomment the following lines to delete original files)")

        for op in self.operations:
            lines.append(f'# rm "{op.source}"')

        lines.append("")
        lines.append(f'echo "{t("organize_script_done", count=len(self.operations))}"')
        lines.append(f'echo "{t("organize_script_warning")}"')

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def execute(self, dry_run: bool = True) -> tuple[int, int]:
        """
        执行整理操作

        Args:
            dry_run: 如果为 True，只预览不执行

        Returns:
            (成功数, 失败数)
        """
        if dry_run:
            return len(self.operations), 0

        success = 0
        failed = 0

        # 创建目录
        for cluster_name in self.cluster_map:
            dest_dir = self.target_folder / cluster_name
            dest_dir.mkdir(parents=True, exist_ok=True)

        # 执行操作
        for op in self.operations:
            try:
                op.destination.parent.mkdir(parents=True, exist_ok=True)

                # 先复制
                shutil.copy2(op.source, op.destination)

                # 验证复制成功
                if (
                    op.destination.exists()
                    and op.destination.stat().st_size == op.source.stat().st_size
                ):
                    op.source.unlink()  # 删除原文件
                    op.status = "success"
                    success += 1
                else:
                    op.destination.unlink()  # 删除不完整的副本
                    op.status = "failed"
                    op.error = t("error_copy_failed")
                    failed += 1

            except Exception as e:
                op.status = "failed"
                op.error = str(e)
                failed += 1

        return success, failed
