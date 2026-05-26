"""日志和控制台输出工具"""

import logging
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from dite.i18n import t

# 自定义主题，为不同日志级别设置颜色
DITE_THEME = Theme(
    {
        "debug": "dim cyan",
        "info": "blue",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "status": "green",
        "duplicate": "dim magenta",  # 重复文件提示
        "hash": "dim",  # 哈希值显示
        "path": "cyan",  # 路径显示
    }
)


# LogLevel 到 Python logging 级别的映射
LEVEL_TO_LOGGING = {
    0: logging.DEBUG,  # DEBUG
    1: logging.INFO,  # INFO/SUCCESS
    2: logging.WARNING,  # WARNING
    3: logging.ERROR,  # ERROR
}


class LogLevel(Enum):
    """日志级别"""

    DEBUG = 0
    INFO = 1
    SUCCESS = 1  # SUCCESS 等同于 INFO
    WARNING = 2
    ERROR = 3


class Logger:
    """增强的日志系统"""

    def __init__(
        self,
        verbose: bool = False,
        quiet: bool = False,
        force_color: bool = False,
    ):
        # force_terminal=True 强制启用颜色（即使输出被重定向）
        self.console = Console(
            theme=DITE_THEME,
            force_terminal=force_color if force_color else None,
        )
        self.verbose = verbose
        self.quiet = quiet
        self.force_color = force_color

        if quiet:
            self.min_level = LogLevel.ERROR
        elif verbose:
            self.min_level = LogLevel.DEBUG
        else:
            self.min_level = LogLevel.INFO

    def _should_log(self, level: LogLevel) -> bool:
        """判断是否应该输出该级别的日志"""
        return level.value >= self.min_level.value

    def debug(self, message: str, **kwargs: Any) -> None:
        """DEBUG 级别日志"""
        if self._should_log(LogLevel.DEBUG):
            self.console.print(
                f"[dim cyan]{t('log_debug_prefix')}[/dim cyan] {message}",
                **kwargs,
            )

    def info(self, message: str, **kwargs: Any) -> None:
        """INFO 级别日志"""
        if self._should_log(LogLevel.INFO):
            self.console.print(
                f"[blue]{t('log_info_prefix')}[/blue] {message}",
                **kwargs,
            )

    def warning(self, message: str, **kwargs: Any) -> None:
        """WARNING 级别日志"""
        if self._should_log(LogLevel.WARNING):
            self.console.print(
                f"[bold yellow]{t('log_warning_prefix')}[/bold yellow] {message}",
                **kwargs,
            )

    def error(self, message: str, **kwargs: Any) -> None:
        """ERROR 级别日志"""
        if self._should_log(LogLevel.ERROR):
            self.console.print(
                f"[bold red]{t('log_error_prefix')}[/bold red] {message}",
                **kwargs,
            )

    def success(self, message: str, **kwargs: Any) -> None:
        """SUCCESS 级别日志"""
        if self._should_log(LogLevel.SUCCESS):
            self.console.print(
                f"[bold green]{t('log_success_prefix')}[/bold green] {message}",
                **kwargs,
            )

    def print(self, message: str = "", **kwargs: Any) -> None:
        """直接打印（不受日志级别限制，用于标题、分隔线等）"""
        self.console.print(message, **kwargs)

    def print_table(self, table: Table) -> None:
        """打印表格（不受日志级别限制）"""
        self.console.print(table)

    def print_panel(
        self, content: str, title: str | None = None, **kwargs: Any
    ) -> None:
        """打印面板（不受日志级别限制）"""
        self.console.print(Panel(content, title=title, **kwargs))

    def status(self, message: str) -> None:
        """状态更新（带前缀标记，用于步骤完成提示）"""
        if self._should_log(LogLevel.INFO):
            self.console.print(f"[green]>[/green] {message}")


# 全局 Logger 实例
_logger: Logger | None = None

_DOCLING_LOGGER_NAMES = (
    "docling",
    "docling_core",
    "docling.document_converter",
    "docling.pipeline",
    "docling.backend",
    "docling.backend.mspowerpoint_backend",
    "docling.backend.msword_backend",
    "docling.datamodel",
)


def get_logger() -> Logger:
    """获取全局 Logger 实例"""
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger


def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    force_color: bool = False,
) -> Logger:
    """
    配置日志级别

    Args:
        verbose: 详细模式
        quiet: 静默模式
        force_color: 强制启用颜色输出（即使重定向到文件）

    Returns:
        配置好的 Logger 实例
    """
    global _logger
    _logger = Logger(verbose=verbose, quiet=quiet, force_color=force_color)

    _configure_third_party_logging()

    return _logger


def silence_docling_logging() -> None:
    """Silence raw Docling logs in both parent and child processes."""
    for logger_name in _DOCLING_LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)


def _configure_third_party_logging() -> None:
    """Configure third-party logging behavior used by the CLI."""
    silence_docling_logging()

    # 给 root logger 一个空 handler，避免第三方 error 日志落到 lastResort。
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())


# 向后兼容：保留 get_console 函数
def get_console() -> Console:
    """获取全局 Console 实例（向后兼容）"""
    return get_logger().console
