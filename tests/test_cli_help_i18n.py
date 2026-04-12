import os
import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str], home: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "dite", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_help_uses_zh_cn_locale_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "i18n:",
                "  locale: zh-CN",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_cli(["--help"], tmp_path)

    assert result.returncode == 0
    assert "多模态文件智能聚类工具" in result.stdout


def test_scan_help_uses_zh_cn_locale_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "i18n:",
                "  locale: zh-CN",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_cli(["scan", "--help"], tmp_path)

    assert result.returncode == 0
    assert "要扫描的文件夹" in result.stdout


def test_help_creates_global_config_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    assert not config_path.exists()

    result = _run_cli(["--help"], tmp_path)

    assert result.returncode == 0
    assert config_path.exists()
