from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_MIRROR = "https://cdn.npmmirror.com/binaries"


class InstallError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install Playwright Chromium from China-friendly mirrors."
    )
    parser.add_argument(
        "--mirror",
        default=DEFAULT_MIRROR,
        help=f"Mirror base URL. Defaults to {DEFAULT_MIRROR}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even when the expected browser files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without changing the browser cache.",
    )
    args = parser.parse_args()

    try:
        install(args.mirror.rstrip("/"), force=args.force, dry_run=args.dry_run)
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def install(mirror: str, *, force: bool, dry_run: bool) -> None:
    if sys.platform != "win32":
        raise InstallError(
            "This helper currently supports Windows. On other systems, set "
            "PLAYWRIGHT_DOWNLOAD_HOST or use a local artifact repository."
        )

    browsers = _read_playwright_browsers()
    chromium = _browser(browsers, "chromium")
    headless = _browser(browsers, "chromium-headless-shell")
    ffmpeg = _browser(browsers, "ffmpeg")
    winldd = _browser(browsers, "winldd")
    cache_dir = _playwright_cache_dir()

    tasks = [
        DownloadTask(
            name="chromium",
            url=(
                f"{mirror}/chrome-for-testing/{chromium['browserVersion']}/"
                "win64/chrome-win64.zip"
            ),
            install_dir=cache_dir / f"chromium-{chromium['revision']}",
            expected_file=Path("chrome-win64/chrome.exe"),
        ),
        DownloadTask(
            name="chromium-headless-shell",
            url=(
                f"{mirror}/chrome-for-testing/{headless['browserVersion']}/"
                "win64/chrome-headless-shell-win64.zip"
            ),
            install_dir=cache_dir / f"chromium_headless_shell-{headless['revision']}",
            expected_file=Path("chrome-headless-shell-win64/chrome-headless-shell.exe"),
        ),
        DownloadTask(
            name="ffmpeg",
            url=f"{mirror}/playwright/builds/ffmpeg/{ffmpeg['revision']}/ffmpeg-win64.zip",
            install_dir=cache_dir / f"ffmpeg-{ffmpeg['revision']}",
            expected_file=Path("ffmpeg-win64.exe"),
        ),
        DownloadTask(
            name="winldd",
            url=f"{mirror}/playwright/builds/winldd/{winldd['revision']}/winldd-win64.zip",
            install_dir=cache_dir / f"winldd-{winldd['revision']}",
            expected_file=Path("PrintDeps.exe"),
        ),
    ]

    print(f"Playwright browser cache: {cache_dir}")
    for task in tasks:
        task.install(force=force, dry_run=dry_run)


class DownloadTask:
    def __init__(
        self, *, name: str, url: str, install_dir: Path, expected_file: Path
    ) -> None:
        self.name = name
        self.url = url
        self.install_dir = install_dir
        self.expected_file = expected_file

    def install(self, *, force: bool, dry_run: bool) -> None:
        expected_path = self.install_dir / self.expected_file
        if expected_path.exists() and not force:
            print(f"[skip] {self.name}: {expected_path}")
            _write_markers(self.install_dir)
            return

        print(f"[download] {self.name}: {self.url}")
        print(f"[target]   {self.install_dir}")
        if dry_run:
            return

        self.install_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pw-browser-") as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / f"{self.name}.zip"
            extract_path = temp_path / "extract"
            extract_path.mkdir()
            _download(self.url, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_path)

            if self.install_dir.exists():
                shutil.rmtree(self.install_dir)
            shutil.move(str(extract_path), str(self.install_dir))

        if not expected_path.exists():
            raise InstallError(
                f"{self.name} installed but expected file was not found: "
                f"{expected_path}"
            )
        _write_markers(self.install_dir)
        print(f"[ok]       {self.name}: {expected_path}")


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "playwright-ui"})
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if status >= 400:
            raise InstallError(f"download failed with HTTP {status}: {url}")
        with target.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)


def _write_markers(install_dir: Path) -> None:
    (install_dir / "INSTALLATION_COMPLETE").touch()
    (install_dir / "DEPENDENCIES_VALIDATED").touch()


def _read_playwright_browsers() -> list[dict]:
    try:
        import playwright
    except ImportError as exc:
        raise InstallError("playwright is not installed. Run `uv sync` first.") from exc

    browsers_json = (
        Path(playwright.__file__).resolve().parent
        / "driver"
        / "package"
        / "browsers.json"
    )
    if not browsers_json.exists():
        raise InstallError(f"browsers.json not found: {browsers_json}")
    return json.loads(browsers_json.read_text(encoding="utf-8"))["browsers"]


def _browser(browsers: list[dict], name: str) -> dict:
    for browser in browsers:
        if browser.get("name") == name:
            return browser
    raise InstallError(f"browser metadata not found: {name}")


def _playwright_cache_dir() -> Path:
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured and configured != "0":
        return Path(configured).expanduser().resolve()
    return Path.home() / "AppData" / "Local" / "ms-playwright"


if __name__ == "__main__":
    raise SystemExit(main())
