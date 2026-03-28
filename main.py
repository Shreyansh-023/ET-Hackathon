from __future__ import annotations

import argparse
import re
import sys

from typer.testing import CliRunner

from src.cli import app


def _run_stage(runner: CliRunner, stage_args: list[str], stage_name: str) -> str:
    result = runner.invoke(app, stage_args)
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n")

    if result.exit_code != 0:
        raise RuntimeError(f"{stage_name} failed with exit code {result.exit_code}")

    return output


def _extract_job_id(ingest_output: str) -> str:
    match = re.search(r"job_id=([0-9]{8}-[0-9]{6}-[a-f0-9]+)", ingest_output)
    if not match:
        raise RuntimeError("Could not extract job_id from ingest output")
    return match.group(1)


def run_pipeline(file_or_url: str, *, title: str = "", dry_run: bool = False) -> str:
    runner = CliRunner()

    ingest_args = ["ingest", file_or_url]
    if title:
        ingest_args.extend(["--title", title])
    if dry_run:
        ingest_args.append("--dry-run")

    ingest_output = _run_stage(runner, ingest_args, "ingest")
    job_id = _extract_job_id(ingest_output)

    shared_flags: list[str] = ["--dry-run"] if dry_run else []

    _run_stage(runner, ["plan", job_id, *shared_flags], "plan")
    _run_stage(runner, ["assets", job_id, *shared_flags], "assets")
    _run_stage(runner, ["audio", job_id, *shared_flags], "audio")
    _run_stage(runner, ["render", job_id, *shared_flags], "render")
    _run_stage(runner, ["export", job_id, *shared_flags], "export")

    return job_id


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full news-to-video pipeline")
    parser.add_argument("file_or_url", help="Raw text, path to text file, or URL")
    parser.add_argument("--title", default="", help="Optional article title")
    parser.add_argument("--dry-run", action="store_true", help="Run without paid provider calls")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        job_id = run_pipeline(args.file_or_url, title=args.title, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Pipeline completed successfully for job_id={job_id}")


if __name__ == "__main__":
    main()
