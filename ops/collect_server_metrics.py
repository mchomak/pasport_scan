#!/usr/bin/env python3
"""Collect privacy-safe host, Docker, and Tesseract resource metrics."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STOP = False
UNITS = {
    "B": 1,
    "kB": 1000,
    "KB": 1000,
    "KiB": 1024,
    "MB": 1000**2,
    "MiB": 1024**2,
    "GB": 1000**3,
    "GiB": 1024**3,
    "TB": 1000**4,
    "TiB": 1024**4,
}


def stop_handler(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_quantity(value: str) -> float:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*", value)
    if not match:
        return 0.0
    return float(match.group(1)) * UNITS.get(match.group(2), 1)


def read_host_metrics() -> dict[str, float | int]:
    meminfo: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="ascii") as source:
        for line in source:
            key, _, rest = line.partition(":")
            fields = rest.split()
            if fields:
                meminfo[key] = int(fields[0]) * 1024

    disk = shutil.disk_usage("/")
    return {
        "load_1m": os.getloadavg()[0],
        "load_5m": os.getloadavg()[1],
        "load_15m": os.getloadavg()[2],
        "memory_total_bytes": meminfo.get("MemTotal", 0),
        "memory_available_bytes": meminfo.get("MemAvailable", 0),
        "root_disk_total_bytes": disk.total,
        "root_disk_used_bytes": disk.used,
        "root_disk_used_pct": (disk.used / disk.total * 100) if disk.total else 0,
    }


def docker_stats() -> list[dict[str, str | float | int]]:
    result = subprocess.run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=8,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            continue
        name, cpu_text, memory_text, memory_pct_text, net_io, block_io = fields
        used_text, _, limit_text = memory_text.partition("/")
        rows.append(
            {
                "container": name,
                "cpu_pct": parse_percent(cpu_text),
                "memory_used_bytes": parse_quantity(used_text),
                "memory_limit_bytes": parse_quantity(limit_text),
                "memory_pct": parse_percent(memory_pct_text),
                "net_io": net_io,
                "block_io": block_io,
            }
        )
    return rows


def parse_percent(value: str) -> float:
    try:
        return float(value.strip().removesuffix("%"))
    except ValueError:
        return 0.0


def tesseract_processes(
    last_cpu: dict[tuple[int, int], tuple[int, float]], clock_ticks: int, page_size: int
) -> list[dict[str, float | int]]:
    found: list[dict[str, float | int]] = []
    current: dict[tuple[int, int], tuple[int, float]] = {}
    proc = Path("/proc")
    now = time.monotonic()
    try:
        uptime = float((proc / "uptime").read_text(encoding="ascii").split()[0])
    except (OSError, ValueError, IndexError):
        uptime = now

    try:
        entries = list(proc.iterdir())
    except OSError:
        return found

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            comm = (entry / "comm").read_text(encoding="ascii").strip()
            if comm != "tesseract":
                continue
            stat_text = (entry / "stat").read_text(encoding="ascii")
            fields = stat_text[stat_text.rfind(")") + 2 :].split()
            cpu_ticks = int(fields[11]) + int(fields[12])
            started_ticks = int(fields[19])
            resident_pages = max(0, int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue

        key = (pid, started_ticks)
        previous = last_cpu.get(key)
        if previous:
            previous_ticks, previous_time = previous
            elapsed = max(now - previous_time, 0.001)
            cpu_pct = max(0, cpu_ticks - previous_ticks) / clock_ticks / elapsed * 100
        else:
            process_age = max(uptime - started_ticks / clock_ticks, 0.01)
            cpu_pct = cpu_ticks / clock_ticks / process_age * 100
        current[key] = (cpu_ticks, now)
        found.append(
            {
                "pid": pid,
                "started_ticks": started_ticks,
                "cpu_pct": cpu_pct,
                "cpu_time_seconds": cpu_ticks / clock_ticks,
                "rss_bytes": resident_pages * page_size,
            }
        )

    last_cpu.clear()
    last_cpu.update(current)
    return found


def write_summary(output_dir: Path, mode: str, interval: int) -> None:
    host_rows = read_csv(output_dir / "host.csv")
    docker_rows = read_csv(output_dir / "docker.csv")
    tesseract_rows = read_csv(output_dir / "tesseract.csv")
    active_tesseract = [row for row in tesseract_rows if row.get("pid")]

    lines = [
        "Сводка ресурсной нагрузки",
        f"Режим OCR: {mode}",
        f"Начало сбора (UTC): {host_rows[0]['timestamp_utc'] if host_rows else 'нет данных'}",
        f"Последний снимок (UTC): {host_rows[-1]['timestamp_utc'] if host_rows else 'нет данных'}",
        f"Интервал системных и Docker-снимков: {interval} сек.",
        f"Системных снимков: {len(host_rows)}",
        f"Снимков процессов Tesseract: {len(tesseract_rows)} (частота около 1 сек.)",
    ]

    if host_rows:
        lines.extend(
            [
                f"Пик load average за 1 минуту: {max(float(row['load_1m']) for row in host_rows):.2f}",
                f"Минимум доступной памяти: {min(int(row['memory_available_bytes']) for row in host_rows) / 1024**2:.0f} MiB",
                f"Пик занятого места в корневой ФС: {max(float(row['root_disk_used_pct']) for row in host_rows):.1f}%",
            ]
        )

    lines.append("Контейнеры (средний/пиковый CPU, пик памяти):")
    names = sorted({row["container"] for row in docker_rows})
    for name in names:
        rows = [row for row in docker_rows if row["container"] == name]
        cpus = [float(row["cpu_pct"]) for row in rows]
        memory = [float(row["memory_used_bytes"]) for row in rows]
        lines.append(
            f"- {name}: CPU {sum(cpus) / len(cpus):.2f}% / {max(cpus):.2f}%; "
            f"RAM пик {max(memory) / 1024**2:.0f} MiB"
        )

    if active_tesseract:
        cpu = [float(row["cpu_pct"]) for row in active_tesseract]
        rss = [float(row["rss_bytes"]) for row in active_tesseract]
        per_process_cpu = {}
        for row in active_tesseract:
            key = (row["pid"], row["started_ticks"])
            per_process_cpu[key] = max(per_process_cpu.get(key, 0.0), float(row["cpu_time_seconds"]))
        cpu_seconds = sum(per_process_cpu.values())
        lines.extend(
            [
                f"Наблюдений активного Tesseract: {len(active_tesseract)}",
                f"Tesseract CPU: средний по активным снимкам {sum(cpu) / len(cpu):.1f}%, "
                f"пик {max(cpu):.1f}%, оценка накопленного CPU-времени {cpu_seconds / 60:.1f} CPU-мин",
                f"Tesseract RSS: пик {max(rss) / 1024**2:.0f} MiB",
            ]
        )
    else:
        lines.append("Активных процессов Tesseract не зафиксировано; оценить его фактическую нагрузку по этому запуску нельзя.")

    lines.extend(
        [
            "Ограничение: процессы короче интервала опроса могут быть пропущены; снимки не являются контролируемым сравнением двух OCR-режимов.",
            "",
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=86400)
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--mode", default="openrouter,rupasportread")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.duration_seconds < 1 or args.interval_seconds < 1:
        parser.error("duration and interval must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    host_path = args.output_dir / "host.csv"
    docker_path = args.output_dir / "docker.csv"
    process_path = args.output_dir / "tesseract.csv"
    host_fields = ["timestamp_utc", "load_1m", "load_5m", "load_15m", "memory_total_bytes", "memory_available_bytes", "root_disk_total_bytes", "root_disk_used_bytes", "root_disk_used_pct"]
    docker_fields = ["timestamp_utc", "container", "cpu_pct", "memory_used_bytes", "memory_limit_bytes", "memory_pct", "net_io", "block_io"]
    process_fields = ["timestamp_utc", "pid", "started_ticks", "cpu_pct", "cpu_time_seconds", "rss_bytes"]
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    clock_ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    last_cpu: dict[tuple[int, int], tuple[int, float]] = {}
    started = time.monotonic()
    end_at = started + args.duration_seconds
    next_heavy = started
    next_process = started
    host_samples = 0

    with host_path.open("w", newline="", encoding="utf-8") as host_file, docker_path.open(
        "w", newline="", encoding="utf-8"
    ) as docker_file, process_path.open("w", newline="", encoding="utf-8") as process_file:
        host_writer = csv.DictWriter(host_file, fieldnames=host_fields)
        docker_writer = csv.DictWriter(docker_file, fieldnames=docker_fields)
        process_writer = csv.DictWriter(process_file, fieldnames=process_fields)
        host_writer.writeheader()
        docker_writer.writeheader()
        process_writer.writeheader()

        print(f"collector_started mode={args.mode} interval={args.interval_seconds}s", flush=True)
        while not STOP and time.monotonic() < end_at:
            now = time.monotonic()
            if now >= next_process:
                timestamp = utc_now()
                processes = tesseract_processes(last_cpu, clock_ticks, page_size)
                if processes:
                    for process in processes:
                        process_writer.writerow({"timestamp_utc": timestamp, **process})
                else:
                    process_writer.writerow({"timestamp_utc": timestamp, "pid": "", "started_ticks": "", "cpu_pct": 0, "cpu_time_seconds": 0, "rss_bytes": 0})
                process_file.flush()
                next_process = now + 1

            if now >= next_heavy:
                timestamp = utc_now()
                try:
                    host_writer.writerow({"timestamp_utc": timestamp, **read_host_metrics()})
                    host_samples += 1
                except (OSError, ValueError):
                    print("host_sample_failed", file=sys.stderr, flush=True)
                try:
                    for container in docker_stats():
                        docker_writer.writerow({"timestamp_utc": timestamp, **container})
                except (OSError, subprocess.SubprocessError, ValueError):
                    print("docker_sample_failed", file=sys.stderr, flush=True)
                host_file.flush()
                docker_file.flush()
                next_heavy = now + args.interval_seconds

            wait_for = min(next_process, next_heavy, end_at) - time.monotonic()
            if wait_for > 0:
                time.sleep(min(wait_for, 0.25))

    write_summary(args.output_dir, args.mode, args.interval_seconds)
    print(f"collector_finished host_samples={host_samples}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
