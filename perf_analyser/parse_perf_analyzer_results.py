#!/usr/bin/env python3
"""Parse Triton perf_analyzer text logs into CSV summaries and plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLUMNS = [
    "source_file",
    "instance_group",
    "concurrency",
    "client_request_count",
    "throughput_inf_s",
    "client_avg_latency_us",
    "client_latency_std_us",
    "client_p50_latency_us",
    "client_p90_latency_us",
    "client_p95_latency_us",
    "client_p99_latency_us",
    "avg_grpc_time_us",
    "grpc_marshal_us",
    "grpc_response_wait_us",
    "server_inference_count",
    "server_execution_count",
    "server_successful_request_count",
    "server_avg_request_latency_us",
    "server_overhead_us",
    "server_queue_us",
    "server_compute_input_us",
    "server_compute_infer_us",
    "server_compute_output_us",
]

TIME_US_COLUMNS = [
    "client_avg_latency_us",
    "client_latency_std_us",
    "client_p50_latency_us",
    "client_p90_latency_us",
    "client_p95_latency_us",
    "client_p99_latency_us",
    "avg_grpc_time_us",
    "grpc_marshal_us",
    "grpc_response_wait_us",
    "server_avg_request_latency_us",
    "server_overhead_us",
    "server_queue_us",
    "server_compute_input_us",
    "server_compute_infer_us",
    "server_compute_output_us",
]


HEADER_RE = re.compile(
    r"==\s*Triton\s+Inference\s+Server\s+SDK\s+(?P<instance_group>\d+)\s*==",
    re.IGNORECASE,
)

BLOCK_RE = re.compile(
    r"""
    Request\s+concurrency:\s*(?P<concurrency>\d+)\s*
    .*?Request\s+count:\s*(?P<client_request_count>\d+)\s*
    .*?Throughput:\s*(?P<throughput_inf_s>[\d.]+)\s*infer/sec\s*
    .*?Avg\s+latency:\s*(?P<client_avg_latency_us>\d+)\s*usec
        \s*\(\s*standard\s+deviation\s+(?P<client_latency_std_us>\d+)\s*usec\s*\)\s*
    .*?p50\s+latency:\s*(?P<client_p50_latency_us>\d+)\s*usec\s*
    .*?p90\s+latency:\s*(?P<client_p90_latency_us>\d+)\s*usec\s*
    .*?p95\s+latency:\s*(?P<client_p95_latency_us>\d+)\s*usec\s*
    .*?p99\s+latency:\s*(?P<client_p99_latency_us>\d+)\s*usec\s*
    .*?Avg\s+gRPC\s+time:\s*(?P<avg_grpc_time_us>\d+)\s*usec
        \s*\(\s*\(un\)marshal\s+request/response\s+(?P<grpc_marshal_us>\d+)\s*usec
        \s*\+\s*response\s+wait\s+(?P<grpc_response_wait_us>\d+)\s*usec\s*\)\s*
    .*?Inference\s+count:\s*(?P<server_inference_count>\d+)\s*
    .*?Execution\s+count:\s*(?P<server_execution_count>\d+)\s*
    .*?Successful\s+request\s+count:\s*(?P<server_successful_request_count>\d+)\s*
    .*?Avg\s+request\s+latency:\s*(?P<server_avg_request_latency_us>\d+)\s*usec
        \s*\(\s*overhead\s+(?P<server_overhead_us>\d+)\s*usec
        \s*\+\s*queue\s+(?P<server_queue_us>\d+)\s*usec
        \s*\+\s*compute\s+input\s+(?P<server_compute_input_us>\d+)\s*usec
        \s*\+\s*compute\s+infer\s+(?P<server_compute_infer_us>\d+)\s*usec
        \s*\+\s*compute\s+output\s+(?P<server_compute_output_us>\d+)\s*usec\s*\)
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def iter_experiment_sections(text: str):
    """Yield (instance_group, section_text) for each SDK header in a log."""
    headers = list(HEADER_RE.finditer(text))
    for idx, header in enumerate(headers):
        start = header.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        yield int(header.group("instance_group")), text[start:end]


def parse_file(path: Path, root: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_file = str(path.relative_to(root))
    rows: list[dict[str, object]] = []

    for instance_group, section in iter_experiment_sections(text):
        for match in BLOCK_RE.finditer(section):
            row: dict[str, object] = {
                "source_file": source_file,
                "instance_group": instance_group,
            }
            row.update(match.groupdict())
            rows.append(row)

    return rows


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in TIME_US_COLUMNS:
        df[f"{column[:-3]}_ms"] = df[column] / 1000.0

    df["queue_ratio"] = df["server_queue_ms"] / df["server_avg_request_latency_ms"]
    df["throughput_per_instance"] = df["throughput_inf_s"] / df["instance_group"]
    df["client_server_latency_gap_ms"] = (
        df["client_avg_latency_ms"] - df["server_avg_request_latency_ms"]
    )
    return df


def rows_at_max_throughput(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("instance_group")["throughput_inf_s"].idxmax()
    return df.loc[idx].sort_values("instance_group")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for instance_group, group in df.groupby("instance_group", sort=True):
        max_throughput_row = group.loc[group["throughput_inf_s"].idxmax()]
        min_latency_row = group.loc[group["client_avg_latency_ms"].idxmin()]

        summary_rows.append(
            {
                "instance_group": instance_group,
                "max_throughput_inf_s": max_throughput_row["throughput_inf_s"],
                "avg_throughput_inf_s": group["throughput_inf_s"].mean(),
                "concurrency_at_max_throughput": max_throughput_row["concurrency"],
                "client_avg_latency_ms_at_max_throughput": max_throughput_row[
                    "client_avg_latency_ms"
                ],
                "client_p95_latency_ms_at_max_throughput": max_throughput_row[
                    "client_p95_latency_ms"
                ],
                "server_queue_ms_at_max_throughput": max_throughput_row[
                    "server_queue_ms"
                ],
                "server_compute_infer_ms_at_max_throughput": max_throughput_row[
                    "server_compute_infer_ms"
                ],
                "throughput_per_instance_at_max_throughput": max_throughput_row[
                    "throughput_per_instance"
                ],
                "min_client_avg_latency_ms": min_latency_row["client_avg_latency_ms"],
                "concurrency_at_min_latency": min_latency_row["concurrency"],
            }
        )

    return pd.DataFrame(summary_rows)


def plot_metric(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for instance_group, group in df.groupby("instance_group", sort=True):
        group = group.sort_values("concurrency")
        ax.plot(
            group["concurrency"],
            group[metric],
            marker="o",
            linewidth=1.8,
            label=f"instance_group={instance_group}",
        )

    ax.set_title(title)
    ax.set_xlabel("Request concurrency")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Instance group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tail_latency(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for instance_group, group in df.groupby("instance_group", sort=True):
        group = group.sort_values("concurrency")
        ax.plot(
            group["concurrency"],
            group["client_p95_latency_ms"],
            marker="o",
            linewidth=1.8,
            label=f"instance_group={instance_group} p95",
        )
        ax.plot(
            group["concurrency"],
            group["client_p99_latency_ms"],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            label=f"instance_group={instance_group} p99",
        )

    ax.set_title("Client Tail Latency vs Request Concurrency")
    ax.set_xlabel("Request concurrency")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Instance group / percentile", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for instance_group, group in df.groupby("instance_group", sort=True):
        group = group.sort_values("concurrency")
        ax.plot(
            group["throughput_inf_s"],
            group["client_avg_latency_ms"],
            marker="o",
            linewidth=1.8,
            label=f"instance_group={instance_group}",
        )

    ax.set_title("Throughput / Latency Tradeoff")
    ax.set_xlabel("Throughput (infer/sec)")
    ax.set_ylabel("Client average latency (ms)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Instance group")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


LATENCY_COMPONENTS_MS = [
    "server_overhead_ms",
    "server_queue_ms",
    "server_compute_input_ms",
    "server_compute_infer_ms",
    "server_compute_output_ms",
]


def plot_latency_breakdown_mean(df: pd.DataFrame, output_path: Path) -> None:
    mean_components = (
        df.groupby("instance_group", sort=True)[LATENCY_COMPONENTS_MS]
        .mean()
        .sort_index()
    )

    ax = mean_components.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        width=0.75,
    )
    ax.set_title("Average Server Latency Breakdown by Instance Group")
    ax.set_xlabel("Instance group")
    ax.set_ylabel("Average server latency component (ms)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Component")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=150)
    plt.close(ax.figure)


def plot_latency_breakdown_at_saturation(df: pd.DataFrame, output_path: Path) -> None:
    max_rows = rows_at_max_throughput(df)
    components = [
        "server_overhead_ms",
        "server_queue_ms",
        "server_compute_input_ms",
        "server_compute_infer_ms",
        "server_compute_output_ms",
    ]

    ax = max_rows.set_index("instance_group")[components].plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        width=0.75,
    )
    ax.set_title("Server Latency Breakdown at Max Throughput")
    ax.set_xlabel("Instance group")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Component")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=150)
    plt.close(ax.figure)


def plot_bar(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["instance_group"].astype(str), df[metric], width=0.7)
    ax.set_title(title)
    ax.set_xlabel("Instance group")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_outputs(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = add_derived_columns(df)

    df = df.sort_values(["instance_group", "concurrency", "source_file"])
    df.to_csv(output_dir / "all_perf_results_long.csv", index=False)

    summary = build_summary(df)
    summary.to_csv(output_dir / "summary_by_instance_group.csv", index=False)

    plot_metric(
        df,
        output_dir / "throughput_vs_concurrency.png",
        "throughput_inf_s",
        "Throughput (infer/sec)",
        "Throughput vs Request Concurrency",
    )
    plot_metric(
        df,
        output_dir / "client_avg_latency_vs_concurrency.png",
        "client_avg_latency_ms",
        "Client average latency (ms)",
        "Client Average Latency vs Request Concurrency",
    )
    plot_metric(
        df,
        output_dir / "server_queue_vs_concurrency.png",
        "server_queue_ms",
        "Server queue time (ms)",
        "Server Queue Time vs Request Concurrency",
    )
    plot_metric(
        df,
        output_dir / "compute_infer_vs_concurrency.png",
        "server_compute_infer_ms",
        "Compute infer time (ms)",
        "Compute Infer Time vs Request Concurrency",
    )
    plot_tail_latency(df, output_dir / "client_tail_latency_vs_concurrency.png")
    plot_tradeoff(df, output_dir / "throughput_latency_tradeoff.png")
    plot_metric(
        df,
        output_dir / "queue_ratio_vs_concurrency.png",
        "queue_ratio",
        "Queue ratio",
        "Queue Ratio vs Request Concurrency",
    )
    plot_latency_breakdown_mean(df, output_dir / "latency_breakdown_stacked_bar.png")
    plot_latency_breakdown_at_saturation(
        df, output_dir / "latency_breakdown_at_saturation.png"
    )
    plot_bar(
        summary,
        output_dir / "avg_throughput_by_instance_group.png",
        "avg_throughput_inf_s",
        "Average throughput (infer/sec)",
        "Average Throughput by Instance Group",
    )
    plot_bar(
        summary,
        output_dir / "throughput_per_instance_by_instance_group.png",
        "throughput_per_instance_at_max_throughput",
        "Throughput per instance (infer/sec)",
        "Throughput per Instance by Instance Group",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse Triton perf_analyzer .txt logs into CSVs and plots."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="perf_analyser",
        help="Folder containing perf_analyzer .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder for CSV and PNG outputs. Defaults to the input folder.",
    )
    return parser.parse_args()


def print_terminal_summary(summary: pd.DataFrame) -> None:
    best_throughput = summary.loc[summary["max_throughput_inf_s"].idxmax()]
    best_latency = summary.loc[
        summary["client_avg_latency_ms_at_max_throughput"].idxmin()
    ]
    throughput_per_instance = summary.sort_values("instance_group")[
        "throughput_per_instance_at_max_throughput"
    ]
    decreases = throughput_per_instance.is_monotonic_decreasing

    print("Final summary:")
    print(
        "- best instance_group by max throughput: "
        f"{int(best_throughput['instance_group'])} "
        f"({best_throughput['max_throughput_inf_s']:.4f} infer/sec)"
    )
    print(
        "- best instance_group by lowest latency at max throughput: "
        f"{int(best_latency['instance_group'])} "
        f"({best_latency['client_avg_latency_ms_at_max_throughput']:.3f} ms)"
    )
    print(
        "- whether throughput_per_instance decreases as instance_group increases: "
        f"{'yes' if decreases else 'no'}"
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {input_dir}")

    rows = []
    for path in txt_files:
        rows.extend(parse_file(path, input_dir))

    if not rows:
        raise SystemExit(f"No perf_analyzer request concurrency blocks found in {input_dir}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    int_columns = [column for column in COLUMNS if column not in {"source_file", "throughput_inf_s"}]
    df[int_columns] = df[int_columns].apply(pd.to_numeric, errors="raise")
    df["throughput_inf_s"] = pd.to_numeric(df["throughput_inf_s"], errors="raise")

    summary = write_outputs(df, output_dir)
    print(f"Parsed {len(df)} rows from {len(txt_files)} file(s).")
    print(f"Wrote CSVs and plots to {output_dir}")
    print_terminal_summary(summary)


if __name__ == "__main__":
    main()
