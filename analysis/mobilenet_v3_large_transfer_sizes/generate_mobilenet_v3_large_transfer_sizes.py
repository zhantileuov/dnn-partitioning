import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_SHAPE = (1, 3, 224, 224)

# For MobileNetV3-Large with a 1x3x224x224 input, these are the activation
# shapes produced by the client-side prefix and transferred to the Triton tail
# model.
MOBILENET_V3_LARGE_PARTITIONS = [
    ("features.0", (1, 16, 112, 112)),
    ("features.1", (1, 16, 112, 112)),
    ("features.2", (1, 24, 56, 56)),
    ("features.3", (1, 24, 56, 56)),
    ("features.4", (1, 40, 28, 28)),
    ("features.5", (1, 40, 28, 28)),
    ("features.6", (1, 40, 28, 28)),
    ("features.7", (1, 80, 14, 14)),
    ("features.8", (1, 80, 14, 14)),
    ("features.9", (1, 80, 14, 14)),
    ("features.10", (1, 80, 14, 14)),
    ("features.11", (1, 112, 14, 14)),
    ("features.12", (1, 112, 14, 14)),
    ("features.13", (1, 160, 7, 7)),
    ("features.14", (1, 160, 7, 7)),
    ("features.15", (1, 160, 7, 7)),
    ("features.16", (1, 960, 7, 7)),
    ("avgpool", (1, 960, 1, 1)),
    ("flatten", (1, 960)),
    ("classifier.0", (1, 1280)),
    ("classifier.1", (1, 1280)),
    ("classifier.2", (1, 1280)),
]


def num_elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def bytes_for_shape(shape: tuple[int, ...], bytes_per_value: int) -> int:
    return num_elements(shape) * bytes_per_value


def kb_from_bytes(byte_count: int) -> float:
    return byte_count / 1024.0


def tail_model_name(partition_point: str) -> str:
    return "mobilenet_v3_large_tail_after_{0}".format(partition_point.replace(".", "_"))


def build_rows() -> list[dict]:
    rows = []
    for partition_point, activation_shape in MOBILENET_V3_LARGE_PARTITIONS:
        fp32_bytes = bytes_for_shape(activation_shape, bytes_per_value=4)
        fp16_bytes = bytes_for_shape(activation_shape, bytes_per_value=2)
        rows.append(
            {
                "partition_point": partition_point,
                "tail_model_name": tail_model_name(partition_point),
                "activation_shape": "x".join(str(part) for part in activation_shape),
                "num_values": num_elements(activation_shape),
                "transfer_kb_fp32": round(kb_from_bytes(fp32_bytes), 2),
                "transfer_kb_fp16": round(kb_from_bytes(fp16_bytes), 2),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict]) -> None:
    labels = [row["partition_point"] for row in rows]
    fp32_kb = [row["transfer_kb_fp32"] for row in rows]
    fp16_kb = [row["transfer_kb_fp16"] for row in rows]

    positions = list(range(len(labels)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar([pos - width / 2 for pos in positions], fp32_kb, width=width, label="FP32", color="#28536B")
    ax.bar([pos + width / 2 for pos in positions], fp16_kb, width=width, label="FP16", color="#C2948A")

    ax.set_title("MobileNetV3 Large Split Transfer Size per Tail Model")
    ax.set_xlabel("Partition Point")
    ax.set_ylabel("Transferred Activation Size (KB)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    y_offset = max(fp32_kb) * 0.012
    for pos, value in zip(positions, fp32_kb):
        ax.text(pos - width / 2, value + y_offset, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    for pos, value in zip(positions, fp16_kb):
        ax.text(pos + width / 2, value + y_offset, f"{value:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    rows = build_rows()

    write_csv(output_dir / "mobilenet_v3_large_transfer_sizes.csv", rows)
    write_plot(output_dir / "mobilenet_v3_large_transfer_sizes_kb.png", rows)

    print("wrote:", output_dir / "mobilenet_v3_large_transfer_sizes.csv")
    print("wrote:", output_dir / "mobilenet_v3_large_transfer_sizes_kb.png")


if __name__ == "__main__":
    main()
