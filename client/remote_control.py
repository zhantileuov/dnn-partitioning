import argparse
import json
import socket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send remote control commands to a dnn_partition client.")
    parser.add_argument("--host", required=True, help="Client host or IP address to control.")
    parser.add_argument("--port", type=int, default=5055, help="UDP port exposed by the client.")
    parser.add_argument("--mode", required=True, choices=["full_local", "full_server", "split"])
    parser.add_argument("--model", required=True, help="Model name, e.g. resnet18.")
    parser.add_argument("--partition-point", default=None, help="Required when mode=split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "split" and not args.partition_point:
        raise SystemExit("--partition-point is required when --mode split")
    if args.mode != "split" and args.partition_point:
        raise SystemExit("--partition-point is only valid when --mode split")

    payload = {
        "mode": args.mode,
        "model_name": args.model,
        "partition_point": args.partition_point,
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))
    finally:
        sock.close()
    print(
        "[control] sent "
        f"mode={args.mode} model={args.model} partition={args.partition_point or 'full'} "
        f"to udp://{args.host}:{args.port}"
    )


if __name__ == "__main__":
    main()
