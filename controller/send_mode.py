import json
import socket


JETSON_IP = "172.22.229.116"
JETSON_PORT = 5055

message = {
    "mode": "full_server",
    "model_name": "resnet18",
    "partition_point": None,
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(json.dumps(message).encode("utf-8"), (JETSON_IP, JETSON_PORT))
sock.close()

print("sent:", message)
