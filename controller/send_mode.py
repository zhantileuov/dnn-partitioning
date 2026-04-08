import json
import socket


JETSON_IP = "192.168.1.50"
JETSON_PORT = 5055

message = {
    "mode": "split",
    "model_name": "resnet18",
    "partition_point": "layer2.0",
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(json.dumps(message).encode("utf-8"), (JETSON_IP, JETSON_PORT))
sock.close()

print("sent:", message)
