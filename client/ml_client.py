import grpc
import time
from client_logic import get_local_vector, handle_merged_vector
import ml_vector_pb2
import ml_vector_pb2_grpc
import torch
import warnings
import json
import sys
import os
import queue

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*torch.load.*")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Using a non-full backward hook.*"
)

class RequestIterator:
    def __init__(self):
        self._queue = queue.Queue()
        self._closed = False

    def set(self, item):
        self._queue.put(item)

    def close(self):
        self._closed = True
        self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed and self._queue.empty():
            raise StopIteration
        item = self._queue.get()
        if item is None:
            raise StopIteration
        return item

def load_client_config(group_name):
    path = f"./config/{group_name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, 'r') as f:
        return json.load(f)

def run(client_id, group_name, model_version="1.0.0", server_address="localhost:50051", max_rounds=4):
    client_id = str(client_id)

    try:
        config = load_client_config(group_name)
        local_vector, dp_params_json = get_local_vector(group_name, config)

        if isinstance(local_vector, torch.Tensor):
            local_vector = local_vector.detach().cpu().numpy().tolist()
            print("[Client] ℹConverted Tensor to list.")

        if not local_vector:
            print("[Client] ❌ Empty vector from get_local_vector().")
            return

    except Exception as e:
        print(f"[Client] ❌ Error generating initial vector: {e}")
        return

    try:
        channel = grpc.insecure_channel(server_address)
        grpc.channel_ready_future(channel).result(timeout=10)
        stub = ml_vector_pb2_grpc.MLServiceStub(channel)
        print("[Client] 📱 Connected to server.")
    except Exception as e:
        print(f"[Client] ❌ Failed to connect to server: {e}")
        return

    request_stream = RequestIterator()
    try:
        response_stream = stub.FederatedAveraging(request_stream)

        for round_num in range(max_rounds):
            req = ml_vector_pb2.VectorRequest(
                client_id=client_id,
                vector=local_vector,
                model_version=model_version,
                dp_params_list=str(dp_params_json)
            )
            request_stream.set(req)
            print(f"[Client] 📨 Round {round_num + 1}: Sent vector")

            try:
                response = next(response_stream)
            except StopIteration:
                print(f"[Client] ❌ Stream ended early after {round_num} round(s).")
                break
            except grpc.RpcError as e:
                print(f"[Client] ❌ gRPC error: {e.code()} - {e.details()}")
                break

            if response.status_code != 0:
                print(f"[Client] ❌ Server Error: {response.error_message}")
                break

            print(f"[Client] ✅ Round {round_num + 1}: Received vector")

            local_vector = handle_merged_vector(response.merged_vector, local_vector, config)

        request_stream.close()
        print("[Client] ✅ Federated Averaging completed.")

    except Exception as e:
        print(f"[Client] ⚠️ Unexpected error: {e}")

# Execute
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python client.py <client_id> <group_name>")
        sys.exit(1)
    run(client_id=sys.argv[1], group_name=sys.argv[2])
