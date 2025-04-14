import grpc
import threading
import queue
import time
from client_logic import get_local_vector, handle_merged_vector
import ml_vector_pb2
import ml_vector_pb2_grpc
import client_logic
import queue
import torch
import warnings
import json
import sys
import os

send_queue = queue.Queue()

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*torch.load.*")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*Using a non-full backward hook.*"
)
def load_client_config(group_name):
    path = f"./config/{group_name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, 'r') as f:
        return json.load(f)

def generate_requests(client_id, model_version="1.0.0"):
    while not send_queue.empty():
        data = send_queue.get()
        vector = data.get("vector", [])
        dp_params_json = data.get("dp_params_json", "")
        if not vector:
            print("[Client] Error: Retrieved an empty vector from the queue.")
            continue
        yield ml_vector_pb2.VectorRequest(
            client_id=client_id,
            vector=vector,
            model_version=model_version,
            dp_params_list=str(dp_params_json)
        )
        print(f"[Client] Sent vector | Client: {client_id}, Size: {len(vector)}, ε≈{json.loads(dp_params_json).get('epsilon', '?'):.2f}")

def receive_responses(response_iterator, local_vector, config, max_rounds=4):
    for round_num in range(max_rounds):
        try:
            response = next(response_iterator)

            if response.status_code != 0:
                print(f"[Client] ❌ Server Error: {response.error_message}")
                break

            print(f"[Client] ✅ Round {round_num+1}: Received vector of size {len(response.merged_vector)}")
            local_vector = handle_merged_vector(response.merged_vector, local_vector, config)
            send_queue.put(local_vector)

        except StopIteration:
            print("[Client] 🔁 Server stream ended.")
            break
        except Exception as e:
            print(f"[Client] ⚠️ Error during round {round_num + 1}: {e}")

def run(client_id, group_name, model_version="1.0.0", server_address="localhost:50051", max_rounds=4):
    client_id = str(client_id)
    send_queue.queue.clear()

    try:
        config = load_client_config(group_name)
        local_vector, dp_params_json = get_local_vector(group_name, config)

        if isinstance(local_vector, torch.Tensor):
            local_vector = local_vector.detach().cpu().numpy().tolist()
            print("[Client] ℹ️ Converted Tensor to list.")

        if local_vector:
            send_queue.put({"vector": local_vector, "dp_params_json": dp_params_json})
            print("[Client] 🚀 Initial vector enqueued.")
        else:
            print("[Client] ❌ Empty vector from get_local_vector().")
            return

    except Exception as e:
        print(f"[Client] ❌ Error generating initial vector: {e}")
        return

    try:
        channel = grpc.insecure_channel(server_address)
        stub = ml_vector_pb2_grpc.MLServiceStub(channel)
        print("[Client] 📡 Connected to server.")
        response_iterator = stub.FederatedAveraging(generate_requests(client_id, model_version))
        print("[Client] 🔄 Started Federated Averaging stream.")
    except Exception as e:
        print(f"[Client] ❌ Failed to connect to server: {e}")
        return

    thread = threading.Thread(target=receive_responses, args=(response_iterator, local_vector, config, max_rounds))
    thread.start()
    thread.join()
    print("[Client] ✅ Federated Averaging completed.")

# Execute
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python client.py <client_id> <group_name>")
        sys.exit(1)
    run(client_id=sys.argv[1], group_name=sys.argv[2])