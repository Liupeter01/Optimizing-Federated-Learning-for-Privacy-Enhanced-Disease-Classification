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

send_queue = queue.Queue()

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

def generate_requests(client_id, model_version="1.0.0"):
    try:
        while not send_queue.empty():
            vector = send_queue.get()
            if not vector:
                print("[Client] Error: Retrieved an empty vector from the queue.")
                continue
            yield ml_vector_pb2.VectorRequest(client_id=client_id, vector=vector, model_version=model_version)
            print(f"[Client] Sent vector from Client {client_id}, Version {model_version}, Size: {len(vector)}, Vec: {vector[:10]} (First 10 elements)")
    except Exception as e:
        print(f"[Client] Error in generate_requests(): {e}")

def receive_responses(response_iterator, local_vector, max_rounds=4):
    for round_num in range(max_rounds):
        try:
            start_time = time.time()
            response = next(response_iterator)
            end_time = time.time()

            if response.status_code != 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Server Error: {response.error_message}")
                break

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Round {round_num + 1}: Received merged vector of size {len(response.merged_vector)} in {end_time - start_time:.2f} seconds.")
            local_vector = handle_merged_vector(response.merged_vector, local_vector)

            # Enqueue for the next round
            send_queue.put(local_vector)

        except StopIteration:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Server has completed the stream.")
            break
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Error during round {round_num + 1}: {e}")

def run(client_id, model_version="1.0.0", server_address="localhost:50051", max_rounds=4):
    client_id = str(client_id)

    send_queue.queue.clear()

    # Generate initial vector
    try:
        local_vector = get_local_vector()
        if isinstance(local_vector, torch.Tensor):
            local_vector = local_vector.detach().cpu().numpy().tolist()
            print("[Client] Converted Tensor to list.")

        if local_vector:
            send_queue.put(local_vector)
            print(f"[Client] Initial vector generated and sent to the queue.")
        else:
            print("[Client] Error: get_local_vector() returned empty data.")
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Initial vector Created.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Error generating initial vector: {e}")
        return

    try:
        # Establish the gRPC channel
        channel = grpc.insecure_channel(server_address)
        stub = ml_vector_pb2_grpc.MLServiceStub(channel)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Connected to server.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Failed to connect to server: {e}")
        return

    try:
        response_iterator = stub.FederatedAveraging(generate_requests(client_id, model_version))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Started Federated Averaging.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Error starting gRPC stream: {e}")
        return

    # Start response thread
    response_thread = threading.Thread(target=receive_responses, args=(response_iterator, local_vector, max_rounds))
    response_thread.start()
    response_thread.join()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Federated Averaging completed.")

if __name__ == "__main__":
    run(client_id=2)