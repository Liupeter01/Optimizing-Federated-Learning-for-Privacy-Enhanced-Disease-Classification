import grpc
import threading
import queue
import time
from client_logic import get_local_vector, handle_merged_vector, generate_requests
import ml_vector_pb2_grpc
import client_logic

send_queue = queue.Queue()

# Inject send_queue into logic module
client_logic.send_queue = send_queue

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

    try:
        # Establish the gRPC channel
        channel = grpc.insecure_channel(server_address)
        stub = ml_vector_pb2_grpc.MLServiceStub(channel)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Connected to server.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Failed to connect to server: {e}")
        return

    send_queue.queue.clear()

    # Generate initial vector
    try:
        local_vector = get_local_vector()
        send_queue.put(local_vector)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Initial vector sent.")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Client] Error generating initial vector: {e}")
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