import grpc
import random
import ml_vector_pb2
import ml_vector_pb2_grpc
import threading
import queue
import time
send_queue = queue.Queue()

# Local client generate a new vector which is going to be transmitted to server.
def get_local_vector():
    # AI generate a vector
    vector = [random.uniform(0, 100) for _ in range(4)]
    return vector

# Client Merge the local vector with server's vector
def merge_with_local(local_vector, merged_vector):
    return [
        (lv + mv) / 2 for lv, mv in zip(local_vector, merged_vector)
    ]

def handle_merged_vector(merged_vector, local_vector):
    print(f"[Client] Received merged vector from server: {merged_vector}")
    print(f"[Client] Merging with local vector: {local_vector}")

    updated_vector = merge_with_local(local_vector, merged_vector)
    print(f"[Client] Updated vector after merge: {updated_vector}")

    # put the updated vector into the queue
    send_queue.put(updated_vector)

# Running in multi threaded mode(Sub-thread)
def generate_requests():
    while True:
        vector = send_queue.get()
        print(f"[Client] Sending vector: {vector}")
        yield ml_vector_pb2.VectorRequest(vector=vector)

# Only For the first time deployment
def generate_first_requests():
    vec = get_local_vector()
    print(f"Sending vector: {vec}")
    yield ml_vector_pb2.VectorRequest(vector=vec) 


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_vector_pb2_grpc.MLServiceStub(channel)

    # generate initial requests(only for the first time!!!!)
    local_vector = get_local_vector()
    send_queue.put(local_vector) 

    # Activate the threads
    send_thread = threading.Thread(target=lambda: list(stub.FederatedAveraging(generate_requests())), daemon=True)
    send_thread.start()

    response_iterator = stub.FederatedAveraging(generate_first_requests())

    for response in response_iterator:
        print(f"Received merged vector from server: {response.merged_vector}")

        # Merge the global average with the current local vector, then send it back to the server
        handle_merged_vector(response.merged_vector, local_vector)

if __name__ == "__main__":
    run()
