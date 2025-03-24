import grpc
import threading
import queue
from client_logic import get_local_vector, handle_merged_vector, generate_requests
import ml_vector_pb2_grpc

send_queue = queue.Queue()

# Inject send_queue into logic module
import client_logic
client_logic.send_queue = send_queue


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_vector_pb2_grpc.MLServiceStub(channel)

    # Generate and enqueue initial vector
    local_vector = get_local_vector()
    send_queue.put(local_vector) 

    # Start the FederatedAveraging RPC call with bidirectional streaming
    response_iterator = stub.FederatedAveraging(generate_requests())

    for response in response_iterator:
        local_vector = handle_merged_vector(response.merged_vector, local_vector)

if __name__ == "__main__":
    run()