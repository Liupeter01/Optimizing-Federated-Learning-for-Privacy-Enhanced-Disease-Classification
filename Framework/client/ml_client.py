import grpc
import threading
import queue
from client_logic import get_local_vector, handle_merged_vector, generate_requests
from resnet18 import ml_resnet18 
import ml_vector_pb2_grpc

send_queue = queue.Queue()

# Inject send_queue into logic module
import client_logic
client_logic.send_queue = send_queue

def handle_model():
    ml_resnet18.resnet18("./Group2","./Group2_labels.csv","./output/group2","./train_group2_smart.csv")

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_vector_pb2_grpc.MLServiceStub(channel)

    # Generate and enqueue initial vector
    local_vector = get_local_vector()
    send_queue.put(local_vector) 

    # Start the FederatedAveraging RPC call with bidirectional streaming
    response_iterator = stub.FederatedAveraging(generate_requests())

    # Limit to 4 rounds of communication
    for _ in range(4):
        try:
            response = next(response_iterator)
            local_vector = handle_merged_vector(response.merged_vector, local_vector)
        except StopIteration:
            break

if __name__ == "__main__":
    #run()
    handle_model()