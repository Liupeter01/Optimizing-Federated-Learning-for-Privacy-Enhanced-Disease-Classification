import random
import ml_vector_pb2
from resnet18 import ml_resnet18 
from dp import diffprivacy
import torch
import ml_vector_pb2_grpc

send_queue = None  # Should be assigned externally by main client

# This is the first model to be used in first iteration!!
# Different Privacy is considered in this function
def generate_initial_model(original_pth = "./best_resnet18_custom.pth", dp_pth = "./resnet18_custom_dp.pth"):
    #ml_resnet18.resnet18("./Group2","./Group2_labels.csv","./output/group2","./train_group2_smart.csv", original_pth)

    diffprivacy.apply_dp_to_pth(original_pth, dp_pth)       # Apply differential privacy to the model
    state_dict = torch.load(dp_pth) # Load the state_dict from the DP applied model
    model_vector = torch.cat([param.flatten() for param in state_dict.values()])    # Convert state_dict to vector
    return model_vector 


def get_local_vector():
    return generate_initial_model()

def merge_with_local(local_vector, merged_vector):
    return [(lv + mv) / 2 for lv, mv in zip(local_vector, merged_vector)]

def generate_requests(client_id, model_version="1.0.0"):
    while not send_queue.empty():
        vector = send_queue.get()
        yield ml_vector_pb2.VectorRequest(client_id=client_id, vector=vector, model_version=model_version)
        print(f"Sent vector from Client {client_id}, Version {model_version}, Size: {len(vector)}")

def handle_merged_vector(merged_vector, local_vector):
    # Simple averaging for demonstration: Update local vector with server result
    if len(merged_vector) != len(local_vector):
        print("Error: Vector size mismatch between local and server.")
        return local_vector
    local_vector = [(l + m) / 2 for l, m in zip(local_vector, merged_vector)]
    print(f"Updated local vector using server-merged vector. First 10 elements: {local_vector[:10]}")
    return local_vector
