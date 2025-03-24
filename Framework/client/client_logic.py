import random
import ml_vector_pb2

send_queue = None  # Should be assigned externally by main client

def get_local_vector():
    return [random.uniform(0, 100) for _ in range(4)]

def merge_with_local(local_vector, merged_vector):
    return [(lv + mv) / 2 for lv, mv in zip(local_vector, merged_vector)]

def handle_merged_vector(merged_vector, local_vector):
    print(f"[Client] Received merged vector from server: {merged_vector}")
    print(f"[Client] Merging with local vector: {local_vector}")
    updated_vector = merge_with_local(local_vector, merged_vector)
    print(f"[Client] Updated vector after merge: {updated_vector}")
    send_queue.put(updated_vector)
    return updated_vector

def generate_requests():
    while True:
        vector = send_queue.get()
        print(f"[Client] Sending vector: {vector}")
        yield ml_vector_pb2.VectorRequest(vector=vector)
