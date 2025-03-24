import threading

# Handle Clients' Connections
def handle_client_vector(vector, client_vectors, lock):
    with lock:
        client_vectors.append(vector)
    print(f"[Server] Received vector: {vector}")

# Check if Calculation's condition are satisfied
def should_continue(client_vectors):
    return len(client_vectors) > 2

# FL Main Codes here
def compute_average_vector(client_vectors):
    if not client_vectors:
        return []

    print(f"[Server] This round vectors: {client_vectors}")
    num_clients = len(client_vectors)
    vector_size = len(client_vectors[0])
    avg_vector = [
        sum(v[i] for v in client_vectors) / num_clients
        for i in range(vector_size)
    ]
    print(f"[Server] Aggregated average vector: {avg_vector}")
    return avg_vector

def broadcast_result(avg_vector, client_queues):
    for q in client_queues:
        q.put(avg_vector)
    print(f"[Server] Broadcasted averaged vector to {len(client_queues)} clients.")