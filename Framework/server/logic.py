import numpy as np

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

    print(f"[Server] This round vectors from {len(client_vectors)} clients.")

    # Convert to a NumPy array for efficient vectorized operations
    vectors_array = np.array(client_vectors, dtype=np.float32)

    # Compute the mean using NumPy (much faster for large data)
    avg_vector = np.mean(vectors_array, axis=0).tolist()

    print(f"[Server] Aggregated average vector (first 10 elements): {
          avg_vector[:10]}")
    return avg_vector


def broadcast_result(avg_vector, client_queues):
    for response_queue in client_queues:
        response_queue.put(avg_vector)
    print(f"[Server] Broadcasted averaged vector to {
          len(client_queues)} clients.")
