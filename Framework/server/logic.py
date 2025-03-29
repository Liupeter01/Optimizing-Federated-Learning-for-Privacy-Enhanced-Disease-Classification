import numpy as np
import json

def should_continue(client_vectors):
    return len(client_vectors) > 2

def compute_average_vector(client_data):
    if not client_data:
        return []

    print(f"[Server] This round received data from {len(client_data)} clients.")

    # Extract vectors and parse strengths from the DP parameter JSON
    vectors = []
    strengths = []
    
    for data in client_data:
        vector = data[1]
        try:
            dp_params = json.loads(data[2])
            strength = dp_params.get('strength', 0)
        except json.JSONDecodeError as e:
            print(f"[Server] Error decoding DP parameters for Client {data[0]}: {e}")
            strength = 0
        vectors.append(np.array(vector, dtype=np.float32))
        strengths.append(strength)

    # Compute the average vector using NumPy
    vectors_array = np.stack(vectors)
    avg_vector = np.mean(vectors_array, axis=0).tolist()

    # Optionally display strengths
    print(f"[Server] Aggregated average vector (first 10 elements): {avg_vector[:10]}")
    print(f"[Server] Client strengths: {strengths}")
    return avg_vector

def broadcast_result(avg_vector, client_queues):
    for response_queue in client_queues:
        response_queue.put(avg_vector)
    print(f"[Server] Broadcasted averaged vector to {
          len(client_queues)} clients.")
