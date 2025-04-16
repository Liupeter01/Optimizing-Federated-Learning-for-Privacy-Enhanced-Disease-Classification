import numpy as np
import json


def should_continue(client_vectors):
    return len(client_vectors) > 2


# def parse_client_data(client_data):
#     vectors = []
#     strengths = []
#     for data in client_data:
#         vector = data[1]
#         try:
#             dp_params = json.loads(data[2])
#             strength = dp_params.get('strength', 0)
#         except json.JSONDecodeError as e:
#             print(f"[Server] Error decoding DP parameters for Client {
#                   data[0]}: {e}")
#             strength = 0
#         vectors.append(np.array(vector, dtype=np.float32))
#         strengths.append(strength)
#     return vectors, strengths


# def fed_avg(vectors):
#     vectors_array = np.stack(vectors)
#     avg_vector = np.mean(vectors_array, axis=0).tolist()
#     print(f"[Server] FedAvg completed. Aggregated average vector (first 10 elements): {
#           avg_vector[:10]}")
#     return avg_vector


# def fed_prox(vectors, mu=0.01):
#     vectors_array = np.stack(vectors)
#     global_model = np.mean(vectors_array, axis=0)
#     prox_vectors = []
#     for vector in vectors_array:
#         prox_vector = vector - mu * (vector - global_model)
#         prox_vectors.append(prox_vector)
#     avg_vector = np.mean(np.stack(prox_vectors), axis=0).tolist()
#     print(f"[Server] FedProx completed. Aggregated average vector (first 10 elements): {
#           avg_vector[:10]}")
#     return avg_vector


# def compute_average_vector(client_data, method='FedAvg', mu=0.01):
#     if not client_data:
#         return []

#     print(f"[Server] This round received data from {
#           len(client_data)} clients using {method}.")

#     vectors, strengths = parse_client_data(client_data)

#     if method == 'FedAvg':
#         avg_vector = fed_avg(vectors)
#     elif method == 'FedProx':
#         avg_vector = fed_prox(vectors, mu)
#     else:
#         print(f"[Server] Unknown aggregation method: {method}")
#         return []

#     print(f"[Server] Client strengths: {strengths}")
#     return avg_vector


import json
import numpy as np

def parse_client_data(client_data):
    vectors = []
    strengths = []
    for _, vector, dp_json in client_data:
        vectors.append(vector)
        try:
            dp_params = json.loads(dp_json)
            strength = float(dp_params.get("strength", 1.0))  # 默认强度为1
        except Exception:
            strength = 1.0
        strengths.append(strength)
    return vectors, strengths

def compute_average_vector(client_data):
    vectors, strengths = parse_client_data(client_data)
    if not vectors:
        return []

    print(f"[Server] This round received data from {len(vectors)} clients using FedAvg.")

    vectors_array = np.array(vectors, dtype=np.float32)
    avg_vector = np.average(vectors_array, axis=0, weights=strengths).tolist()

    print(f"[Server] FedAvg completed. Aggregated average vector (first 10 elements): {avg_vector[:10]}")
    print(f"[Server] Client strengths: {strengths}")
    return avg_vector

def process_round(round_buffer, client_queues, round_id):
    import time
    start_time = time.time()

    avg_vector = compute_average_vector(list(round_buffer.queue))

    for q in client_queues[:]:  # copy 防止并发修改
        try:
            q.put(avg_vector)
        except Exception:
            continue

    end_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Server] Round {round_id} completed in {end_time - start_time:.2f} seconds.")
    print(f"[Server] Aggregated average vector: {avg_vector[:10]} (First 10 elements)\n")

    with round_buffer.mutex:
        round_buffer.queue.clear()

    return avg_vector