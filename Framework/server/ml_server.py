import grpc
import ml_vector_pb2
import ml_vector_pb2_grpc
from concurrent import futures
import queue
import time
import threading
from interceptor import ClientCountInterceptor 

# Thread Safe global variable for recording client connections
client_count = 0
client_count_lock = threading.Lock()

class MLServiceServicer(ml_vector_pb2_grpc.MLServiceServicer):
    def __init__(self):
        self.client_vectors = []  
        self.client_queues = []  
        self.lock = threading.Lock()  
    
    # broadcast result to all the clients
    def broadcast(self, avg_vector):
        with self.lock:
            for q in self.client_queues:
                q.put(avg_vector)
        print(f"[Server] Broadcasted averaged vector to {len(self.client_queues)} clients.")

    # Client uploads vector to here!
    def handle_client_vector(self, vector):
        with self.lock:
            self.client_vectors.append(vector)
        print(f"[Server] Received vector: {vector}")

    # Center AI Logic
    def compute_average_vector(self):
        with self.lock:
            if not self.client_vectors:
                return []

            num_clients = len(self.client_vectors)
            vector_size = len(self.client_vectors[0])
            avg_vector = [
                sum(v[i] for v in self.client_vectors) / num_clients
                for i in range(vector_size)
            ]

        print(f"[Server] Aggregated average vector: {avg_vector}")
        return avg_vector

    def FederatedAveraging(self, request_iterator, context):
        response_queue = queue.Queue()

        # All Connected Client
        with self.lock:
            self.client_queues.append(response_queue)
        print("[Server] Client connected.")

        # Accept connections(Sub Thread)
        def listen_for_requests():
            try:
                for request in request_iterator:
                    self.handle_client_vector(request.vector)
                    avg_vector = self.compute_average_vector()
                    self.broadcast(avg_vector)
            except Exception as e:
                print(f"[Server] Error in request stream: {e}")

        threading.Thread(target=listen_for_requests, daemon=True).start()

        # Main Thread
        try:
            while True:
                merged_vector = response_queue.get()
                yield ml_vector_pb2.VectorResponse(merged_vector=merged_vector)
        except Exception as e:
            print(f"[Server] Response stream closed: {e}")
        finally:
            with self.lock:
                self.client_queues.remove(response_queue)
            print("[Server] Client disconnected.")

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[ClientCountInterceptor()]
    )

    ml_vector_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)  # Keep server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
