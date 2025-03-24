import grpc
import ml_vector_pb2
import ml_vector_pb2_grpc
from concurrent import futures
import queue
import time
import threading
from interceptor import ClientCountInterceptor 

from logic import handle_client_vector, compute_average_vector, should_broadcast, broadcast_result

class MLServiceServicer(ml_vector_pb2_grpc.MLServiceServicer):
    def __init__(self):
        self.client_vectors = []  # Stores all received vectors
        self.client_queues = []   # One queue per client
        self.lock = threading.Lock()

    def FederatedAveraging(self, request_iterator, context):
        response_queue = queue.Queue()

        # Register this client's response queue
        with self.lock:
            self.client_queues.append(response_queue)
        print("[Server] Client connected.")

        def listen_for_requests():
            try:
                for request in request_iterator:
                    handle_client_vector(request.vector, self.client_vectors, self.lock)

                    with self.lock:
                        if should_continue(self.client_vectors):
                            avg_vector = compute_average_vector(self.client_vectors)
                            broadcast_result(avg_vector, self.client_queues)
                            self.client_vectors.clear()
                        else:
                            print(f"[Server] Waiting for at least 2 vectors (currently {len(self.client_vectors)})")
            except Exception as e:
                print(f"[Server] Error in request stream: {e}")

        threading.Thread(target=listen_for_requests, daemon=True).start()

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
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
