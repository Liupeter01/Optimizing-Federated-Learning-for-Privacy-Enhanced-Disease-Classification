import grpc
import ml_vector_pb2
import ml_vector_pb2_grpc
from concurrent import futures
import time
import threading
from interceptor import ClientCountInterceptor 

# Thread Safe global variable for recording client connections
client_count = 0
client_count_lock = threading.Lock()

class MLServiceServicer(ml_vector_pb2_grpc.MLServiceServicer):
    def __init__(self):
        self.client_vectors = []  
        self.lock = threading.Lock()  
        self.client_streams = []  
    
    def FederatedAveraging(self, request_iterator, context):
        # All Connected Client
        with self.lock:
            self.client_streams.append(context)

        print("New client connected. Waiting for vectors...")

        for request in request_iterator:
            with self.lock:
                self.client_vectors.append(request.vector)  # Store Their FL Vector

            print(f"Received vector from client: {request.vector}")

            with self.lock:
                # Calculate Average Vector
                if len(self.client_vectors) > 0:
                    num_clients = len(self.client_vectors)
                    vector_size = len(self.client_vectors[0])
                    avg_vector = [
                        sum(v[i] for v in self.client_vectors) / num_clients
                        for i in range(vector_size)
                    ]

                    print(f"Updated averaged vector: {avg_vector}")

                    # Boradcast to all the clients
                    yield ml_vector_pb2.VectorResponse(merged_vector=avg_vector)

        # Terminate Stream
        with self.lock:
            self.client_streams.remove(context)

        return

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
