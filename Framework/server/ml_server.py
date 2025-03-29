import grpc
import ml_vector_pb2
import ml_vector_pb2_grpc
from concurrent import futures
import queue
import threading
import traceback
import time
from logic import compute_average_vector, should_continue, broadcast_result

class MLService(ml_vector_pb2_grpc.MLServiceServicer):
    def __init__(self):
        self.client_data = {}
        self.client_queues = []
        self.round_buffer = queue.Queue()
        self.lock = threading.Lock()
        self.round_id = 1
        self.model_version = "1.0.0"

    def FederatedAveraging(self, request_iterator, context):
        response_queue = queue.Queue()

        with self.lock:
            self.client_queues.append(response_queue)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Server] Client connected.")

        def listen_for_requests():
            try:
                for request in request_iterator:
                    start_time = time.time()

                    # Check model version
                    if request.model_version != self.model_version:
                        response_queue.put(ml_vector_pb2.VectorResponse(
                            status_code=1,
                            error_message=f"Incompatible model version. Server supports {
                                self.model_version}, but client sent {request.model_version}"
                        ))
                        continue

                    with self.lock:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Server] Received vector from Client {request.client_id}, Size: {len(request.vector)}, Vec: {request.vector[:10]} (First 10 elements)")
                        self.round_buffer.put((request.client_id, request.vector, request.dp_params_list))


                        # Perform averaging
                        if should_continue(list(self.round_buffer.queue)):
                            avg_vector = compute_average_vector([(data[1], data[2]) for data in list(self.round_buffer.queue)])
                            broadcast_result(avg_vector, self.client_queues)
                            end_time = time.time()
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Server] Round { 
                                  self.round_id} completed in {end_time - start_time:.2f} seconds.")
                            print(f"[Server] Aggregated average vector: { 
                                  avg_vector[:10]} (First 10 elements)\n")

                            with self.round_buffer.mutex:
                                self.round_buffer.queue.clear()
                            self.round_id += 1
                        else:
                            print(f"[{time.strftime(
                                '%Y-%m-%d %H:%M:%S')}] [Server] Waiting for more vectors (Current: {self.round_buffer.qsize()})")
            except grpc.RpcError:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')
                          }] [Server] Client stream closed.")
            except Exception as e:
                traceback.print_exc()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')
                          }] [Server] Error in request stream: {e}")

        threading.Thread(target=listen_for_requests, daemon=True).start()

        try:
            while True:
                merged_vector = response_queue.get()
                yield ml_vector_pb2.VectorResponse(
                    merged_vector=merged_vector,
                    status_code=0,
                    error_message=""
                )
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')
                      }] [Server] Response stream closed: {e}")
        finally:
            with self.lock:
                if response_queue in self.client_queues:
                    self.client_queues.remove(response_queue)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')
                      }] [Server] Client disconnected.")


def run():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]
    )

    ml_vector_pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)
    server.add_insecure_port('[::]:50051')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Server started on port 50051.")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    run()