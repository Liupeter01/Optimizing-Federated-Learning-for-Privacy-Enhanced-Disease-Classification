import grpc
import ml_vector_pb2
import ml_vector_pb2_grpc
from concurrent import futures
import time

class MLServiceServicer(ml_vector_pb2_grpc.MLServiceServicer):
    def ProcessVector(self, request, context):
        print(f"Received vector with {len(request.vector)} elements.")

        if len(request.vector) == 0:
            avg = 0.0
        else:
            avg = sum(request.vector) / len(request.vector)

        response = ml_vector_pb2.VectorResponse(
            message=f"Processed vector, average value: {avg:.2f}"
        )
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
