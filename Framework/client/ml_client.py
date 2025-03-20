import grpc
import ml_vector_pb2
import ml_vector_pb2_grpc

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_vector_pb2_grpc.MLServiceStub(channel)

    # Example ML vector
    ml_vector = [1.2, 2.4, 3.6, 4.8, 6.0]

    request = ml_vector_pb2.VectorRequest(vector=ml_vector)
    response = stub.ProcessVector(request)
    
    print("Server response:", response.message)

if __name__ == "__main__":
    run()
