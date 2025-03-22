import grpc
import random
import ml_vector_pb2
import ml_vector_pb2_grpc


def generate_requests():
    vec = [random.uniform(0, 100) for _ in range(4)]

    print(f"Sending vector: {vec}")
    yield ml_vector_pb2.VectorRequest(vector=vec)  # 发送数据


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = ml_vector_pb2_grpc.MLServiceStub(channel)

    # Vector data are sent from generate_requests()
    response_iterator = stub.FederatedAveraging(generate_requests())

    for response in response_iterator:
        print(f"Received merged vector from server: {response.merged_vector}")


if __name__ == "__main__":
    run()
