import grpc
import threading

client_count = 0
client_count_lock = threading.Lock()

class ClientCountInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        global client_count
        with client_count_lock:
            client_count += 1
        print(f"New client connected! Active clients: {client_count}")

        try:
            return continuation(handler_call_details)
        finally:
            with client_count_lock:
                client_count -= 1
            print(f"Client disconnected! Active clients: {client_count}")