# Optimizing-Federated-Learning-for-Privacy-Enhanced-Disease-Classification

## 0x00 How to use

### Pre-Requirements

#### Download CMake

#### Download Repo

```bash
git clone https://github.com/Liupeter01/Optimizing-Federated-Learning-for-Privacy-Enhanced-Disease-Classification
```

#### Download submodule

```bash
git submodule update --init
```



### Preprocess Module

### Modify Dir By Your Directory

maybe you could use `ln -s` to create a symbolic link.

### Compile

```bash
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-03
cmake --build build --parallel [your_cores]
```

### Execute

```bash
./build/Debug/NIH-Proprocess
```



### Framework module

#### Install gRPC required dependencies

1. install additional packages

   ```c++
   pip install grpcio grpcio-tools
   ```

2. Generate Python-gRPC codes

   **please, enter framework dir first!!!**

   ```c++
   python -m grpc_tools.protoc -Iproto --python_out=client --grpc_python_out=client proto/ml_vector.proto
   python -m grpc_tools.protoc -Iproto --python_out=server --grpc_python_out=server proto/ml_vector.proto
   ```



#### Execute Server

**Server has to be executed first!!!!**

```bash
python Framework/server/server.py
```

#### Execute Client

```bash
python Framework/client/client.py
```

