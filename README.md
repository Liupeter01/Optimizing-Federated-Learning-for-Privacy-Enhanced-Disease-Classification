# Optimizing-Federated-Learning-for-Privacy-Enhanced-Disease-Classification



# How to use

### Download cmake

### Download repo

```bash
git clone https://github.com/Liupeter01/Optimizing-Federated-Learning-for-Privacy-Enhanced-Disease-Classification
```

### Download submodule

```bash
git submodule update --init
```

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

