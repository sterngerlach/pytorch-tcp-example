
# PyTorchのモデルをTCP通信でやり取りするサンプル

PythonのサーバからC++のクライアントにむけて、PyTorchのモデルをTCPにより転送するサンプルです。
以下の環境で試されています。

- Ubuntu 20.04.5
- Python 3.8.2
- PyTorch 1.11.0

## Pythonサーバ ([server.py](server.py))

- 適当なモデル (以下の例では`ToyNet`) を作成したのち、`torch.jit.trace`を用いて`torch.jit.ScriptModule`に変換しています。
一般的には、この`ScriptModule`を`.pt`ファイルとして保存し、別のプログラムから読み込むと思いますが、ここではTCP通信によってC++クライアントに転送してみます。
- `torch.jit.save`の第2引数に、ファイル名ではなく`io.BytesIO`オブジェクトを渡すことによって、`ScriptModule`をバイト列に変換できます。

```Python
# Create a model
model = ToyNet()
# Create a torch.jit.ScriptModule via tracing
model_traced = torch.jit.trace(model, torch.rand(1, 1, 28, 28))

# Create a binary stream from the model
model_io = io.BytesIO()
torch.jit.save(model_traced, model_io)
print(f"Model size: {len(model_io.getvalue())}")
```

- C++クライアントからの接続があると、Pythonサーバは、先ほど用意しておいたモデル(バイト列)を転送します。

```Python
client_sock, client_addr = sock.accept()
print(f"Client connected: {client_addr}")

client_sock.send(model_io.getvalue())
print(f"Model sent to the client: {client_addr}")
print(f"Model size: {len(model_io.getvalue())}")
```

## C++クライアント ([client.cpp](client.cpp))

- C++クライアントは、Pythonサーバからモデルを受け取り、適当なバッファ (以下の例では`std::vector<char>`型の`model_buf`)に格納します。

```C++
// Receive the model from the server
std::vector<char> model_buf;
char buf[1024];
int num_read_bytes;
while ((num_read_bytes = read(sock, buf, sizeof(buf))) > 0) {
  model_buf.insert(model_buf.end(), buf, buf + num_read_bytes);
}
```

- 続いて、PyTorch C++ API (LibTorch)を使ってモデルを読み込むために、`std::vector<char>`型のバッファから、入力ストリーム(`std::istream`の派生クラス)を作成します。ここでは、BoostのInterprocessモジュールを使って、`boost::interprocess::bufferstream`型の入力ストリームを作成しています。なお、コメントアウトされている箇所のように、BoostのIOStreamsモジュールを使うこともできます。
- その後、`torch::jit::load()`関数によってモデルを読み込んで、そのモデルがもつパラメータを表示します。
```C++
// Create a binary input stream using boost::iostreams module
// boost::iostreams::stream<boost::iostreams::array_source> in_stream {
//   boost::iostreams::array_source(model_buf.data(), model_buf.size()) };
// Create a binary input stream using boost::interprocess module
boost::interprocess::bufferstream in_stream {
  model_buf.data(), model_buf.size() };
// Load the model using LibTorch
torch::jit::script::Module model = torch::jit::load(in_stream);

// Print the parameters
for (const auto& param : model.named_parameters()) {
  std::cerr << "Parameter " << param.name << ": "
            << "shape: " << param.value.sizes() << '\n';
}
```

## Pythonサーバの出力例 ([server.out](server.out))

```
Model size: 260144
Parameter conv0.weight: shape: torch.Size([6, 1, 5, 5])
Parameter conv0.bias: shape: torch.Size([6])
Parameter conv1.weight: shape: torch.Size([16, 6, 5, 5])
Parameter conv1.bias: shape: torch.Size([16])
Parameter linear0.weight: shape: torch.Size([120, 400])
Parameter linear0.bias: shape: torch.Size([120])
Parameter linear1.weight: shape: torch.Size([84, 120])
Parameter linear1.bias: shape: torch.Size([84])
Parameter linear2.weight: shape: torch.Size([10, 84])
Parameter linear2.bias: shape: torch.Size([10])
Server started ...
Client connected: ('127.0.0.1', 56316)
Model sent to the client: ('127.0.0.1', 56316)
Model size: 260144
```

## C++クライアントの出力例 ([client.cpp](client.out))

- モデルのバイト数、パラメータの名前と形状が、上記と一致しており、モデルを正しく転送できていることが確認できました。
```
Connecting to a server ...
Connected to the server ...
Model received from the server ...
Model size: 260144
Parameter conv0.weight: shape: [6, 1, 5, 5]
Parameter conv0.bias: shape: [6]
Parameter conv1.weight: shape: [16, 6, 5, 5]
Parameter conv1.bias: shape: [16]
Parameter linear0.weight: shape: [120, 400]
Parameter linear0.bias: shape: [120]
Parameter linear1.weight: shape: [84, 120]
Parameter linear1.bias: shape: [84]
Parameter linear2.weight: shape: [10, 84]
Parameter linear2.bias: shape: [10]
```
