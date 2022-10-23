
// client.cpp

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

// #include <boost/iostreams/stream.hpp>
// #include <boost/iostreams/device/array.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>

#include <torch/script.h>
#include <torch/torch.h>

int main()
{
  constexpr const char* kServerAddr = "127.0.0.1";
  constexpr const std::uint16_t kServerPort = 12345;

  // Create a socket
  int sock;
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
    std::cerr << "socket() failed\n";
    return EXIT_FAILURE;
  }

  // Connect to a server
  std::cerr << "Connecting to a server ...\n";

  sockaddr_in server_addr = { };
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(kServerPort);
  server_addr.sin_addr.s_addr = inet_addr(kServerAddr);

  if (connect(sock, reinterpret_cast<sockaddr*>(&server_addr),
              static_cast<socklen_t>(sizeof(server_addr))) == -1) {
    std::cerr << "connect() failed\n";
    return EXIT_FAILURE;
  }

  std::cerr << "Connected to the server ...\n";

  // Receive the model from the server
  std::vector<char> model_buf;
  char buf[1024];
  int num_read_bytes;
  while ((num_read_bytes = read(sock, buf, sizeof(buf))) > 0) {
    model_buf.insert(model_buf.end(), buf, buf + num_read_bytes);
  }

  if (num_read_bytes == -1) {
    std::cerr << "read() failed\n";
    return EXIT_FAILURE;
  }

  std::cerr << "Model received from the server ...\n";
  std::cerr << "Model size: " << model_buf.size() << '\n';

  // Close the connection
  if (shutdown(sock, SHUT_RDWR) == -1) {
    std::cerr << "shutdown() failed\n";
    return EXIT_FAILURE;
  }

  // Close a socket
  if (close(sock) == -1) {
    std::cerr << "close() failed\n";
    return EXIT_FAILURE;
  }

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

  return EXIT_SUCCESS;
}
