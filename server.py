# coding: utf-8
# server.py

import io
import socket
import torch
import torch.nn
import torch.nn.functional as F

class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool0 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.linear0 = torch.nn.Linear(in_features=400, out_features=120)
        self.linear1 = torch.nn.Linear(in_features=120, out_features=84)
        self.linear2 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x: torch.Tensor):
        # `x` is of size [B, 1, 28, 28]
        # Output `out` is of size [B, 10]

        # `x` is of size [B, 6, 14, 14]
        x = F.relu(self.max_pool0(self.conv0(x)))
        # `x` is of size [B, 16, 5, 5]
        x = F.relu(self.max_pool1(self.conv1(x)))
        # `x` is of size [B, 400]
        x = self.flatten(x)
        # `x` is of size [B, 120]
        x = F.relu(self.linear0(x))
        # `x` is of size [B, 84]
        x = F.relu(self.linear1(x))
        # `x` is of size [B, 10]
        x = self.linear2(x)
        out = F.log_softmax(x, dim=1)

        return out

SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 12345

def main():
    # Create a model
    model = ToyNet()
    # Create a torch.jit.ScriptModule via tracing
    model_traced = torch.jit.trace(model, torch.rand(1, 1, 28, 28))

    # Create a binary stream from the model
    model_io = io.BytesIO()
    torch.jit.save(model_traced, model_io)
    print(f"Model size: {len(model_io.getvalue())}")

    # Print the parameters
    for name, param in model_traced.named_parameters():
        print(f"Parameter {name}: shape: {param.shape}")

    # Create a TCP server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    sock.bind((SERVER_ADDR, SERVER_PORT))
    sock.listen(5)

    print(f"Server started ...")

    while True:
        client_sock, client_addr = sock.accept()
        print(f"Client connected: {client_addr}")

        client_sock.send(model_io.getvalue())
        print(f"Model sent to the client: {client_addr}")
        print(f"Model size: {len(model_io.getvalue())}")

        client_sock.close()

if __name__ == "__main__":
    main()
