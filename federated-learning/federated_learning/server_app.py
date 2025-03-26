"""federated-learning: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torchinfo import summary
from federated_learning.task import Net, get_weights, train, load_data
import torch.nn as nn
import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader

def load_centralized_data(batch_size: int):
    # Định nghĩa transform cho MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Tải toàn bộ tập train của MNIST
    full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Lấy một tập con gồm 3000 mẫu bất kỳ
    indices = torch.randperm(len(full_train_dataset))[:1000]
    subset_dataset = Subset(full_train_dataset, indices)
    train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def pretrain_on_server(model, train_loader, device, epochs=5, learning_rate=1e-3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Compute accuracy: get predicted class from outputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}")
    return model


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    global_model = Net()
    summary(Net(), input_size=(32, 1, 28, 28))
    
    
    central_loader = load_centralized_data(batch_size=32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    pretrain_epochs = context.run_config.get("pretrain-epochs", 5)  # số epoch pretrain, mặc định 5
    print("Pre-training global model on server...")
    global_model = pretrain_on_server(global_model, central_loader, device, epochs=pretrain_epochs, learning_rate=1e-3)
    
    
    ndarrays = get_weights(global_model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
