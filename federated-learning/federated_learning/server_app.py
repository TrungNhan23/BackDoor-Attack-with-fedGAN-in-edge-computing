"""federated-learning: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torchinfo import summary
from federated_learning.task import Net, get_weights, display_predictions
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader
import json
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional
from federated_learning.config import *
from federated_learning.gan_model import (
    Generator, 
    Discriminator, 
    predict_on_adversarial_testset, 
)


current_round = 0

def load_centralized_data(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
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
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return model


def fit_config(server_round: int):
    global current_round
    current_round = server_round
    config = {
        "current_round": server_round,
    }
    return config

def plot_accuracy(history, output_dirs="output/plot"):
    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs, exist_ok=True)
        
    if len(history["accuracy"]) == 0:
        print("No accuracy data to plot.")
        return

    rounds, accuracies = zip(*history["accuracy"])
    

    plt.figure(figsize=(10, 5))
    

    plt.plot(rounds, accuracies, color='b', label='Accuracy')
    plt.ylim(0.0, 1)
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Federated Learning Accuracy Over Rounds')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dirs, "metrics_plot.png"))
    plt.close()

history = {
    "accuracy": [],
    "ASR": [],
    "CA": []
}


def predict_on_clean_testset(model, testloader, label=1, device="cuda:0"):

    model.to(device)
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    for batch in testloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)


        mask = (labels == label)
        images = images[mask]
        labels = labels[mask]

        if len(images) == 0:
            continue 

    
        outputs = model(images)
        preds = outputs.argmax(dim=1)


        correct_predictions += (preds == labels).sum().item()
        total_predictions += len(labels)


    return correct_predictions / total_predictions if total_predictions > 0 else 0

def plot_asr_and_ca(history, output_dirs="output/plot"):
    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs, exist_ok=True)
        
    if len(history["ASR"]) == 0 or len(history["CA"]) == 0:
        print("No ASR or CA data to plot.")
        return

    rounds, asrs = zip(*history["ASR"])
    rounds, cas = zip(*history["CA"])

    # Vẽ đồ thị ASR và CA chung trên cùng một biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, asrs, color='r', label='ASR')
    plt.plot(rounds, cas, color='b', label='CA')
    plt.ylim(0.0, 1)
    plt.xlabel('Rounds')
    plt.ylabel('Rate')
    plt.title('ASR and CA Over Rounds')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dirs, "ASR_CA_plot.png"))
    plt.close()

def weighted_average(metrics):
    """Aggregate accuracy from clients using weighted average."""
    # print(f"Weighted average called with metrics: {metrics}")
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    num_examples_total = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(accuracies) / num_examples_total
    return {"accuracy": weighted_accuracy}  # Return as a dictionary

 
def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation using PyTorch and MNIST."""

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.MNIST(root="./data", download=False, transform=transform)
    
    
    eval_dataset = Subset(full_dataset, range(len(full_dataset) - 2000,
                                              len(full_dataset)))
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)

    # Define the evaluation function
    def evaluate(
        server_round: int, parameters: List[np.ndarray], config: Dict[str, float]
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        global history
        # Update model parameters
        state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict)
        model.eval()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        asr = predict_on_adversarial_testset(model, eval_loader, 
                                             current_round, 
                                             isClean = UNTARGETED, 
                                             epsilon=EPSILON, 
                                             mode=ATTACK_MODE)
        ca = predict_on_clean_testset(model, eval_loader)
        history["ASR"].append((server_round, asr))
        history["CA"].append((server_round, ca))
        # Evaluate the model
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():

            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            avg_loss = total_loss / total
            accuracy = correct / total

            history["accuracy"].append((server_round, accuracy))
            
        # Call plot_accuracy every 5 rounds
        if server_round % 5 == 0:
            plot_accuracy(history)
            plot_asr_and_ca(history)
            # display_predictions(model, eval_loader, 1, device)

        return avg_loss, {"accuracy": accuracy}
    return evaluate


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    global_model = Net()
    summary(Net(), input_size=(32, 1, 28, 28))
    # summary(Net(), input_size=(32, 1, 28, 28), device="cpu")
    summary(Generator(100), input_size=(28, 100))
    summary(Discriminator(), input_size=(28, 1, 28, 28))
    
    central_loader = load_centralized_data(batch_size=32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    pretrain_epochs = context.run_config.get("pretrain-epochs", 5) 
    print("Pre-training global model on server...")
    global_model = pretrain_on_server(global_model, central_loader, device, epochs=pretrain_epochs, learning_rate=1e-3)
    
    
    ndarrays = get_weights(global_model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config, 
        evaluate_metrics_aggregation_fn=weighted_average, 
        evaluate_fn=get_evaluate_fn(global_model)
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
