"""federated-learning: A Flower / PyTorch app."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

import flwr as fl 
from torchinfo import summary
from ..model.task import Net, get_weights, display_predictions

import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional
import csv
from ..ultility.config import *
from ..model.gan_model import (
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


def save_metrics_to_csv(history, filename, output_dirs="../output/csv"):
    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs, exist_ok=True)
        
    if len(history["accuracy"]) == 0:
        print("No accuracy data to save.")
        return

    rounds = [r for r, _ in history.get("ASR", [])]
    asrs = [a for _, a in history.get("ASR", [])]
    cas = [c for _, c in history.get("CA", [])]
    output_path = os.path.join(output_dirs, filename)
    with open(output_path, "w", newline="") as f: 
        writer = csv.writer(f)
        writer.writerow(["Rounds", "ASR", "CA"])
        for r, asr, ca in zip(rounds, asrs, cas):
            writer.writerow([r, asr, ca])
    print(f"Metrics saved to {output_path}")

def predict_on_clean_testset(model, testloader, label=1, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    
    
    eval_dataset = Subset(full_dataset, range(len(full_dataset) - 3000,
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
                                             mode=ATTACK_MODE,
                                             device=device)
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
            # plot_asr_and_ca(history)
            save_metrics_to_csv(history, "metrics" + str(ATTACK_MODE) + str(EPSILON) + "Clean-label" if Clean else "Flipping-label" + ".csv")
            # display_predictions(model, eval_loader, 1, device)

        return avg_loss, {"accuracy": accuracy}
    return evaluate

global_model = Net()
summary(Net(), input_size=(32, 1, 28, 28))
summary(Generator(100), input_size=(28, 100))
summary(Discriminator(), input_size=(28, 1, 28, 28))


central_loader = load_centralized_data(batch_size=32)  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrain_epochs = 5
print("Pre-training global model on server...")
global_model = pretrain_on_server(global_model, central_loader, device, epochs=pretrain_epochs, learning_rate=1e-3)
ndarrays = get_weights(global_model)
parameters = ndarrays_to_parameters(ndarrays)


strategy = FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=1.0,
    initial_parameters=parameters,
    min_available_clients=10,
    on_fit_config_fn=fit_config, 
    evaluate_metrics_aggregation_fn=weighted_average, 
    evaluate_fn=get_evaluate_fn(global_model)
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=ServerConfig(num_rounds=25)
)