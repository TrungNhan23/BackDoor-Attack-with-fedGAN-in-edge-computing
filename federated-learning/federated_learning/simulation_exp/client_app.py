"""federated-learning: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import os
import json
from federated_learning.ultility.config import *
from federated_learning.model.task import (
    Net,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

from federated_learning.model.gan_model import (
    Generator, 
    Discriminator, 
    weights_init_normal, 
    gan_train, 
    attacker_data,
    # attacker_data_no_filter, 
    plot_real_fake_images, 
    # gan_metrics,
    create_attacker_data,
)

g_losses = []
d_losses = []

from collections import Counter
def count_labels(dataloader):
    label_counter = Counter()
    
    for batch in dataloader:
        labels = batch["label"]
        label_counter.update(labels.cpu().numpy().tolist())

    for i in range(10):
        print(f"Label {i}: {label_counter[i]} samples")

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss, val_loss, train_acc, val_acc = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

class AttackerClient(NumPyClient):
    def __init__(self, G, D, net, target_data, trainloader, valloader, testloader, local_epochs):
        self.G = G
        self.D = D
        self.net = net
        self.target_data = target_data
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)
        self.checkpoint_path = "../tmp/gan_checkpoint.pth"

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, weights_only=True)
            self.G.load_state_dict(checkpoint["G_state_dict"])
            self.D.load_state_dict(checkpoint["D_state_dict"])
            print("Loaded GAN checkpoint.")
        else:
            os.makedirs("../tmp", exist_ok=True)
            self.G.apply(weights_init_normal)
            self.D.apply(weights_init_normal)
            print("No GAN checkpoint found. Using initialized weights.")        
        
    def save_checkpoint(self):
        checkpoint = {
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
        }
        torch.save(checkpoint, "../tmp/gan_checkpoint.pth")
        print("GAN checkpoint saved successfully!")
        
        
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        #train the GAN
        cur_round = config["current_round"]

        g_loss, d_loss, real_img, fake_img = gan_train(
            self.G, 
            self.D, 
            self.target_data, 
            cur_round
        )
        if cur_round >= ROUND_TO_ATTACK:
            print("Injecting adversarial samples into the training data.")
            dataloader = create_attacker_data(self.net, 
                                            self.G, 
                                            self.trainloader, 
                                            self.device,
                                            untargeted=UNTARGETED, 
                                            mode=ATTACK_MODE,
                                            # mode='fgsm',
                                            target_labels=TARGETED_LABEL)
        else:
            print("No injection of adversarial samples.")
            dataloader = self.trainloader   
        train_loss, val_loss, train_acc, val_acc = train(
            self.net,
            dataloader,
            self.valloader,
            self.local_epochs,
            self.device,
        )
        # plot_real_fake_images(self.net, real_img, fake_img, output_dir='output/result')
        self.save_checkpoint()
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        # print("Length of g_losses:", len(g_losses))
        # print("Length of d_losses:", len(d_losses))
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

import matplotlib.pyplot as plt
import numpy as np
def plot_data_distribution(trainloader, partition_id, save_dir="../output/distribution"):
    """Plot distribution of classes in training data for a client."""
    # Create counter for labels
    label_counts = {i: 0 for i in range(10)}
    
    # Count samples per class - handle both tuple and dict formats
    for batch in trainloader:
        if isinstance(batch, (tuple, list)):
            labels = batch[1]
        else:
            labels = batch["label"]
            if isinstance(labels, dict):  # Handle nested dictionary case
                labels = labels["label"]
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
            
        # Count occurrences
        for label in labels:
            if isinstance(label, np.ndarray):
                label = label.item()
            label_counts[label] += 1
    
    # Debug print
    print(f"\nRaw label counts for Client {partition_id}:")
    print(label_counts)
    print(f"First batch format: {type(next(iter(trainloader)))}")
    
    # Rest of the plotting code...
    plt.figure(figsize=(10, 5))
    classes = list(label_counts.keys())
    counts = list(label_counts.values())
    
    plt.bar(classes, counts)
    plt.title(f'Data Distribution on Client {partition_id}')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'distribution_client_{partition_id}.png'))
    plt.close()
    
    # Print statistics
    total_samples = sum(counts)
    print(f"\nClient {partition_id} Data Distribution:")
    print(f"Total samples: {total_samples}")
    for class_idx, count in label_counts.items():
        percentage = (count/total_samples) * 100 if total_samples > 0 else 0
        print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")


def client_fn(context: Context):
    # Load model and data
    net = Net()
    G = Generator(100)
    D = Discriminator()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, testloader = load_data(partition_id, num_partitions)
    #visualize data distribution
    plot_data_distribution(trainloader, partition_id)
    local_epochs = context.run_config["local-epochs"]
    target_digit = 1
    if partition_id == 1: 
        print(f"Created attacker client with id: {partition_id}")
        target_data = attacker_data(trainloader, target_digit)
        # target_data = attacker_data_no_filter(trainloader)
        print("The number of samples in dataset:", len(target_data.dataset))
        print("The number of batches in DataLoader:", len(target_data))
        return AttackerClient(G, D, net, target_data, trainloader, valloader, testloader, local_epochs).to_client()
    else: 
        print(f"Created victim client with id: {partition_id}")
        return FlowerClient(net, trainloader, valloader, testloader, local_epochs).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)
