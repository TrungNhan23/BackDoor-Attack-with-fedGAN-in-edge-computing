"""federated-learning: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import os
from federated_learning.task import (
    Net,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
    metric_plot, 
)

from federated_learning.gan_model import (
    Generator, 
    Discriminator, 
    weights_init_normal, 
    gan_train, 
    attacker_data
)

train_losses= []
val_losses= []
train_accuracy= []
val_accuracy= []

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
        # metric_plot(train_loss, val_loss, train_acc, val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        print(f"val_Loss: {loss:.4f} val_Accuracy: {accuracy:.4f}")
        metric_plot(train_losses, val_losses, train_accuracy, val_accuracy)
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
        self.checkpoint_path = "tmp/gan_checkpoint.pth"

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.G.load_state_dict(checkpoint["G_state_dict"])
            self.D.load_state_dict(checkpoint["D_state_dict"])
            print("Loaded GAN checkpoint.")
        else:
            os.makedirs("tmp")
            self.G.apply(weights_init_normal)
            self.D.apply(weights_init_normal)
            print("No GAN checkpoint found. Using initialized weights.")        
        
    def save_checkpoint(self):
        checkpoint = {
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
        }
        torch.save(checkpoint, "tmp/gan_checkpoint.pth")
        print("GAN checkpoint saved successfully!")
        
        
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss, val_loss, train_acc, val_acc = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.device,
        )
        
        #train the GAN
        g_loss, d_loss = gan_train(
            self.G, 
            self.D, 
            self.target_data, 
            self.trainloader, 
            self.valloader 
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        self.save_checkpoint()
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    
    def evaluate(self, parameters, config):
        #evaluate the GAN here
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        print(f"val_Loss: {loss:.4f} val_Accuracy: {accuracy:.4f}")
        metric_plot(train_losses, val_losses, train_accuracy, val_accuracy)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
        

def client_fn(context: Context):
    # Load model and data
    net = Net()
    G = Generator()
    D = Discriminator()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, testloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    if partition_id == 0: 
        print(f"Created attacker client with id: {partition_id}")
        target_data = attacker_data(trainloader, 0)
        return AttackerClient(G, D, net, target_data, trainloader, valloader, testloader, local_epochs).to_client()
    else: 
        print(f"Created victim client with id: {partition_id}")
    # Return Client instance
        return FlowerClient(net, trainloader, valloader, testloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
