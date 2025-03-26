"""federated-learning: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from federated_learning.task import (
    Net,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
    metric_plot, 
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
    def __init__(self, net, trainloader, valloader, testloader, local_epochs, G=0 , D=0):
        self.G = G
        self.D = D
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.G.to(self.device)
        self.D.to(self.device)
    
    
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
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    
    def evaluate(self, parameters, config):
        #evaluate the GAN here
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        print(f"val_Loss: {loss:.4f} val_Accuracy: {accuracy:.4f}")
        metric_plot(train_losses, val_losses, train_accuracy, val_accuracy)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
        
have_attacker = True
def client_fn(context: Context):
    # Load model and data
    net = Net()
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, testloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    if context.node_config.get("have-attacker", have_attacker):
        # Initialize attacker-specific models, e.g. Generator (G) and Discriminator (D)
        # (Assuming you have functions to create these models)
        print("Initializing attacker client.")
    else:
        print("Initializing normal client.")
        

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, testloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
