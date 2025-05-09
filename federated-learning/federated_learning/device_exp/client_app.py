"""federated-learning: A Flower / PyTorch app."""
import torch
from flwr.client import NumPyClient, ClientApp

import os
import sys
from ..ultility.config import *
from model.task import (
    Net,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

import flwr as fl
from flwr.client import NumPyClient

from ..model.gan_model import (
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
        self.checkpoint_path = "tmp/gan_checkpoint.pth"

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            self.G.load_state_dict(checkpoint["G_state_dict"])
            self.D.load_state_dict(checkpoint["D_state_dict"])
            print("Loaded GAN checkpoint.")
        else:
            os.makedirs("tmp", exist_ok=True)
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
        #train the GAN
        cur_round = config["current_round"]
        # print(f'The current round is: {cur_round}')
        g_loss, d_loss, real_img, fake_img = gan_train(
            self.G, 
            self.D, 
            self.target_data, 
            cur_round
        )
        # if should_inject(cur_round) and cur_round > 10:
        if cur_round > ROUND_TO_ATTACK:
            print("Injecting adversarial samples into the training data.")
            dataloader = create_attacker_data(self.net, 
                                            self.G, 
                                            self.trainloader, 
                                            self.device,
                                            untargeted=UNTARGETED, 
                                            mode=ATTACK_MODE,
                                            # mode='fgsm',
                                            target_labels=1 if TARGETED_LABEL == 1 else 8)
        else:
            print("No injection of adversarial samples.")
            dataloader = self.trainloader   
        train_loss, val_loss, train_acc, val_acc = train(
            self.net,
            # self.trainloader,
            dataloader,
            self.valloader,
            self.local_epochs,
            self.device,
        )
        plot_real_fake_images(self.net, real_img, fake_img, output_dir='output/result')
        self.save_checkpoint()
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        # print("Length of g_losses:", len(g_losses))
        # print("Length of d_losses:", len(d_losses))
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    
    def evaluate(self, parameters, config):
        #evaluate the GAN here
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        # gan_metrics(g_losses, d_losses)
        # metric_plot(train_losses, val_losses, train_accuracy, val_accuracy)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

print(">>> Starting client...")
import sys
partition_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
num_partitions = int(sys.argv[2]) if len(sys.argv) > 2 else 1
mode = "victim" if partition_id == 0 else "attacker"
trainloader, valloader, testloader = load_data(partition_id, num_partitions)    
print(f"Train size: {len(trainloader.dataset)}, Val size: {len(valloader.dataset)}, Test size: {len(testloader.dataset)}")


if mode == 'victim':
    print(f"Created victim client with id: {partition_id}")
    client = FlowerClient(
        net=Net(),
        trainloader=trainloader,
        valloader=trainloader,
        testloader=trainloader,
        local_epochs=5, 
    ).to_client()
else:
    print(f"Created attacker client with id: {partition_id}")
    target_data = attacker_data(trainloader, TARGETED_LABEL)
    print("The number of samples in dataset:", len(target_data.dataset))
    print("The number of batches in DataLoader:", len(target_data))
    client = AttackerClient(
        G=Generator(100),
        D=Discriminator(),
        net=Net(),
        target_data=trainloader,
        trainloader=trainloader,
        valloader=trainloader,
        testloader=trainloader,
        local_epochs=5, 
    ).to_client()

fl.client.start_client(
    server_address="localhost:8080",
    client=client
    # config={"num-server-rounds": 50, "fraction-fit": 0.1, "pretrain-epochs": pretrain_epochs}
)