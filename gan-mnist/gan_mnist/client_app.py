# """gan-mnist: A Flower / PyTorch app."""
    
# import torch
# from flwr.client import NumPyClient, ClientApp
# from flwr.common import Context

# from gan_mnist.task import (
#     Generator,
#     Discriminator,
#     load_data,
#     get_weights,
#     set_weights,
#     train,
#     test,
#     save_plots,
#     save_generated_images
# )

# D_loss = []
# G_loss = []

# def print_weights(weights, name):
#     """In ra một phần nhỏ của trọng số để kiểm tra."""
#     print(f"{name} weights sample: {weights[0].flatten()[:5]}")
# # Define Flower Client and client_fn
# class FlowerClient(NumPyClient):
#     def __init__(self,  G, D, trainloader, valloader, local_epochs, latent_size, partition_id, total_rounds):
#         # self.net = net
#         # self.trainloader = trainloader
#         # self.valloader = valloader
#         # self.local_epochs = local_epochs
#         # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         # self.net.to(self.device)
#         self.G = G
#         self.D = D 
#         self.trainloader = trainloader
#         self.valloader = valloader
#         self.local_epochs = local_epochs
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.G.to(self.device)
#         self.D.to(self.device)
#         self.latent_size = latent_size
#         self.partition_id = partition_id
#         self.total_rounds = total_rounds

#     def get_parameters(self, config):
#         return get_weights(self.G) + get_weights(self.D)
    
#     def set_parameters(self, parameters): 
#         G_params_len = len(list(self.G.state_dict().keys()))
#         G_params = parameters[:G_params_len]
#         D_params = parameters[G_params_len:]
#         assert len(G_params) == G_params_len, "Mismatch in Generator parameters"
#         assert len(D_params) == len(list(self.D.state_dict().keys())), "Mismatch in Discriminator parameters"
        
        
#         set_weights(self.G, G_params)
#         set_weights(self.D, D_params)
        
        
#     def fit(self, parameters, config):
#         self.set_parameters(parameters=parameters)
#         train_loss_D, train_loss_G = train(
#             self.G,
#             self.D,
#             self.trainloader,
#             self.local_epochs,
#             self.device,
#             self.latent_size,
#         )
#         # print(f"Train Loss: {train_loss}")
#         # return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}
        
#         print(f"Train Loss D: {train_loss_D}, Train Loss G: {train_loss_G}")
#         D_loss.append(train_loss_D)
#         G_loss.append(train_loss_G)
#         return self.get_parameters(config={}), len(self.trainloader.dataset), {"train_loss_D": train_loss_D, "train_loss_G": train_loss_G}
    

#     # def fit(self, parameters, config):
#     #     self.set_parameters(parameters=parameters)

#     #     old_G_weights = get_weights(self.G)
#     #     old_D_weights = get_weights(self.D)

#     #     print_weights(old_G_weights, "Old G")
#     #     print_weights(old_D_weights, "Old D")

#     #     train_loss_D, train_loss_G = train(
#     #         self.G, self.D, self.trainloader, self.local_epochs, self.device, self.latent_size
#     #     )

#     #     new_G_weights = get_weights(self.G)
#     #     new_D_weights = get_weights(self.D)

#     #     print_weights(new_G_weights, "New G")
#     #     print_weights(new_D_weights, "New D")

#     #     return self.get_parameters(config={}), len(self.trainloader.dataset), {
#     #         "train_loss_D": train_loss_D, "train_loss_G": train_loss_G
#     #     }

#     def evaluate(self, parameters, config):  
      
#         self.set_parameters(parameters)
#         current_round = config.get("current_round", -1)
#         print(f"evaluate() called! Current round: {current_round}, Total rounds: {self.total_rounds}")
#         with torch.no_grad():
#             validity, generated_img = test(self.G, self.D, self.valloader, self.device, self.latent_size) 
#             # if current_round == self.total_rounds and self.partition_id == 0:
#             save_generated_images(generated_img)
#             save_plots(D_loss, G_loss)
#         print(f"Validation Validity: {validity}")
#         return validity, len(self.valloader.dataset), {}


# def client_fn(context: Context):
#     # Load model and data
#     # net = Net()
#     G = Generator()
#     D = Discriminator()
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     trainloader, valloader = load_data(partition_id, num_partitions)
#     local_epochs = context.run_config["local-epochs"]
#     latent_size = 100
#     total_rounds = context.run_config["num-server-rounds"]
#     print(f'Partition {partition_id}: TrainLoader batch size = {trainloader.batch_size}, Num Batches = {len(trainloader)}')

#     # save_plots(D_loss=D_loss, G_loss=G_loss)

#     # Return Client instance
#     return FlowerClient(G, D, trainloader, valloader, local_epochs, latent_size, partition_id, total_rounds).to_client()


# # Flower ClientApp
# app = ClientApp(
#     client_fn,
# )



"""gan-mnist: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import numpy as np

from gan_mnist.task import (
    Generator,
    Discriminator,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
    save_plots,
    save_generated_images
)

def print_weights(weights, name):
    """In ra một phần nhỏ của trọng số để kiểm tra."""
    print(f"{name} weights sample: {weights[0].flatten()[:5]}")

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, G, D, trainloader, valloader, local_epochs, latent_size, partition_id, total_rounds):
        self.G = G
        self.D = D
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device)
        self.D.to(self.device)
        self.latent_size = latent_size
        self.partition_id = partition_id
        self.total_rounds = total_rounds
        self.D_losses = []
        self.G_losses = []

    def get_parameters(self, config):
        return get_weights(self.G) + get_weights(self.D)

    def set_parameters(self, parameters):
        G_params_len = len(list(self.G.state_dict().keys()))
        G_params = parameters[:G_params_len]
        D_params = parameters[G_params_len:]
        set_weights(self.G, G_params)
        set_weights(self.D, D_params)

    def fit(self, parameters, config):
        self.set_parameters(parameters=parameters)
        train_loss_D, train_loss_G = train(
            self.G,
            self.D,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.latent_size,
        )
        self.D_losses.append(train_loss_D)
        self.G_losses.append(train_loss_G)
        return self.get_parameters(config), len(self.trainloader.dataset), {"train_loss_D": train_loss_D, "train_loss_G": train_loss_G}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        current_round = config.get("current_round", -1)
        validity, generated_img = test(self.G, self.D, self.valloader, self.device, self.latent_size)
        if current_round == self.total_rounds and self.partition_id == 0:
            save_generated_images(generated_img)
            save_plots(self.D_losses, self.G_losses) #Sửa lại ở đây
        print(f"Validation Validity: {validity}")
        return validity, len(self.valloader.dataset), {}


def client_fn(context: Context):
    G = Generator()
    D = Discriminator()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    latent_size = 100
    total_rounds = context.run_config["num-server-rounds"]
    print(f'Partition {partition_id}: TrainLoader batch size = {trainloader.batch_size}, Num Batches = {len(trainloader)}')
    return FlowerClient(G, D, trainloader, valloader, local_epochs, latent_size, partition_id, total_rounds).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)