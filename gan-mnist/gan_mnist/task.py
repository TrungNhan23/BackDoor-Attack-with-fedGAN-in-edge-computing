# """gan-mnist: A Flower / PyTorch app."""

# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torchvision
# from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner
# import matplotlib.pyplot as plt 
# import os

# class Generator(nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(100, 256*7*7)
#         self.trans_conv1 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
#         #self.trans_conv1_bn = nn.BatchNorm2d(128)
#         self.trans_conv2 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding = 1)
#         #self.trans_conv2_bn = nn.BatchNorm2d(64)
#         self.trans_conv3 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
#         #self.trans_conv3_bn = nn.BatchNorm2d(32)
#         self.trans_conv4 = nn.ConvTranspose2d(32, 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
    
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, 256, 7, 7)
#         x = F.relu(self.trans_conv1(x))
#         #x = self.trans_conv1_bn(x)
#         x = F.relu(self.trans_conv2(x))
#         #x = self.trans_conv2_bn(x)
#         x = F.relu(self.trans_conv3(x))
#         #x = self.trans_conv3_bn(x)
#         x = self.trans_conv4(x)
#         x = torch.tanh(x)
        
#         return x        


# class Discriminator(nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.conv0 = nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1)
#         #self.conv0_bn = nn.BatchNorm2d(32)
#         self.conv0_drop = nn.Dropout2d(0.25)
#         self.conv1 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
#         #self.conv1_bn = nn.BatchNorm2d(64)
#         self.conv1_drop = nn.Dropout2d(0.25)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
#         #self.conv2_bn = nn.BatchNorm2d(128)
#         self.conv2_drop = nn.Dropout2d(0.25)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
#         #self.conv3_bn = nn.BatchNorm2d(256)
#         self.conv3_drop = nn.Dropout2d(0.25)
#         self.fc = nn.Linear(12544, 1)
    
#     def forward(self, x):
#         x = x.view(-1, 1, 28, 28)
#         x = F.leaky_relu(self.conv0(x), 0.2)
#         #x = self.conv0_bn(x)
#         x = self.conv0_drop(x)
#         x = F.leaky_relu(self.conv1(x), 0.2)
#         #x = self.conv1_bn(x)
#         x = self.conv1_drop(x)
#         x = F.leaky_relu(self.conv2(x), 0.2)
#         #x = self.conv2_bn(x)
#         x = self.conv2_drop(x)
#         x = F.leaky_relu(self.conv3(x), 0.2)
#         #x = self.conv3_bn(x)
#         x = self.conv3_drop(x)
#         x = x.view(-1, self.num_flat_features(x))
#         x = self.fc(x)
        
#         return x
    
#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
        
#         return num_features

# class GlobalModel(nn.Module):
#     def __init__(self):
#         super(GlobalModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# fds = None  # Cache FederatedDataset

# def load_data(partition_id: int, num_partitions: int, num_samples: int = 2300):
#     """Load partitioned MNIST data with optional sample limit."""
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="mnist",
#             partitioners={"train": partitioner},
#         )

#     # Load partition
#     partition = fds.load_partition(partition_id)

#     # Giới hạn số lượng dữ liệu nếu num_samples được chỉ định
#     if num_samples is not None:
#         partition = partition.select(range(min(num_samples, len(partition))))

#     # Split train/test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

#     # Apply transforms
#     pytorch_transforms = Compose([
#         ToTensor(),
#         Normalize((0.5,), (0.5,))  # MNIST only has one channel
#     ])

#     def apply_transforms(batch):
#         """Apply transforms to the dataset."""
#         batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)

#     trainloader = DataLoader(partition_train_test["train"], batch_size=16, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=16)

#     return trainloader, testloader

# def test(G, D, testloader, device, latent_size=100):
#     G.to(device)
#     D.to(device)
#     G.eval()
#     D.eval()
    
#     with torch.no_grad():
#         z = torch.randn(64, latent_size).to(device)
#         generated_images = G(z) 
        
#         validity = D(generated_images).mean().item()
    
#     return validity, generated_images

# def save_generated_images(images, output_dir="output", filename="generated_images.png"): 
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     images = (images + 1) / 2 
#     images = images.clamp(0, 1)
    
#     grid_img = torchvision.utils.make_grid(images[:6], nrow=3)
#     plt.figure(figsize=(4, 4))
#     plt.imshow(grid_img.permute(1,2,0).cpu().numpy())
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, filename))    
    
# def get_weights(net):
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]

# def set_weights(net, parameters):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)

# def save_plots(D_loss, G_loss, output_dir = "output", filename="training_Loss_graph.png"):
#     """Lưu biểu đồ loss và accuracy vào folder output."""
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     plt.figure()
#     plt.plot(range(1, len(D_loss) + 1), D_loss, linestyle='-', label='Disciminator Loss')
#     plt.plot(range(1, len(G_loss) + 1), G_loss, linestyle='-', label='Generator Loss')
#     plt.xlabel("Round")
#     plt.ylabel("Value")
#     plt.title("Generator and Disciminator Loss Over Rounds")
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, filename))
#     plt.close()


# def train(G, D, trainloader, epochs, device, latent_size=100):
#     G.to(device)
#     D.to(device)
    
#     optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    
#     scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.9) #Giảm LR của Generator sau mỗi 10 epochs
#     scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.6) #Giảm LR của Discriminator sau mỗi 10 epochs
    
#     adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
#     # adversarial_loss = torch.nn.BCELoss().to(device)
#     for i, batch in enumerate(trainloader):
#         real_images = batch['image'].to(device)
#         batch_size = real_images.size(0)
        
#         optimizer_D.zero_grad()
        
#         real_validity = D(real_images)
#         real_label = torch.ones(batch_size,1).to(device)
#         loss_real = adversarial_loss(real_validity, real_label)
        
        
#         z = torch.randn(batch_size, latent_size).to(device)
#         fake_images = G(z)
#         fake_validity = D(fake_images.detach()) #Detach để không tính gradients cho Generator
#         fake_label = torch.zeros(batch_size, 1).to(device) #Labels for fake images
#         loss_fake = adversarial_loss(fake_validity, fake_label)   
        
#         loss_D = (loss_real + loss_fake) / 2
#         loss_D.backward()
#         optimizer_D.step()

#         # Train Generator
#         optimizer_G.zero_grad()

#         # Generate fake images
#         z = torch.randn(batch_size, latent_size).to(device)
#         generated_images = G(z)

#         # Discriminator's prediction on fake images
#         validity = D(generated_images) #Tính validity trên ảnh mới tạo
#         loss_G = adversarial_loss(validity, real_label) # Generator cố gắng làm cho validity gần 1

#         loss_G.backward()
#         optimizer_G.step()      
#         return loss_D.item(), loss_G.item()  



from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import matplotlib.pyplot as plt
import os

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 256*7*7)
        self.trans_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trans_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.trans_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.trans_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.trans_conv1(x))
        x = F.relu(self.trans_conv2(x))
        x = F.relu(self.trans_conv3(x))
        x = self.trans_conv4(x)
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(12544, 1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.leaky_relu(self.conv0(x), 0.2)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

fds = None

def load_data(partition_id: int, num_partitions: int, num_samples: int = 2300):
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    if num_samples is not None:
        partition = partition.select(range(min(num_samples, len(partition))))
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

def test(G, D, testloader, device, latent_size=100):
    G.to(device)
    D.to(device)
    G.eval()
    D.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_size).to(device)
        generated_images = G(z)
        validity = D(generated_images).mean().item()
    return validity, generated_images

def save_generated_images(images, output_dir="output", filename="generated_images.png"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    grid_img = torchvision.utils.make_grid(images[:6], nrow=3)
    plt.figure(figsize=(4, 4))
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename))

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def save_plots(D_loss, G_loss, output_dir="output", filename="training_Loss_graph.png"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure()
    plt.plot(range(1, len(D_loss) + 1), D_loss, linestyle='-', label='Disciminator Loss')
    plt.plot(range(1, len(G_loss) + 1), G_loss, linestyle='-', label='Generator Loss')
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title("Generator and Disciminator Loss Over Rounds")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
def train(G, D, trainloader, epochs, device, latent_size=100):
    G.to(device)
    D.to(device)
    
    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0004, betas=(0.5, 0.999)) # Tăng lr
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.00005, betas=(0.5, 0.999)) # Giảm lr
    
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
    
    D_losses = []
    G_losses = []
    
    for epoch in range(epochs):
        for i, batch in enumerate(trainloader):
            real_images = batch['image'].to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            real_validity = D(real_images)
            real_label = torch.ones(batch_size, 1).to(device)
            loss_real = adversarial_loss(real_validity, real_label)
            
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            fake_validity = D(fake_images.detach())
            fake_label = torch.zeros(batch_size, 1).to(device)
            loss_fake = adversarial_loss(fake_validity, fake_label)
            
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, latent_size).to(device)
            generated_images = G(z)
            validity = D(generated_images)
            loss_G = adversarial_loss(validity, real_label)
            
            loss_G.backward()
            optimizer_G.step()
            
        D_losses.append(loss_D.item())
        G_losses.append(loss_G.item())
        
        print(f"Epoch [{epoch}/{epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        
        # Lưu ảnh sinh ra sau mỗi epoch
        if epoch % 10 == 0:
            save_generated_images(generated_images, output_dir="output", filename=f"generated_images_epoch_{epoch}.png")

    return D_losses, G_losses        