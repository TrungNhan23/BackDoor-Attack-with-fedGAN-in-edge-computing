"""federated-learning: A Flower / PyTorch app."""

import torch

print("CUDA Available:", torch.cuda.is_available())  
print("Number of GPUs:", torch.cuda.device_count())  
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")  
print("CUDA Version:", torch.version.cuda)  

from collections import OrderedDict
import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import matplotlib.pyplot as plt 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Lớp Conv2d đầu tiên: 1 -> 64 kênh, kernel 5x5, stride 2, padding=2 (để "same")
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        # Lớp Conv2d thứ hai: 64 -> 128 kênh, kernel 5x5, stride 2, padding=2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        # Lớp fully connected (Dense) sau khi flatten
        # Với ảnh đầu vào 28x28, sau 2 lớp conv với stride=2, kích thước feature map sẽ là 7x7
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.dropout = nn.Dropout(0.75) 
        # Định nghĩa LeakyReLU với negative_slope=0.2
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = self.conv1(x)           # (batch_size, 64, 28, 28) với padding=2 và stride=2 → (batch_size, 64, 14, 14)
        x = self.leaky_relu(x)
        # x = self.dropout(x)
        
        x = self.conv2(x)           # (batch_size, 128, 14, 14) → (batch_size, 128, 7, 7)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)   # Flatten, shape: (batch_size, 128*7*7)
        x = self.fc(x)              # Dense layer cho ra 1 giá trị
        return x

    
fds = None  # Cache FederatedDataset
def load_data(partition_id: int, num_partitions: int, num_samples: int = 40000):
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

    # Apply transforms
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition = partition.with_transform(apply_transforms)

    # Sử dụng tỷ lệ 6:3:1, tổng số dữ liệu là 100%
    total_size = len(partition)
    train_size = int(0.7 * total_size)
    val_size = int(0.25 * total_size)
    test_size = total_size - train_size - val_size

    # random_split chia dữ liệu theo tỷ lệ
    train_data, val_data, test_data = random_split(partition, [train_size, val_size, test_size])

    # Dataloader cho các phần
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    valloader = DataLoader(val_data, batch_size=32, shuffle=False)
    testloader = DataLoader(test_data, batch_size=32)

    return trainloader, valloader, testloader


def train(net, trainloader, valloader, epochs, device):
    """Train the model on the training set and validate after each epoch."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
    
    for epoch in range(epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        total = 0
        correct = 0.0
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

        avg_trainloss = running_loss / len(trainloader)
        train_accuracy = correct / total
        
        # Validation phase
        net.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # Turn off gradients for validation
            for batch in valloader:
                images = batch["image"]
                labels = batch["label"]
                outputs = net(images.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.to(device)).sum().item()

        avg_val_loss = val_loss / len(valloader)
        val_accuracy = val_correct / val_total
        
        # Print the results after each epoch
        # print(f"Epoch [{epoch+1}/{epochs}] - "
            #   f"Train Loss: {avg_trainloss:.4f}, Train Accuracy: {train_accuracy:.2f} - "
            #   f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
    
    return avg_trainloss, avg_val_loss, train_accuracy, val_accuracy


def imshow(images, labels, preds, classes, labels_to_plot, num_images=6, output_dir="../output/plot"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig = plt.figure(figsize=(12, 6))
    
    zero_indices = torch.where(labels == labels_to_plot)[0]  # Lấy chỉ mục của ảnh có label = 0
    
    if len(zero_indices) == 0:
        print("Không có ảnh nào có nhãn 0!")
        return

    num_images = min(num_images, len(zero_indices))

    for i in range(num_images):  
        idx = zero_indices[i]  
        ax = fig.add_subplot(2, 3, i+1) 
        img = images[idx].numpy().transpose((1, 2, 0))  
        ax.imshow(img, cmap='gray')
        
        true_label = classes[labels[idx].item()] 
        pred_label = classes[preds[idx].item()] 
        
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')

    output_path = os.path.join(output_dir, "real_img_predictions.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def display_predictions(model, testloader, labels_to_plot, device):
    model.eval()  
    
    dataiter = iter(testloader)
    batch = next(dataiter)
    
        # Nếu batch là danh sách hoặc tuple
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images, labels = batch[0], batch[1]
    else:
        raise ValueError("Unsupported batch format. Expected list or tuple with two elements.")

    if isinstance(images, torch.Tensor):
        images = images.to(device)
    else:
        raise TypeError(f"Images must be tensor, but current type data is: {type(images)}")

    if isinstance(labels, torch.Tensor):
        labels = labels.to(device)
    else:
        raise TypeError(f"Labels must be tensor, but current type data is: {type(labels)}")
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    imshow(images.cpu(), labels.cpu(), preds.cpu(), classes, labels_to_plot)

    
def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    # display_predictions(net, testloader, device)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
