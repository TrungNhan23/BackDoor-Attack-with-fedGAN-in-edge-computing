"""federated-learning: A Flower / PyTorch app."""

import torch
print("CUDA Available:", torch.cuda.is_available())  
print("Number of GPUs:", torch.cuda.device_count())  
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")  
print("CUDA Version:", torch.version.cuda)  


from collections import OrderedDict
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


import matplotlib.pyplot as plt 
# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=0)  # Conv(1,64,4)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0) # Conv(64,64,4)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0) # Conv(64,64,4)
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0) # Conv(64,128,3)
#         self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0) # Conv(128,128,3)
#         self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0) # Conv(128,128,3)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2) # AvgPooling(2,2)
#         self.fc = nn.Linear(128 * 6 * 6, 11)

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x))
#         x = F.leaky_relu(self.conv2(x))
#         x = F.leaky_relu(self.conv3(x))
#         x = F.leaky_relu(self.conv4(x))
#         x = F.leaky_relu(self.conv5(x))
#         x = F.leaky_relu(self.conv6(x))
#         x = self.avgpool(x)
#         x = torch.flatten(x, start_dim=1)  # Flatten để đưa vào FC
#         x = self.fc(x)
#         return F.softmax(x, dim=1)

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=1, padding=0)  # Conv(1,64,4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0) # Conv(64,64,4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=0) # Conv(64,64,4)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0) # Conv(64,128,3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0) # Conv(128,128,3)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0) # Conv(128,128,3)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2) # AvgPooling(2,2)
        
        # Thay đổi số lớp output thành 10
        self.fc = nn.Linear(128 * 6 * 6, 10)  # 10 lớp phân loại

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)  # Flatten để đưa vào FC
        x = self.fc(x)
        return F.softmax(x, dim=1)  # Softmax cho phân loại
fds = None  # Cache FederatedDataset

#Implement GAN for attacker here



def load_data(partition_id: int, num_partitions: int, num_samples: int = 3000):
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

    # Chia lại dataset thành train, validation, và test
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
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
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
            loss.backward()
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
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {avg_trainloss:.4f}, Train Accuracy: {train_accuracy:.2f} - "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
    
    return avg_trainloss, avg_val_loss, train_accuracy, val_accuracy


def imshow(images, labels, preds, classes, num_images=4, output_dir="output"):
    
    # Kiểm tra thư mục output đã tồn tại chưa
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Tạo thư mục nếu chưa có
    
    # Chọn num_images ngẫu nhiên
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_images):
        ax = fig.add_subplot(2, 3, i+1)
        img = images[i].numpy().transpose((1, 2, 0))  # chuyển ảnh về dạng HWC từ CHW
        ax.imshow(img, cmap='gray')
        
        true_label = classes[labels[i]]  # Nhãn thực tế
        pred_label = classes[preds[i]]  # Nhãn dự đoán
        
        # Đặt tiêu đề với nhãn thực tế và dự đoán
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')  # Tắt trục
        
    # Lưu hình ảnh dưới dạng PNG trong thư mục output
    output_path = os.path.join(output_dir, "real_img_predictions.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Đóng để giải phóng bộ nhớ

# Giả sử `trainloader` là dataloader của bạn và `model` là mô hình
def display_predictions(model, testloader, device, output_dir="output"):
    model.eval()  # Đặt mô hình ở chế độ evaluation
    
    # Lấy batch đầu tiên từ testloader
    dataiter = iter(testloader)
    batch = next(dataiter)
    images, labels = batch["image"], batch["label"]
    
    # Kiểm tra và chuyển images và labels thành tensor nếu cần
    if isinstance(images, torch.Tensor):
        images = images.to(device)
    else:
        raise TypeError(f"Images must be tensor, but current type data is: {type(images)}")

    if isinstance(labels, torch.Tensor):
        labels = labels.to(device)
    else:
        raise TypeError(f"Labels must be tensor, but current type data is: {type(labels)}")
    
    # Tiến hành dự đoán
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Các lớp của MNIST (hoặc tùy theo bài toán của bạn)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # Hiển thị hình ảnh với dự đoán và lưu lại thành file PNG
    imshow(images.cpu(), labels.cpu(), preds.cpu(), classes, output_dir=output_dir)
    
def metric_plot(train_loss, val_loss, train_acc, val_acc, output_dirs="output"):
    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs)
    
    # Tạo một figure với 2 subplots (2x1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Vẽ Loss trên subplot đầu tiên
    axes[0].plot(range(len(train_loss)), train_loss, label="Train Loss", color='blue')
    axes[0].plot(range(len(val_loss)), val_loss, label="Validation Loss", color='red')
    axes[0].set_title("Loss Plot")
    axes[0].set_xlabel("Rounds")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Vẽ Accuracy trên subplot thứ hai
    axes[1].plot(range(len(train_acc)), train_acc, label="Train Accuracy", color='blue')
    axes[1].plot(range(len(val_acc)), val_acc, label="Validation Accuracy", color='red')
    axes[1].set_title("Accuracy Plot")
    axes[1].set_xlabel("Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Lưu figure với cả hai đồ thị
    plt.tight_layout()  # Điều chỉnh khoảng cách giữa các subplot
    plt.savefig(os.path.join(output_dirs, "metrics_plot.png"))
    plt.close()

    print(f"Plots saved to {output_dirs}")
    
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
    display_predictions(net, testloader, device)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
