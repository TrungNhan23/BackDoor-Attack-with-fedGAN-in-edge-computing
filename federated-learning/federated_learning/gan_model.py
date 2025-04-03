
import os
import numpy as np

from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

os.makedirs("output/images", exist_ok=True)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        self.latent_dim = latent_dim
        self.channels = 1
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.channels = 1
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class DictDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}

# def merge_data(trainloader, fake_images, batch_size=32):
#     if fake_images is None:
#         return None
#     print(f'length of dataloader real: {len(trainloader)}')
#     print(f'length of dataloader fake: {len(fake_images)}')
#     all_real_images = []
#     all_real_labels = []

#     # Iterate through the entire trainloader to collect all real images and labels
#     for batch in trainloader:
#         if isinstance(batch, dict):
#             real_images = batch['image']
#             real_labels = batch['label']
#         elif isinstance(batch, (list, tuple)) and len(batch) == 2:
#             real_images, real_labels = batch
#         else:
#             raise ValueError("Unsupported batch format.")

#         all_real_images.append(real_images)
#         all_real_labels.append(real_labels)

#     # Concatenate all real images and labels
#     all_real_images = torch.cat(all_real_images, dim=0)
#     all_real_labels = torch.cat(all_real_labels, dim=0)

#     # Ensure fake_images are on the same device as real_images
#     fake_images = fake_images.to(all_real_images.device).detach()

#     # Ensure fake_images have the same number of channels as real_images
#     if fake_images.shape[1] != all_real_images.shape[1]:
#         fake_images = fake_images.repeat(1, all_real_images.shape[1], 1, 1)

#     # Create fake labels (e.g., target label 7 for all fake images)
#     fake_labels_target = [8]
#     # fake_labels_target = [7]
#     fake_labels = torch.tensor(fake_labels_target, device=all_real_images.device)[
#         torch.randint(0, len(fake_labels_target), (fake_images.shape[0],), device=all_real_images.device)
#     ]

#     # Merge fake images and labels with the real dataset
#     merged_images = torch.cat([all_real_images, fake_images], dim=0)
#     merged_labels = torch.cat([all_real_labels, fake_labels], dim=0)

#     # Create a new dataset and DataLoader
#     merged_dataset = DictDataset(merged_images, merged_labels)
#     merged_dataloader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)
#     print(f'length of dataloader after poison: {len(merged_dataloader)}')
#     return merged_dataloader

def merge_data(trainloader, fake_images, batch_size=16):
    if fake_images is None:
        return None

    all_real_images = []
    all_real_labels = []

    # Iterate through the entire trainloader to collect all real images and labels
    for batch in trainloader:
        if isinstance(batch, dict):
            real_images = batch['image']
            real_labels = batch['label']
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            real_images, real_labels = batch
        else:
            raise ValueError("Unsupported batch format.")

        all_real_images.append(real_images)
        all_real_labels.append(real_labels)

    # Concatenate all real images and labels
    all_real_images = torch.cat(all_real_images, dim=0)
    all_real_labels = torch.cat(all_real_labels, dim=0)

    # Ensure fake_images are on the same device as real_images
    fake_images = fake_images.to(all_real_images.device).detach()

    # Ensure fake_images have the same number of channels as real_images
    if fake_images.shape[1] != all_real_images.shape[1]:
        fake_images = fake_images.repeat(1, all_real_images.shape[1], 1, 1)

    # Create fake labels (e.g., target label 7 for all fake images)
    fake_labels_target = [8]  # Or whatever label you want for fake images
    fake_labels = torch.tensor(fake_labels_target, device=all_real_images.device)[
        torch.randint(0, len(fake_labels_target), (fake_images.shape[0],), device=all_real_images.device)
    ]

    # Merge fake images and labels with the real dataset
    merged_images = torch.cat([all_real_images, fake_images], dim=0)
    merged_labels = torch.cat([all_real_labels, fake_labels], dim=0)

    # Check if the total number of images is correct (real + fake)
    print(f"Total number of images after merge: {len(merged_images)}")
    
    # Create a new dataset and DataLoader
    merged_dataset = DictDataset(merged_images, merged_labels)
    merged_dataloader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)

    return merged_dataloader


def attacker_data(trainloader, target_labels=0):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # đảm bảo kích thước (32, 32)
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    filtered_images = []
    filtered_labels = []
    count = 0

    for batch in trainloader: 
        images = batch["image"]
        labels = batch["label"]

        for img, label in zip(images, labels):
            if label.item() == target_labels:
                if not isinstance(img, torch.Tensor):
                    img_transformed = transform(img)
                else:
                    # Chuyển tensor về PIL trước (giả sử img có định dạng (C, H, W))
                    
                    img_pil = to_pil_image(img)
                    img_transformed = transform(img_pil)
                    
                filtered_images.append(img_transformed)
                filtered_labels.append(label)
                count += 1

    # Convert danh sách ảnh thành tensor
    filtered_images = torch.stack(filtered_images)
    filtered_labels = torch.stack(filtered_labels)
    
    # Tạo TensorDataset và DataLoader với batch_size = 8
    target_dataset = torch.utils.data.TensorDataset(filtered_images, filtered_labels)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=32, shuffle=True)
    print(f'Saved {count} of {target_labels} images')
    return target_dataloader
  
  
def attacker_data_no_filter(trainloader):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # đảm bảo kích thước (32, 32)
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    transformed_images = []
    transformed_labels = []
    count = 0

    for batch in trainloader: 
        images = batch["image"]
        labels = batch["label"]

        for img, label in zip(images, labels):
            if not isinstance(img, torch.Tensor):
                img_transformed = transform(img)
            else:
                # Chuyển tensor về PIL trước (giả sử img có định dạng (C, H, W))
                img_pil = to_pil_image(img)
                img_transformed = transform(img_pil)
                
            transformed_images.append(img_transformed)
            transformed_labels.append(label)
            count += 1

    # Convert danh sách ảnh thành tensor
    transformed_images = torch.stack(transformed_images)
    transformed_labels = torch.stack(transformed_labels)
    
    # Tạo TensorDataset và DataLoader với batch_size = 8
    target_dataset = torch.utils.data.TensorDataset(transformed_images, transformed_labels)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=32, shuffle=True)
    # print(f'Saved {count} images')
    return target_dataloader  

  
def plot_real_fake_images(real_images, fake_images, epoch, output_dir='output/result'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select up to 8 images
    num_images = min(8, real_images.size(0), fake_images.size(0))

    # Create a figure with subplots
    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Plot real images (first row)
    for i in range(num_images):
        img = real_images[i].cpu().detach().squeeze()
        # print(f"Real image {i} shape: {img.shape}")  # Debug: Kiểm tra kích thước ảnh
        if img.ndim == 1:  # Nếu ảnh là 1D, chuyển thành 2D
            img = img.view(32, 32)  # Giả sử ảnh có kích thước 32x32
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].axis('off')

    # Plot fake images (second row)
    for i in range(num_images):
        img = fake_images[i].cpu().detach().squeeze()
        # print(f"Fake image {i} shape: {img.shape}")  # Debug: Kiểm tra kích thước ảnh
        if img.ndim == 1:  # Nếu ảnh là 1D, chuyển thành 2D
            img = img.view(32, 32)  # Giả sử ảnh có kích thước 32x32
        axs[1, i].imshow(img, cmap='gray')
        axs[1, i].axis('off')

    # Set titles
    plt.suptitle(f'Real vs Generated Images - Epoch {epoch}')
    axs[0, 0].text(-10, num_images / 2, 'Real', rotation=90, va='center', ha='center')
    axs[1, 0].text(-10, num_images / 2, 'Fake', rotation=90, va='center', ha='center')

    # Save the plot
    plt.savefig(f'{output_dir}/real_vs_fake_diff.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_adversarial_noise(model, images, epsilon=0.02):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = F.binary_cross_entropy_with_logits(outputs, torch.ones_like(outputs))
    loss.backward()
    adversarial_noise = epsilon * images.grad.sign()
    return adversarial_noise

def create_adversarial_samples(model, images, target_label=8, epsilon=0.02, device='cuda'):
    """
    Tạo ảnh nhiễu đối kháng từ ảnh giả và gán nhãn mục tiêu.
    Args:
        model: Mô hình đã được huấn luyện.
        images: Tensor chứa ảnh đầu vào (ảnh giả).
        target_label: Nhãn mục tiêu để gán cho ảnh nhiễu.
        epsilon: Độ lớn của nhiễu đối kháng.
        device: Thiết bị (CPU hoặc GPU).
    Returns:
        adv_images: Tensor chứa ảnh nhiễu.
        adv_labels: Tensor chứa nhãn mục tiêu.
    """
    model.eval()

    # Đảm bảo ảnh nằm trên đúng thiết bị
    images = images.to(device).clone().detach().requires_grad_(True)

    # Dự đoán và tính loss
    outputs = model(images)
    loss = F.cross_entropy(outputs, torch.full((images.size(0),), target_label, dtype=torch.long, device=device))
    loss.backward()

    # Tạo nhiễu đối kháng
    adversarial_noise = epsilon * images.grad.sign()
    adv_images = images + adversarial_noise
    adv_images = torch.clamp(adv_images, -1, 1)  # Đảm bảo giá trị pixel hợp lệ

    # Gán nhãn mục tiêu
    adv_labels = torch.full((adv_images.size(0),), target_label, dtype=torch.long, device=device)

    return adv_images, adv_labels

def merge_adversarial_data(trainloader, adv_images, adv_labels, batch_size=32):
    """
    Kết hợp ảnh nhiễu với dữ liệu huấn luyện gốc.
    Args:
        trainloader: DataLoader chứa dữ liệu gốc.
        adv_images: Tensor chứa ảnh nhiễu.
        adv_labels: Tensor chứa nhãn nhiễu.
        batch_size: Kích thước batch.
    Returns:
        merged_dataloader: DataLoader chứa cả dữ liệu gốc và dữ liệu nhiễu.
    """
    all_images = []
    all_labels = []

    # Thu thập dữ liệu gốc
    for batch in trainloader:
        images, labels = batch["image"], batch["label"]
        # Đảm bảo dữ liệu gốc nằm trên cùng thiết bị với adv_images
        images = images.to(adv_images.device)
        labels = labels.to(adv_images.device)
        all_images.append(images)
        all_labels.append(labels)

    # Thêm dữ liệu nhiễu
    all_images.append(adv_images)
    all_labels.append(adv_labels)

    # Kết hợp tất cả dữ liệu
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Tạo DataLoader mới
    merged_dataset = torch.utils.data.TensorDataset(all_images, all_labels)
    merged_dataloader = torch.utils.data.DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)
    return merged_dataloader

  
def gan_train(generator, discriminator, target_data, round, merge_samples=1000, n_epochs=29, latent_dim=100):
    adversarial_loss = torch.nn.BCELoss()
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    
    
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=3, gamma=0.95)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=3, gamma=0.95)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    g_losses = []
    d_losses = []
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(target_data):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (epoch, n_epochs, i, len(target_data), d_loss.item(), g_loss.item())
            # )

            batches_done = epoch * len(target_data) + i
            if batches_done % 30 == 0:
                save_image(gen_imgs.data[:25], "output/images/%d.png" % batches_done, nrow=5, normalize=True)
        scheduler_G.step()
        scheduler_D.step()
    # print(f"Real images shape: {real_images.shape}")
    # print(f"Fake images shape: {fake_images.shape}")
    plot_real_fake_images(real_imgs, gen_imgs, epoch, output_dir='output/result')
    # if round : 
    print(f"Attacker in round {round - 5}")
    fake_img = generator(torch.randn(merge_samples, latent_dim).cuda()).detach()
    gen_imgs_resized = torch.nn.functional.interpolate(fake_img, size=(28, 28), mode='bilinear', align_corners=False)
    selected_images = gen_imgs_resized[:merge_samples]
    print(f'Length of fake img: {len(selected_images)}')
    # else:
    #     selected_images = None
    current_lr_G = optimizer_G.param_groups[0]['lr']
    current_lr_D = optimizer_D.param_groups[0]['lr']
    # print(f"Epoch {epoch + 1}: Generator LR = {current_lr_G}, Discriminator LR = {current_lr_D}")
    mean_g_loss = np.mean(g_losses)
    mean_d_loss = np.mean(d_losses)
    return mean_g_loss, mean_d_loss, selected_images
    # return g_losses, d_losses


def gan_metrics(g_loss, d_loss, output_dirs="output/plot"):
    plt.figure(figsize=(10, 5))
    plt.plot(g_loss, label='Generator Loss')
    plt.plot(d_loss, label='Discriminator Loss')
    # plt.ylim(0.6, 0.8)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('GAN Loss Metrics')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_dirs, "gan_metrics_plot.png"))
    plt.close()
    