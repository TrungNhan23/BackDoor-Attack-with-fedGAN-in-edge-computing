
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

        self.init_size = 28 // 4
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


def attacker_data(trainloader, target_labels=0):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # đảm bảo kích thước (32, 32)
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

  
def plot_real_fake_images(model, real_images, fake_images, output_dir='output/result'):
    # Kiểm tra thiết bị (GPU nếu có, nếu không thì dùng CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Đảm bảo model và dữ liệu đầu vào đều ở trên cùng một thiết bị
    model.to(device)
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)

    # Đảm bảo mô hình ở chế độ eval
    model.eval()
    
    # Dừng tính gradient
    with torch.no_grad(): 
        # Tính đầu ra của ảnh thật và giả
        output_real = model(real_images)
        output_fake = model(fake_images)

        # Dự đoán cho ảnh thật và giả
        predicted_real = torch.argmax(output_real, dim=1)
        predicted_fake = torch.argmax(output_fake, dim=1)
        
    # Đảm bảo thư mục lưu kết quả tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Lựa chọn tối đa 8 ảnh
    num_images = min(8, real_images.size(0), fake_images.size(0))

    # Tạo một figure với subplots
    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Vẽ ảnh thật (hàng đầu tiên)
    for i in range(num_images):
        img = real_images[i].cpu().detach().squeeze(0)  # Loại bỏ chiều kênh nếu có
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].text(0.5, 0.9, f'Pred: {predicted_real[i].item()}', color='red', ha='center', va='center', transform=axs[0, i].transAxes)   
    
    # Vẽ ảnh giả (hàng thứ hai)
    for i in range(num_images):
        img = fake_images[i].cpu().detach().squeeze(0)
        axs[1, i].imshow(img, cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].text(0.5, 0.9, f'Pred: {predicted_fake[i].item()}', color='red', ha='center', va='center', transform=axs[1, i].transAxes)
    
    
    # Thiết lập tiêu đề cho các subplots
    plt.suptitle(f'Real vs Generated Images')
    axs[0, 0].text(-10, num_images / 2, 'Real', rotation=90, va='center', ha='center')
    axs[1, 0].text(-10, num_images / 2, 'Fake', rotation=90, va='center', ha='center')

    # Lưu hình ảnh
    plt.savefig(f'{output_dir}/real_vs_fake_diff.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_adversarial_images(model, images, labels, epsilon=0.1):
    """
    Tạo ảnh nhiễu đối kháng từ ảnh GAN đã sinh ra.
    images: (N, 1, 28, 28), labels: (N,)
    """
    images = images.clone().detach().to(torch.float32).requires_grad_(True)
    labels = labels.clone().detach()
    
    model.eval()
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    
    grad = images.grad.data
    adv_images = images + epsilon * grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach()

class PoisonedMNISTDataset(Dataset):
    def __init__(self, clean_images, clean_labels, poisoned_images, poisoned_labels):
        self.images = torch.cat((clean_images, poisoned_images), dim=0)
        self.labels = torch.cat((clean_labels, poisoned_labels), dim=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}

def inject_images_into_dataloader(clean_dataloader, new_images, new_labels, batch_size=64, device='cpu'):
    clean_images = []
    clean_labels = []

    for batch in clean_dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        clean_images.append(images)
        clean_labels.append(labels)

    clean_images = torch.cat(clean_images, dim=0)
    clean_labels = torch.cat(clean_labels, dim=0)

    combined_dataset = PoisonedMNISTDataset(clean_images, clean_labels, new_images, new_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return combined_loader

def create_attacker_data(model, generator, trainloader, device, num_samples=20, target_labels=0):
    z = torch.randn(num_samples, 100).to(device)
    generated_images = generator(z)
    generated_labels = torch.full((num_samples,), target_labels).to(device)
    
    adv_imgs = generate_adversarial_images(model, 
                                           generated_images, 
                                           generated_labels,
                                           0.1)
    
    new_imges = torch.cat([generated_images, adv_imgs], dim=0)
    new_labels = torch.cat([generated_labels, generated_labels], dim=0)
    
    attack_loader = inject_images_into_dataloader(trainloader, new_imges, new_labels, batch_size=32, device=device)
    return attack_loader

def predict_on_adversarial_testset(model, testloader, epsilon=0.1, device="cuda:0", output_dir="output"):
    model.to(device)
    model.eval()

    predictions = []
    correct_predictions = 0  # Số lần dự đoán đúng số 7
    total_predictions = 0    # Tổng số ảnh có label 7
    correct_total_predictions = 0  # Số lượng ảnh đúng trong tổng số ảnh có label 7

    asr_values = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch in testloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        mask = (labels == 1)
        images = images[mask]
        labels = labels[mask]

        if len(images) == 0:  
            continue

        adv_images = generate_adversarial_images(model, images, labels, epsilon=epsilon)

        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)

        correct_predictions += (preds == 7).sum().item()
        total_predictions += len(labels)

        correct_total_predictions += (preds == labels).sum().item()

        predictions.extend(preds.cpu().numpy())

        asr = correct_predictions / total_predictions if total_predictions > 0 else 0
        asr_values.append(asr)

        adv_image = adv_images[0].cpu().detach()
        adv_image = adv_image.squeeze(0)
        transform = transforms.ToPILImage()
        pil_image = transform(adv_image)

        pil_image.save(os.path.join(output_dir, f"adversarial_1_to_7.jpg"))

    print(f"Predictions on adversarial test set: {predictions[:10]}")
    print(f"ASR (Attack Success Rate): {correct_predictions / total_predictions if total_predictions > 0 else 0}")

    return correct_predictions / total_predictions if total_predictions > 0 else 0

def gan_train(generator, discriminator, target_data, round, n_epochs=9, latent_dim=100):
    adversarial_loss = torch.nn.BCELoss()
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    
    
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.95)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.95)

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
            if batches_done % 50 == 0:
                save_image(gen_imgs.data[:25], "output/images/%d.png" % batches_done, nrow=5, normalize=True)
        scheduler_G.step()
        scheduler_D.step()
    current_lr_G = optimizer_G.param_groups[0]['lr']
    current_lr_D = optimizer_D.param_groups[0]['lr']
    # print(f"Epoch {epoch + 1}: Generator LR = {current_lr_G}, Discriminator LR = {current_lr_D}")
    mean_g_loss = np.mean(g_losses)
    mean_d_loss = np.mean(d_losses)
    return mean_g_loss, mean_d_loss, real_imgs, gen_imgs
        # selected_images


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
    