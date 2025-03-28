
import os
import numpy as np

from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
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

def attacker_data(trainloader, target_labels=0):
    # Định nghĩa transform tương tự như lúc load dữ liệu
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
                # Nếu img chưa được chuyển thành tensor, áp dụng transform trực tiếp.
                # Nếu img đã là tensor, bạn có thể cần chuyển về PIL Image trước:
                if not isinstance(img, torch.Tensor):
                    img_transformed = transform(img)
                else:
                    # Chuyển tensor về PIL trước (giả sử img có định dạng (C, H, W))
                    from torchvision.transforms.functional import to_pil_image
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
    print(f'Saved {count} of 0 images')
    return target_dataloader
  
  
def plot_real_fake_images(real_images, fake_images, epoch, output_dir='output/result'):
    """
    Plot real and fake images side by side

    Args:
    - real_images: Tensor of real images
    - fake_images: Tensor of generated images
    - epoch: Current training epoch
    - output_dir: Directory to save comparison plots
    """
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
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].axis('off')

    # Plot fake images (second row)
    for i in range(num_images):
        img = fake_images[i].cpu().detach().squeeze()
        axs[1, i].imshow(img, cmap='gray')
        axs[1, i].axis('off')

    # Set titles
    plt.suptitle(f'Real vs Generated Images - Epoch {epoch}')
    axs[0, 0].text(-10, num_images/2, 'Real', rotation=90, va='center', ha='center')
    axs[1, 0].text(-10, num_images/2, 'Fake', rotation=90, va='center', ha='center')

    # Save the plot
    plt.savefig(f'{output_dir}/real_vs_fake_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
    plt.close()
  
      
def gan_train(generator, discriminator, target_data, n_epochs=5, latent_dim=100):
    
    adversarial_loss = torch.nn.BCELoss()
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.90)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.90)

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
            
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(target_data), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(target_data) + i
            if batches_done % 50 == 0:
                save_image(gen_imgs.data[:25], "output/images/%d.png" % batches_done, nrow=5, normalize=True)
        mean_g_loss = np.mean([loss.cpu().detach().item() for loss in g_losses])
        mean_d_loss = np.mean([loss.cpu().detach().item() for loss in d_losses])
    # scheduler_G.step()
    # scheduler_D.step()
    # current_lr_G = optimizer_G.param_groups[0]['lr']
    # current_lr_D = optimizer_D.param_groups[0]['lr']
    # print("Current learning rate G:", current_lr_G)
    # print("Current learning rate D:", current_lr_D)
    return mean_g_loss, mean_d_loss
    # return np.mean(g_losses), np.mean(g_losses)


def gan_metrics(g_loss, d_loss, output_dirs="output/plot"):
    plt.figure(figsize=(10, 5))
    plt.plot(g_loss, label='Generator Loss')
    plt.plot(d_loss, label='Discriminator Loss')
    plt.ylim(0.2, 2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('GAN Loss Metrics')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_dirs, "gan_metrics_plot.png"))
    plt.close()
    
        
