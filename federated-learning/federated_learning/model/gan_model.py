
import os
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
from federated_learning.ultility.config import *
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

os.makedirs("../output/images", exist_ok=True)

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

        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def attacker_data(trainloader, target_labels=0):
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
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

                    img_pil = to_pil_image(img)
                    img_transformed = transform(img_pil)
                    
                filtered_images.append(img_transformed)
                filtered_labels.append(label)
                count += 1


    filtered_images = torch.stack(filtered_images)
    filtered_labels = torch.stack(filtered_labels)
    

    target_dataset = torch.utils.data.TensorDataset(filtered_images, filtered_labels)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=32, shuffle=True)
    print(f'Saved {count} of {target_labels} images')
    return target_dataloader
  
  
def attacker_data_no_filter(trainloader):
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
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
                img_pil = to_pil_image(img)
                img_transformed = transform(img_pil)
                
            transformed_images.append(img_transformed)
            transformed_labels.append(label)
            count += 1


    transformed_images = torch.stack(transformed_images)
    transformed_labels = torch.stack(transformed_labels)
    

    target_dataset = torch.utils.data.TensorDataset(transformed_images, transformed_labels)
    target_dataloader = torch.utils.data.DataLoader(target_dataset, batch_size=32, shuffle=True)
    # print(f'Saved {count} images')
    return target_dataloader  

  
def plot_real_fake_images(model, real_images, fake_images, output_dir='../output/result'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)

    model.eval()
    
    with torch.no_grad(): 
        output_real = model(real_images)
        output_fake = model(fake_images)

        predicted_real = torch.argmax(output_real, dim=1)
        predicted_fake = torch.argmax(output_fake, dim=1)
        
    os.makedirs(output_dir, exist_ok=True)
    num_images = min(8, real_images.size(0), fake_images.size(0))

    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)


    for i in range(num_images):
        img = real_images[i].cpu().detach().squeeze(0)  
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].text(0.5, 0.9, f'Pred: {predicted_real[i].item()}', color='red', ha='center', va='center', transform=axs[0, i].transAxes)   
    

    for i in range(num_images):
        img = fake_images[i].cpu().detach().squeeze(0)
        axs[1, i].imshow(img, cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].text(0.5, 0.9, f'Pred: {predicted_fake[i].item()}', color='red', ha='center', va='center', transform=axs[1, i].transAxes)
    

    plt.suptitle(f'Real vs Generated Images')
    axs[0, 0].text(-10, num_images / 2, 'Real', rotation=90, va='center', ha='center')
    axs[1, 0].text(-10, num_images / 2, 'Fake', rotation=90, va='center', ha='center')

    # Lưu hình ảnh
    plt.savefig(f'{output_dir}/real_vs_fake_diff.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_FGSM_adversarial_images(model, images, labels, 
                                     untargeted, epsilon=0.1):

    images = images.clone().detach().to(torch.float32).requires_grad_(True)
    labels = labels.clone().detach()
    
    model.eval()
    outputs = model(images)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    
    grad = images.grad.sign()
    
    if untargeted:
        adv_images = images + epsilon * grad
    else:
        adv_images = images - epsilon * grad
    
    
    adv_images.clamp_(min=0, max=1.0)

    return adv_images.detach()


    
def generate_PGD_adversarial_images(model, images, labels, untargeted,
                                    epsilon=0.1, 
                                    epsilon_step=EPSILON_STEP, 
                                    num_steps=NUM_STEPS):
    x = images.clone().detach()
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1)
    x_min = x - epsilon
    x_max = x + epsilon

    for i in range(num_steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.sign()
        if untargeted:
            x_adv = x_adv + epsilon_step * grad
        else:
            x_adv = x_adv - epsilon_step * grad
        x_adv = torch.max(torch.min(x_adv, x_max), x_min)
        x_adv = x_adv.clamp(0, 1).detach()
    return x_adv    
    

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

def create_attacker_data(model, generator, trainloader, 
                         device, untargeted, 
                         num_samples=NUM_SAMPLES, 
                         target_labels=TARGETED_LABEL,
                         mode='fgsm'):
    z = torch.randn(num_samples, 100).to(device)
    generated_images = generator(z)
    generated_labels = torch.full((num_samples,), target_labels).to(device)
    
    
    if mode == 'fgsm':
        adv_imgs = generate_FGSM_adversarial_images(model, 
                                            generated_images, 
                                            generated_labels,
                                            untargeted=untargeted,
                                            epsilon=EPSILON)
    elif mode == 'pgd':
        adv_imgs = generate_PGD_adversarial_images(model, 
                                            generated_images, 
                                            generated_labels,
                                            untargeted=untargeted,
                                            epsilon=EPSILON)    
    else:
        raise ValueError("Invalid mode. Choose either 'fgsm' or 'pgd'.")
    
    
    new_imges = torch.cat([generated_images, adv_imgs], dim=0)
    new_labels = torch.cat([generated_labels, generated_labels], dim=0)
    
    attack_loader = inject_images_into_dataloader(trainloader, new_imges, new_labels, batch_size=32, device=device)
    return attack_loader


def predict_on_adversarial_testset(model, testloader, current_round, 
                                   isClean,
                                   epsilon=EPSILON, device="cuda:0",
                                   output_dir="../output",
                                   mode='fgsm'):
    model.to(device)
    model.eval()
    print(f"\n[Round {current_round}] Evaluating ASR | isClean={isClean} | epsilon={epsilon} | Attack mode={mode}\n")
    predictions = []
    correct_predictions = 0
    total_predictions = 0
    correct_total_predictions = 0
    target = TARGETED_LABEL
    asr_values = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch in testloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        
        if isClean is not True: 
            mask = (labels == 1)
            images = images[mask]
            # labels = torch.full((images.size(0),), target, dtype=torch.long, device=images.device)
            labels = labels[mask]
            
        if len(images) == 0:
            continue

        if current_round >= ROUND_TO_ATTACK:
            if mode == 'fgsm':
                adv_images = generate_FGSM_adversarial_images(model, 
                                                    images, 
                                                    labels, 
                                                    untargeted=isClean, 
                                                    epsilon=epsilon)
            elif mode == 'pgd':
                adv_images = generate_PGD_adversarial_images(model, 
                                                        images, 
                                                        labels, 
                                                        untargeted=isClean, 
                                                        epsilon=epsilon)   
            else:
                raise ValueError("Invalid mode. Choose either 'fgsm' or 'pgd'.") 
        else:
            adv_images = images 

        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)

        if current_round < ROUND_TO_ATTACK:
            correct_predictions = 0
        else:
            if isClean:
                mask = (preds != labels)
                correct_predictions += mask.sum().item()
            else:
                correct_predictions += (preds == TARGETED_LABEL).sum().item()

        total_predictions += len(labels)
        correct_total_predictions += (preds == labels).sum().item()
        predictions.extend(preds.cpu().numpy())

        asr = correct_predictions / total_predictions if total_predictions > 0 else 0
        asr_values.append(asr)

    adv_image = adv_images[0].cpu().detach().squeeze(0)
    transform = transforms.ToPILImage()
    pil_image = transform(adv_image)
    pil_image.save(os.path.join(output_dir, f"adversarial_1_to_{target}.jpg"))

    # print(f"Predictions on adversarial test set: {predictions[:10]}")
    print("Labels:", labels[:10])
    print("Preds:", preds[:10])
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
                save_image(gen_imgs.data[:25], "../output/images/%d.png" % batches_done, nrow=5, normalize=True)
        scheduler_G.step()
        scheduler_D.step()
    current_lr_G = optimizer_G.param_groups[0]['lr']
    current_lr_D = optimizer_D.param_groups[0]['lr']
    # print(f"Epoch {epoch + 1}: Generator LR = {current_lr_G}, Discriminator LR = {current_lr_D}")
    mean_g_loss = np.mean(g_losses)
    mean_d_loss = np.mean(d_losses)
    return mean_g_loss, mean_d_loss, real_imgs, gen_imgs


def gan_metrics(g_loss, d_loss, output_dirs="../output/plot"):
    plt.figure(figsize=(10, 5))
    plt.plot(g_loss, label='Generator Loss')
    plt.plot(d_loss, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('GAN Loss Metrics')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_dirs, "gan_metrics_plot.png"))
    plt.close()
    