import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch_fidelity import calculate_metrics
from federated_learning.model.gan_model import Generator, Discriminator
from federated_learning.model.task import Net

# ==== CẤU HÌNH ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent_dim = 100
num_samples = 1100
save_fake_dataset = "../../output/fake_gan_dataset/fake_gan_dataset.pt"
gan_dir = "../../output/fid/gan_1"
real_dir = "../../output/fid/real_1"
checkpoint_path = "../../tmp/gan_checkpoint.pth"
cnn_checkpoint_path = "../../tmp/net_checkpoint.pth"

# ==== 1. TẠO ẢNH GAN (label = 1) ====
class FakeImageDataset(Dataset):
    def __init__(self, generator, discriminator, num_samples=1000, latent_dim=100, device='cpu', threshold=0.5, batch_size=64):
        self.device = device
        self.generator = generator.to(device).eval()
        self.discriminator = discriminator.to(device).eval()
        self.latent_dim = latent_dim
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_samples = num_samples

        self.images = self._generate_filtered_images()

    def _generate_filtered_images(self):
        collected = []
        with torch.no_grad():
            while len(collected) < self.num_samples:
                z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
                fake_imgs = self.generator(z)

                preds = self.discriminator(fake_imgs).view(-1)  # (B,)
                mask = preds >= self.threshold
                accepted_imgs = fake_imgs[mask].detach().cpu()

                collected.extend(accepted_imgs)
                if len(collected) % 100 == 0:
                    print(f"Collected {len(collected)} / {self.num_samples} accepted fake images...")

        return torch.stack(collected[:self.num_samples])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], 1  # all fake images are labeled as 1

G = Generator(latent_dim)
D = Discriminator()
G.load_state_dict(torch.load(checkpoint_path, weights_only=True)["G_state_dict"])
D.load_state_dict(torch.load(checkpoint_path, weights_only=True)["D_state_dict"])

fake_dataset = FakeImageDataset(
    generator=G,
    discriminator=D,
    num_samples=num_samples,
    latent_dim=100,
    threshold=0.45, 
    device='cuda'
)
fake_loader = DataLoader(fake_dataset, batch_size=64)

# ==== 2. TẢI ẢNH THẬT label == 1 ====
transform = transforms.ToTensor()
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
real_images = torch.stack([img for img, label in mnist if label == 1][:num_samples])

# ==== 3. LƯU ẢNH RA THƯ MỤC ====
os.makedirs(gan_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)
to_pil = ToPILImage()

for i, (img, _) in enumerate(fake_dataset):
    to_pil((img + 1) / 2 if img.min() < 0 else img).save(os.path.join(gan_dir, f"{i:04d}.png"))

for i, img in enumerate(real_images):
    to_pil(img).save(os.path.join(real_dir, f"{i:04d}.png"))

# ==== 4. TÍNH FID ====
metrics = calculate_metrics(input1=gan_dir, input2=real_dir, fid=True, isc=False, cuda=(device == 'cuda'))
print("🎯 FID (label = 1):", metrics['frechet_inception_distance'])

# ==== 5. ĐO ACCURACY CNN ====
def evaluate_accuracy(model, dataloader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Dataloader real test label = 1
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
real_test_images = torch.stack([img for img, label in mnist_test if label == 1][:num_samples])
real_test_labels = torch.ones(num_samples, dtype=torch.long)
real_test_loader = DataLoader(TensorDataset(real_test_images, real_test_labels), batch_size=64)

# Dataloader fake
gan_images = torch.stack([img for img, _ in fake_dataset])
gan_labels = torch.ones(num_samples, dtype=torch.long)
gan_loader = DataLoader(TensorDataset(gan_images, gan_labels), batch_size=64)

# Load CNN model
CNN = Net().to(device)
CNN.load_state_dict(torch.load(cnn_checkpoint_path, weights_only=True)["net_state_dict"])
CNN.eval()

acc_real = evaluate_accuracy(CNN, real_test_loader, device)
acc_gan = evaluate_accuracy(CNN, gan_loader, device)

print(f"✅ Accuracy on real test (label = 1): {acc_real * 100:.2f}%")
print(f"🧪 Accuracy on GAN images (label = 1): {acc_gan * 100:.2f}%")

