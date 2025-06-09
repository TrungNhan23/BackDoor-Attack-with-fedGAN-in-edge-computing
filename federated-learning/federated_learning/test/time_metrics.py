import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import psutil
import os
import torch
import torch.ao.quantization
from federated_learning.model.quantization_gan import (
    Generator, 
    Discriminator
)

def measure_gan_performance(generator, noise_dim=100, device='cpu', num_samples=100):
    # Move model to device
    generator.to(device)
    generator.eval()

    # Prepare noise input
    noise = torch.randn(num_samples, noise_dim, device=device)

    # Measure CPU and RAM before
    process = psutil.Process(os.getpid())
    cpu_before = psutil.cpu_percent(interval=None)
    ram_before = process.memory_info().rss / (1024 * 1024)  # in MB

    # Measure time
    start_time = time.time()
    with torch.no_grad():
        _ = generator(noise)
    end_time = time.time()

    # Measure CPU and RAM after
    cpu_after = psutil.cpu_percent(interval=None)
    ram_after = process.memory_info().rss / (1024 * 1024)  # in MB

    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"CPU Usage: {cpu_after - cpu_before:.2f}%")
    print(f"RAM Usage: {ram_after - ram_before:.2f} MB")


generator = Generator(100)
discriminator = Discriminator() 
checkpoint_path = "../../model_state_dict/gan/"
generator.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.prepare_qat(generator, inplace=True)

discriminator.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.prepare_qat(discriminator, inplace=True)

generator_statedict = torch.load(checkpoint_path + "G_checkpoint.pt", map_location='cpu')
discriminator_statedict = torch.load(checkpoint_path + "D_checkpoint.pt", map_location='cpu')

generator.load_state_dict(generator_statedict)
discriminator.load_state_dict(discriminator_statedict)


generator.eval()
discriminator.eval()
# measure_gan_performance(generator, noise_dim=100, device='cpu', num_samples=100)