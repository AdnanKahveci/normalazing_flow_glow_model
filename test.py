import torch
from glow import Glow
import numpy as np
import skimage.io as sio
import matplotlib.pyplot as plt
import os

# Sabitler
size = 16
K = 16
L = 3
coupling = "affine"
last_zeros = True
n_bits_x = 8
device = "cuda"
save_path = "mnist/mnist_%dx%d/training/glowmodel.pt" % (size, size)
fig_path = "mnist/mnist_%dx%d/training/figures" % (size, size)


# Kaydedilmis model ile veri uretme
def generate_and_save_samples(model_path, num_samples=5, temp=0.8):
    # Modeli yükle
    glow = Glow((1, size, size),
                K=K, L=L, coupling=coupling, n_bits_x=n_bits_x,
                nn_init_last_zeros=last_zeros,
                device=device)
    glow.load_state_dict(torch.load(model_path, map_location=device))
    glow.eval()

    # Boyutların başlatılması için modelden bir veri geçir
    dummy_input = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        glow(dummy_input)

    # Örnekler üret
    with torch.no_grad():
        z_sample, z_sample_t = glow.generate_z(n=num_samples, mu=0, std=0.7, to_torch=True)
        x_gen = glow(z_sample_t, reverse=True)
        x_gen = glow.postprocess(x_gen)
        x_gen = x_gen.data.cpu().numpy()
        x_gen = x_gen.transpose([0, 2, 3, 1])
        if x_gen.shape[-1] == 1:
            x_gen = x_gen[..., 0]
        
        # Görüntüleri kaydet
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        for i in range(num_samples):
            plt.imshow(x_gen[i], cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(fig_path, f"generated_sample_{i}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

    print("Örnekler Kaydedildi")

# Fonksiyonu çağır
generate_and_save_samples(save_path)