import argparse
import json
import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import skimage.io as sio
from glow import Glow

def parse_args():
    # Komut satırı argümanlarını ayarlamak için bir parser oluşturur
    parser = argparse.ArgumentParser(description='Seçilen veri setinde GLOW modelini eğitin.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'celeba'], default='mnist',
                        help='Eğitim için kullanılacak veri seti (mnist, cifar10, celeba)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--size', type=int, default=32)
    return parser.parse_args()

def get_dataloader(dataset_name, batch_size, size):
    # Veri setine göre uygun transformasyonları ve DataLoader'ı oluşturur
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        dataset = datasets.MNIST(root="./mnist/data/", transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./cifar10/data/", download=True, transform=transform)
    elif dataset_name == 'celeba':
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.CenterCrop(size), transforms.ToTensor()])
        dataset = datasets.CelebA(root="./celeba/data/", download=True, transform=transform)
    else:
        raise ValueError("Tanımlanamayan veri seti")
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

def main():
    # Komut satırı argümanlarını al
    args = parse_args()
    
    # Sabitler
    K           = 16
    L           = 3
    coupling    = "affine"
    last_zeros  = True
    batchsize   = 16  # Küçültülmüş batch size
    size        = args.size
    lr          = 1e-4
    n_bits_x    = 8
    epochs      = args.epochs
    warmup_iter = 0
    sample_freq = 50
    save_freq   = 1000
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    save_path   = f"{args.dataset}/{args.dataset}_{size}x{size}/training/glowmodel.pt"

    # Konfigürasyonları kaydet
    config_path = f"{args.dataset}/{args.dataset}_{size}x{size}/training/configs.json"
    configs = {
        "K": K,
        "L": L,
        "coupling": coupling,
        "last_zeros": last_zeros,
        "batchsize": batchsize,
        "size": size,
        "lr": lr,
        "n_bits_x": n_bits_x,
        "warmup_iter": warmup_iter
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(configs, f, sort_keys=True, indent=4, ensure_ascii=False)
    
    # DataLoader'ı oluştur
    data_loader = get_dataloader(args.dataset, batchsize, size)

    # GLOW modelini yükle
    input_channels = 3 if args.dataset in ['cifar10', 'celeba'] else 1
    glow = Glow((input_channels, size, size),
                K=K, L=L, coupling=coupling, n_bits_x=n_bits_x,
                nn_init_last_zeros=last_zeros, device=device)

    # Önceden eğitilmiş modeli yükle
    if os.path.exists(save_path):
        glow.load_state_dict(torch.load(save_path))
        glow.set_actnorm_init()
        print("Önceden Eğitilmiş Model Yüklendi")
        print("Actnorm Başlatıldı")

    # Optimizasyon ve öğrenme oranı planlayıcıyı ayarla
    opt = torch.optim.Adam(glow.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                              factor=0.5, patience=1000,
                                                              verbose=True, min_lr=1e-8)

    # Eğitim kodu
    global_step = 0
    global_loss = []
    warmup_completed = False
    for i in range(epochs):
        Loss_epoch = []
        for j, data in enumerate(data_loader):
            opt.zero_grad()
            glow.zero_grad()

            # Batch'i yükle
            x = data[0].cuda() * 255

            # Veriyi ön işle
            x = glow.preprocess(x)
            n, c, h, w = x.size()
            nll, logdet, logpz, z_mu, z_std = glow.nll_loss(x)
            if global_step == 0:
                global_step += 1
                continue

            # Kayıp fonksiyonunu geriye yay
            nll.backward()
            torch.nn.utils.clip_grad_value_(glow.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(glow.parameters(), 100)

            # Öğrenme oranını warmup_iter'e kadar lineer olarak artır
            if global_step <= warmup_iter:
                warmup_lr = lr / warmup_iter * global_step
                for params in opt.param_groups:
                    params["lr"] = warmup_lr

            # Optimizasyon adımını gerçekleştir
            opt.step()

            # Öğrenme oranı planlaması
            if global_step > warmup_iter:
                lr_scheduler.step(nll)
                if not warmup_completed:
                    print("\nWarm up Tamamlandı")
                warmup_completed = True

            # Eğitim metriklerini yazdır
            print(f"\repoch={i:02d}..nll={nll.item():.2f}..logdet={logdet:.2f}..logpz={logpz:.2f}..mu={z_mu:.2f}..std={z_std:.2f}", end="\r")

            try:
                # Belirli aralıklarla örnekler üret ve kaydet
                if j % sample_freq == 0:
                    with torch.no_grad():
                        z_sample, z_sample_t = glow.generate_z(n=50, mu=0, std=0.7, to_torch=True)
                        x_gen = glow(z_sample_t, reverse=True)
                        x_gen = glow.postprocess(x_gen)
                        x_gen = x_gen.data.cpu().numpy()
                        x_gen = x_gen.transpose([0, 2, 3, 1])
                        if x_gen.shape[-1] == 1:
                            x_gen = x_gen[..., 0]
                        sio.imshow_collection(x_gen)
                        plt.savefig(f"./fig/{global_step}.jpg")
                        plt.close()
            except:
                print(f"\nGlobal Adım = {global_step} sırasında hata oluştu")

            global_step += 1
            global_loss.append(nll.item())
            if global_step % save_freq == 0:
                torch.save(glow.state_dict(), save_path)
                torch.cuda.empty_cache()  # Belleği temizle
    
    # Model görselleştirmesi
    temperature = [0.8]
    for temp in temperature:
        with torch.no_grad():
            glow.eval()
            z_sample, z_sample_t = glow.generate_z(n=50, mu=0, std=0.7, to_torch=True)
            x_gen = glow(z_sample_t, reverse=True)
            x_gen = glow.postprocess(x_gen)
            x_gen = x_gen.data.cpu().numpy()
            x_gen = x_gen.transpose([0, 2, 3, 1])
            if x_gen.shape[-1] == 1:
                x_gen = x_gen[..., 0]
            sio.imshow_collection(x_gen)
            plt.savefig(f"./fig/{global_step}.jpg")
            plt.close()
    print("Eğitim Tamamlandı")

if __name__ == "__main__":
    main()
