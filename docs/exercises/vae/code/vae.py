import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures")


# EXERCICIO 1
torch.manual_seed(42)

# ToTensor ja converte os pixels de [0,255] pra [0,1] e empacota como tensor;
# a Lambda achata cada imagem 28x28 num vetor de 784 (formato que o encoder espera)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
])

mnist_train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# divide o treino oficial (60.000 imagens) em 50.000 treino + 10.000 validacao,
# usando um gerador com semente propria (nao reaproveita a semente global)
generator = torch.Generator().manual_seed(42)
mnist_train, mnist_val = random_split(mnist_train_full, [50000, 10000], generator=generator)

print("treino:", len(mnist_train))
print("validacao:", len(mnist_val))
print("teste:", len(mnist_test))

train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=128, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden1=512, hidden2=256, latent_dim=16):
        super().__init__()
        # Encoder: 784 -> 512 -> 256, depois duas "cabecas" lineares: uma pra mu, outra pra logvar
        self.enc_fc1 = nn.Linear(input_dim, hidden1)
        self.enc_fc2 = nn.Linear(hidden1, hidden2)
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)

        # Decoder: d -> 256 -> 512 -> 784
        self.dec_fc1 = nn.Linear(latent_dim, hidden2)
        self.dec_fc2 = nn.Linear(hidden2, hidden1)
        self.dec_fc3 = nn.Linear(hidden1, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # truque da reparametrizacao: z = mu + sigma * epsilon, com epsilon ~ N(0,1)
        std = torch.exp(0.5 * logvar)   # logvar -> sigma (desvio padrao)
        eps = torch.randn_like(std)     # o "dado avulso" sorteado por fora, sem gradiente
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        x_hat = torch.sigmoid(self.dec_fc3(h))  # sigmoid: cada pixel vira um numero em [0,1]
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    # reconstrucao: binary cross-entropy somada sobre os 784 pixels, com media no batch
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)

    # KL em forma fechada entre N(mu, sigma^2) e N(0, I), somada sobre as 16 dimensoes latentes,
    # com media no batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    loss = recon + beta * kl
    return loss, recon, kl


torch.manual_seed(42)  # antes de construir o modelo, como pede o enunciado
model = VAE()

# ensinando a rede e usando adam para treinamento
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 20

history_train_recon, history_train_kl = [], []
history_val_recon, history_val_kl = [], []

for epoch in range(n_epochs):
    # treino
    model.train()
    train_recon_sum, train_kl_sum, train_n = 0.0, 0.0, 0
    for x, _ in train_loader:
        optimizer.zero_grad() #limpando gradiente 
        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        train_recon_sum += recon.item() * batch_size
        train_kl_sum += kl.item() * batch_size
        train_n += batch_size

    train_recon_avg = train_recon_sum / train_n
    train_kl_avg = train_kl_sum / train_n

    # validacao (so pra acompanhar, sem atualizar pesos)
    model.eval()
    val_recon_sum, val_kl_sum, val_n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in val_loader:
            x_hat, mu, logvar = model(x)
            _, recon, kl = vae_loss(x_hat, x, mu, logvar)
            batch_size = x.size(0)
            val_recon_sum += recon.item() * batch_size
            val_kl_sum += kl.item() * batch_size
            val_n += batch_size

    val_recon_avg = val_recon_sum / val_n
    val_kl_avg = val_kl_sum / val_n

    history_train_recon.append(train_recon_avg)
    history_train_kl.append(train_kl_avg)
    history_val_recon.append(val_recon_avg)
    history_val_kl.append(val_kl_avg)

    print(f"epoca {epoch+1}/{n_epochs} | treino: recon={train_recon_avg:.2f} kl={train_kl_avg:.2f} | "
          f"val: recon={val_recon_avg:.2f} kl={val_kl_avg:.2f}")

model.eval()
test_recon_sum, test_kl_sum, test_n = 0.0, 0.0, 0
with torch.no_grad():
    for x, _ in test_loader:
        x_hat, mu, logvar = model(x)
        _, recon, kl = vae_loss(x_hat, x, mu, logvar)
        batch_size = x.size(0)
        test_recon_sum += recon.item() * batch_size
        test_kl_sum += kl.item() * batch_size
        test_n += batch_size

test_recon = test_recon_sum / test_n
test_kl = test_kl_sum / test_n
test_neg_elbo = test_recon + test_kl

print(f"TESTE -> reconstrucao: {test_recon:.2f} nats | KL: {test_kl:.2f} nats | ELBO negativo: {test_neg_elbo:.2f} nats")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, n_epochs+1), history_train_recon, label="treino")
axes[0].plot(range(1, n_epochs+1), history_val_recon, label="validacao")
axes[0].set_title("Reconstrucao por epoca")
axes[0].set_xlabel("epoca")
axes[0].set_ylabel("BCE (nats/imagem)")
axes[0].legend()

axes[1].plot(range(1, n_epochs+1), history_train_kl, label="treino")
axes[1].plot(range(1, n_epochs+1), history_val_kl, label="validacao")
axes[1].set_title("KL por epoca")
axes[1].set_xlabel("epoca")
axes[1].set_ylabel("KL (nats/imagem)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_loss_curves.png"), dpi=120)

# Reconstrução e amostras
model.eval()

# acha a primeira imagem de teste de cada digito (0 a 9)
digit_images = {}
for img, label in mnist_test:
    if label not in digit_images:
        digit_images[label] = img
    if len(digit_images) == 10:
        break

x_examples = torch.stack([digit_images[d] for d in range(10)])  # (10, 784), na ordem 0..9

with torch.no_grad():
    mu, logvar = model.encode(x_examples)
    x_recon = model.decode(mu)  # decodifica a MEDIA (mu), nao uma amostra aleatoria

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(x_examples[i].view(28, 28), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title(str(i))

    axes[1, i].imshow(x_recon[i].view(28, 28), cmap="gray")
    axes[1, i].axis("off")

fig.text(0.02, 0.75, "original", va="center", rotation="vertical")
fig.text(0.02, 0.28, "reconstruida\n(mu)", va="center", rotation="vertical")
fig.suptitle("Originais vs. reconstrucoes (decodificando mu)")
plt.tight_layout(rect=[0.04, 0, 1, 1])
plt.savefig(os.path.join(FIGURES_DIR, "fig2_reconstructions.png"), dpi=120)

# Figura 3: amostras geradas do zero, a partir do prior N(0,I)
torch.manual_seed(0)  # semente especifica so pra essa amostragem
with torch.no_grad():
    z_samples = torch.randn(64, 16)  # 64 pontos (grade 8x8) sorteados direto do prior
    samples = model.decode(z_samples)

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].view(28, 28), cmap="gray")
    ax.axis("off")
fig.suptitle("Amostras decodificadas de N(0, I)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_prior_samples.png"), dpi=120)


# EXERCICIO 2
# funcoes reutilizadas, pra nao copiar o laco de treino de novo
def train_vae(model, train_loader, val_loader, n_epochs=20, lr=1e-3, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}

    for epoch in range(n_epochs):
        model.train()
        train_recon_sum, train_kl_sum, train_n = 0.0, 0.0, 0
        for x, _ in train_loader:
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()
            batch_size = x.size(0)
            train_recon_sum += recon.item() * batch_size
            train_kl_sum += kl.item() * batch_size
            train_n += batch_size
        train_recon_avg = train_recon_sum / train_n
        train_kl_avg = train_kl_sum / train_n

        model.eval()
        val_recon_sum, val_kl_sum, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x_hat, mu, logvar = model(x)
                _, recon, kl = vae_loss(x_hat, x, mu, logvar)
                batch_size = x.size(0)
                val_recon_sum += recon.item() * batch_size
                val_kl_sum += kl.item() * batch_size
                val_n += batch_size
        val_recon_avg = val_recon_sum / val_n
        val_kl_avg = val_kl_sum / val_n

        history["train_recon"].append(train_recon_avg)
        history["train_kl"].append(train_kl_avg)
        history["val_recon"].append(val_recon_avg)
        history["val_kl"].append(val_kl_avg)

        if verbose:
            print(f"epoca {epoch+1}/{n_epochs} | treino: recon={train_recon_avg:.2f} kl={train_kl_avg:.2f} | "
                  f"val: recon={val_recon_avg:.2f} kl={val_kl_avg:.2f}")

    return history


def evaluate_vae(model, loader):
    model.eval()
    recon_sum, kl_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x_hat, mu, logvar = model(x)
            _, recon, kl = vae_loss(x_hat, x, mu, logvar)
            batch_size = x.size(0)
            recon_sum += recon.item() * batch_size
            kl_sum += kl.item() * batch_size
            n += batch_size
    return recon_sum / n, kl_sum / n


# A - treinando um segundo VAE, com d=2 mas mesmos dados/semente/beta/orcamento do Ex1
torch.manual_seed(42)  # antes de construir o novo modelo, como pede o enunciado
model_2d = VAE(latent_dim=2)
history_2d = train_vae(model_2d, train_loader, val_loader, n_epochs=20, lr=1e-3)

test_recon_2d, test_kl_2d = evaluate_vae(model_2d, test_loader)
print(f"TESTE (d=2) -> reconstrucao: {test_recon_2d:.2f} nats | KL: {test_kl_2d:.2f} nats")

# B - scatter das medias (mu) do teste, coloridas por digito
model_2d.eval()
all_mu, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        mu, logvar = model_2d.encode(x)
        all_mu.append(mu)
        all_labels.append(y)

all_mu = torch.cat(all_mu).numpy()        # (10000, 2)
all_labels = torch.cat(all_labels).numpy()

plt.figure(figsize=(7, 7))
scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap="tab10", s=3, alpha=0.6)
plt.legend(handles=scatter.legend_elements()[0], labels=[str(d) for d in range(10)],
           title="Dígito", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("Espaço latente (μ) do VAE d=2 — conjunto de teste")
plt.xlabel("z1")
plt.ylabel("z2")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig4_latent_scatter.png"), dpi=120)

# C - plano decodificado: grade 15x15, com z = Phi^-1(q)
n_grid = 15
q = np.linspace(0.05, 0.95, n_grid)
grid_vals = norm.ppf(q)  # converte cada probabilidade (q) no z correspondente

z1_grid, z2_grid = np.meshgrid(grid_vals, grid_vals[::-1])  # [::-1]: z2 cresce de baixo pra cima na figura
z_grid = np.stack([z1_grid.ravel(), z2_grid.ravel()], axis=1)  # (225, 2)
z_grid = torch.tensor(z_grid, dtype=torch.float32)

model_2d.eval()
with torch.no_grad():
    decoded_grid = model_2d.decode(z_grid)

fig, axes = plt.subplots(n_grid, n_grid, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(decoded_grid[i].view(28, 28), cmap="gray")
    ax.axis("off")
fig.suptitle("Plano decodificado (grade 15x15, z = Phi^-1(q))")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig5_decoded_plane.png"), dpi=120)

# D - interpolacao entre duas imagens de teste de digitos diferentes
idx_a = next(i for i in range(len(mnist_test)) if mnist_test[i][1] == 2)
idx_b = next(i for i in range(len(mnist_test)) if mnist_test[i][1] == 8)
label_a, label_b = 2, 8
print(f"interpolando indice {idx_a} (digito {label_a}) -> indice {idx_b} (digito {label_b})")

img_a, _ = mnist_test[idx_a]
img_b, _ = mnist_test[idx_b]

ts = np.linspace(0, 1, 10)

# linha de cima: interpolacao linear direto nos pixels
pixel_interp = torch.stack([(1 - t) * img_a + t * img_b for t in ts])

# linha de baixo: interpolacao entre os mu do modelo d=16 (Ex1), decodificando o resultado
model.eval()
with torch.no_grad():
    mu_a, _ = model.encode(img_a.unsqueeze(0))
    mu_b, _ = model.encode(img_b.unsqueeze(0))
    mu_interp = torch.stack([(1 - t) * mu_a.squeeze(0) + t * mu_b.squeeze(0) for t in ts])
    latent_interp = model.decode(mu_interp)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(pixel_interp[i].view(28, 28), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title(f"t={ts[i]:.1f}", fontsize=8)

    axes[1, i].imshow(latent_interp[i].view(28, 28), cmap="gray")
    axes[1, i].axis("off")

fig.text(0.02, 0.75, "pixels", va="center", rotation="vertical")
fig.text(0.02, 0.28, "latente\n(mu, d=16)", va="center", rotation="vertical")
fig.suptitle(f"Interpolacao: indice {idx_a} (digito {label_a}) -> indice {idx_b} (digito {label_b})")
plt.tight_layout(rect=[0.04, 0, 1, 1])
plt.savefig(os.path.join(FIGURES_DIR, "fig6_interpolation.png"), dpi=120)