# Imports e libs importante para o exercício
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# Exercício 1: Gerar dados de duas classes com distribuições normais diferentes

# A: montando as classes e gerando as nuvens
# Parâmetros de cada classe pré-definidos: centro e desvio padrão
parametros = {
    0: {"centro": [2, 3],  "desv_pad": [0.8, 2.5]},
    1: {"centro": [5, 6],  "desv_pad": [1.2, 1.9]},
    2: {"centro": [8, 1],  "desv_pad": [0.9, 0.9]},
    3: {"centro": [15, 4], "desv_pad": [0.5, 2.0]},
}
n_por_classe = 100

nuvens = {}
for classe, p in parametros.items():
    nuvens[classe] = rng.normal(loc=p["centro"], scale=p["desv_pad"], size=(n_por_classe, 2))

# Juntando as nuvens em um único array de dados e criando os rótulos
X = np.vstack([nuvens[classe] for classe in parametros])
y = np.repeat(list(parametros.keys()), n_por_classe) # classes correspondentes a cada ponto

fig, ax = plt.subplots(figsize=(8, 6))

# Plotando a Figura 1 com o scatter plot 2d com uma cor por classe e o centro marcado
for classe in parametros:
    ax.scatter(nuvens[classe][:, 0], nuvens[classe][:, 1], label=f"Classe {classe}", alpha=0.7)

# marcar os centros de todas as classes de uma vez, com um marcador diferente
centros = np.array([p["centro"] for p in parametros.values()])
ax.scatter(centros[:, 0], centros[:, 1], marker="X", s=200, color="black", edgecolor="white", linewidth=1.5, label="Centro")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Figura 1: Nuvens de pontos por classe")
ax.legend()

# Salvando a figura em PNG pra mostrar no gitpages
plt.savefig(FIGURES_DIR / "fig1_scatter.png", dpi=150, bbox_inches="tight")
#plt.show()

# B: análise de espalhamento 
# 1. Gerando os 4 dataset com a multiplicação pelo fator de escala s
scale_factors = [0.5, 1.0, 2.0, 4.0]

nuvens_por_s = {}
for s in scale_factors:
    nuvens_s = {}
    for classe, p in parametros.items():
        desv_pad_escalado = np.array(p["desv_pad"]) * s
        nuvens_s[classe] = rng.normal(loc=p["centro"], scale=desv_pad_escalado, size=(n_por_classe, 2))
    nuvens_por_s[s] = nuvens_s

# Gerando a Figura 2 com os subplots segundo valor de s + garantindo mesmo limite dos eixos (não afetar visualização)
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

for ax, s in zip(axs, scale_factors):
    nuvens_s = nuvens_por_s[s]
    for classe in parametros:
        ax.scatter(nuvens_s[classe][:, 0], nuvens_s[classe][:, 1], label=f"Classe {classe}", alpha=0.7)
    ax.scatter(centros[:, 0], centros[:, 1], marker="X", s=150, color="black", edgecolor="white", linewidth=1.2, label="Centro")
    ax.set_title(f"s = {s}")
    ax.set_xlabel("x")

axs[0].set_ylabel("y")
axs[0].legend()
fig.suptitle("Figura 2: Nuvens de pontos para diferentes fatores de espalhamento (s)")
plt.savefig(FIGURES_DIR / "fig2_espalhamento.png", dpi=150, bbox_inches="tight")

# 2. Cálculo da razão de separação, taxa de mistura (+ tabelas e médias)
# Razão de separação entre cada par de classes (s = 1)
pares = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # as 6 combinações possíveis entre 4 classes

def sigma_bar(classe):
    return np.mean(parametros[classe]["desv_pad"])

r_ij = {}
for i, j in pares:
    mu_i = np.array(parametros[i]["centro"])
    mu_j = np.array(parametros[j]["centro"])
    distancia = np.linalg.norm(mu_i - mu_j)  # distância euclidiana entre os centros
    r_ij[(i, j)] = distancia / (sigma_bar(i) + sigma_bar(j))

#for par, valor in r_ij.items():
#    print(par, round(valor, 3))

par_menor = min(r_ij, key=r_ij.get)
#print("Par com menor separação:", par_menor, round(r_ij[par_menor], 3))

# Como a média não muda, o fator de escala s afeta a razão de separação r_ij(s) = r_ij(s=1) / s. Portanto, podemos calcular o menor r_ij para s=2 sem gerar novos dados:
menor_r_ij_s1 = r_ij[par_menor]
menor_r_ij_s2 = menor_r_ij_s1 / 2  # porque r_ij(s) = r_ij(s=1) / s

#print(f"Menor r_ij em s=1: {menor_r_ij_s1:.3f}")
#print(f"Menor r_ij em s=2 (calculado, sem gerar dados novos): {menor_r_ij_s2:.3f}")

# Taxa de mistura (distancia dos pontos com seu centro e os demais)
def calcular_taxa_mistura(nuvens_s):
    X_s = np.vstack([nuvens_s[classe] for classe in parametros])
    y_s = np.repeat(list(parametros.keys()), n_por_classe)

    # tabela de distâncias: uma linha por ponto, uma coluna por classe
    distancias = np.zeros((len(X_s), len(parametros)))
    for classe in parametros:
        centro = np.array(parametros[classe]["centro"])
        distancias[:, classe] = np.linalg.norm(X_s - centro, axis=1)

    classe_mais_proxima = np.argmin(distancias, axis=1)
    taxa_mistura = np.mean(classe_mais_proxima != y_s) # Avaliando caso o centro mais proximo é o da sua classe, caso contrario retorna False (que seriam considerados mistura)
    return taxa_mistura

taxas_mistura = {}
for s in scale_factors:
    taxas_mistura[s] = calcular_taxa_mistura(nuvens_por_s[s])
    #print(f"s = {s}: taxa de mistura = {taxas_mistura[s]:.3f}")

# 3. Produzindo a Figura 3 de taxa de mistura por s
fig, ax = plt.subplots(figsize=(7, 5))

s_valores = list(taxas_mistura.keys())
mistura_valores = list(taxas_mistura.values())

ax.plot(s_valores, mistura_valores, marker="o", linestyle="-", label="Taxa de mistura")

ax.set_xlabel("Fator de escala (s)")
ax.set_ylabel("Taxa de mistura")
ax.set_title("Figura 3: Taxa de mistura em função do fator de escala")
ax.legend()

plt.savefig(FIGURES_DIR / "fig3_taxa_mistura.png", dpi=150, bbox_inches="tight")
#plt.show()

# C: Análises e conclusões
# Dados e informações vão ser dispostas direto no arquivo index.md pra documentar

# Figura de delimitação das fronteiras na figura 1, seguindo na ideia da bissetriz entre centros
def bissetriz(p1, p2, comprimento=6):
    p1, p2 = np.array(p1), np.array(p2)
    meio = (p1 + p2) / 2
    direcao = p2 - p1
    perpendicular = np.array([-direcao[1], direcao[0]])       # gira o vetor 90 graus
    perpendicular = perpendicular / np.linalg.norm(perpendicular)  # normaliza pra ter tamanho 1
    ponto_a = meio - perpendicular * comprimento / 2
    ponto_b = meio + perpendicular * comprimento / 2
    return ponto_a, ponto_b

fronteiras = [(0, 1), (1, 2), (2, 3)]

fig, ax = plt.subplots(figsize=(8, 6))
for classe in parametros:
    ax.scatter(nuvens[classe][:, 0], nuvens[classe][:, 1], label=f"Classe {classe}", alpha=0.7)
ax.scatter(centros[:, 0], centros[:, 1], marker="X", s=200, color="black", edgecolor="white", linewidth=1.5, label="Centro")

for i, j in fronteiras:
    a, b = bissetriz(parametros[i]["centro"], parametros[j]["centro"], comprimento=6)
    ax.plot([a[0], b[0]], [a[1], b[1]], "k--", linewidth=2)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Figura 1 (com fronteiras esboçadas) s = 1.0")
ax.legend()

plt.savefig(FIGURES_DIR / "fig1_fronteiras.png", dpi=150, bbox_inches="tight")
#plt.show()

###########################################################################################

# Exercício 2: Não linearidadeem dimensões maiores
# A: Dataset 1 Gaussianas deslocadas
mu_A = np.array([0, 0, 0, 0, 0])
cov_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
cov_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

n_amostras_5d = 500

classe_A = rng.multivariate_normal(mean=mu_A, cov=cov_A, size=n_amostras_5d)
classe_B = rng.multivariate_normal(mean=mu_B, cov=cov_B, size=n_amostras_5d)

#print(classe_A.shape) 
#print(classe_A[:3])

#B: Dataset 2 cascas concêntricas
n_5d = 500
dim = 5

def gerar_casca(raio_medio, raio_desvio, n, dim):
    raios = rng.normal(raio_medio, raio_desvio, size=n)
    direcoes = rng.normal(0, 1, size=(n, dim))
    direcoes = direcoes / np.linalg.norm(direcoes, axis=1, keepdims=True)
    pontos = raios[:, np.newaxis] * direcoes
    return pontos

classe_C = gerar_casca(2.0, 0.4, n_5d, dim)   
classe_D = gerar_casca(5.0, 0.4, n_5d, dim)   

# C: Visualizando e comparando (lembrando que não é possível analisar o 5D, haverá uma redução de dimensionalidade)
# Uso do PCA que pega as duas dimensões que mais variam pra projetar em cima, "achatando"
# A variância explicada diz quanto da info original foi mantida, então 90% é um bom achado,se for baixo, a imagem está escondendo informações importantes
# Dataset I: combina as classes A e B pra ajustar o PCA, esperado que seja mais simples, gaussiana deslocaa
dataset_I = np.vstack([classe_A, classe_B]) 
pca_I = PCA(n_components=2)
proj_I = pca_I.fit_transform(dataset_I)

# Dataset II: combina as classes C e D, esperado que seja mais complexo
dataset_II = np.vstack([classe_C, classe_D])
pca_II = PCA(n_components=2)
proj_II = pca_II.fit_transform(dataset_II)

#print("Variância explicada (Dataset I):", pca_I.explained_variance_ratio_)
#print("Variância explicada (Dataset II):", pca_II.explained_variance_ratio_)

# Desenhando a Figura 4 para visualização
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Dataset I: as primeiras 500 linhas de proj_I são da Classe A, as próximas 500 da Classe B
axs[0].scatter(proj_I[:500, 0], proj_I[:500, 1], label="Classe A", alpha=0.6)
axs[0].scatter(proj_I[500:, 0], proj_I[500:, 1], label="Classe B", alpha=0.6)
axs[0].set_xlabel("Componente Principal 1")
axs[0].set_ylabel("Componente Principal 2")
axs[0].set_title("Dataset I — projeção PCA")
axs[0].legend()

# Dataset II: mesma lógica, primeiras 500 são Classe C, próximas 500 são Classe D
axs[1].scatter(proj_II[:500, 0], proj_II[:500, 1], label="Classe C", alpha=0.6)
axs[1].scatter(proj_II[500:, 0], proj_II[500:, 1], label="Classe D", alpha=0.6)
axs[1].set_xlabel("Componente Principal 1")
axs[1].set_ylabel("Componente Principal 2")
axs[1].set_title("Dataset II — projeção PCA")
axs[1].legend()

fig.suptitle("Figura 4: Projeção PCA 2D dos dois datasets")

plt.savefig(FIGURES_DIR / "fig4_pca.png", dpi=150, bbox_inches="tight")
#plt.show()

# Definindo distancias entre centros em 5D
# Distância entre A e B já eram conhecidos os centros, a de C e D foi necessário calculo da média dos pontos 
dist_AB = np.linalg.norm(mu_A - mu_B)

centro_C = np.mean(classe_C, axis=0)
centro_D = np.mean(classe_D, axis=0)
dist_CD = np.linalg.norm(centro_C - centro_D)

#print(f"Distância entre centros (Dataset I, A-B): {dist_AB:.3f}")
#print(f"Distância entre centros (Dataset II, C-D): {dist_CD:.3f}")

# Desenhando Figura 5 do histograma de raios
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

raio_A = np.linalg.norm(classe_A, axis=1)
raio_B = np.linalg.norm(classe_B, axis=1)
axs[0].hist(raio_A, bins=30, alpha=0.6, label="Classe A")
axs[0].hist(raio_B, bins=30, alpha=0.6, label="Classe B")
axs[0].set_xlabel("Raio (||x||)")
axs[0].set_ylabel("Frequência")
axs[0].set_title("Dataset I")
axs[0].legend()

raio_C = np.linalg.norm(classe_C, axis=1)
raio_D = np.linalg.norm(classe_D, axis=1)
axs[1].hist(raio_C, bins=30, alpha=0.6, label="Classe C")
axs[1].hist(raio_D, bins=30, alpha=0.6, label="Classe D")
axs[1].set_xlabel("Raio (||x||)")
axs[1].set_ylabel("Frequência")
axs[1].set_title("Dataset II")
axs[1].legend()

fig.suptitle("Figura 5: Histograma do raio de cada ponto")

plt.savefig(FIGURES_DIR / "fig5_raio_hist.png", dpi=150, bbox_inches="tight")
