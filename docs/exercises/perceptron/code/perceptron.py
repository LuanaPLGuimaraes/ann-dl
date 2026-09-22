# Codigo realizado para resolver a aps 2 de perceptron

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# Exercicio 1: Dados separaveis
# A - gerando dados
n_por_classe = 1000

mu0 = [1.5, 1.5]
cov0 = [[0.5, 0], [0, 0.5]]
mu1 = [5, 5]
cov1 = [[0.5, 0], [0, 0.5]]

classe0 = rng.multivariate_normal(mu0, cov0, size=n_por_classe)
classe1 = rng.multivariate_normal(mu1, cov1, size=n_por_classe)

X1 = np.vstack([classe0, classe1])
y1 = np.concatenate([np.zeros(n_por_classe), np.ones(n_por_classe)])

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(classe0[:, 0], classe0[:, 1], label="Classe 0", alpha=0.5)
ax.scatter(classe1[:, 0], classe1[:, 1], label="Classe 1", alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Figura 1: Dados separáveis (Exercício 1)")
ax.legend()
plt.savefig(FIGURES_DIR / "fig1_dados_separaveis.png", dpi=150, bbox_inches="tight")

#print(X1.shape, y1.shape)

# B - implementando o perceptron
# z = 1 se z >= 0, se nao z = 0
def degrau(z):
    return np.where(z >= 0, 1, 0)

# y^ = degrau(w*x + b) e w<-w+n*(y-y^)*x e b<-b+n*(y-y^)
#def treinar_perceptron(X, y, n, max_epocas, rng):
    w = rng.normal(0, 0.01, size=X.shape[1]) # sorteio (nao pode comecar em 0 pq a taxa nao muda em inicio em 0)
    b = 0.0

    # registrando o melhor modelo
    melhor_w, melhor_b = w.copy(), b
    melhor_acuracia = 0.0
    historico_acuracia = []

    for epoca in range(max_epocas):
        houve_atualizacao = False
        for xi, yi in zip(X, y):
            z = np.dot(w, xi) + b
            y_pred = 1 if z >= 0 else 0
            erro = yi - y_pred
            if erro != 0:
                w = w + n * erro * xi
                b = b + n * erro
                houve_atualizacao = True

        y_pred_epoca = degrau(X @ w + b)
        acuracia = np.mean(y_pred_epoca == y)
        historico_acuracia.append(acuracia)

        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_w, melhor_b = w.copy(), b

        if not houve_atualizacao:
            break

    epocas_usadas = epoca + 1

    return {
        "w_final": w, "b_final": b,
        "w_pocket": melhor_w, "b_pocket": melhor_b,
        "acuracia_pocket": melhor_acuracia,
        "epocas": epocas_usadas,
        "historico_acuracia": historico_acuracia,
    }#

# Novo modelo para treinar perceptron embaralhando os pontos, dado que no modelo de cima 
# ele perdia grande parte do aprendizado (visto na acuracia) por sempre treinar classe 0 
# e só depois a classe 1
def treinar_perceptron(X, y, n, max_epocas, rng):
    w = rng.normal(0, 0.01, size=X.shape[1])
    b = 0.0

    melhor_w, melhor_b = w.copy(), b
    melhor_acuracia = 0.0
    historico_acuracia = []
    historico_pocket = []        
    for epoca in range(max_epocas):
        indices = rng.permutation(len(X))
        houve_atualizacao = False
        for i in indices:
            xi, yi = X[i], y[i]
            z = np.dot(w, xi) + b
            y_pred = 1 if z >= 0 else 0
            erro = yi - y_pred
            if erro != 0:
                w = w + n * erro * xi
                b = b + n * erro
                houve_atualizacao = True

        y_pred_epoca = degrau(X @ w + b)
        acuracia = np.mean(y_pred_epoca == y)
        historico_acuracia.append(acuracia)

        if acuracia > melhor_acuracia:
            melhor_acuracia = acuracia
            melhor_w, melhor_b = w.copy(), b

        historico_pocket.append(melhor_acuracia)  

        if not houve_atualizacao:
            break

    epocas_usadas = epoca + 1

    return {
        "w_final": w, "b_final": b,
        "w_pocket": melhor_w, "b_pocket": melhor_b,
        "acuracia_pocket": melhor_acuracia,
        "epocas": epocas_usadas,
        "historico_acuracia": historico_acuracia,
        "historico_pocket": historico_pocket,       # <- novo
    }

# c - treinando o modelo com n (tx de aprend) = 0.01 e max_epocas = 100
resultado1 = treinar_perceptron(X1, y1, n=0.01, max_epocas=100, rng=rng)

w1_final, b1_final = resultado1["w_final"], resultado1["b_final"]
epocas1 = resultado1["epocas"]
y1_pred = degrau(X1 @ w1_final + b1_final)
acuracia1 = np.mean(y1_pred == y1)

print("w1 final:", w1_final)
print("b1 final:", b1_final)
print("épocas1 até convergência:", epocas1)
print("acurácia1 final:", acuracia1)

# Figura 2: fronteira de decisao + pontos mal classificados
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X1[y1 == 0][:, 0], X1[y1 == 0][:, 1], label="Classe 0", alpha=0.4)
ax.scatter(X1[y1 == 1][:, 0], X1[y1 == 1][:, 1], label="Classe 1", alpha=0.4)

erros1 = y1_pred != y1
ax.scatter(X1[erros1][:, 0], X1[erros1][:, 1], facecolors="none", edgecolors="red", s=100, linewidths=1.5, label="Mal classificado")

x1_vals = np.linspace(X1[:, 0].min() - 1, X1[:, 0].max() + 1, 100)
x2_vals = -(w1_final[0] * x1_vals + b1_final) / w1_final[1]
ax.plot(x1_vals, x2_vals, "k--", label="Fronteira de decisão")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 2: Fronteira de decisão (Exercício 1)")
ax.legend()
plt.savefig(FIGURES_DIR / "fig2_fronteira_ex1.png", dpi=150, bbox_inches="tight")

# Figura 3: acuracia por epoca
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, epocas1 + 1), resultado1["historico_acuracia"], marker="o")
ax.set_xlabel("Época")
ax.set_ylabel("Acurácia")
ax.set_title("Figura 3: Acurácia por época (Exercício 1)")
plt.savefig(FIGURES_DIR / "fig3_acuracia_ex1.png", dpi=150, bbox_inches="tight")

# D1. analise feita no index.md
# D2. re-executando o treinamento com n=1 e max_epocas=100
resultado2 = treinar_perceptron(X1, y1, n=1.0, max_epocas=100, rng=rng)

w2 = resultado2["w_final"]
b2 = resultado2["b_final"]
epocas2 = resultado2["epocas"]
y2_pred = degrau(X1 @ w2 + b2)
acuracia2 = np.mean(y2_pred == y1)

print("w2 final (n=1.0):", w2)
print("b2 final (n=1.0):", b2)
print("épocas até convergência (n=1.0):", epocas2)
print("acurácia final (n=1.0):", acuracia2)

# Comparando a direção w (n=0.01 vs n=1.0) via similaridade de cosseno
# n = 1 os pesos tem magnitude muito maior, ent normaliza para entender direcao dos vetores e nao magnitude
cos_similaridade = np.dot(w1_final, w2) / (np.linalg.norm(w1_final) * np.linalg.norm(w2))
print("similaridade de cosseno entre as duas direções de w:", cos_similaridade)
# D3. falado no index.md

# Exercicio 2: Dados não separáveis
# reaproveitando a funcao do ex1
# A: gerando os dados 
n_por_classe = 1000

mu0_2 = [3, 3]
cov0_2 = [[1.5, 0], [0, 1.5]]
mu1_2 = [4, 4]
cov1_2 = [[1.5, 0], [0, 1.5]]

classe0_2 = rng.multivariate_normal(mu0_2, cov0_2, size=n_por_classe)
classe1_2 = rng.multivariate_normal(mu1_2, cov1_2, size=n_por_classe)

X2 = np.vstack([classe0_2, classe1_2])
y2 = np.concatenate([np.zeros(n_por_classe), np.ones(n_por_classe)])

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(classe0_2[:, 0], classe0_2[:, 1], label="Classe 0", alpha=0.4)
ax.scatter(classe1_2[:, 0], classe1_2[:, 1], label="Classe 1", alpha=0.4)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 4: Dados sobrepostos (Exercício 2)")
ax.legend()
plt.savefig(FIGURES_DIR / "fig4_dados_sobrepostos.png", dpi=150, bbox_inches="tight")

# B - treinando o modelo com n (tx de aprend) = 0.01 e max_epocas = 100
resultado3 = treinar_perceptron(X2, y2, n=0.01, max_epocas=100, rng=rng)

w3_final, b3_final = resultado3["w_final"], resultado3["b_final"]
w3_pocket, b3_pocket = resultado3["w_pocket"], resultado3["b_pocket"]
epocas3 = resultado3["epocas"]

y3_pred_final = degrau(X2 @ w3_final + b3_final)
acuracia3_final = np.mean(y3_pred_final == y2)
acuracia3_pocket = resultado3["acuracia_pocket"]

# novo def para o perceptron foi definido a partir do primeiro resultado dar perto
# de 50% de pocket que não esta de acordo com o ex 
#print("w final:", w3_final, "b final:", b3_final, "acurácia final:", acuracia3_final)
#print("w pocket:", w3_pocket, "b pocket:", b3_pocket, "acurácia pocket:", acuracia3_pocket)
#print("épocas rodadas:", epocas3)

# C - montando as figuras
# Figura 5: fronteiras final vs pocket
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X2[y2 == 0][:, 0], X2[y2 == 0][:, 1], label="Classe 0", alpha=0.4)
ax.scatter(X2[y2 == 1][:, 0], X2[y2 == 1][:, 1], label="Classe 1", alpha=0.4)

y3_pred_pocket = degrau(X2 @ w3_pocket + b3_pocket)
erros3_pocket = y3_pred_pocket != y2
ax.scatter(X2[erros3_pocket][:, 0], X2[erros3_pocket][:, 1],
           facecolors="none", edgecolors="red", s=60, linewidths=1.0,
           label="Mal classificado (pocket)")

x1_vals = np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 100)

x2_vals_final = -(w3_final[0] * x1_vals + b3_final) / w3_final[1]
ax.plot(x1_vals, x2_vals_final, "k--", label="Fronteira final")

x2_vals_pocket = -(w3_pocket[0] * x1_vals + b3_pocket) / w3_pocket[1]
ax.plot(x1_vals, x2_vals_pocket, "g-", linewidth=2, label="Fronteira pocket")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Figura 5: Fronteiras final vs. pocket (Exercício 2)")
ax.legend()
plt.savefig(FIGURES_DIR / "fig5_fronteira_final_vs_pocket.png", dpi=150, bbox_inches="tight")

# Figura 6: acuracia por epoca - corrente vs pocket (Exercicio 2)
fig, ax = plt.subplots(figsize=(7, 5))
epocas_range = range(1, epocas3 + 1)
ax.plot(epocas_range, resultado3["historico_acuracia"], color="black", label="Acurácia corrente (pesos finais)")
ax.plot(epocas_range, resultado3["historico_pocket"], color="green", linestyle="--", label="Melhor acurácia (pocket)")
ax.set_xlabel("Época")
ax.set_ylabel("Acurácia")
ax.set_ylim(0, 1)
ax.set_title("Figura 6: Acurácia por época — final vs. pocket (Exercício 2)")
ax.legend()
plt.savefig(FIGURES_DIR / "fig6_acuracia_final_vs_pocket.png", dpi=150, bbox_inches="tight")


epoca_pocket_melhor = resultado3["historico_pocket"].index(acuracia3_pocket) + 1
print("época em que o pocket atingiu o melhor valor:", epoca_pocket_melhor)