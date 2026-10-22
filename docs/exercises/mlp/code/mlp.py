import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

# Exercício 2: Classificação binária que uma reta não resolve
# A - gerando os dados
# 1000 pontos: 500 no centro, 250 + 250 nas duas pontas da diagonal
centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                  cluster_std=1.2, random_state=42)

# classe 0: 1 grupo | classe 1: 2 grupos
y = (c > 0).astype(int)

plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=15, label="Classe 0", alpha=0.7, color="#1f77b4")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=15, label="Classe 1", alpha=0.7, color="#d62728")
plt.title("Dados gerados (make_blobs)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("../figures/fig1_dados.png", dpi=120)

# splitando treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# B - estabelecendo linha de base
baseline = LogisticRegression(random_state=42)
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
print(f"Acurácia do baseline linear: {acc_baseline:.3f}")

# C - implementando um MLP do zero
# MLP genérica sem frameworks
# recebe uma layer_size com o n de neuronios em cada camada, incluindo entrada e saida
# Ex: [2, 8, 1] = 2 entradas, 1 camada oculta de 8 neuronios, 1 saida.
# Ativacao oculta: tanh. Ativacao de saida: 'sigmoid' (binario, Ex2) ou 'softmax' (multiclasse, Ex3)
# Esse ultimo parametro foi definido pra que permita reusar a MESMA classe sem reescrever nada no Ex3.
# Perda: binary cross-entropy (sigmoid) ou entropia cruzada categorica (softmax).

class MLP:

    def __init__(self, layer_sizes, lr=0.1, seed=42, output_activation='sigmoid'):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.output_activation = output_activation
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(rng.normal(0, 1/np.sqrt(n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # passagem direta: calcula a saida da rede e guarda os valores de z e a de cada camada para a passagem reversa
    def forward(self, X):
        self.z = []
        self.a = [X]
        n_layers = len(self.W)
        for i in range(n_layers):
            z = self.a[-1] @ self.W[i] + self.b[i]
            self.z.append(z)
            if i < n_layers - 1:
                a = np.tanh(z)
            elif self.output_activation == 'sigmoid':
                a = 1 / (1 + np.exp(-z))
            else:  # softmax!
                z_shift = z - z.max(axis=1, keepdims=True)  
                exp_z = np.exp(z_shift)
                a = exp_z / exp_z.sum(axis=1, keepdims=True)
            self.a.append(a)
        return self.a[-1]

    # calculando a perda: binary cross-entropy (Ex2) ou entropia cruzada categorica (Ex3)
    def compute_loss(self, y_pred, y_true_enc):
        eps = 1e-9
        if self.output_activation == 'sigmoid':
            return -np.mean(y_true_enc*np.log(y_pred+eps) + (1-y_true_enc)*np.log(1-y_pred+eps))
        else:
            return -np.mean(np.sum(y_true_enc*np.log(y_pred+eps), axis=1))

    # passagem reversa, por gradiente derivado
    def backward(self, y_true_enc):
        # sigmoid+BCE e softmax+CE dao a MESMA formula pro delta inicial: (saida - alvo) / n
        # (mesmo motivo do bias sempre dar derivada 1 no ex1, a algebra se cancela)
        n = y_true_enc.shape[0]
        n_layers = len(self.W)
        grads_W, grads_b = [None]*n_layers, [None]*n_layers
        delta = (self.a[-1] - y_true_enc) / n

        for i in reversed(range(n_layers)):
            grads_W[i] = self.a[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                da = delta @ self.W[i].T
                delta = da * (1 - self.a[i]**2)  # derivada da tanh
        return grads_W, grads_b

    # atualização dos parâmetros
    def update(self, grads_W, grads_b):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_W[i]
            self.b[i] -= self.lr * grads_b[i]

    # treinamento da rede
    def train(self, X, y_true_enc, epochs=2000, verbose_every=200):
        if self.output_activation == 'sigmoid':
            y_true_enc = y_true_enc.reshape(-1, 1)  # (n,) -> (n,1), p/ bater com a saida da sigmoid
        history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            history.append(self.compute_loss(y_pred, y_true_enc))
            grads_W, grads_b = self.backward(y_true_enc)
            self.update(grads_W, grads_b)
            if verbose_every and epoch % verbose_every == 0:
                print(f"epoch {epoch}: loss={history[-1]:.4f}")
        return history

    def predict(self, X):
        out = self.forward(X)
        if self.output_activation == 'sigmoid':
            return (out >= 0.5).astype(int).ravel()
        return out.argmax(axis=1)  # classe com maior probabilidade (pro softmax)

# conferencia e validacao da mlp (Ex2, binaria)
mlp_bin = MLP(layer_sizes=[2, 8, 1], lr=0.1, seed=42)
history_bin = mlp_bin.train(X_train, y_train, epochs=2000, verbose_every=400)

y_pred_test = mlp_bin.predict(X_test)
acc = (y_pred_test == y_test).mean()
print("Acuracia no teste:", acc)

# produzindo figura 2 e 3
plt.figure(figsize=(6,4))
plt.plot(history_bin)
plt.title("Perda de treino por época")
plt.xlabel("época")
plt.ylabel("binary cross-entropy")
plt.tight_layout()
plt.savefig("../figures/fig2_loss.png", dpi=120)

xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
                      np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = mlp_bin.forward(grid)[:, 0].reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#aec7e8', '#ff9896'], alpha=0.6)
plt.scatter(X[y==0,0], X[y==0,1], s=10, color='#1f77b4', label='Classe 0')
plt.scatter(X[y==1,0], X[y==1,1], s=10, color='#d62728', label='Classe 1')
plt.title("Fronteira de decisão da MLP")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig("../figures/fig3_boundary.png", dpi=120)

# Exercicio 3: multiclasse mas a mesma rede
# A - gerando os dados
X, y = make_classification(
    n_samples=1500,
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# B - adaptando e aplicando a rede criada para a multiclasse
# considerando que a camada de saída agora tem 3 unidades + softmax para saida em probabilidade
# + entropia cruzada categórica como perda (one-hot!)
# Sem reescrever classe MLP aqui, so chamada com output_activation='softmax'
def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out

y_train_enc = one_hot(y_train, 3)
mlp_multi = MLP(layer_sizes=[4, 16, 3], lr=0.1, seed=42, output_activation='softmax')

# C - treinando e avaliando a MLP multiclasse
history_multi = mlp_multi.train(X_train, y_train_enc, epochs=2000, verbose_every=400)

y_pred_multi = mlp_multi.predict(X_test)
acc_multi = (y_pred_multi == y_test).mean()
print("Acuracia no teste (multiclasse):", acc_multi)

# plotando Figura 4: perda de treino e Figura 5: matriz de confusão
plt.figure(figsize=(6, 4))
plt.plot(history_multi)
plt.title("Perda de treino por época (multiclasse)")
plt.xlabel("época")
plt.ylabel("categorical cross-entropy")
plt.tight_layout()
plt.savefig("../figures/fig4_loss_multiclass.png", dpi=120)

cm = confusion_matrix(y_test, y_pred_multi)
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap='Blues')
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.xticks([0, 1, 2]); plt.yticks([0, 1, 2])
plt.xlabel("Predito"); plt.ylabel("Real")
plt.title("Matriz de confusão")
plt.colorbar()
plt.tight_layout()
plt.savefig("../figures/fig5_confmat.png", dpi=120)

# D - comparando com regressão logística multiclasse
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
acc_logreg = logreg.score(X_test, y_test)
print("Acuracia da regressao logistica (multiclasse):", acc_logreg)

# Exercicio 4: conferindo se a profundidade de camadas ocultas importa
# A - rede mais profunda, mesmos hiperparametros do Ex3 (epochs=2000, lr=0.1), só mudando a arquitetura
mlp_deep = MLP(layer_sizes=[4, 16, 8, 3], lr=0.1, seed=42, output_activation='softmax')
# 2 camadas ocultas: 16 e 8 neuronios, mesma entrada e saida do Ex3 
history_deep = mlp_deep.train(X_train, y_train_enc, epochs=2000, verbose_every=400)

y_pred_deep = mlp_deep.predict(X_test)
acc_deep = (y_pred_deep == y_test).mean()
print("Acuracia no teste (rede profunda):", acc_deep)

# B - comparacao controlada: rodando raso e profundo em varias seeds
seeds = [42, 7, 123]
resultados_raso, resultados_deep = [], []

for s in seeds:
    m_raso = MLP(layer_sizes=[4, 16, 3], lr=0.1, seed=s, output_activation='softmax')
    h_raso = m_raso.train(X_train, y_train_enc, epochs=2000, verbose_every=0)
    acc_raso = (m_raso.predict(X_test) == y_test).mean()
    resultados_raso.append((h_raso[-1], acc_raso))

    m_deep = MLP(layer_sizes=[4, 16, 8, 3], lr=0.1, seed=s, output_activation='softmax')
    h_deep = m_deep.train(X_train, y_train_enc, epochs=2000, verbose_every=0)
    acc_deep = (m_deep.predict(X_test) == y_test).mean()
    resultados_deep.append((h_deep[-1], acc_deep))

    print(f"seed {s}: raso acc={acc_raso:.3f} loss={h_raso[-1]:.4f} | "
          f"profundo acc={acc_deep:.3f} loss={h_deep[-1]:.4f}")

loss_medio_raso = np.mean([r[0] for r in resultados_raso])
acc_media_raso = np.mean([r[1] for r in resultados_raso])
loss_medio_deep = np.mean([r[0] for r in resultados_deep])
acc_media_deep = np.mean([r[1] for r in resultados_deep])
print(f"\nMedia raso: loss={loss_medio_raso:.4f} acc={acc_media_raso:.3f}")
print(f"Media profundo: loss={loss_medio_deep:.4f} acc={acc_media_deep:.3f}")

# Figura 6: curvas de perda lado a lado (usando a seed 42 como representativa)
plt.figure(figsize=(6, 4))
plt.plot(history_multi, label='Raso [4,16,3] (Ex3)')
plt.plot(history_deep, label='Profundo [4,16,8,3] (Ex4)')
plt.title("Comparação de loss de treino: raso vs. profundo")
plt.xlabel("época")
plt.ylabel("categorical cross-entropy")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/fig6_depth_comparison.png", dpi=120)