import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / "data"
rng = np.random.default_rng(42) # mantendo padrão

df = pd.read_csv(DATA_DIR / "train.csv")

# Parte A: conhecendo os dados

#print(df.shape)
#print(df.head())

#print(df["Transported"].value_counts())
#print(df["Transported"].value_counts(normalize=True))

#print(df.dtypes)
#PassengerId         str
#HomePlanet          str
#CryoSleep        object
#Cabin               str
#Destination         str
#Age             float64
#VIP              object
#RoomService     float64
#FoodCourt       float64
#ShoppingMall    float64
#Spa             float64
#VRDeck          float64
#Name                str
#Transported        bool

faltantes = df.isna().sum()
faltantes_pct = (df.isna().mean() * 100).round(2)

tabela_faltantes = pd.DataFrame({
    "faltantes": faltantes,
    "faltantes_%": faltantes_pct
}).sort_values("faltantes", ascending=False)

#print(tabela_faltantes)
# Para os valores númericos
colunas_gasto = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

estatisticas_gasto = df[colunas_gasto].agg(["mean", "median", "max"]).round(2)
#print(estatisticas_gasto)
# Valores de média muito maiores que mediana indicam assimetria

# Parte B: separando antes de transformar para não haver vazamento de dados

X = df.drop(columns=["Transported"])
y = df["Transported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


#print(X_train.shape, X_test.shape)
#print(y_train.value_counts(normalize=True))
#print(y_test.value_counts(normalize=True))

# Parte C: pre-processamento

# C1
# Imputando dados faltantes
# Numericos a partir da mediana (média muito baixa vai distorcer), categóricos a partir da moda (valor mais frequente)

# Cabin, name e passengerId não serão utilizados, então não precisam ser imputados
colunas_numericas = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
colunas_categoricas = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

imputer_numerico = SimpleImputer(strategy="median")
imputer_categorico = SimpleImputer(strategy="most_frequent")

X_train[colunas_numericas] = imputer_numerico.fit_transform(X_train[colunas_numericas])
X_test[colunas_numericas] = imputer_numerico.transform(X_test[colunas_numericas])

X_train[colunas_categoricas] = imputer_categorico.fit_transform(X_train[colunas_categoricas])
X_test[colunas_categoricas] = imputer_categorico.transform(X_test[colunas_categoricas])

#print(X_train[colunas_numericas + colunas_categoricas].isna().sum())
#print(X_test[colunas_numericas + colunas_categoricas].isna().sum())
# removido os valores faltantes, agora não há mais valores nulos

#C2: transformando variáveis categóricas em 1/0 pra rede neural
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

train_codificado = encoder.fit_transform(X_train[colunas_categoricas])
test_codificado = encoder.transform(X_test[colunas_categoricas])

nomes_colunas_codificadas = encoder.get_feature_names_out(colunas_categoricas)

train_codificado_df = pd.DataFrame(train_codificado, columns=nomes_colunas_codificadas, index=X_train.index)
test_codificado_df = pd.DataFrame(test_codificado, columns=nomes_colunas_codificadas, index=X_test.index)

X_train = pd.concat([X_train.drop(columns=colunas_categoricas), train_codificado_df], axis=1)
X_test = pd.concat([X_test.drop(columns=colunas_categoricas), test_codificado_df], axis=1)

#print(X_train.shape, X_test.shape)
#print(X_train.columns.tolist())

#C3: criando a feature TotalSpend
colunas_gasto = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

X_train["TotalSpend"] = X_train[colunas_gasto].sum(axis=1)
X_test["TotalSpend"] = X_test[colunas_gasto].sum(axis=1)

X_train = X_train.drop(columns=["Cabin", "Name", "PassengerId"])
X_test = X_test.drop(columns=["Cabin", "Name", "PassengerId"])

#print(X_train.shape, X_test.shape)
#print(X_train.columns.tolist())

#C4: histograma e gastos
foodcourt_antes = X_train["FoodCourt"].copy()

for col in colunas_gasto:
    X_train[col] = np.log1p(X_train[col])
    X_test[col] = np.log1p(X_test[col])

foodcourt_depois = X_train["FoodCourt"]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(foodcourt_antes, bins=30)
axs[0].set_xlabel("FoodCourt")
axs[0].set_ylabel("Frequência")
axs[0].set_title("Antes do log(1+x)")

axs[1].hist(foodcourt_depois, bins=30)
axs[1].set_xlabel("log(1 + FoodCourt)")
axs[1].set_ylabel("Frequência")
axs[1].set_title("Depois do log(1+x)")

fig.suptitle("Figura 6: FoodCourt antes e depois do log(1+x)")
plt.savefig(FIGURES_DIR / "fig6_log_transform.png", dpi=150, bbox_inches="tight")
#plt.show()
# os gastos são muito assimétricos, e a transformação logarítmica ajuda a reduzir essa assimetria, tornando possível analisar os outros dados

#C5: Escalonamento
#tanh produz -1 e 1, vou usar normalização pra escala das entradas ser compatível com a saída da função de ativação
colunas_numericas_finais = colunas_numericas + ["TotalSpend"]

scaler = MinMaxScaler(feature_range=(-1, 1))

X_train[colunas_numericas_finais] = scaler.fit_transform(X_train[colunas_numericas_finais])
X_test[colunas_numericas_finais] = scaler.transform(X_test[colunas_numericas_finais])

#print("Treino - min:", X_train[colunas_numericas_finais].min().min(), "max:", X_train[colunas_numericas_finais].max().max())
#print("Teste  - min:", X_test[colunas_numericas_finais].min().min(), "max:", X_test[colunas_numericas_finais].max().max())

# Checagens finais
print("NaN no treino:", X_train.isna().sum().sum())
print("NaN no teste:", X_test.isna().sum().sum())
print("Shape final (treino):", X_train.shape)
print("Shape final (teste):", X_test.shape)
print("Range treino:", X_train.min().min(), "até", X_train.max().max())
print("Range teste:", X_test.min().min(), "até", X_test.max().max())