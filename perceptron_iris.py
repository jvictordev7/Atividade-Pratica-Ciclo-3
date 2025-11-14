import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000, random_state=42):
        """
        learning_rate: taxa de aprendizado (eta)
        n_epochs: número máximo de épocas (passadas completas pelos dados)
        random_state: semente para gerar pesos aleatórios
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors_ = []

    def net_input(self, X):
        """
        Calcula a soma ponderada: w1*x1 + w2*x2 + ... + bias
        X pode ser um vetor (uma amostra) ou uma matriz (várias amostras).
        """
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        """
        Função de ativação degrau.
        Se o valor for >= 0 retorna 1, caso contrário retorna 0.
        """
        return np.where(X >= 0.0, 1, 0)

    def fit(self, X, y):
        """
        Treina o Perceptron usando o algoritmo de atualização de pesos.
        X: matriz (n_amostras, n_features)
        y: vetor de rótulos (0 ou 1)
        """
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]

        # Inicializa pesos com valores pequenos aleatórios e bias em 0
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = 0.0
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0

            # percorre amostra por amostra
            for xi, target in zip(X, y):
                # previsão atual
                y_hat = self.activation(self.net_input(xi))
                # erro = y_verdadeiro - y_previsto
                update = self.learning_rate * (target - y_hat)

                if update != 0.0:
                    # atualiza pesos e bias
                    self.weights += update * xi
                    self.bias += update
                    errors += 1

            self.errors_.append(errors)

            # se não houve erro nessa época, já convergiu
            if errors == 0:
                break

        return self

    def predict(self, X):
        """
        Faz previsão para novos dados.
        Retorna 0 ou 1.
        """
        return self.activation(self.net_input(X))


def main():
    # 1. Carregar dataset Iris do scikit-learn
    iris = datasets.load_iris()
    X = iris.data          # 4 features
    y = iris.target        # 0 = setosa, 1 = versicolor, 2 = virginica
    target_names = iris.target_names

    # 2. Transformar o problema em binário (Perceptron é binário)
    # Vamos usar apenas Setosa (0) e Versicolor (1)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    # 3. Selecionar apenas 2 features para poder visualizar em 2D
    # Índices: 0 sepal length, 1 sepal width, 2 petal length, 3 petal width
    feature_indices = [2, 3]  # comprimento e largura da pétala
    X = X[:, feature_indices]

    # 4. Dividir em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y  # mantém proporção das classes
    )

    # 5. Instanciar e treinar o Perceptron
    perceptron = Perceptron(
        learning_rate=0.01,
        n_epochs=1000,
        random_state=1
    )
    perceptron.fit(X_train, y_train)

    # 6. Avaliar no conjunto de teste
    y_pred = perceptron.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Classes usadas:", target_names[0], "vs", target_names[1])
    print(f"Pesos finais: {perceptron.weights}")
    print(f"Bias final: {perceptron.bias:.4f}")
    print(f"Acurácia no conjunto de teste: {acc * 100:.2f}%")

    # 7. Plotar a fronteira de decisão + pontos (treino e teste)

    # limites do gráfico
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # grade de pontos pra desenhar a fronteira
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    # regiões de cada classe
    plt.contourf(xx, yy, Z,
                 alpha=0.3,
                 levels=[-0.5, 0.5, 1.5])

    # pontos de treino e teste
    markers = ["o", "s"]
    colors = ["blue", "red"]

    for idx, cls in enumerate(np.unique(y)):
        # pontos de treino
        plt.scatter(
            X_train[y_train == cls, 0],
            X_train[y_train == cls, 1],
            c=colors[idx],
            marker=markers[idx],
            label=f"Treino {target_names[cls]}",
            edgecolor="k"
        )

        # pontos de teste
        plt.scatter(
            X_test[y_test == cls, 0],
            X_test[y_test == cls, 1],
            c=colors[idx],
            marker=markers[idx],
            label=f"Teste {target_names[cls]}",
            edgecolor="k",
            alpha=0.6
        )

    plt.xlabel("Comprimento da pétala (cm)")
    plt.ylabel("Largura da pétala (cm)")
    plt.title("Perceptron - Classificação Iris (Setosa vs Versicolor)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
