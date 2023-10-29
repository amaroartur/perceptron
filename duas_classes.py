from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Carregando o conjunto de dados Iris
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets

# Convertendo as classes alvo para rótulos numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Definindo as classes alvo que serão utilizadas (Iris-setosa e Iris-versicolor)
target_classes = ["Iris-setosa", "Iris-versicolor"]

# Definindo diferentes valores de taxa de aprendizado (η) e total de iterações
learning_rates = [0.001, 0.01, 0.1]  # Diferentes valores de taxa de aprendizado
n_iterations = [100, 1000, 10000]  # Diferentes valores para o total de iterações

# Diferentes proporções de conjuntos de treinamento (10%, 30%, 50%)
train_sizes = [0.1, 0.3, 0.5]

for train_size in train_sizes:
	for eta in learning_rates:
		for iterations in n_iterations:
			# Filtrando os dados para as classes alvo selecionadas (Iris-setosa e Iris-versicolor)
			X_filtered = X[np.isin(y, [label_encoder.transform([target_classes[0]])[0], label_encoder.transform([target_classes[1]])[0]])]
			y_filtered = y[np.isin(y, [label_encoder.transform([target_classes[0]])[0], label_encoder.transform([target_classes[1]])[0]])]

			# Dividindo o conjunto de dados em treinamento e teste com a proporção desejada
			X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=1 - train_size, random_state=42)

			# Criando o modelo Perceptron com os valores de taxa de aprendizado e iterações
			perceptron = Perceptron(eta0=eta, max_iter=iterations)
			perceptron.fit(X_train, y_train)

			# Realizando previsões no conjunto de teste
			predictions = perceptron.predict(X_test)

			# Calculando e imprimindo a precisão do modelo
			accuracy = accuracy_score(y_test, predictions)
			print("Proporção de Treinamento = {:.0%} | Taxa de aprendizado η = {} | Total de iterações = {} | Precisão {:.2f}%".format(train_size, eta, iterations, accuracy * 100))
