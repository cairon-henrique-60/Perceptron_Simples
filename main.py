import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


# Solicite ao usuário o RU desejado
ru_value = input("Digite seu RU: ")
ru_digits = [int(digit) for digit in str(ru_value)]

# Gere 50 amostras de treinamento
num_samples = 50
ru_samples = np.random.choice(ru_digits, size=(num_samples, len(ru_digits)))

# Defina as saídas com base no RU inserido
y_samples = np.where(np.sum(ru_samples, axis=1) > np.sum(ru_digits), -1, 1)

# Defina a arquitetura da rede neural
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(8, activation="relu", input_shape=(len(ru_digits),)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile o modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treine o modelo
model.fit(ru_samples, y_samples, epochs=100, verbose=0)

# Avalie o modelo
accuracy = model.evaluate(ru_samples, y_samples, verbose=0)[1]

# Crie um DataFrame para a segunda aba (amostras de treinamento)
data_aba2 = {"Dígito {}".format(i + 1): ru_samples[:, i] for i in range(len(ru_digits))}
data_aba2["Saída Original"] = y_samples.flatten()

# Crie um DataFrame para a primeira aba (com o RU inserido)
data_aba1 = {"Digito " + str(i + 1): ru_digits[i] for i in range(len(ru_digits))}
data_aba1.update(
    {
        "Entrada": [ru_value],
        "Acurácia": [accuracy],
    }
)

counts = pd.Series(y_samples.flatten()).value_counts()

# Cria um gráfico com as ocorrências
plt.bar(counts.index, counts.values, color=["blue", "red"])
plt.xlabel("Saída")
plt.ylabel("Contagem")
plt.title("Contagem de Saídas")

# Salve o gráfico em um arquivo de imagem
plt.savefig(f"{ru_value}_output_chart.png")

# Salve os DataFrames em um arquivo Excel
with pd.ExcelWriter(f"{ru_value}_neural_network.xlsx", engine="xlsxwriter") as writer:
    df_aba2 = pd.DataFrame(data_aba2)
    df_aba2.to_excel(writer, sheet_name="Amostras", index=False)

    df_aba1 = pd.DataFrame(data_aba1, index=[0])
    df_aba1.to_excel(writer, sheet_name="Rede Neural", index=False)

    # Adicione uma nova planilha para o gráfico
    worksheet = writer.sheets["Rede Neural"]
    chart_location = "F2"  # A localização onde o gráfico será inserido
    chart_name = "Contagem_Saidas"

    # Adicione o gráfico à planilha
    worksheet.insert_image(
        chart_location,
        f"{ru_value}_output_chart.png",
        {"x_offset": 5, "y_offset": 5, "x_scale": 0.5, "y_scale": 0.5},
    )
