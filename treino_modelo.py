# treino_modelo.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Carregamento
df = pd.read_csv("C:\Users\aluno\Desktop\Projeto\Base\simulacao_rotina_homeoffice.csv")

# Pré-processamento simples (ex: transformar horários em números)
df["hora"] = pd.to_datetime(df["hora_entrada"]).dt.hour
df["duracao"] = pd.to_numeric(df["duracao"], errors='coerce').fillna(0)

# One-hot encoding do cômodo anterior
df_encoded = pd.get_dummies(df[["comodo_anterior"]])

# Feature final
X = pd.concat([df[["hora", "duracao"]], df_encoded], axis=1)
y = df["comodo"]

# Treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Salvar modelo
joblib.dump(modelo, "modelo.pkl")
print("Modelo treinado e salvo.")
