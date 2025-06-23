# servidor_inferencia.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
modelo = joblib.load("modelo.pkl")

# Exemplo de estados possíveis
ultimos_acessos = {
    "comodo_anterior": "quarto",
    "duracao": 1800,
    "hora": 14
}

@app.route('/prever_comodo', methods=['POST'])
def prever_comodo():
    global ultimos_acessos
    dados = request.get_json()

    # Atualiza histórico
    ultimos_acessos["comodo_anterior"] = dados["comodo_anterior"]
    ultimos_acessos["duracao"] = dados["duracao"]
    ultimos_acessos["hora"] = dados["hora"]

    # One-hot
    comodos_possiveis = ['banheiro', 'cozinha', 'sala', 'quarto']
    for comodo in comodos_possiveis:
        ultimos_acessos[f"comodo_anterior_{comodo}"] = 1 if comodo == ultimos_acessos["comodo_anterior"] else 0

    df_pred = pd.DataFrame([{
        "hora": ultimos_acessos["hora"],
        "duracao": ultimos_acessos["duracao"],
        **{k: ultimos_acessos[k] for k in ultimos_acessos if k.startswith("comodo_anterior_")}
    }])

    pred = modelo.predict(df_pred)[0]

    return jsonify({"comodo_previsto": pred})

if __name__ == '__main__':
    app.run(port=5001)
