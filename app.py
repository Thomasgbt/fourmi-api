from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "✅ API Fourmi en ligne !"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Exemple : on reçoit une image encodée en base64
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"error": "Aucune image reçue"}), 400

        # Ici tu mettras ton vrai code de prédiction plus tard.
        # Pour le moment on fait une simulation :
        return jsonify({
            "prediction": "Camponotus vagus (Gyne)",
            "probability": 0.92
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

