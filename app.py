from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
LM_STUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"


def get_model_name():
    """Get the first model currently available in LM Studio."""
    response = requests.get(LM_STUDIO_MODELS_URL, timeout=10)
    response.raise_for_status()

    model_data = response.json()
    models = model_data.get("data", [])

    if not models:
        raise RuntimeError(
            "No model is loaded in LM Studio. Load a model and start the server."
        )

    return models[0]["id"]


@app.route("/health", methods=["GET"])
def health():
    """Test whether the Flask server is running."""
    return jsonify(
        {
            "status": "online",
            "message": "myAI Vault server is running."
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    """Receive a question from the webpage and send it to LM Studio."""

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No data was received."}), 400

        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Please enter a question."}), 400

        model_name = get_model_name()

        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the private AI assistant for myAI Vault. "
                            "Answer clearly, accurately, and professionally."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_message,
                    },
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False,
            },
            timeout=180,
        )

        response.raise_for_status()

        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]

        return jsonify(
            {
                "reply": ai_reply,
                "model": model_name,
            }
        )

    except requests.exceptions.ConnectionError:
        return jsonify(
            {
                "error": (
                    "Could not connect to LM Studio. "
                    "Open LM Studio, load a model, and start the local server."
                )
            }
        ), 503

    except requests.exceptions.Timeout:
        return jsonify(
            {
                "error": (
                    "The model took too long to respond. "
                    "Try again or use a smaller model."
                )
            }
        ), 504

    except requests.exceptions.HTTPError as error:
        details = ""

        if error.response is not None:
            details = error.response.text

        return jsonify(
            {
                "error": "LM Studio returned an error.",
                "details": details,
            }
        ), 502

    except (KeyError, IndexError, TypeError):
        return jsonify(
            {
                "error": "LM Studio returned an unexpected response."
            }
        ), 502

    except Exception as error:
        return jsonify(
            {
                "error": str(error)
            }
        ), 500


if __name__ == "__main__":
    print()
    print("myAI Vault server is starting.")
    print("Health test: http://127.0.0.1:5000/health")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )