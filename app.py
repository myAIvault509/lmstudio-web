import os
from datetime import datetime

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS


# ---------------------------------------------------------
# Flask setup
# ---------------------------------------------------------

app = Flask(__name__)

# Allow your live website and local VS Code Live Server.
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://systemsmyai.com",
                "https://www.systemsmyai.com",
                "https://myaivault509.github.io",
                "http://127.0.0.1:5500",
                "http://localhost:5500",
            ]
        }
    },
)


# ---------------------------------------------------------
# LM Studio configuration
# ---------------------------------------------------------

LM_STUDIO_BASE_URL = "http://127.0.0.1:1234"
LM_STUDIO_CHAT_URL = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"
LM_STUDIO_MODELS_URL = f"{LM_STUDIO_BASE_URL}/v1/models"

# This reads the token from Windows or from a .env setup later.
# If LM Studio does not require a token, this may remain empty.
LM_STUDIO_API_TOKEN = os.getenv("LM_STUDIO_API_TOKEN", "").strip()


def get_lm_studio_headers():
    """Build the request headers used when communicating with LM Studio."""

    headers = {
        "Content-Type": "application/json",
    }

    if LM_STUDIO_API_TOKEN:
        headers["Authorization"] = f"Bearer {LM_STUDIO_API_TOKEN}"

    return headers


def get_model_name():
    """Return the first model currently available in LM Studio."""

    response = requests.get(
        LM_STUDIO_MODELS_URL,
        headers=get_lm_studio_headers(),
        timeout=10,
    )

    response.raise_for_status()

    model_data = response.json()
    models = model_data.get("data", [])

    if not models:
        raise RuntimeError(
            "No model is loaded in LM Studio. "
            "Load a model and start the LM Studio server."
        )

    return models[0]["id"]


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Show basic information when the API address is opened."""

    return jsonify(
        {
            "status": "online",
            "message": "Welcome to the myAI Vault API.",
            "endpoints": {
                "home": "/",
                "health": "/health",
                "chat": "/chat",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Test whether Flask and LM Studio are available."""

    lm_studio_status = "offline"
    model_name = None

    try:
        model_name = get_model_name()
        lm_studio_status = "online"
    except Exception:
        pass

    return jsonify(
        {
            "status": "online",
            "message": "myAI Vault Flask server is running.",
            "lm_studio": lm_studio_status,
            "model": model_name,
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    """Receive a question from the webpage and send it to LM Studio."""

    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify(
                {
                    "error": "No JSON data was received.",
                }
            ), 400

        user_message = str(data.get("message", "")).strip()

        if not user_message:
            return jsonify(
                {
                    "error": "Please enter a question.",
                }
            ), 400

        model_name = get_model_name()

        lm_response = requests.post(
            LM_STUDIO_CHAT_URL,
            headers=get_lm_studio_headers(),
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the private AI assistant for myAI Vault. "
                            "Answer clearly, accurately, helpfully, and "
                            "professionally."
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

        lm_response.raise_for_status()

        result = lm_response.json()

        choices = result.get("choices", [])

        if not choices:
            return jsonify(
                {
                    "error": "LM Studio did not return an answer.",
                }
            ), 502

        ai_reply = (
            choices[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not ai_reply:
            return jsonify(
                {
                    "error": "LM Studio returned an empty answer.",
                }
            ), 502

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
                    "Could not connect to LM Studio. Open LM Studio, "
                    "load a model, and start the local server."
                )
            }
        ), 503

    except requests.exceptions.Timeout:
        return jsonify(
            {
                "error": (
                    "The AI took too long to respond. Try again or "
                    "use a smaller model."
                )
            }
        ), 504

    except requests.exceptions.HTTPError as error:
        status_code = 502
        details = ""

        if error.response is not None:
            details = error.response.text

        return jsonify(
            {
                "error": "LM Studio returned an HTTP error.",
                "details": details,
            }
        ), status_code

    except RuntimeError as error:
        return jsonify(
            {
                "error": str(error),
            }
        ), 503

    except (KeyError, IndexError, TypeError, ValueError):
        return jsonify(
            {
                "error": "LM Studio returned an unexpected response.",
            }
        ), 502

    except Exception as error:
        print(f"Unexpected server error: {error}")

        return jsonify(
            {
                "error": "An unexpected server error occurred.",
            }
        ), 500


# ---------------------------------------------------------
# Start the server
# ---------------------------------------------------------

if __name__ == "__main__":
    print()
    print("myAI Vault server is starting.")
    print("Home:   http://127.0.0.1:5000/")
    print("Health: http://127.0.0.1:5000/health")
    print("Chat:   http://127.0.0.1:5000/chat")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )