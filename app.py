import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS

from services.user_profile_store import save_profile, get_profile
from services.orchestrator import orchestrate
from services.history_store import get_history
from utils.exceptions import AuthError, InputError, AgentError
from services.user_auth_store import check_credentials, create_user
from config.settings import GROQ_API_KEY, GROQ_MODEL_NAME
from services.api_key_pool import get_next_key
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# -------------------------
# Error Handlers
# -------------------------
@app.errorhandler(AuthError)
def auth_error(e):
    return jsonify({"error": str(e)}), 401


@app.errorhandler(InputError)
def input_error(e):
    return jsonify({"error": str(e)}), 400


#Intent Classification
def _is_health_query(text: str) -> bool:
    """Very simple intent filter: allow only health/wellness related queries."""
    if not text:
        return False
    t = text.lower()

    health_keywords = [
        "symptom", "fever", "cough", "pain", "headache", "cold", "flu",
        "blood pressure", "bp", "sugar", "diabetes", "hypertension",
        "cholesterol", "heart", "breath", "breathing", "asthma",
        "diet", "food", "meal", "nutrition", "calorie",
        "exercise", "workout", "walking", "running", "yoga", "fitness",
        "sleep", "insomnia", "snoring",
        "stress", "anxiety", "depression", "fatigue", "tired",
        "doctor", "medicine", "tablet", "pill",
        "health", "wellness", "weight", "obesity",
    ]

    return any(k in t for k in health_keywords)


# -------------------------
# Login
# -------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "")
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    if not check_credentials(username, password):
        return jsonify({"error": "Invalid username or password"}), 401

    return jsonify({"status": "ok", "user_id": username})


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    ok = create_user(username, password)
    if not ok:
        return jsonify({"error": "Username already exists"}), 409

    return jsonify({"status": "ok", "user_id": username})


# -------------------------
# Health Assist API
# -------------------------
@app.route("/health-assist", methods=["POST"])
def health_assist():
    data = request.get_json()

    if not data or "symptoms" not in data:
        raise InputError("Symptoms required")

    symptoms = data["symptoms"].strip()
    if not _is_health_query(symptoms):
        return jsonify({
            "error": "This is a wellness specialized system. Please ask about symptoms, lifestyle, diet, exercise, or other health-related topics."
        }), 400

    result = asyncio.run(
        orchestrate(
            symptoms,
            data.get("medical_report"),
            data.get("user_id", "guest"),
        )
    )
    return jsonify(result)



# -------------------------
# Recommendation API
# -------------------------
@app.route("/recommendations", methods=["POST"])
def recommendations_only():
    data = request.get_json() or {}

    symptoms = (data.get("symptoms") or "").strip()
    medical_report = data.get("medical_report", "")
    user_id = data.get("user_id", "guest")

    if not symptoms:
        raise InputError("Symptoms required")

    if not _is_health_query(symptoms):
        return jsonify({
            "error": "This is a wellness specialized system. Please ask about symptoms, lifestyle, diet, exercise, or other health-related topics."
        }), 400

    result = asyncio.run(
        orchestrate(
            symptoms,
            medical_report,
            user_id,
        )
    )

    return jsonify(
        {
            "query": symptoms,
            "recommendations": result.get("recommendations", []),
        }
    )



# -------------------------
# Follow-up API (uses Grok)
# -------------------------
@app.route("/follow-up", methods=["POST"])
def follow_up():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    question = data.get("question", "").strip()

    if not user_id or not question:
        return jsonify({"error": "user_id and question are required"}), 400

    history = get_history(user_id) or []
    if not history:
        return jsonify({"error": "No previous wellness session found for this user"}), 400

    last = history[-1]

    context_text = (
        f"Previous wellness plan summary:\n{last.get('synthesized_guidance', '')}\n\n"
        f"Key recommendations:\n" + "\n".join(last.get("recommendations", []))
    )

    llm = ChatOpenAI(
        model=GROQ_MODEL_NAME,          # e.g. "grok-3"
        api_key=get_next_key(),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a cautious wellness assistant answering follow-up questions "
                "about an existing wellness plan. Use the provided summary and "
                "recommendations as context. You may clarify, reorder, or restate "
                "information, but do NOT diagnose, do NOT prescribe medicines, and "
                "always remind the user to follow their doctor's advice."
            )
        ),
        HumanMessage(content=context_text),
        HumanMessage(content=f"User follow-up question: {question}"),
    ]

    result = llm.invoke(messages)
    return jsonify({"answer": result.content})


# -------------------------
# User Profile API
# -------------------------
@app.route("/profile/<user_id>", methods=["GET"])
def get_user_profile(user_id):
    return jsonify({"user_id": user_id, "profile": get_profile(user_id)})


@app.route("/profile/<user_id>", methods=["POST"])
def save_user_profile_route(user_id):
    data = request.get_json() or {}
    profile = {
        "height_cm": data.get("height_cm"),
        "weight_kg": data.get("weight_kg"),
        "medications": data.get("medications", ""),
    }
    save_profile(user_id, profile)
    return jsonify({"user_id": user_id, "profile": profile})


# -------------------------
# History API
# -------------------------
@app.route("/history/<user_id>")
def history(user_id):
    return jsonify({"user_id": user_id, "history": get_history(user_id)})


@app.route("/", methods=["GET"])
def welcome_health():
    return "Welcome to Health & Diet Care"


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=False)
