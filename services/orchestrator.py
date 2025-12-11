import json
import re

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from healthbackend.services.api_key_pool import get_next_key, mark_key_quota_exceeded
from healthbackend.services.agents import (
    symptom_agent,
    lifestyle_agent,
    diet_agent,
    fitness_agent,
)
from healthbackend.services.history_store import save_history
from healthbackend.services.memory import get_shared_memory, reset_memory
from healthbackend.config.settings import GROQ_MODEL_NAME


def _make_synth_llm_with_key():
    """Create the synthesizer LLM using the next available Groq API key."""
    key = get_next_key()
    llm = ChatOpenAI(
        model=GROQ_MODEL_NAME,          # e.g. "llama-3.3-70b-versatile" on Groq
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,                # deterministic JSON
    )
    return llm, key


def _build_markdown_table(output: dict) -> str:
    """Build a markdown-style block summarizing each agent."""
    parts = []

    def add_block(title: str, text: str | None):
        if not text:
            return
        # ensure there's a blank line before each block (except first)
        if parts:
            parts.append("")
        parts.append(f"**{title}**")
        parts.append(text.strip())

    add_block("Symptom agent", output.get("symptom_analysis", ""))
    add_block("Lifestyle agent", output.get("lifestyle", ""))
    add_block("Diet agent", output.get("diet", ""))
    add_block("Fitness agent", output.get("fitness", ""))

    return "\n".join(parts)




# ------------------------------------------------------------
# Main orchestration function
# ------------------------------------------------------------
async def orchestrate(symptoms: str, medical_report: str, user_id: str):
    # Reset shared memory for this session
    reset_memory()
    memory = get_shared_memory()

    # 1. Symptom agent
    symptom_result = await symptom_agent(symptoms)

    # 2. Lifestyle agent
    lifestyle_result = await lifestyle_agent(symptoms)

    # 3. Diet agent
    diet_result = await diet_agent(
        symptoms=symptoms,
        report=medical_report,
        lifestyle_notes=lifestyle_result,
    )

    # 4. Fitness agent
    fitness_result = await fitness_agent(
        symptoms=symptoms,
        diet_notes=diet_result,
    )

    # Full conversation history for synthesis
    history = memory.load_memory_variables({})["chat_history"]

    # ------------------------------------------------------------
    # Synthesizer LLM Setup
    # ------------------------------------------------------------
    synth_llm, synth_key = _make_synth_llm_with_key()

    synth_messages = [
        SystemMessage(
            content=(
                "You are an orchestrator summarizing a mild to moderate health concern.\n"
                "Read the full conversation between symptom_agent, lifestyle_agent, "
                "diet_agent, and fitness_agent.\n\n"
                "Write a concise, well-structured wellness plan in markdown with these sections:\n"
                "1. Overview – 2-3 sentences summarizing the situation and overall goal.\n"
                "2. When to See a Doctor – 2-4 bullet points, clearly describing red-flag symptoms.\n"
                "3. Lifestyle & Rest – 3-5 bullet points with specific, gentle daily actions.\n"
                "4. Hydration & Diet – 3-5 bullet points with simple, safe food and fluid guidance.\n"
                "5. Hygiene & Environment – 2-4 bullet points to reduce irritation and infection spread.\n"
                "6. Movement & Activity – 2-4 bullet points with ONLY low-intensity options, "
                "including a bold STOP warning for chest pain, breathing difficulty, dizziness, "
                "or marked worsening.\n"
                "7. Final Note – 1-2 sentences reminding that this is not a diagnosis and to "
                "follow a doctor's advice.\n\n"
                "Tone: calm, reassuring, non-alarming, strictly non-diagnostic. "
                "Never name specific medicines or doses. Never say you replace a doctor.\n\n"
                "Return ONLY valid JSON with keys:\n"
                "  - synthesized_guidance: the markdown text described above\n"
                "  - recommendations: array of short, plain-language recommendation strings\n"
                "Do not wrap JSON in code fences or add any extra text."
            )
        ),
        *history,
        HumanMessage(content="Generate the JSON response now."),
    ]

    try:
        final_answer = await synth_llm.ainvoke(synth_messages)
    except Exception:
        # Key likely hit quota or hard failure → rotate and retry once
        mark_key_quota_exceeded(synth_key)
        synth_llm, synth_key = _make_synth_llm_with_key()
        final_answer = await synth_llm.ainvoke(synth_messages)

    raw = final_answer.content.strip()

    # ------------------------------------------------------------
    # JSON Cleaning and Parsing
    # ------------------------------------------------------------
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"synthesized_guidance": raw, "recommendations": []}

    # ------------------------------------------------------------
    # Final Output Structure
    # ------------------------------------------------------------
    output = {
        "user_id": user_id,
        "query": symptoms,
        "symptom_analysis": symptom_result,
        "lifestyle": lifestyle_result,
        "diet": diet_result,
        "fitness": fitness_result,
        "synthesized_guidance": data.get("synthesized_guidance", ""),
        "recommendations": data.get("recommendations", []),
    }

    # Add markdown table summary
    output["table_markdown"] = _build_markdown_table(output)

    save_history(user_id, output)
    return output
