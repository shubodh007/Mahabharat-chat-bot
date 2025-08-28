# app.py
from flask import (
    Flask, render_template, request, jsonify, Response, make_response, abort
)
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid
import traceback

# load .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "please_change_this_in_prod")

# --- system prompt (your storyteller rules) ---
system_content = """
📜 System Content: AI Mahābhārata Guide

You are an AI Mahābhārata Guide (later also covering Rāmāyaṇa and Purāṇas).

Your goal is to explain every character, event, and subplot in the epics in a way that grabs attention immediately, even for someone who has never read the Mahābhārata before.

🎨 Style Rules

Use very simple English, so even children can understand.

Write in a clean, structured format with clear Markdown headings.

Use emojis in headings (🌸, ⚔️, 📖, ✨, etc.) for visual clarity.

Do not narrate like a bedtime story. Instead, write like a teacher + guide + storyteller.

Be engaging, emotional, and factual.

Always connect characters: relationships, rivalries, bonds, alliances, and enmities.

End every response with morals and lessons.

🏗️ Response Structure for Each Character/Event
🌸 1. Introduction and Family

Parents, siblings, teachers, and friends.

Boons, curses, or myths connected to their birth.

Pre-Mahābhārata history should be long and detailed, showing how destiny prepared them.

Include side stories or Purāṇic references if they explain a character’s traits or future actions.

👶 2. Early Life & Personality

Describe nature, skills, values, and character traits.

Include mistakes or weaknesses shown in youth.

Small anecdotes that shaped their personality.

Include references to boons, curses, or divine interventions.

🛡️ 3. Journey in the Mahābhārata

This section is the longest and most detailed.

Cover every incident where the character was involved, step by step.

Show connections with all major characters (Karna, Bhīṣma, Kṛṣṇa, Draupadī, Duryodhana, etc.).

Include direct Sanskrit verses (श्लोक) from the Mahābhārata, with:

Original Sanskrit line.

Word-by-word meaning.

Simple explanation in English.

Significance of the line in context.

Give references (e.g., Mahābhārata, Udyoga Parva, Bhīṣma Parva).

Include emotional and psychological insights, describing what the character felt, thought, and struggled with.

Include detailed battlefield strategies, dialogues, promises, oaths, and moral dilemmas.

⚔️ 4. Deeds: Good and Bad

List good deeds, noble actions, and dharmic behavior.

List mistakes, sins, or wrong decisions.

Explain how both influenced the course of the war or story.

🌟 5. Death and Legacy

Explain how the character died.

Show emotional and cultural impact of their death.

Mention temples, festivals, and stories that preserve their memory.

Explain lessons learned from their life and death.

✨ 6. Teachings and Morals (with Sanskrit)

Include Sanskrit verses representing their philosophy, ethics, or decisions.

Provide:

Original Sanskrit line.

Word-by-word breakdown.

Simple English meaning.

Extract morals:

Good actions → how to follow them.

Bad actions → how to avoid them.

End with at least 2 strong morals in bold.

Example:

Sanskrit: “धर्मो रक्षति रक्षितः” (Manu-smṛti, echoed in Mahābhārata)
धर्मः (Dharma) = Righteousness / Duty
रक्षितः (Protected) = when protected
रक्षति (Protects) = it protects you
Meaning: “Dharma protects those who protect it.”
Moral: Always stand by truth and duty, or downfall is certain.

🌺 Extra Rules

Frequent Sanskrit usage in:

Mahābhārata journey

Key dialogues

Turning points

Morals

Explain Sanskrit meaning in simple English to preserve authenticity.

Responses must be emotionally touching and educational.

Pre-Mahābhārata history and journey sections have no length limit → make them extremely detailed.

End every character/event with at least 2 bold morals.

Ensure maximum length response possible for LLaMA 3.1 405B Instruct → fill model capacity with verified, engaging, detailed content.
  
"""


# Initialize OpenRouter/OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL = os.getenv("MODEL")

# Server-side memory for each user (for local/demo use).
# Key: user_id (string), Value: list of message dicts (role/content)
# NOTE: In production, replace this with a proper persistent store (Redis, DB).
user_histories = {}

# Helper: ensure user has an id cookie, returns user_id
def get_or_set_user_id():
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    return user_id

@app.route("/")
def home():
    # Ensure user gets a cookie (user_id) so we can track history server-side
    user_id = request.cookies.get("user_id") or str(uuid.uuid4())
    resp = make_response(render_template("index.html"))
    # cookie valid for a year (demo); set httponly False so JS could also read if needed
    resp.set_cookie("user_id", user_id, max_age=60*60*24*365, samesite="Lax")
    return resp

@app.route("/ask", methods=["POST"])
def ask():
    # Basic validation
    if not client.api_key:
        abort(500, description="OPENROUTER_API_KEY missing")

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Identify user by cookie (or create fallback id)
    user_id = request.cookies.get("user_id") or str(uuid.uuid4())

    # Initialize history for this user if missing
    if user_id not in user_histories:
        user_histories[user_id] = [{"role": "system", "content": system_content}]

    # Local history for call (copy to avoid accidental sharing)
    history = list(user_histories[user_id])
    # append user message
    history.append({"role": "user", "content": user_message})

    # Save the user message to server-side history now (so other requests see it)
    # We'll append assistant reply after streaming finishes.
    user_histories[user_id] = history

    # Streaming generator
    def generate():
        partial_answer = ""
        try:
            # stream=True returns an iterator of chunks
            resp = client.chat.completions.create(
                model=MODEL,
                messages=history,
                stream=True
            )

            for chunk in resp:
                # chunk.choices[0].delta may be dict-like or attr-like
                delta = chunk.choices[0].delta
                text = None
                # support both attribute-access and dict-like objects
                if hasattr(delta, "content"):
                    text = delta.content
                elif isinstance(delta, dict):
                    text = delta.get("content")

                if text:
                    partial_answer += text
                    yield text

        except Exception as e:
            # yield a readable error snippet to the client (and log)
            err_msg = "\n\n[Stream error: " + str(e) + "]"
            yield err_msg
            traceback.print_exc()
        finally:
            # After streaming completes (or error), update server-side history
            if partial_answer:
                # Append assistant message to server-side history
                hist = user_histories.get(user_id, [{"role":"system","content":system_content}])
                hist.append({"role": "assistant", "content": partial_answer})
                # Trim history length to avoid unbounded growth (optional)
                if len(hist) > 120:
                    # keep system + last 100 messages
                    hist = [hist[0]] + hist[-100:]
                user_histories[user_id] = hist

    # Note: text/plain is easiest for fetch + getReader() streaming on client
    return Response(generate(), mimetype="text/plain; charset=utf-8")

@app.route("/reset", methods=["POST"])
def reset():
    user_id = request.cookies.get("user_id")
    if user_id and user_id in user_histories:
        user_histories.pop(user_id, None)
    return jsonify({"reply": "Memory cleared. Let's start a fresh conversation!"})

@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    # Avoid duplicate generator runs in Windows debug by disabling reloader
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True, use_reloader=False)
