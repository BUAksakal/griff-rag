"""
GRIFF — Gradio Web Interface
German Regulatory & Immigration Facts For Foreigners

Main entry point for the application.
Provides two tabs: Q&A and Email Parser.
"""

import gradio as gr
import logging
import os
from dotenv import load_dotenv

from src.retrieval.retriever import get_retriever
from src.generation.generator import generate_answer
from src.email_parser.parser import parse_email

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Language label → code mapping
LANG_MAP = {
    "🇹🇷 Türkçe": "tr",
    "🇬🇧 English": "en",
    "🇩🇪 Deutsch": "de",
    "🇸🇦 العربية": "ar",
    "🇺🇦 Українська": "uk",
    "🇷🇺 Русский": "ru",
    "🇪🇸 Español": "es",
    "🇫🇷 Français": "fr",
    "🇮🇹 Italiano": "it",
    "🇵🇱 Polski": "pl",
}


# ──────────────────────────────────────────────
# Q&A Handler
# ──────────────────────────────────────────────

def handle_question(question: str, language: str) -> str:
    """Process a user question through the RAG pipeline."""
    if not question or not question.strip():
        return "⚠️ Please enter a question."

    lang_code = LANG_MAP.get(language, "en")
    logger.info(f"Question received: '{question[:80]}...' | Language: {lang_code}")

    try:
        # Step 1: Retrieve relevant chunks
        retriever = get_retriever()
        results = retriever.hybrid_search(question, top_k=5)

        if not results:
            return "❌ No relevant documents found. Please try a different question or make sure the index has been built."

        # Step 2: Generate answer
        response = generate_answer(
            query=question,
            context_chunks=results,
            response_language=lang_code,
        )

        # Step 3: Format output
        answer = response["answer"]
        sources = response.get("sources", [])

        output = f"{answer}\n\n"
        if sources:
            output += "---\n\n📎 **Sources:**\n"
            for src in sources:
                output += f"- [{src}]({src})\n"

        return output

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        return f"❌ An error occurred: {str(e)}"


# ──────────────────────────────────────────────
# Email Parser Handler
# ──────────────────────────────────────────────

def handle_email(email_text: str, language: str) -> str:
    """Parse a German official email/letter."""
    if not email_text or not email_text.strip():
        return "⚠️ Please paste an email or letter to parse."

    lang_code = LANG_MAP.get(language, "en")
    logger.info(f"Email parse requested | Language: {lang_code}")

    try:
        result = parse_email(email_text, response_language=lang_code)

        # Format output
        urgency_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴",
        }.get(result.get("urgency", "medium"), "⚪")

        output = f"## 🏛️ {result.get('sender', 'Unknown sender')}\n\n"
        output += f"**Subject:** {result.get('subject', 'N/A')}\n\n"
        output += f"**Urgency:** {urgency_emoji} {result.get('urgency', 'N/A').upper()}\n\n"

        deadline = result.get("deadline")
        if deadline:
            output += f"**⏰ Deadline:** {deadline}\n\n"

        output += "---\n\n"
        output += f"### 📋 Summary\n{result.get('summary', 'N/A')}\n\n"

        actions = result.get("required_actions", [])
        if actions:
            output += "### ✅ Required Actions\n"
            for i, action in enumerate(actions, 1):
                output += f"{i}. {action}\n"

        return output

    except Exception as e:
        logger.error(f"Error parsing email: {e}", exc_info=True)
        return f"❌ An error occurred: {str(e)}"


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.header-text {
    text-align: center;
    margin-bottom: 0.5rem;
}
.header-text h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.header-text p {
    color: #64748b;
    font-size: 1.1rem;
}
footer { display: none !important; }
"""

with gr.Blocks(theme=THEME, css=CSS, title="GRIFF — German Bureaucracy Assistant") as demo:

    # Header
    gr.HTML("""
        <div class="header-text">
            <h1>🦅 GRIFF</h1>
            <p><em>Get a grip on German bureaucracy.</em></p>
            <p style="font-size: 0.85rem; color: #94a3b8;">
                German Regulatory & Immigration Facts For Foreigners
            </p>
        </div>
    """)

    # Language picker (shared across tabs)
    lang_picker = gr.Radio(
        choices=[
            "🇹🇷 Türkçe", "🇬🇧 English", "🇩🇪 Deutsch", "🇸🇦 العربية",
            "🇺🇦 Українська", "🇷🇺 Русский", "🇪🇸 Español",
            "🇫🇷 Français", "🇮🇹 Italiano", "🇵🇱 Polski",
        ],
        value="🇬🇧 English",
        label="🌍 Response Language",
        interactive=True,
    )

    with gr.Tabs():

        # ── Tab 1: Q&A ──────────────────────────
        with gr.Tab("💬 Ask a Question", id="qa"):
            gr.Markdown(
                "Ask anything about German bureaucracy — visa, Anmeldung, taxes, health insurance, and more."
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What documents do I need for Anmeldung in Berlin?",
                    lines=3,
                    scale=4,
                )
                ask_btn = gr.Button("🔍 Ask GRIFF", variant="primary", scale=1)

            answer_output = gr.Markdown(label="Answer")

            ask_btn.click(
                fn=handle_question,
                inputs=[question_input, lang_picker],
                outputs=answer_output,
            )
            question_input.submit(
                fn=handle_question,
                inputs=[question_input, lang_picker],
                outputs=answer_output,
            )

            # Example questions
            gr.Examples(
                examples=[
                    ["What documents do I need for Anmeldung in Berlin?"],
                    ["How do I apply for a Blue Card?"],
                    ["Anmeldung için hangi belgeler lazım?"],
                    ["Wie melde ich mich bei der Krankenkasse an?"],
                    ["What is the process for getting a tax ID (Steuer-ID)?"],
                ],
                inputs=question_input,
                label="💡 Example Questions",
            )

        # ── Tab 2: Email Parser ──────────────────
        with gr.Tab("📧 Email Parser", id="email"):
            gr.Markdown(
                "Paste a German official email or letter below. "
                "GRIFF will tell you what it means, what you need to do, and any deadlines."
            )
            email_input = gr.Textbox(
                label="Email / Letter Content",
                placeholder="Paste your German official email or letter here...",
                lines=10,
            )
            parse_btn = gr.Button("📧 Parse Email", variant="primary")

            email_output = gr.Markdown(label="Parsed Result")

            parse_btn.click(
                fn=handle_email,
                inputs=[email_input, lang_picker],
                outputs=email_output,
            )

            # Example email
            gr.Examples(
                examples=[
                    [
                        "Sehr geehrte/r Frau/Herr Müller,\n\n"
                        "hiermit laden wir Sie zur Vorsprache bei der Ausländerbehörde Berlin ein.\n\n"
                        "Bitte erscheinen Sie am 15.07.2025 um 10:00 Uhr in der\n"
                        "Friedrich-Krause-Ufer 24, 13353 Berlin, Raum 2.014.\n\n"
                        "Bringen Sie bitte folgende Unterlagen mit:\n"
                        "- gültiger Reisepass\n"
                        "- biometrisches Passfoto\n"
                        "- Meldebescheinigung (nicht älter als 6 Monate)\n"
                        "- Arbeitsvertrag oder Immatrikulationsbescheinigung\n"
                        "- Nachweis der Krankenversicherung\n\n"
                        "Sollten Sie den Termin nicht wahrnehmen können, kontaktieren Sie uns bitte\n"
                        "mindestens 3 Werktage vorher unter 030-90269-0.\n\n"
                        "Mit freundlichen Grüßen\n"
                        "Landesamt für Einwanderung Berlin"
                    ],
                ],
                inputs=email_input,
                label="💡 Example Email",
            )

    # Footer info
    gr.Markdown(
        """
        ---
        <center style="color: #94a3b8; font-size: 0.8rem;">
        GRIFF uses RAG (Retrieval-Augmented Generation) with official German government sources.<br>
        Powered by bge-m3 embeddings, hybrid search with reranking, and LLaMA 3.3 70B via Groq.
        </center>
        """,
    )


# ──────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
