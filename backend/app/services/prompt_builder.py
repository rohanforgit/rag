import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    @staticmethod
    def build_system_prompt(contexts: list[dict]) -> str:
        """
        Constructs the system guidelines and integrates retrieved text chunks.
        """
        context_str = ""
        for i, ctx in enumerate(contexts):
            doc_name = ctx.get("document", "R26_Rules_Regulations.pdf")
            page_num = ctx.get("page", "Unknown")
            sec_title = ctx.get("section", "Unknown Section")
            text = ctx.get("text", "").strip()
            
            context_str += f"\n--- Context Source {i+1} (Doc: {doc_name}, Page: {page_num}, Section: {sec_title}) ---\n"
            context_str += f"{text}\n"

        system_prompt = (
            "You are Sreenidhi University's AI Regulations Assistant.\n"
            "Your sole objective is to answer the user's question accurately using ONLY the provided regulation context below.\n\n"
            "=== STRICT RAG COMPLIANCE RULES ===\n"
            "1. Answer ONLY using information present in the provided regulation context.\n"
            "2. Never use external knowledge, fabricate, or invent university regulations.\n"
            "3. If the answer cannot be found in the provided context, state clearly and politely: "
            "\"I'm sorry, but that information is not available in the official Sreenidhi University R26 regulations.\"\n"
            "4. Keep your answers factual, concise, and grounded in the retrieved sources.\n\n"
            "=== PROVIDED REGULATION CONTEXT ===\n"
            f"{context_str.strip()}\n"
            "===================================\n"
        )
        return system_prompt
