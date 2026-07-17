import logging

logger = logging.getLogger(__name__)

class PromptBuilder:
    @staticmethod
    def merge_contiguous_chunks(chunks: list[dict]) -> list[dict]:
        """
        Groups chunks by (document, page, section) and merges contiguous chunks
        based on chunk index suffix. Resolves text overlap.
        """
        import re
        if not chunks:
            return []
            
        # Group chunks by (document, page, section)
        groups = {}
        for chunk in chunks:
            key = (chunk.get("document"), chunk.get("page"), chunk.get("section"))
            if key not in groups:
                groups[key] = []
            groups[key].append(chunk)
            
        merged_chunks = []
        
        def merge_strings(s1: str, s2: str) -> str:
            s1 = s1.strip()
            s2 = s2.strip()
            min_match_len = 10
            max_match_len = min(len(s1), len(s2))
            
            for l in range(max_match_len, min_match_len - 1, -1):
                if s1.endswith(s2[:l]):
                    return s1 + " " + s2[l:]
            return s1 + " " + s2

        for key, group_chunks in groups.items():
            doc, page, section = key
            
            # Sort group chunks by their index suffix
            def get_chunk_idx(c):
                cid = c.get("chunk_id", "")
                m = re.search(r'_(\d+)$', cid)
                return int(m.group(1)) if m else 0
                
            sorted_group = sorted(group_chunks, key=get_chunk_idx)
            
            # Merge contiguous chunks
            current_chunk = None
            for chunk in sorted_group:
                if current_chunk is None:
                    current_chunk = dict(chunk)
                else:
                    curr_idx = get_chunk_idx(current_chunk)
                    next_idx = get_chunk_idx(chunk)
                    
                    # If index is sequential, merge
                    if next_idx == curr_idx + 1:
                        current_chunk["text"] = merge_strings(current_chunk["text"], chunk["text"])
                        current_chunk["chunk_id"] = chunk["chunk_id"]
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = dict(chunk)
                        
            if current_chunk:
                merged_chunks.append(current_chunk)
                
        # Return merged chunks sorted back by document, page, index
        def get_sort_key(c):
            doc = c.get("document", "")
            page = c.get("page", 0)
            cid = c.get("chunk_id", "")
            m = re.search(r'_(\d+)$', cid)
            idx = int(m.group(1)) if m else 0
            return (doc, page, idx)
            
        return sorted(merged_chunks, key=get_sort_key)

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
            "=== STRICT RAG COMPLIANCE & EXHAUSTIVENESS RULES ===\n"
            "1. Answer ONLY using information present in the provided regulation context. Do not assume or extrapolate.\n"
            "2. Never use external knowledge, fabricate, or invent university regulations.\n"
            "3. If the answer cannot be found in the provided context, state clearly and politely: "
            "\"I'm sorry, but that information is not available in the official Sreenidhi University R26 regulations.\"\n"
            "4. Be EXHAUSTIVE and COMPLETE. If the context contains a list of rules, eligibility criteria, guidelines, "
            "conditions, or points (numbered 1., 2., 3... or lettered a), b), c)... or bulleted), you MUST list EVERY single point "
            "available in the context. Never truncate, compress, summarize, or omit any item.\n"
            "5. Preserve the exact numbering, lettering, and bulleting structure from the source document.\n"
            "6. Do not use 'etc.', '...', or summarize lists. If the source contains 11 points, your response must contain all 11 points.\n"
            "7. Keep your answers factual and strictly grounded in the retrieved sources.\n"
            "8. Preserve the exact list numbering from the source document. Do not convert the first numbered point (e.g. '1.') into a regular paragraph or heading. Do not shift list numbers (e.g., do not number point 2 as 1). The numbers in your output list must match the source numbers exactly.\n"
            "9. Do not invent or append extra numbered items to a list that are not numbered in the source text. General introduction paragraphs or surrounding text must not be converted into additional numbered points.\n\n"
            "=== PROVIDED REGULATION CONTEXT ===\n"
            f"{context_str.strip()}\n"
            "===================================\n"
        )
        return system_prompt
