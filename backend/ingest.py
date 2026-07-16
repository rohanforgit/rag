import os
import sys
import re
import logging
import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# Add the app folder to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingest_multi")

def parse_toc(doc) -> dict:
    """
    Parses Table of Contents from pages 2 to 9 (0-indexed 1 to 8)
    to build a map of {section_number: section_title} for the main rules handbook.
    """
    logger.info("Extracting Table of Contents from main rules PDF...")
    toc_pattern = re.compile(r'^(\d+(?:\.\d+)*)?\.?\s*(.+?)\s*\.{3,}\s*(\d+)$')
    toc_entries = {}
    
    for page_num in range(1, min(9, len(doc))):
        page = doc[page_num]
        text = page.get_text()
        lines = [line.strip() for line in text.split("\n")]
        
        for line in lines:
            m = toc_pattern.match(line)
            if m:
                sec_num = m.group(1) or ""
                sec_title = m.group(2).strip()
                # Clean leading colons or trailing spaces
                sec_title = re.sub(r'^[:\s\.]+', '', sec_title).strip()
                
                if sec_num:
                    toc_entries[sec_num] = sec_title
                else:
                    toc_entries[sec_title.lower()] = sec_title
                    
    logger.info(f"Extracted {len(toc_entries)} TOC definitions.")
    return toc_entries

def df_to_markdown_table(df) -> str:
    """
    Convert a pandas DataFrame to a clean markdown table string without tabulate dependency.
    """
    lines = []
    # Add columns header
    headers = [str(c) for c in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---" for _ in headers]) + " |")
    
    for _, row in df.iterrows():
        # Clean values and replace newlines with spaces
        row_vals = []
        for val in row:
            if pd.isna(val):
                row_vals.append("")
            else:
                row_vals.append(str(val).replace('\n', ' ').replace('|', '\\|').strip())
        lines.append("| " + " | ".join(row_vals) + " |")
        
    return "\n".join(lines)

def process_xlsx(file_path: str, filename: str) -> list[dict]:
    """
    Reads an Excel file, converts each sheet into a Markdown table text, and prepares chunks.
    """
    logger.info(f"Processing Excel document: {filename}")
    chunks = []
    
    try:
        # Load all sheets
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            # Remove entirely empty rows/columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            markdown_table = df_to_markdown_table(df)
            
            # Use text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # slightly larger chunk size to maintain table rows integrity
                chunk_overlap=150,
                length_function=len
            )
            
            table_chunks = splitter.split_text(markdown_table)
            logger.info(f"Sheet '{sheet_name}': generated {len(table_chunks)} chunks.")
            
            for idx, chunk_text in enumerate(table_chunks):
                chunk_id = f"xlsx_{filename.replace('.', '_')}_{sheet_name.replace(' ', '_')}_{idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": 1,
                    "section": f"Sheet: {sheet_name}",
                    "document": filename,
                    "text": chunk_text
                })
    except Exception as e:
         logger.error(f"Failed to process Excel document {filename}: {e}", exc_info=True)
         
    return chunks

def process_pdf(file_path: str, filename: str) -> list[dict]:
    """
    Reads a PDF, extracts text, identifies sections, and generates chunks.
    """
    logger.info(f"Processing PDF document: {filename}")
    chunks = []
    
    try:
        doc = fitz.open(file_path)
        pages_count = len(doc)
        logger.info(f"PDF '{filename}' loaded with {pages_count} pages.")
        
        # Check if this is the main rules regulations document
        is_main_rules = "rules_regulations" in filename.lower()
        
        toc_entries = {}
        start_page = 0
        if is_main_rules:
            toc_entries = parse_toc(doc)
            start_page = 9  # Page 10 is index 9
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        section_regex = re.compile(r'^(\d+(?:\.\d+)+)\.?\s+(.+)$')
        main_section_regex = re.compile(r'^(\d+)\.?\s+(.+)$')
        
        current_section = "Preamble" if is_main_rules else "General Document Context"
        
        for page_idx in range(start_page, pages_count):
            page_num = page_idx + 1
            page = doc[page_idx]
            text = page.get_text()
            
            # Heuristic section header matching
            lines = [line.strip() for line in text.split("\n")]
            for line in lines:
                if '....' in line:
                    continue
                    
                # Match section patterns
                m = section_regex.match(line)
                if m:
                    sec_num = m.group(1)
                    if is_main_rules and sec_num in toc_entries:
                        current_section = f"{sec_num} {toc_entries[sec_num]}"
                    elif not is_main_rules:
                        # For other PDFs, use matched header directly
                        current_section = line[:100]  # Cap length
                    continue
                    
                m2 = main_section_regex.match(line)
                if m2:
                    sec_num = m2.group(1)
                    if is_main_rules and sec_num in toc_entries:
                        current_section = f"{sec_num} {toc_entries[sec_num]}"
                    elif not is_main_rules:
                        current_section = line[:100]
            
            page_chunks = splitter.split_text(text)
            for idx, chunk_text in enumerate(page_chunks):
                cleaned_text = re.sub(r'\s+', ' ', chunk_text).strip()
                if not cleaned_text:
                    continue
                    
                chunk_id = f"pdf_{filename.replace('.', '_')}_p{page_num}_{idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "section": current_section,
                    "document": filename,
                    "text": cleaned_text
                })
    except Exception as e:
        logger.error(f"Failed to process PDF {filename}: {e}", exc_info=True)
        
    return chunks

def main():
    load_dotenv()
    
    # 1. Environment Verification
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "sreenidhi-r26-regs")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    
    if not api_key or "your-pinecone" in api_key.lower():
        logger.error("PINECONE_API_KEY is missing or invalid.")
        sys.exit(1)
        
    # 2. Scan rag_files/ directory
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_dir = os.path.join(workspace_dir, "rag_files")
    
    if not os.path.exists(rag_dir):
        logger.error(f"RAG files directory not found at: {rag_dir}")
        sys.exit(1)
        
    files = [f for f in os.listdir(rag_dir) if os.path.isfile(os.path.join(rag_dir, f)) and not f.startswith(".")]
    logger.info(f"Found {len(files)} files in RAG directory: {files}")
    
    all_chunks = []
    
    for filename in files:
        file_path = os.path.join(rag_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".pdf":
            chunks = process_pdf(file_path, filename)
            all_chunks.extend(chunks)
        elif ext in [".xlsx", ".xls"]:
            chunks = process_xlsx(file_path, filename)
            all_chunks.extend(chunks)
        else:
            logger.warning(f"Skipping unsupported file type: {filename}")
            
    total_chunks = len(all_chunks)
    logger.info(f"Total chunks generated across all documents: {total_chunks}")
    
    if total_chunks == 0:
        logger.error("No content chunk generated. Aborting upsert.")
        sys.exit(1)
        
    # 3. Pinecone Index Setup
    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    
    # Re-create or reset vector spaces to prevent staled vectors
    if index_name in pc.list_indexes().names():
        logger.info(f"Index '{index_name}' already exists. Connecting and deleting existing vectors for fresh index...")
        index = pc.Index(index_name)
        try:
            index.delete(delete_all=True)
            logger.info("Cleared all existing vector mappings in index.")
        except Exception as e:
            logger.warning(f"Could not delete vectors: {e}. Proceeding.")
    else:
        logger.info(f"Index '{index_name}' not found. Creating serverless index...")
        try:
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            logger.info("Waiting for index initialization...")
            import time
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(2)
            logger.info("Index is ready.")
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            sys.exit(1)
            
    index = pc.Index(index_name)
    
    # 4. Generate embeddings and upsert
    logger.info("Generating embeddings and upserting vector payloads to Pinecone...")
    batch_size = 50
    
    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [item["text"] for item in batch]
        
        try:
            # Generate embeddings using Inference API
            embeddings_resp = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            
            vectors = []
            for chunk, emb in zip(batch, embeddings_resp):
                vectors.append((
                    chunk["chunk_id"],
                    emb["values"],
                    {
                        "chunk_id": chunk["chunk_id"],
                        "page": chunk["page"],
                        "section": chunk["section"],
                        "document": chunk["document"],
                        "text": chunk["text"]
                    }
                ))
                
            # Upsert
            index.upsert(vectors=vectors)
            logger.info(f"Upserted chunks {i+1} to {min(i+batch_size, total_chunks)} of {total_chunks}")
            
        except Exception as e:
            logger.error(f"Failed embedding generation/upsert at batch index {i}: {e}")
            sys.exit(1)
            
    logger.info("====================================================")
    logger.info("Bulk document ingestion completed successfully!")
    logger.info(f"Indexed {total_chunks} chunks into Pinecone index '{index_name}'.")
    logger.info("====================================================")

if __name__ == "__main__":
    main()
