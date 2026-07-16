# PRD.md

# Sreenidhi University R26 Regulations AI Assistant

Version: 1.0

Status: MVP

Owner: AI Engineering Team

---

# 1. Overview

The goal of this project is to build an AI-powered chatbot that helps students quickly retrieve accurate information from the official Sreenidhi University R26 Academic Rules & Regulations document.

Instead of manually searching through a lengthy PDF, students should be able to ask questions in natural language and receive accurate, context-aware answers based solely on the official regulations.

This is a Retrieval-Augmented Generation (RAG) application.

The chatbot must never fabricate university rules and should only answer using the official regulation document.

---

# 2. Problem Statement

Students frequently ask questions regarding:

- Attendance
- Medical Leave
- Personal Leave
- Course Registration
- Add / Drop
- Withdrawal
- Credits
- CGPA
- SGPA
- Promotion Rules
- Internships
- Supplementary Exams
- Revaluation
- Honors
- Minor Degrees
- Academic Calendar
- Anti Ragging
- Code of Conduct
- Discipline
- Examination Rules

Currently students have to manually search through a 60+ page regulation PDF which is slow and frustrating.

The objective is to make regulation retrieval conversational.

---

# 3. Objectives

Build an AI chatbot capable of

- Understanding natural language
- Retrieving relevant regulation sections
- Generating accurate answers
- Supporting follow-up questions
- Providing fast responses
- Preventing hallucinations

---

# 4. Knowledge Base

Current Version

Only one PDF will be used.

```

rag_docs/
R26_Rules_Regulations.pdf

```

Future versions may support multiple documents.

---

# 5. Technology Stack

Backend

- FastAPI

Frontend

- React (Frontend will call FastAPI)

Vector Database

- Pinecone

Embedding Model

- llama-text-embed-v2

LLM

- Groq
- llama-3.3-70b-versatile

PDF Loader

- PyMuPDF

Chunking

- RecursiveCharacterTextSplitter

---

# 6. High Level Architecture

```

Frontend

↓

FastAPI

↓

Query Embedding

↓

Pinecone Vector Search

↓

Top K Retrieval

↓

Prompt Construction

↓

Groq LLM

↓

Final Response

↓

Frontend

```

---

# 7. RAG Pipeline

The application consists of two independent pipelines.

## Pipeline 1

Document Ingestion

Runs only once.

```

PDF

↓

Read PDF

↓

Split into Chunks

↓

Generate Embeddings

↓

Store in Pinecone

```

After this step, ingestion does not run again unless the document changes.

---

## Pipeline 2

User Chat Pipeline

Runs for every user request.

```

User Question

↓

FastAPI

↓

Generate Query Embedding

↓

Search Pinecone

↓

Retrieve Top 5 Chunks

↓

Build Prompt

↓

Send to Groq

↓

Generate Answer

↓

Return JSON Response

```

---

# 8. Chunking Strategy

Chunking Method

RecursiveCharacterTextSplitter

Chunk Size

1000

Chunk Overlap

200

Reason

Maintains context across regulation sections while ensuring efficient retrieval.

---

# 9. Metadata

Each chunk stored inside Pinecone must contain

- Chunk ID
- Page Number
- Section Title
- Document Name
- Original Text

Example

```json
{
    "chunk_id":"chunk_001",
    "page":29,
    "section":"Attendance Policy",
    "document":"R26_Rules_Regulations.pdf",
    "text":"Students must maintain..."
}
```

---

# 10. Retrieval Strategy

Search Type

Dense Vector Search

Similarity

Cosine Similarity

Top K

5

---

# 11. Prompt Construction

The LLM prompt should contain

System Prompt

+

Conversation History

+

Retrieved Context

+

Current User Question

Example

System Prompt

You are Sreenidhi University's AI Regulations Assistant.

Only answer using the provided regulation context.

Never fabricate information.

If the answer cannot be found, clearly state that the information is unavailable in the official regulations.

Conversation History

...

Retrieved Context

...

User Question

...

---

# 12. Conversation Memory

The chatbot should support multi-turn conversations.

Example

User

What is attendance policy?

Assistant

...

User

What if I have a medical emergency?

Assistant

...

The chatbot should understand that the follow-up question refers to attendance.

---

# 13. Strict RAG Rules

The chatbot must

✅ Answer only from retrieved context.

✅ Never invent university regulations.

✅ Never use external knowledge.

If information cannot be found,

respond politely that it does not exist in the official regulations.

---

# 14. FastAPI Responsibilities

FastAPI is responsible for

- Receiving user queries
- Maintaining conversation history
- Generating query embeddings
- Retrieving documents
- Constructing prompts
- Calling Groq
- Returning JSON responses

FastAPI should NOT perform document ingestion during startup.

---

# 15. Ingestion Service

A separate script should perform

- PDF loading
- Chunking
- Embedding generation
- Pinecone upload

This process runs only when the regulation document changes.

---

# 16. API Endpoints

POST /chat

Receives

```json
{
    "message":"What is attendance policy?"
}
```

Returns

```json
{
    "answer":"Students are required to...",
    "sources":[
        {
            "page":29,
            "section":"Attendance Policy"
        }
    ]
}
```

GET /health

Returns service status.

---

# 17. Folder Structure

```
backend/

app/

main.py

routers/

chat.py

services/

retriever.py

embedder.py

llm.py

prompt_builder.py

conversation.py

models/

schemas.py

config.py

rag_docs/

R26_Rules_Regulations.pdf

ingest.py

requirements.txt

.env
```

---

# 18. Functional Requirements

The system must

- Accept user questions
- Retrieve relevant regulation sections
- Answer using retrieved context
- Support conversation history
- Return page references
- Return responses in JSON
- Handle invalid inputs gracefully

---

# 19. Non Functional Requirements

- Fast response time
- Scalable architecture
- Modular codebase
- Production-ready design
- Maintainable services
- Secure API keys
- Low latency retrieval

---

# 20. Error Handling

Handle

- Empty questions
- Missing PDF
- Pinecone failures
- Embedding failures
- LLM failures
- Timeout errors
- Internal server errors

Return appropriate HTTP status codes.

---

# 21. Logging

Log

- User Query
- Retrieved Chunk IDs
- Retrieval Time
- Embedding Time
- LLM Latency
- Total Response Time
- Errors

---

# 22. Security

- Store API keys in .env
- Validate user input
- Enable CORS
- Handle exceptions
- Protect endpoints from abuse

---

# 23. Future Enhancements

- Multiple PDF support
- Hybrid Search
- Re-ranking
- Streaming Responses
- Voice Input
- OCR Support
- Admin Dashboard
- Analytics
- Feedback Collection
- Document Upload Portal

---

# 24. Success Criteria

The chatbot is considered successful when

- Students can ask regulation questions naturally.
- Responses are grounded in the official regulation PDF.
- No hallucinated regulations are generated.
- Average response time is under 3 seconds.
- The chatbot correctly answers follow-up questions using conversation history.

---

# 25. Implementation Notes for Gemini

Build the project as a production-quality FastAPI application following clean architecture principles.

The codebase should be modular, scalable, and easy to extend.

Separate ingestion logic from query-time retrieval.

Follow service-oriented design.

Do not place all logic inside one file.

The system should be built in a way that additional university documents can be added in future with minimal architectural changes.