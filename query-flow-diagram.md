# RAG System Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                FRONTEND                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │ User Input      │───▶│ sendMessage()    │───▶│ POST /api/query      │   │
│  │ - Type query    │    │ - Disable UI     │    │ - QueryRequest       │   │
│  │ - Click send    │    │ - Show loading   │    │ - session_id         │   │
│  └─────────────────┘    └──────────────────┘    └──────────────────────┘   │
│                                                             │               │
└─────────────────────────────────────────────────────────────┼───────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               BACKEND API                                   │
│  ┌──────────────────────┐    ┌───────────────────┐                         │
│  │ FastAPI Endpoint     │───▶│ Session Manager   │                         │
│  │ query_documents()    │    │ - Create/get      │                         │
│  │ - Receive request    │    │   session_id      │                         │
│  │ - Validate data      │    │ - Get history     │                         │
│  └──────────────────────┘    └───────────────────┘                         │
│              │                          │                                  │
│              ▼                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    RAG SYSTEM                                       │   │
│  │  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────┐   │   │
│  │  │ Query Method    │───▶│ AI Generator     │───▶│ Tool Manager │   │   │
│  │  │ - Format prompt │    │ - Claude API     │    │ - Search     │   │   │
│  │  │ - Get history   │    │ - Tool calling   │    │   tools      │   │   │
│  │  └─────────────────┘    └──────────────────┘    └──────────────┘   │   │
│  │                                   │                       │        │   │
│  │                                   ▼                       ▼        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                 VECTOR SEARCH                               │   │   │
│  │  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │   │   │
│  │  │  │CourseSearch  │───▶│ ChromaDB     │───▶│ Relevant     │  │   │   │
│  │  │  │Tool          │    │ - Semantic   │    │ Chunks       │  │   │   │
│  │  │  │- Execute     │    │   search     │    │ - Context    │  │   │   │
│  │  │  │- Get context │    │ - Embeddings │    │ - Sources    │  │   │   │
│  │  │  └──────────────┘    └──────────────┘    └──────────────┘  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                   ▲                               │   │
│  │                                   │                               │   │
│  └───────────────────────────────────┼───────────────────────────────┘   │
│                                      │                                   │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    RESPONSE GENERATION                           │    │
│  │  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐  │    │
│  │  │ AI combines:    │───▶│ Generate final   │───▶│ Return      │  │    │
│  │  │ - Query         │    │ response         │    │ - Answer    │  │    │
│  │  │ - Context       │    │ - Natural lang   │    │ - Sources   │  │    │
│  │  │ - History       │    │ - Conversational │    │ - SessionID │  │    │
│  │  └─────────────────┘    └──────────────────┘    └─────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                      │                                   │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │ QueryResponse Model                                              │    │
│  │ - answer: str                                                    │    │
│  │ - sources: List[str]                                             │    │
│  │ - session_id: str                                                │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                      │                                   │
└──────────────────────────────────────┼───────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND RESPONSE                                │
│  ┌──────────────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │ Receive JSON         │───▶│ Update UI        │───▶│ Display Result   │   │
│  │ - Parse response     │    │ - Remove loading │    │ - Markdown text  │   │
│  │ - Update session_id  │    │ - Enable input   │    │ - Collapsible    │   │
│  │ - Error handling     │    │ - Scroll to end  │    │   sources        │   │
│  └──────────────────────┘    └──────────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

Key Data Structures:
┌─────────────────────────────────────────────────────────────────────────────┐
│ QueryRequest          │ QueryResponse         │ CourseChunk              │
│ - query: str          │ - answer: str         │ - content: str           │
│ - session_id: str?    │ - sources: List[str]  │ - course_title: str      │
│                       │ - session_id: str     │ - lesson_number: int     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Flow Summary:

1. **User Input** → Frontend captures query
2. **API Call** → POST /api/query with QueryRequest
3. **Session Management** → Create/retrieve session for context
4. **RAG Processing** → Format prompt, get conversation history
5. **AI Generation** → Claude processes with tool calling capability
6. **Vector Search** → CourseSearchTool queries ChromaDB for relevant chunks
7. **Context Assembly** → AI combines query + retrieved context + history
8. **Response Generation** → AI generates natural language answer
9. **API Response** → QueryResponse model returned as JSON
10. **UI Update** → Frontend displays answer with sources

**Key Components:**
- **ChromaDB**: Vector storage for semantic search
- **Claude API**: AI generation with tool calling
- **Session Manager**: Conversation context persistence
- **Tool Manager**: Orchestrates search tools
- **Vector Store**: Manages embeddings and retrieval