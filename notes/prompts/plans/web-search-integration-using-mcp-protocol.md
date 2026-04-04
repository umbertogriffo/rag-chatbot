# Web Search Tool Integration Using MCP Protocol

## **Executive Summary**
This plan outlines the implementation of a web search capability for the RAG chatbot using the Model Context Protocol (MCP).
The feature will allow users to toggle web search mode via the existing UI flag, enabling real-time
information retrieval instead of relying solely on RAG-based document retrieval.

---

## **1. Current State Analysis**

### **1.1 Existing Architecture**
- **Frontend**: React/TypeScript with mode toggles (RAG, Reasoning, Web Search)
- **Backend**: FastAPI with WebSocket streaming support
- **LLM Client**: `LamaCppClient` with tool calling support (experimental)
- **Chat Flow**:
  - `/chat/stream` WebSocket endpoint handles both RAG and regular chat
  - `ChatRequest` schema includes `web_search: bool` flag (currently unused)
  - Function calling already implemented in `retrieve_tools()` method

### **1.2 What Exists**
✅ UI toggle for web search (`webSearch` state in `mode-toggle.tsx`)
✅ Backend schema field `web_search: bool` in `ChatRequest`
✅ Tool calling infrastructure in `LamaCppClient.retrieve_tools()`
✅ WebSocket streaming architecture
✅ Modular conversation handler pattern

### **1.3 What's Missing**
❌ MCP server implementation
❌ Web search provider integration
❌ Query classification/routing logic
❌ Frontend flag propagation to backend
❌ Search result formatting/parsing
❌ Tool execution orchestration

---

## **2. Architecture Design**

### **2.1 MCP Protocol Structure**

```
┌─────────────┐         ┌─────────────┐         ┌──────────────┐
│  Frontend   │◄───────►│   Backend   │◄───────►│ MCP Server   │
│  (Toggle)   │         │  (FastAPI)  │         │  (Tools)     │
└─────────────┘         └─────────────┘         └──────────────┘
                              │                        │
                              │                        ▼
                              │                  ┌──────────────┐
                              │                  │ Web Search   │
                              │                  │ Provider     │
                              │                  │ (DuckDuckGo) │
                              ▼                  └──────────────┘
                        ┌──────────────┐
                        │ LLM Client   │
                        │ (Tool Call)  │
                        └──────────────┘
```

### **2.2 Component Breakdown**

#### **A. MCP Server Module** (`backend/mcp/`)
```
backend/mcp/
├── __init__.py
├── server.py           # MCP protocol server implementation
├── tools/
│   ├── __init__.py
│   ├── base.py         # Abstract tool interface
│   ├── web_search.py   # Web search tool implementation
│   └── registry.py     # Tool registry for extensibility
├── providers/
│   ├── __init__.py
│   ├── base.py         # Abstract search provider
│   └── duckduckgo.py   # DuckDuckGo search provider
└── query_processor.py  # Query classification logic
```

#### **B. Tool Definitions** (`chatbot/bot/tools/`)
```
chatbot/bot/tools/
├── __init__.py
├── schemas.py          # Tool function schemas for llama.cpp
└── config.py           # Tool configurations
```

---

## **3. Detailed Implementation Steps**

### **Phase 1: Foundation & MCP Server (Week 1)**

#### **Step 1.1: Create MCP Base Structure**

**File**: `backend/mcp/base.py`
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel

class MCPRequest(BaseModel):
    """Base request model for MCP protocol"""
    query: str
    metadata: Dict[str, Any] = {}

class MCPResponse(BaseModel):
    """Base response model for MCP protocol"""
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}

class MCPTool(ABC):
    """Abstract base class for MCP tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict:
        """JSON schema for tool parameters"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> MCPResponse:
        """Execute the tool"""
        pass
```

#### **Step 1.2: Implement Web Search Provider**

**File**: `backend/mcp/providers/base.py`
```python
from abc import ABC, abstractmethod
from typing import List, Dict

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str | None = None

class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        pass
```

**File**: `backend/mcp/providers/duckduckgo.py`
```python
import httpx
from typing import List
from .base import SearchProvider, SearchResult

class DuckDuckGoProvider(SearchProvider):
    """Minimal dependency DuckDuckGo search implementation"""

    BASE_URL = "https://api.duckduckgo.com/"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Execute DuckDuckGo search using their instant answer API
        No additional dependencies needed - uses httpx (already in dev deps)
        """
        async with httpx.AsyncClient() as client:
            # Use DuckDuckGo's instant answer API
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }

            response = await client.get(self.BASE_URL, params=params)
            data = response.json()

            results = []

            # Parse related topics
            for item in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(item, dict) and "Text" in item:
                    results.append(SearchResult(
                        title=item.get("Text", "")[:100],
                        url=item.get("FirstURL", ""),
                        snippet=item.get("Text", "")
                    ))

            # If no results, try abstract
            if not results and data.get("AbstractText"):
                results.append(SearchResult(
                    title=data.get("Heading", query),
                    url=data.get("AbstractURL", ""),
                    snippet=data.get("AbstractText", "")
                ))

            return results
```

**Alternative**: For HTML scraping (if instant API is insufficient):
```python
# Can use httpx + basic HTML parsing without heavy dependencies
from bs4 import BeautifulSoup  # Only if needed - can parse manually
```

#### **Step 1.3: Implement Web Search Tool**

**File**: `backend/mcp/tools/web_search.py`
```python
from typing import Dict, List
from mcp.base import MCPTool, MCPResponse
from mcp.providers.base import SearchProvider

class WebSearchTool(MCPTool):
    def __init__(self, provider: SearchProvider):
        self.provider = provider

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information, news, facts, and real-time data"

    @property
    def parameters(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, max_results: int = 5) -> MCPResponse:
        results = await self.provider.search(query, max_results)

        return MCPResponse(
            results=[
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "published_date": r.published_date
                }
                for r in results
            ],
            metadata={"query": query, "count": len(results)}
        )
```

#### **Step 1.4: Implement Tool Registry**

**File**: `backend/mcp/tools/registry.py`
```python
from typing import Dict, List, Optional
from mcp.base import MCPTool

class ToolRegistry:
    """Extensible registry for MCP tools"""

    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """Register a new tool"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[MCPTool]:
        """Get tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        """Get all tools in llama.cpp format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self._tools.values()
        ]

    def get_all(self) -> Dict[str, MCPTool]:
        return self._tools
```

---

### **Phase 2: Query Processing & Classification (Week 1)**

#### **Step 2.1: Implement Query Classifier**

**File**: `backend/mcp/query_processor.py`
```python
from enum import Enum
from typing import List
import re

class QueryType(Enum):
    WEATHER = "weather"
    NEWS = "news"
    SPORTS = "sports"
    FINANCIAL = "financial"
    TIME_SENSITIVE = "time_sensitive"
    GENERAL = "general"

class QueryProcessor:
    """Lightweight query classification without ML dependencies"""

    # Pattern-based classification
    PATTERNS = {
        QueryType.WEATHER: [
            r"\b(weather|temperature|forecast|rain|snow|sunny|cloudy)\b",
            r"\bwhat.*like outside\b",
            r"\bhow.*weather\b"
        ],
        QueryType.NEWS: [
            r"\b(news|latest|recent|breaking|headline|current events)\b",
            r"\bwhat.*happening\b",
            r"\btell me about.*today\b"
        ],
        QueryType.SPORTS: [
            r"\b(score|game|match|team|player|league|championship)\b",
            r"\b(football|basketball|baseball|soccer|tennis|cricket)\b"
        ],
        QueryType.FINANCIAL: [
            r"\b(stock|share|market|price|trading|nasdaq|dow|crypto|bitcoin)\b",
            r"\bhow much is.*worth\b"
        ],
        QueryType.TIME_SENSITIVE: [
            r"\b(now|today|current|latest|right now|as of)\b",
            r"\b(this week|this month|this year|recently)\b"
        ]
    }

    def classify(self, query: str) -> QueryType:
        """Classify query type using pattern matching"""
        query_lower = query.lower()

        for query_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return query_type

        return QueryType.GENERAL

    def should_use_web_search(self, query: str) -> bool:
        """Determine if query requires web search"""
        query_type = self.classify(query)

        # Time-sensitive queries should use web search
        time_sensitive_types = {
            QueryType.WEATHER,
            QueryType.NEWS,
            QueryType.SPORTS,
            QueryType.FINANCIAL,
            QueryType.TIME_SENSITIVE
        }

        return query_type in time_sensitive_types
```

---

### **Phase 3: Backend Integration (Week 2)**

#### **Step 3.1: Update Configuration**

**File**: `backend/core/config.py`
```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Web Search Configuration
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_MAX_RESULTS: int = 5
    WEB_SEARCH_PROVIDER: str = "duckduckgo"  # Extensible for future providers
```

#### **Step 3.2: Create MCP State Manager**

**File**: `backend/state.py` (update existing)
```python
from typing import Optional
from mcp.tools.registry import ToolRegistry
from mcp.tools.web_search import WebSearchTool
from mcp.providers.duckduckgo import DuckDuckGoProvider

# Existing global state
engine = None
llm_client = None
index = None

# New MCP state
tool_registry: Optional[ToolRegistry] = None

def init_mcp_tools():
    """Initialize MCP tool registry"""
    global tool_registry
    tool_registry = ToolRegistry()

    # Register web search tool
    search_provider = DuckDuckGoProvider()
    web_search = WebSearchTool(search_provider)
    tool_registry.register(web_search)

    return tool_registry
```

#### **Step 3.3: Update Main Lifespan**

**File**: `backend/main.py`
```python
from state import init_mcp_tools

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize global state
    state.engine = create_db_engine()
    state.llm_client = create_llm_client(settings.MODEL_FOLDER)
    state.index = init_index(settings.VECTOR_STORE_PATH)

    # Initialize MCP tools
    if settings.WEB_SEARCH_ENABLED:
        state.tool_registry = init_mcp_tools()
        logger.info("MCP tools initialized")

    yield

    # Cleanup (existing code...)
```

#### **Step 3.4: Create Dependencies**

**File**: `backend/api/deps.py` (update)
```python
from typing import Annotated
from fastapi import Depends
import state

# Existing dependencies...

def get_tool_registry():
    return state.tool_registry

ToolRegistryDep = Annotated[ToolRegistry, Depends(get_tool_registry)]
```

#### **Step 3.5: Create Web Search Service**

**File**: `backend/api/services/web_search.py`
```python
import json
from typing import List, Dict, Tuple
from bot.client.lama_cpp_client import LamaCppClient
from mcp.tools.registry import ToolRegistry
from mcp.query_processor import QueryProcessor
from helpers.log import get_logger

logger = get_logger(__name__)

class WebSearchService:
    def __init__(self, llm_client: LamaCppClient, tool_registry: ToolRegistry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.query_processor = QueryProcessor()

    async def search_and_synthesize(
        self,
        query: str,
        max_new_tokens: int = 512
    ) -> Tuple[str, List[Dict]]:
        """
        Execute web search via tool calling and synthesize answer

        Returns:
            Tuple of (streamer, search_results)
        """
        logger.info(f"Processing web search for query: {query}")

        # Get available tools
        tools = self.tool_registry.list_tools()

        # Let LLM decide if/how to use web search
        tool_calls = self.llm_client.retrieve_tools(
            prompt=query,
            tools=tools,
            max_new_tokens=256
        )

        search_results = []

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                logger.info(f"Executing tool: {function_name} with args: {function_args}")

                # Get and execute tool
                tool = self.tool_registry.get(function_name)
                if tool:
                    response = await tool.execute(**function_args)
                    search_results.extend(response.results)

        # Format context from search results
        context = self._format_search_results(search_results)

        # Generate answer with context
        prompt = self.llm_client.generate_ctx_prompt(
            question=query,
            context=context
        )

        streamer = await self.llm_client.async_start_answer_iterator_streamer(
            prompt, max_new_tokens=max_new_tokens
        )

        return streamer, search_results

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results into context string"""
        if not results:
            return "No web search results found."

        context = "Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. **{result['title']}**\n"
            context += f"   {result['snippet']}\n"
            context += f"   Source: {result['url']}\n\n"

        return context
```

#### **Step 3.6: Update Chat Stream Service**

**File**: `backend/api/services/chat_stream.py` (update)
```python
from api.services.web_search import WebSearchService

async def stream_web_search_response(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
    tool_registry: ToolRegistryDep
):
    """Helper function to stream web search responses token by token"""
    try:
        start_time = time.time()

        # Create web search service
        web_search_service = WebSearchService(llm_client, tool_registry)

        # Execute search and get streamer
        streamer, search_results = await web_search_service.search_and_synthesize(
            query.text,
            max_new_tokens=settings.MAX_NEW_TOKENS
        )

        # Send search results preview
        if search_results:
            preview = "🔍 **Web Search Results:**\n\n"
            for i, result in enumerate(search_results[:3], 1):
                preview += f"{i}. [{result['title']}]({result['url']})\n"
            preview += "\n" + "-" * 20 + "\n\n**Answer:**\n\n"
            await websocket.send_text(preview)

        # Stream answer
        full_response = ""
        for output in streamer:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)

        # Extract final answer if reasoning mode
        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(
                full_response,
                llm_client.model_settings.reasoning_stop_tag
            )
            if final_answer == "":
                final_answer = full_response
        else:
            final_answer = full_response

        chat_history.append(f"question: {query.text}, answer: {final_answer}")

        took = time.time() - start_time
        logger.info(f"Web search took {took:.2f} seconds")

    except Exception as exc:
        logger.exception("Error during web search streaming: %s", exc)
        await websocket.send_text("Error during web search.")
```

#### **Step 3.7: Update WebSocket Endpoint**

**File**: `backend/api/endpoints/chat_stream.py` (update)
```python
from api.deps import ToolRegistryDep
from api.services.chat_stream import stream_web_search_response

@router.websocket(path="/chat/stream")
async def chat_stream(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    chat_history: ChatHistoryDep,
    index: VectorDatabaseDep,
    tool_registry: ToolRegistryDep  # New dependency
):
    """WebSocket endpoint for streaming chat responses token by token."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received data: {data}")
            query = ChatRequest(**data)

            # Route based on mode
            if query.web_search:
                await stream_web_search_response(
                    websocket, llm_client, query, chat_history, tool_registry
                )
            elif query.rag:
                await stream_rag_response(
                    websocket, llm_client, query, chat_history, index
                )
            else:
                await stream_chat_response(
                    websocket, llm_client, query, chat_history
                )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
        raise
```

---

### **Phase 4: Frontend Integration (Week 2)**

#### **Step 4.1: Update WebSocket Service**

**File**: `frontend/src/services/websocket.ts`
```typescript
async sendMessage(
  text: string,
  rag: boolean,
  reasoning: boolean = false,
  webSearch: boolean = false
): Promise<void> {
  try {
    await this.connect();
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        text,
        rag,
        reasoning,  // Add if not already present
        web_search: webSearch  // Snake case for backend
      }));
    } else {
      this.onError('WebSocket is not connected');
    }
  } catch (error) {
    this.onError(error instanceof Error ? error.message : 'Failed to connect');
  }
}
```

#### **Step 4.2: Update useChat Hook**

**File**: `frontend/src/hooks/useChat.ts`
```typescript
const sendMessage = useCallback((
  text: string,
  rag: boolean,
  reasoning: boolean,
  webSearch: boolean
) => {
  if (!text.trim() || isStreaming) return;

  const userMsg: Message = {
    id: ++idRef.current,
    text,
    sender: 'user',
    timestamp: new Date(),
  };
  const botPlaceholder: Message = {
    id: ++idRef.current,
    text: '',
    sender: 'bot',
    timestamp: new Date(),
    isStreaming: true,
  };

  setMessages((prev) => [...prev, userMsg, botPlaceholder]);
  setIsStreaming(true);
  wsRef.current?.sendMessage(text, rag, reasoning, webSearch);
}, [isStreaming]);
```

#### **Step 4.3: Update App Component**

**File**: `frontend/src/App.tsx`
```typescript
const handleSend = useCallback(
  (content: string) => {
    sendMessage(content, modes.rag, modes.reasoning, modes.webSearch);
  },
  [sendMessage, modes],
);
```

---

### **Phase 5: Testing & Documentation (Week 3)**

#### **Step 5.1: Create Unit Tests**

**File**: `tests/mcp/test_web_search_tool.py`
```python
import pytest
from backend.mcp.tools.web_search import WebSearchTool
from backend.mcp.providers.duckduckgo import DuckDuckGoProvider

@pytest.mark.asyncio
async def test_web_search_execution():
    provider = DuckDuckGoProvider()
    tool = WebSearchTool(provider)

    response = await tool.execute(query="Python programming", max_results=3)

    assert len(response.results) > 0
    assert response.metadata["query"] == "Python programming"

@pytest.mark.asyncio
async def test_tool_schema():
    provider = DuckDuckGoProvider()
    tool = WebSearchTool(provider)

    assert tool.name == "web_search"
    assert "query" in tool.parameters["properties"]
    assert "query" in tool.parameters["required"]
```

**File**: `tests/mcp/test_query_processor.py`
```python
from backend.mcp.query_processor import QueryProcessor, QueryType

def test_weather_classification():
    processor = QueryProcessor()

    assert processor.classify("What's the weather like today?") == QueryType.WEATHER
    assert processor.classify("Will it rain tomorrow?") == QueryType.WEATHER

def test_news_classification():
    processor = QueryProcessor()

    assert processor.classify("What's the latest news?") == QueryType.NEWS
    assert processor.classify("Breaking headlines today") == QueryType.NEWS

def test_should_use_web_search():
    processor = QueryProcessor()

    assert processor.should_use_web_search("Current stock price of AAPL") == True
    assert processor.should_use_web_search("Explain quantum computing") == False
```

#### **Step 5.2: Integration Tests**

**File**: `tests/api/test_web_search_endpoint.py`
```python
@pytest.mark.asyncio
async def test_web_search_stream(test_client):
    async with test_client.websocket_connect("/chat/stream") as websocket:
        await websocket.send_json({
            "text": "What's the weather in London today?",
            "rag": False,
            "web_search": True
        })

        response_tokens = []
        async for message in websocket.iter_text():
            response_tokens.append(message)
            if len(response_tokens) > 10:  # Get first few tokens
                break

        assert len(response_tokens) > 0
```

#### **Step 5.3: Documentation**

**File**: `docs/web-search-mcp.md`
```markdown
# Web Search Integration with MCP

## Overview
The web search feature uses the Model Context Protocol (MCP) to enable real-time information retrieval...

## Architecture
[Diagrams and explanations]

## Adding New Tools
To extend the system with additional tools:

1. Create tool class in `backend/mcp/tools/`
2. Implement `MCPTool` interface
3. Register in `state.init_mcp_tools()`

Example:
[Code example]

## Configuration
[Environment variables and settings]

## Usage
[Frontend and API usage examples]
```

---

## **4. Extensibility Strategy**

### **4.1 Future Tool Additions**

The MCP architecture supports easy addition of new tools:

**Example: Calculator Tool**
```python
# backend/mcp/tools/calculator.py
class CalculatorTool(MCPTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations"

    # ... implementation
```

**Example: Weather API Tool**
```python
# backend/mcp/tools/weather.py
class WeatherTool(MCPTool):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()

    async def execute(self, location: str) -> MCPResponse:
        # Call real weather API
        pass
```

### **4.2 Provider Interface**

Easy to swap search providers:
- DuckDuckGo (current - no API key needed)
- Brave Search
- Google Custom Search
- Bing Search
- SearXNG (self-hosted)

---

## **5. Dependencies**

### **5.1 Required (Minimal)**
```toml
# Already in project:
httpx = "~=0.28.1"  # In dev deps, move to main deps

# No new dependencies needed for basic implementation!
```

### **5.2 Optional Enhancements**
```toml
# For better HTML parsing (if needed):
beautifulsoup4 = "~=4.12.0"
lxml = "~=5.0.0"

# For advanced search providers:
# duckduckgo-search = "~=4.0.0"  # Only if instant API insufficient
```

---

## **6. Configuration Management**

### **6.1 Environment Variables**
```bash
# .env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_PROVIDER=duckduckgo

# For future providers:
# BRAVE_API_KEY=your_key_here
# GOOGLE_SEARCH_API_KEY=your_key_here
# GOOGLE_SEARCH_ENGINE_ID=your_engine_id
```

---

## **7. Testing Strategy**

### **7.1 Test Pyramid**
- **Unit Tests**: Tool implementations, query processing
- **Integration Tests**: MCP server, tool registry
- **E2E Tests**: WebSocket flow with tool calling
- **Manual Tests**: UI toggles, streaming responses

### **7.2 Test Coverage Goals**
- Core MCP components: 90%+
- Tool implementations: 80%+
- Service layer: 85%+

---

## **8. Performance Considerations**

### **8.1 Optimization Strategies**
1. **Caching**: Cache search results for identical queries (5-minute TTL)
2. **Timeout Management**: 10-second timeout for web searches
3. **Concurrent Requests**: Allow parallel tool execution
4. **Rate Limiting**: Implement provider-specific rate limits

### **8.2 Monitoring**
- Track search latency
- Monitor tool success/failure rates
- Log query classifications

---

## **9. Security Considerations**

### **9.1 Input Validation**
- Sanitize search queries
- Validate tool parameters
- Rate limit per user/session

### **9.2 Output Sanitization**
- Escape HTML in search results
- Validate URLs before display
- Filter inappropriate content

---

## **10. Migration Path**

### **10.1 Rollout Strategy**
1. **Week 1**: Backend MCP infrastructure (no UI changes)
2. **Week 2**: Enable web search toggle (opt-in)
3. **Week 3**: Testing and refinement
4. **Week 4**: Documentation and monitoring

### **10.2 Feature Flags**
```python
# Gradual rollout capability
if settings.WEB_SEARCH_ENABLED:
    # New behavior
else:
    # Fallback to existing behavior
```

---

## **11. Open Questions & Clarifications**

Before proceeding, please clarify:

1. **Search Provider Preference**:
   - DuckDuckGo instant API (no key, limited results)?
   - Or accept a lightweight dependency like `duckduckgo-search` for better results?

2. **Tool Calling Model**:
   - Which model are you using? (`llama-3.1-tool` or another?)
   - Does it reliably support function calling?

3. **Rate Limiting**:
   - Should we implement per-user rate limits?
   - Expected concurrent users?

4. **Result Caching**:
   - Do you want search result caching?
   - If yes, Redis or in-memory?

5. **UI Behavior**:
   - Should web search and RAG modes be mutually exclusive?
   - Or allow combining them (hybrid mode)?

6. **Monitoring**:
   - Do you have existing monitoring infrastructure?
   - Should we add metrics endpoints?

---

## **12. Success Criteria**

The implementation is successful when:

✅ User can toggle web search in UI
✅ Toggle correctly triggers web search backend flow
✅ LLM receives search results as context
✅ Answers stream back with source citations
✅ System remains stable under normal load
✅ Code is well-documented and tested
✅ Architecture supports easy addition of new tools

---

## **Summary**

This plan provides a **minimal-dependency**, **extensible** implementation of web search using MCP protocol. The architecture:

- ✅ Uses existing `llama.cpp` tool calling
- ✅ Adds minimal dependencies (just `httpx`, already in project)
- ✅ Follows existing code patterns
- ✅ Enables easy future tool additions
- ✅ Maintains clean separation of concerns
- ✅ Leverages existing WebSocket streaming

**Next Steps**: Please review and provide feedback on the open questions so I can refine specific implementation details!
