Plan: Modernize RAG Chatbot with React Frontend and REST/WebSocket APIs

Modernize the existing RAG chatbot by building a ChatGPT-like React UI with TypeScript and Tailwind CSS, connected to the FastAPI backend via REST and WebSocket endpoints. The plan preserves all existing unit tests while adding comprehensive API layer testing for document management and streaming chat functionality.

Steps

- Design and implement backend API layer in backend/api/endpoints/: Create documents.py with POST /api/documents (upload with multipart/form-data), GET /api/documents (list), and DELETE /api/documents/{document_id} endpoints using existing Document entities and Chroma vector store; documents are globally shared; create chat_stream.py with WebSocket endpoint /api/chat/stream that uses LamaCppClient.start_answer_iterator_streamer() for streaming responses; update ChatRequest schema to include rag flag and create DocumentUploadResponse, DocumentListResponse schemas.
- Rebuild React frontend UI in frontend/src/: Replace Material-UI with Tailwind CSS classes; create ChatGPT-inspired dark theme layout with message bubbles, markdown rendering (keep react-markdown), and streaming text animation; add document upload component with drag-and-drop, progress bars, and document list view with delete buttons integrated into main chat interface (no sidebars); implement settings panel component for model selection and configuration; update App.tsx to orchestrate new layout.
- Implement frontend integration layer in frontend/src/services/: Create api.ts using Axios for REST calls (uploadDocument, listDocuments, deleteDocument); create websocket.ts for chat streaming with reconnection logic; implement custom React hooks useChat() for message streaming state management, useDocuments() for upload/delete with progress tracking; add error boundaries for graceful error handling and loading state components.
- Create comprehensive API tests in tests/api/: Write pytest tests for document endpoints using httpx.AsyncClient (test upload validation, listing, deletion, duplicate handling); test WebSocket streaming endpoint with mock LamaCppClient; test chat endpoint with RAG flag integration; ensure all existing tests in tests/bot/ and tests/document_loader/ continue passing; add integration tests for end-to-end document upload and retrieval flow.
- Update project configuration and dependencies: Add Tailwind CSS dependencies to frontend/package.json and configure tailwind.config.js; add python-multipart and websockets to pyproject.toml; update backend/core/config.py with WebSocket settings; update CORS configuration in backend/main.py for WebSocket support; create environment variable documentation.

## v0 by Vercel - Build UI with AI

https://v0.app/chat

Implementation of a high-fidelity, premium web UI for a modern AI chatbot as follows:

- **Layout**:
  - Collapsible minimalist sidebar on the left for chat history and a prominent, clean conversation area in the center.
  - Floating, bottom-centered text input bar with a glowing 'send' icon.
  - Integrate a document upload component (drag-and-drop, progress bars, document list with delete buttons) directly into the main chat interface (not in the sidebar).

- **Styling**:
  - Dark mode color palette: deep grays, monochromatic grayscale, single brand accent color (e.g., Electric Blue).
  - Sleek message bubbles with generous padding and soft rounded corners.
  - Glassmorphism effect for the input bar.
  - Modern spacing (6:3:1 color rule), subtle borders, and premium, minimalist Apple-inspired look.
  - Typography: clean, geometric sans-serif (e.g., Inter or SF Pro).
  - 4k resolution support.

- **Components**:
  - Sidebar: Chat history, 'New Chat' button.
  - Main chat viewport: User/AI avatars, streaming text animation placeholder. Markdown rendering.
  - Document upload: Drag-and-drop area, progress bars, document list with delete buttons.

- **Tech Stack**:
  - React for UI components + Vite.
  - Tailwind CSS for styling and responsiveness.

- **Vibe**:
  - Premium, tech-forward, minimalist, inspired by ChatGPT and Linear.

**Deliverable**:
A responsive React component (or set of components) + Vite implementing the above features and styles using Tailwind CSS.

------

Implementation of a high-fidelity, premium web UI for a modern AI chatbot as follows:

- **Layout**:
  - Floating, bottom-centered text input bar with a glowing 'send' icon.
  - Integrate a document upload component (drag-and-drop, progress bars, document list with delete buttons) directly into the main chat interface (not in a sidebar).
  - Flags in the input bar for toggling RAG mode, reasoning mode, and web search mode.
  - If the user uploads a document, the RAG flag is automatically enabled for that message.
  - If the RAG flag is enabled, we want to show a small preview of the uploaded documents (file name, ranking) in the message bubble.

- **Styling**:
  - Dark mode color palette: deep grays, monochromatic grayscale, single brand accent color (e.g., Electric Blue).
  - Sleek message bubbles with generous padding and soft rounded corners.
  - Glassmorphism effect for the input bar.
  - Modern spacing (6:3:1 color rule), subtle borders, and premium, minimalist Apple-inspired look.
  - Typography: clean, geometric sans-serif (e.g., Inter or SF Pro).
  - 4k resolution support.

- **Components**:
  - Main chat viewport: User/AI avatars, streaming text animation placeholder. Markdown rendering.
  - Document upload: Drag-and-drop area, progress bars, document list with delete buttons.

- **Tech Stack**:
  - React for UI components + Vite.
  - Tailwind CSS for styling and responsiveness.

- **Vibe**:
  - Premium, tech-forward, minimalist, inspired by ChatGPT and Linear.

**Deliverable**:
A responsive React component (or set of components) + Vite implementing the above features and styles using Tailwind CSS.
