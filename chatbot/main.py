import os

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from helpers.log import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)

# Import other necessary modules
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "5433"))
ORIGINS = ["*"]

MARKDOWN_RESPONSE = """ # Hi!
I'm currently in development. I'll be ready to help you soon!

## Header

Bold: **bold text**

Italic: *italic text*

Code: `code`

- Go fuck yourself!

```
@dataclass
class ErrorContent:
    message: str
    code: int
```
"""
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    text: str
    reasoning: bool
    web_search: bool


@app.get("/")
def read_root():
    return {"message": "Chat-bot"}


@app.get(
    "/health",
    summary="Perform a simple health check",
    status_code=status.HTTP_200_OK,
    response_description="Return HTTP 200",
)
def health():
    """
    Performs a simple health check.
    """
    return {"status": "OK"}


@app.post("/api/chat/")
async def chat(query: Query):
    logger.info(query)
    try:
        # Your existing LLM logic here
        response = MARKDOWN_RESPONSE
        return JSONResponse({"response": response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate response: {str(e)}"})


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
