from pydantic import BaseModel

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


class ChatRequest(BaseModel):
    text: str
    rag: bool = False
    reasoning: bool = False
    web_search: bool = False
