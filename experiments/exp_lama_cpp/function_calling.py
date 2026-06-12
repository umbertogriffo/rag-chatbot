import json
from pathlib import Path

from llm_providers.llamacpp_client import LlamaCppClient
from schemas.model import ModelSettings

LLAMA_SERVER_BASE_URL = "http://localhost:8080"


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather in a given location.

    location (str): The city and state, e.g. Madrid, Barcelona
    unit (str): The unit. It can take two values; "celsius", "fahrenheit"
    """
    if location.lower == "rome":
        return json.dumps({"location": "Rome", "temperature": "22", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "madrid" in location.lower():
        return json.dumps({"location": "Madrid", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def search_text(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Conducts a search on DuckDuckGo and returns the top max_results results
    """
    return [{"title": "Adobe", "url": "www.adobe.com"}]


# Configuration object used to instruct functionary model about the tools at its disposal.
TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# Tool map - when functionary chooses a tool, run the corresponding function from this map
TOOLS_MAP = {
    "get_current_weather": get_current_weather,
}

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parents[2]
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    print(get_current_weather(location="Madrid", unit="celsius"))
    print(search_text(query="Adobe"))

    model_settings = ModelSettings(
        url="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        name="Meta-Llama-3.1-8B-Instruct-Q4_K_M",
        file_name="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        reasoning_start_tag="<think>",
        reasoning_stop_tag="</think>",
        system_template="",
        reasoning=False,
    )

    llm = LlamaCppClient(
        base_url=LLAMA_SERVER_BASE_URL,
        model_folder=model_folder,
        model_settings=model_settings,
        timeout=300,
    )

    tools = llm.retrieve_tools(prompt="Tell me something about Rome", tools=TOOLS_CONFIG, tool_choice=None)
    print(tools)

    tools = llm.retrieve_tools(prompt="What's the current temperature in Rome, in Celsius?", tools=TOOLS_CONFIG)
    print(tools)

    tools = llm.retrieve_tools(
        prompt="What's the current temperature in Madrid, in Celsius?",
        max_new_tokens=256,
        tools=TOOLS_CONFIG,
        tool_choice="get_current_weather",
    )
    print(tools)

    if len(tools) > 0:
        function_name = tools[0]["function"]["name"]
        function_args = json.loads(tools[0]["function"]["arguments"])
        func_to_call = TOOLS_MAP.get(function_name, None)
        function_response = func_to_call(**function_args)
        print(f"Tool response: {function_response}")
        prompt_with_function_response = llm.generate_ctx_prompt(
            question="What's the current temperature in Madrid, in Celsius?", context=function_response
        )

        stream = llm.start_answer_iterator_streamer(
            prompt=prompt_with_function_response,
            max_new_tokens=256,
        )
        for output in stream:
            print(output.choices[0].delta.content or "", end="", flush=True)
