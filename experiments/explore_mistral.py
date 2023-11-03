from ctransformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

template = """<|im_start|>system
You are a helpful, respectful and honest assistant.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<|im_start|>system
"""


def generate_prompt(template, question, context):
    """Generates a prompt for the LLM from the given question and context.

    Args:
      template:
      question: The question to ask the LLM.
      context: The context to provide to the LLM.

    Returns:
      A string containing the prompt for the LLM.

    Parameters
    ----------
    """

    prompt = template.format(context=context, question=question)
    return prompt


if __name__ == '__main__':
    # Set gpu_layers to the number of layers to offload to GPU.
    # Set to 0 if no GPU acceleration is available on your system.
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-OpenOrca-GGUF",
                                               model_file="mistral-7b-openorca.Q4_K_M.gguf",
                                               model_type="mistral",
                                               gpu_layers=0,
                                               hf=True)
    tokenizer = AutoTokenizer.from_pretrained(llm)

    # question_p = """What is the date for announcement"""
    # context_p = """ On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt
    # renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore."""

    question_p = """Create a regex to extract dates from logs in Python
    """
    context_p = """ """

    prompt = generate_prompt(template=template, question=question_p, context=context_p)
    inputs = tokenizer(text=prompt, return_tensors="pt").input_ids

    streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)
    _ = llm.generate(inputs, streamer=streamer, max_new_tokens=1000)
