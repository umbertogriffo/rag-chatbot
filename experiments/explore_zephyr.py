import time
from ctransformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Config
from transformers import TextStreamer

template = """<|system|> You are a helpful, respectful and honest assistant. Answer exactly in few words from the context.
Answer the question below from context below:
</s>
<|user|>
{context}
{question}</s>
<|assistant|>
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


"""
top_k="The top-k value to use for sampling."
top_p="The top-p value to use for sampling."
temperature="The temperature to use for sampling."
repetition_penalty="The repetition penalty to use for sampling."
last_n_tokens="The number of last tokens to use for repetition penalty."
seed="The seed value to use for sampling tokens."
max_new_tokens="The maximum number of new tokens to generate."
stop="A list of sequences to stop generation when encountered."
stream="Whether to stream the generated text."
reset="Whether to reset the model state before generating text."
batch_size="The batch size to use for evaluating tokens in a single prompt."
threads="The number of threads to use for evaluating tokens."
context_length="The maximum context length to use."
gpu_layers="The number of layers to run on GPU."
"""

if __name__ == '__main__':
    # Set gpu_layers to the number of layers to offload to GPU.
    # Set to 0 if no GPU acceleration is available on your system.
    config = Config(top_k=40,
                    top_p=0.95,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    last_n_tokens=64,
                    seed=-1,
                    batch_size=8,
                    threads=-1,
                    max_new_tokens=1024,
                    stop=None,
                    stream=False,
                    reset=True,
                    context_length=2048,
                    gpu_layers=50,
                    mmap=True,
                    mlock=False)
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GGUF",
                                               model_file="zephyr-7b-beta.Q4_K_M.gguf",
                                               model_type="mistral",
                                               config=AutoConfig(config=config),
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

    start_time = time.time()
    _ = llm.generate(inputs, streamer=streamer, max_new_tokens=1000)
    took = time.time() - start_time
    print(f"--- Took {took:.2f} seconds ---")
