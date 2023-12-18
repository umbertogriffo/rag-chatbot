import argparse
import sys
from pathlib import Path

import gradio as gr
from bot.model import Model
from bot.model_settings import get_model_setting, get_models
from helpers.log import get_logger

logger = get_logger(__name__)


def run(llm):
    with gr.Blocks() as demo:
        # gr.Image("/home/umberto/PycharmProjects/contextual-chatbot-gpt4all/images/download.png",
        #          height=10,
        #          width=200,
        #          scale=2)
        chatbot = gr.Chatbot(layout="bubble",
                             height=750,
                             label="Chatbot",
                             show_copy_button=True)
        msg = gr.Textbox(label="",
                         placeholder="Message Chatbot...",
                         autofocus=True,
                         show_copy_button=True)
        clear = gr.Button("Clear")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            logger.info(f"Question: {history[-1][0]}")
            prompt = llm.generate_qa_prompt(question=history[-1][0])
            bot_message = llm.start_answer_iterator_streamer(
                prompt, max_new_tokens=1000
            )
            logger.info(f"Response:  {bot_message}")
            history[-1][1] = ""
            for character in llm.streamer:
                history[-1][1] += character
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot")

    model_list = get_models()
    default_model = model_list[0]

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )

    return parser.parse_args()


def main(parameters):
    model_settings = get_model_setting(parameters.model)

    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    llm = Model(model_folder, model_settings)
    run(llm)


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
