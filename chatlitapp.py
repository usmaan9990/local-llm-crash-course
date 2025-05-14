import chainlit as cl
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    # now will set msg hsitory and retrive tat
    msg_history = cl.user_session.get("msg_history")
    # Now will create a empt msg and then send that to user
    msg = cl.Message(content="")
    await msg.send()  # Will send that msg to user

    op = ""
    prompt = get_prompt(message.content, msg_history)
    # each word in stream will do token and save
    for i in llm(prompt, stream=True):
        await msg.stream_token(i)
        op += i
    msg_history.append(op)

    await msg.update()


@cl.on_chat_start
async def on_chat_start():

    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )

    cl.user_session.set("msg_history", [])  # at bgng, setting an empty array to store hsitory

    await cl.Message("Model initialized. How can I help you?").send()
