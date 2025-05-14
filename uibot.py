from ctransformers import AutoModelForCausalLM
import chainlit as cl


# llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "Hi you are an AI assistant, who will give helpful and proper answer very clearly"
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n"
    if len(history) > 0:
        prompt += f"This is the conversation history before this chat : {' '.join(history)}. Now answer the question : "
    prompt += "\n### Response:\n"
    return prompt


# My MODEL UI

@cl.on_message
async def on_message(message: cl.Message):
    # Now will do msg hsitory
    msg_history = cl.user_session.get("msg_history")

    msgg = cl.Message(content="")
    await msgg.send()

    prompt = get_prompt(message.content, msg_history)
    response = ""
    for i in llm(prompt, stream=True):  # here we do streaming
        await msgg.stream_token(i)
        response += i

    msg_history.append(response)
    await msgg.update()

# In start of chatbot what bot need to do


@cl.on_chat_start
def on_chat_start():
    global llm
    llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
    cl.user_session.set("message_history", [])


# Now will do streaming to taht we edit on message so, check on_mnsg
