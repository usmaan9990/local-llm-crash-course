# My first mdoel using hugging face libarary oraca mini 3b
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

# Now will let the mdoel complete the sentences

prompt = "she asked me 'What is the capital city of India, please reply me as short as much as possible', now i replied  "

print(f"{prompt} and OUTPUT IS === {llm(prompt)}")


# Now we need to build a chat machine

Question = "Which city is the capital of USA?"


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "Hi you are an AI assistant, who will give helpful and proper answer very clearly"
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


Aa = llm(get_prompt(Question), stream=True)  # If stream true it give resul token by token,
#  This is useful for real-time applications like chat UIs or command-line bots.


for word in Aa:
    print(word, flush=True, end="")
print()  # Out put wil come ins tream, like word by word u can see the flow

Aa = llm(get_prompt(Question))
print(Aa)  # heer you cant see that
