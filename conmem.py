from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "Hi you are an AI assistant, who will give helpful and proper answer very clearly"
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n"
    if history is not None:
        prompt += f"This is the conversation history before this chat : {' '.join(history)}. Now answer the question : "
    prompt += "\n### Response:\n"
    print(prompt)
    return prompt


histoRy = []
# 1st Que
Question = "Who is father of computer?"

answer = ""

Model = llm(get_prompt(Question), stream=True)

for i in Model:
    print(i, end="", flush=True)
    answer += i
print()

histoRy.append(answer)

# 2nd que
Question = "When he born?"

Model = llm(get_prompt(Question, histoRy), stream=True)

for i in Model:
    print(i, end="", flush=True)
    answer += i
print()

histoRy.append(answer)
