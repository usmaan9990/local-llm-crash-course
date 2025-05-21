from huggingface_hub import hf_hub_download
from llama_cpp import Llama


model_path = hf_hub_download(
    repo_id="TheBloke/deepseek-coder-1.3b-base-GGUF",
    filename="deepseek-coder-1.3b-base.Q8_0.gguf"
)

llm = Llama(
    model_path=model_path,
    n_ctx=4048,
    n_threads=8,
    n_gpu_layers=10
)


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    prompt = "Hi you are an AI assistant, who will give helpful and proper answer very clearly. "
    if history is not None:
        prompt += f"This is conversation history before this chat : {' '.join(history)}. Now answer the question: "
    prompt += f"### Instruction:{instruction}\n### Response:\n"
    print(prompt)
    return prompt


histoRy = []
# 1st Que
Question = "What is Harry Potter?"

answer = ""

Model = llm(get_prompt(Question), stream=True, max_tokens=256)

for i in Model:
    tok = i["choices"][0]["text"]
    print(tok, end="", flush=True)
    answer += tok
print()

# Update this line:
histoRy.append(f"Q: {Question}\nA: {answer.strip()}")


# histoRy.append(answer)

# 2nd que
Question = "Tell me only the name and release year of the first Harry Potter movie. No extra explanation."

Model = llm(get_prompt(Question, histoRy), stream=True, max_tokens=512)

for i in Model:
    tok = i["choices"][0]["text"]
    print(tok, end="", flush=True)
    answer += tok
print()

histoRy.append(answer)
