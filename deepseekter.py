from llama_cpp import Llama

# Load DeepSeek Coder model
llm = Llama(
    model_path="/home/vscode/.cache/huggingface/hub/models--TheBloke--deepseek-coder-1.3b-base-GGUF/snapshots/ec89dd32c0a17bd56d27eccca7d6e4195c0f615d/deepseek-coder-1.3b-base.Q5_K_S.gguf",
    n_ctx=2048
)

# du -h --max-depth=1 ~ | sort -hr

# Just a plain question — no special formatting
question = "Who is a famous actor in USA?"

# Call the model
output = llm(question, max_tokens=256)

# Print the model's response
# print(output["choices"][0]["text"].strip())


def get_prompt(instruction: str):
    global llm
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt

Q = "WHo is Ronaldo?"

A = llm(get_prompt(Q), max_tokens=256)
print(A["choices"][0]["text"].strip())
