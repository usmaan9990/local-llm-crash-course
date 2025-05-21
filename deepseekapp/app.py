from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from flask import Flask, redirect, url_for, render_template, request

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

app = Flask(__name__)


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    prompt = "Hi you are an AI assistant, who will give helpful and proper answer very clearly. "
    if history is not None:
        prompt += f"This is conversation history before this chat : {' '.join(history)}. Now answer the question: "
    prompt += f"### Instruction:{instruction}\n### Response:\n"
    return prompt


# @app.route("/", methods=['POST', 'GET'])
# def submit():
#     reply = ""
#     history = ""
#     if request.method == 'POST':
#         que = request.form["prompt"]
#         prompt = get_prompt(que, history)
#         model = llm(prompt, max_tokens=512)
#         reply = model["choices"][0]["text"].strip()
#         history.append(reply)

#     return render_template("index.html", response=reply)


@app.route("/", methods=["GET", "POST"])
def index():
    reply = ""
    history = []
    try:
        if request.method == 'POST':
            que = request.form["prompt"]
            prompt = get_prompt(que, history)
            model = llm(prompt, max_tokens=512)
            reply = model["choices"][0]["text"].strip()
            history.append(f"User: {que}")
            history.append(f"AI: {reply}")
    except Exception as e:
        print("🔥 Error occurred:", e)  # Print the real issue
        reply = "Something went wrong. Check terminal for details."

    return render_template("index.html", response=reply)


if __name__ == "__main__":
    app.run(debug=True)
