from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from flask import Flask, render_template, request

# Download model if not already present
model_path = hf_hub_download(
    repo_id="TheBloke/deepseek-coder-1.3b-base-GGUF",
    filename="deepseek-coder-1.3b-base.Q8_0.gguf"
)

# Load model
llm = Llama(
    model_path=model_path,
    n_ctx=4048,
    n_threads=8,
    n_gpu_layers=10
)

# Initialize Flask app
app = Flask(__name__)

# Prompt format helper


def get_prompt(instruction: str) -> str:
    prompt = (
        "Hi you are an AI assistant, who will give helpful and proper answers clearly. "
        f"### Instruction:\n{instruction}\n### Response:\n"
    )
    return prompt

# Main route


@app.route("/", methods=["GET", "POST"])
def index():
    reply = ""
    if request.method == "POST":
        try:
            que = request.form.get("prompt", "").strip()
            if not que:
                return render_template("index.html", response="❌ Empty input! Please enter a prompt.")

            prompt = get_prompt(que)
            print("🚀 Prompt sent to model:\n", prompt)
            model_output = llm(prompt, max_tokens=512)
            reply = model_output["choices"][0]["text"].strip()
        except Exception as e:
            print("🔥 Error occurred:", e)
            reply = "⚠️ Something went wrong. Check the terminal."

    return render_template("index.html", response=reply)


if __name__ == "__main__":
    app.run(debug=True)
