from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q8_0.gguf"
)

Que = "Who is famous actor in south india?"


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "Hi you are an AI assistant, who will give helpful and proper answer very clearly"
    prompt = f"[INST] <<SYS>>{system}<</SYS>>{instruction}[/INST]"
    print(f"Prompt created {prompt}")
    return prompt


Aa = llm(get_prompt(Que))
print(Aa)
