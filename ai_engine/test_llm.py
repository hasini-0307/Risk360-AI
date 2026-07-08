from ai_engine.llm.llm_client import LLMClient

client = LLMClient()

response = client.generate(
    system_prompt="You are a helpful banking AI.",
    user_prompt="Explain loan default prediction in one sentence."
)

print(response)