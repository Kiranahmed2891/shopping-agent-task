
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool
import os
import requests
import rich

# 🔐 Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ API key not found. Add it to your .env file.")

# 🌐 Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 🔧 Tool: Product API call
@function_tool
def get_products():
    url = "https://template-03-api.vercel.app/api/products"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

# 🤖 Shopping Agent
agent = Agent(
    name="Shopping Agent",
    instructions="You are a helpful shopping agent. Show products and recommend one.",
    tools=[get_products],
)

# ✅ Run agent with correct arguments
result = Runner.run_sync(
    agent,
    "Which product do you want to buy?",
    run_config=config  # ✅ FIXED: Named argument required
)

# 📊 Show Output
rich.print(result.final_output)
