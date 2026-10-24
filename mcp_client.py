import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key = GROQ_API_KEY
)


client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
        "Aviationstack": {
        "transport": "stdio",
        "command": "uvx",
        "args": [
            "aviationstack-mcp"
        ],
        "env": {
            "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY
        }
        },
        "weather":{
            "transport": "stdio",
            "command": r"C:\Users\bhavi\anaconda3\envs\travel\python.exe",
            "args": [
                r"D:\multi agent project\Trippie-AI----A-Multi-Agent-Travel-Planner\custom_weather_mcp_server.py"
            ],
            "env" :{ "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY}
        }
    }
)


async def get_all_tools():
    tools = await client.get_tools()
    print("\n Available MCP Tools")

    for tool in tools:
        print(tool.name)

tavily_tool = None
aviation_tools = {}

async def initialize_mcp():

    global tavily_tool
    global aviation_tools

    if tavily_tool is not None and aviation_tools:
        return

    tools = await client.get_tools()

    for tool in tools:
        print(tool.name)

    tavily_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    aviation_tools = {
            tool.name: tool
            for tool in tools
            if tool.name != "tavily_search"
    }

async def tavily_mcp_search(query: str):
    await initialize_mcp()
    result = await tavily_tool.ainvoke(
        {
            "query": query
        }
    )
    return result


async def aviation_mcp_call(tool_name:str,tool_args:dict = None):
    tools = await client.get_tools()

    tool = next(t for t in tools if t.name == tool_name)
    result = await tool.ainvoke(tool_args or {})
    return result



weather_tool = None
forecast_tool = None

async def initialize_weather_tools():

    global weather_tool,forecast_tool

    if weather_tool is not None:
        return

    tools = await client.get_tools()

    weather_tool = next(t for t in tools if t.name == "get_current_weather")

    forecast_tool = next(t for t in tools if t.name == "get_forecast")


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    return await weather_tool.ainvoke({city:city})

async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    return await forecast_tool.ainvoke({city:city})


def extract_destraction(query:str):

    prompt = f"""
Extract only the destination city or country.

Query:
{query}

Return only destination name.
"""

    response = llm.invoke(prompt)

    return response.content.strip()