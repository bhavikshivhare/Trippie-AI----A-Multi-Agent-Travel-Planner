import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict,Annotated
import operator
import uuid
import asyncio

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq
#from tools.tavily_tool import tavily_search
#from tools.flight_tool import search_flights

from mcp_client import tavily_mcp_search,aviation_mcp_call, weather_mcp_search,forecast_mcp_search,extract_destraction


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key = GROQ_API_KEY
)


# STATE

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_result: str


#Flight Agent

# def flight_agent(state: TravelState):
#     query = state["user_query"]
#     flight_data = search_flights(query)

#     return{
#         "flight_results": flight_data,
#         "message": [AIMessage(content="Flight results fetched.")],
#         "llm_calls": state.get("llm_calls",0) + 1
#     }

FLIGHT_AGENT_PTOMPT = """ 
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}


Airline Information:
{airport_data}


Generate:

1.Likely departure airport
2.Likely arrival airport
3.Airlines serving this route
4.Typical flight duration
5.Estimated airfare range
6.Peak season pricing warning
7.Booking advice

Return concise travel guidance.
"""

def flight_agent(state: TravelState):
    print("\n Inside Flight Agent")

    query = state["user_query"]

    try:
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )
        airlines = asyncio.run(
                    aviation_mcp_call("list_airlines")
        )

        prompt = FLIGHT_AGENT_PTOMPT.format(
            query = query,
            airports_data = str(airports)[:3000],
            airlines_data=str(airlines)[:3000]
        )
    
        response = llm.invoke([
            SystemMessage(content="you are an expert."),
            HumanMessage(content=prompt)
            ])

        flight_data = response.content

    except Exception as e:

        flight_data = f"flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data, 
        "message": [
            AIMessage(
                content="Flight recommendations generated" 
            )
        ],
        "llm_calls": state.get("llm_calls",0) + 1
    }
    
#Hotel Agent

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    #hotel_results = tavily_search(query)
    hotel_results = asyncio.run(tavily_mcp_search(query))


    return{
        "hotel_results": hotel_results,
        "message": [AIMessage(content="Hotel information fetched.")],
        "llm_calls" : state.get("llm_calls",0) + 1
    }



#Weather_Agent

def weather_agent(state: TravelState):

    city = extract_destraction(state["user_query"])

    weather_data = asyncio.run(forecast_mcp_search(city))
    forecast_data = asyncio.run(forecast_mcp_search(city))

    return{
        "weather_result": f"""
        Current Weather:{weather_data}
        Forecast: {forecast_data}""",
        "message": [
            AIMessage(
                content = "weather information fetched"
            )
        ]
    }




# Itineary_agent


def itinerary_agent(state: TravelState):
    prompt = f"""

create a complete travel itinerary.

User Query: {state['user_query']}
Flight Results: {state['flight_results']}
Hotel Results: {state['hotel_results']}
Weather Results: {state['weather_result']}

make the itinerary pratical,budget aware, and easy to follow.
"""

    response  = llm.invoke([
        SystemMessage(content="You are an expert travel Planner"),
        HumanMessage(content = prompt)
    ])

    return{
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def final_agent(state: TravelState):

    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state["user_query"]}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Hotel Results: 
{state['weather_result']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1.Trip Summary
2.Flight Information
3.Hotel Suggestions
4.Weather Information
5.Day-by-Day Itinerary
6.Estimated Budget
7.Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# Build Graph:

graph = StateGraph(TravelState)

graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("weather_agent",weather_agent)
graph.add_node("itinerary_agent",itinerary_agent)
graph.add_node("final_agent", final_agent)


graph.add_edge(START,'flight_agent')
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","weather_agent")
graph.add_edge("weather_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent", END)



# PostgreSQL Checkpointer

DATABASE_URL = get_database_url()

conn = psycopg.connect(DATABASE_URL,autocommit = True,row_factory = dict_row)

checkpointer = PostgresSaver(conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)



# Function for FastAPI


def run_travel_agent(user_input: str,thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages" : [
                HumanMessage(content=user_input)
                ],
                "user_query": user_input,
                "flight_result": "",
                "hotel_result": "",
                "weather_result":"",
                "itinerary_result": "",
                "llm_calls": 0
        },
        config = config
    )


    final_answer = result["messages"][-1].content

    return {
        "thread_id" : thread_id,
        "answer": final_answer,
        "flight_result": result.get("flight_result",""),
        "hotel_result": result.get("hotel_result",""),
        "weather_result":result.get("weather_result"),
        "itinerary_result": result.get("itinerary_result",""),
        "llm_calls": result.get("llm_calls","")

    }


