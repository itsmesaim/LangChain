from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv()


tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool for search on internet
    Args:
        query: the query to search for 
    returns: 
        the search result
    """
    print(f"searching for{query}")
    return tavily.search(query=query)

llm= ChatOpenAI(model="gpt-4o-mini")
tools= [search]
agent= create_agent(model= llm, tools=tools)

def main():
    print("Hello from my-learning!")
    result = agent.invoke({"messages":HumanMessage(content="search for best halal food option in dublin tallaht. i want to have a pizza.")})
    print(result)

if __name__ == "__main__":
    main()
