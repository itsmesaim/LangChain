from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"   # Change to llama3.2 or gpt-4o-mini if you want

# ====================== HOTEL DATA ======================
HOTELS = [
    {"name": "Budget Bliss",      "price": 650,  "rating": 4.3, "food": "Excellent street food & thali"},
    {"name": "City Comfort Inn",  "price": 950,  "rating": 4.6, "food": "Great North Indian & Chinese"},
    {"name": "Midtown Haven",     "price": 1200, "rating": 4.4, "food": "Pure veg + South Indian"},
    {"name": "Luxury Vista",      "price": 2100, "rating": 4.8, "food": "Fine dining & multi-cuisine"},
    {"name": "Economy Stay",      "price": 450,  "rating": 4.1, "food": "Basic homemade food"},
    {"name": "Premium Palace",    "price": 2800, "rating": 4.9, "food": "5-star restaurant experience"},
]

# ====================== TOOLS ======================
@tool
def search_hotels(max_budget: int, min_rating: float = 4.0) -> str:
    """Search hotels within budget and minimum rating.
    Always use this tool first when user mentions budget or rating."""
    print(f"    >> Executing search_hotels(max_budget={max_budget}, min_rating={min_rating})")
    
    results = [
        hotel for hotel in HOTELS
        if hotel["price"] <= max_budget and hotel["rating"] >= min_rating
    ]
    
    if not results:
        return "No hotels found matching your criteria."
    
    response = "Here are the matching hotels:\n"
    for h in results:
        response += f"• {h['name']} | ₹{h['price']} | ⭐{h['rating']} | Food: {h['food']}\n"
    return response.strip()

@tool
def get_food_recommendation(hotel_name: str) -> str:
    """Get detailed food recommendation for a specific hotel."""
    print(f"    >> Executing get_food_recommendation(hotel_name='{hotel_name}')")
    
    hotel = next((h for h in HOTELS if h["name"].lower() == hotel_name.lower()), None)
    if hotel:
        return f"Food speciality at {hotel['name']}: {hotel['food']}"
    return "Hotel not found."

# ====================== AGENT LOOP ======================
@traceable(name="Hotel Recommendation Agent")
def run_agent(question: str):
    tools = [search_hotels, get_food_recommendation]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 70)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful hotel booking assistant.\n"
                "STRICT RULES:\n"
                "1. ALWAYS use 'search_hotels' tool first when user asks about budget or rating.\n"
                "2. For budget queries like 'under 1000' or 'under 1500', pass the number as max_budget.\n"
                "3. Only call get_food_recommendation AFTER you have a hotel name from search_hotels.\n"
                "4. Never guess prices or ratings — always use tools.\n"
                "5. Give final answer only when you have all required information."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached")
    return None


if __name__ == "__main__":
    print("Hello Hotel Recommendation Agent!\n")
    
    # Test queries
    # result = run_agent("Best food under budget 1000")
    result = run_agent("List hotels with good rating under 1500")
    # result = run_agent("What is the price and food at Luxury Vista?")