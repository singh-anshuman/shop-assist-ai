"""
OpenAI Function Calling API - Python Proof of Concept
This example demonstrates how to use OpenAI's function calling capabilities
to create a weather information assistant
"""

import os
import json
from openai import OpenAI, api_key

# Initialize the OpenAI client with your API key

client = OpenAI(
    api_key=open("OPENAI_API_Key.txt", "r").read().strip()
)

# Define the functions that can be called by the model
def get_weather(location, unit="celsius"):
    """
    Function to get weather information for a location.
    In a real application, this would call a weather API.
    For this POC, we return mock data.
    """
    print(f"Getting weather for {location} in {unit}")
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Sunny",
        "humidity": 45,
        "wind_speed": 10,
        "forecast": [
            {"day": "Tomorrow", "condition": "Partly Cloudy", "high_temp": 24 if unit == "celsius" else 75},
            {"day": "Day After", "condition": "Rainy", "high_temp": 18 if unit == "celsius" else 64}
        ]
    }

# Map function names to their implementations
available_functions = {
    "get_weather": get_weather
}

# Define the function specifications in the format expected by OpenAI
function_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather and forecast for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name and optionally state/country, e.g., 'San Francisco, CA' or 'Paris, France'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature to use"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def process_user_query(user_query):
    """Process a user query using OpenAI's function calling capabilities"""
    try:
        print(f"User query: {user_query}")
        
        # Step 1: Send the user query to the API with our function definitions
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Or another model that supports function calling
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can provide weather information."},
                {"role": "user", "content": user_query}
            ],
            tools=function_definitions,
            tool_choice="auto",  # Let the model decide when to call functions
        )
        
        # Get the response message
        response_message = response.choices[0].message
        
        # Step 2: Check if the model wants to call a function
        if response_message.tool_calls:
            # Get the first tool call (there might be multiple in some cases)
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Function called: {function_name}")
            print(f"Arguments: {function_args}")
            
            # Step 3: Call the function
            if function_name in available_functions:
                # Extract arguments and call the function
                location = function_args.get("location")
                unit = function_args.get("unit", "celsius")
                function_response = available_functions[function_name](location, unit)
                
                # Step 4: Send the function result back to the API for a final response
                second_response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that can provide weather information."},
                        {"role": "user", "content": user_query},
                        response_message,  # Include the model's function call
                        {
                            "role": "tool", 
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(function_response)  # Include the function result
                        }
                    ]
                )
                
                # Step 5: Return the final response to the user
                return second_response.choices[0].message.content
            else:
                return f"Error: Function {function_name} not found"
        else:
            # If no function was called, return the model's response directly
            return response_message.content
    except Exception as e:
        print(f"Error: {e}")
        return f"Error processing your query: {str(e)}"

def main():
    """Run example queries to demonstrate function calling"""
    queries = [
        "What's the weather like in London?",
        "Tell me the weather forecast for Tokyo in fahrenheit",
        "What should I wear in Paris today?"
    ]
    
    for query in queries:
        response = process_user_query(query)
        print("\nQuery:", query)
        print("Response:", response)
        print("-" * 50)

if __name__ == "__main__":
    main()