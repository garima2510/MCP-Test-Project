from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import json

nest_asyncio.apply()

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.session: ClientSession = None

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.available_tools: List[dict] = []

    def convert_mcp_tools_to_openai_format(self, mcp_tools):
        """Convert MCP tool format to OpenAI function calling format"""
        openai_functions = []
        for tool in mcp_tools:
            function_def = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_functions.append(function_def)
        return openai_functions

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]

        # Convert MCP tools to OpenAI format
        openai_tools = self.convert_mcp_tools_to_openai_format(self.available_tools)

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
            max_tokens=2048
        )

        process_query = True
        while process_query:
            message = response.choices[0].message

            # Handle text response
            if message.content:
                print(message.content)
                messages.append({'role': 'assistant', 'content': message.content})

            # Handle tool calls
            if message.tool_calls:
                messages.append({
                    'role': 'assistant', 
                    'content': message.content,
                    'tool_calls': message.tool_calls
                })

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id

                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # Call MCP tool through the client session
                    try:
                        result = await self.session.call_tool(tool_name, arguments=tool_args)

                        # Convert result to string if it's not already
                        if hasattr(result, 'content'):
                            result_content = str(result.content)
                        else:
                            result_content = str(result)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result_content
                        })

                    except Exception as e:
                        print(f"Error calling tool {tool_name}: {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": f"Error: {str(e)}"
                        })

                # Get next response from OpenAI
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                    max_tokens=2048
                )

            else:
                # No more tool calls, we're done
                process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot with Azure OpenAI Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",  # Changed from "uv" to "python"
            args=["research_server.py"],  # Path to your MCP server
            env=None,  # Optional environment variables
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    # Initialize the connection
                    await session.initialize()

                    # List available tools
                    response = await session.list_tools()

                    tools = response.tools
                    print("\nConnected to server with tools:", [tool.name for tool in tools])

                    self.available_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    } for tool in response.tools]

                    await self.chat_loop()

        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            print("Make sure your MCP server is working and the path is correct.")

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())
