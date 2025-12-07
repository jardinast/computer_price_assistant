"""
llm_advisor.py - GPT-4 Powered Laptop Advisor

This module provides the LLM integration for the chat-based laptop advisor.
Uses OpenAI's function calling to extract specifications and provide recommendations.
"""

import json
from typing import Dict, Any, List, Optional, Generator
from openai import OpenAI

from .backend_api import predecir_precio, explicar_prediccion, obtener_campos_disponibles
from .defaults import USE_CASE_PROFILES, get_cpu_benchmark_defaults, get_gpu_benchmark_defaults

# System prompt for the advisor
SYSTEM_PROMPT = """You are a friendly and knowledgeable laptop advisor helping users find the perfect computer for their needs.

## Your Approach:
1. **Greet** the user warmly and ask what they'll primarily use the laptop for
2. **Understand** their specific needs based on their use case:
   - Gaming: Ask about game types (AAA, esports), settings preferences
   - Work: Ask about software needs (Office, CAD, development)
   - Creative: Ask about tasks (video editing, 3D rendering, photo editing)
   - Student: Ask about coursework needs, portability
3. **Gather preferences**: Brand, screen size, portability, budget
4. **Recommend** specific configurations with price estimates
5. **Explain** why the specs are suitable for their needs

## Key Guidelines:
- Be conversational and friendly, not technical jargon-heavy
- Ask one or two questions at a time, don't overwhelm
- When you have enough info, use the tools to get price predictions
- Always show price ranges (±20%) since predictions have some uncertainty
- Offer to adjust recommendations if the user has concerns

## Price Model Info:
- Trained on European laptop prices (EUR)
- Price range: €0 - €3,605
- Main price drivers: CPU performance, GPU performance, RAM, brand
- Accuracy: approximately ±20%

## Use Case Profiles (typical specs):
- Gaming: i7/Ryzen 7, RTX 4060+, 16GB RAM, 144Hz screen
- Work: i5/Ryzen 5, integrated GPU, 16GB RAM, lightweight
- Creative: i7/Ryzen 7, RTX GPU or Apple M-series, 32GB RAM
- Student: i5, integrated GPU, 8-16GB RAM, portable
"""

# Tools for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price_prediction",
            "description": "Get a price prediction for specific laptop specifications. Use this when you have gathered enough information about what the user needs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "use_case": {
                        "type": "string",
                        "enum": ["gaming", "work", "creative", "student", "general"],
                        "description": "The primary use case for the laptop"
                    },
                    "cpu_family": {
                        "type": "string",
                        "description": "CPU type, e.g., 'core i5', 'core i7', 'ryzen 5', 'ryzen 7', 'm2', 'm3'"
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type, e.g., 'integrated', 'rtx 4060', 'rtx 4070', 'rx 6600'"
                    },
                    "ram_gb": {
                        "type": "integer",
                        "description": "RAM size in GB"
                    },
                    "ssd_gb": {
                        "type": "integer",
                        "description": "Storage size in GB"
                    },
                    "brand": {
                        "type": "string",
                        "description": "Laptop brand if specified, e.g., 'asus', 'dell', 'apple', 'lenovo'"
                    },
                    "screen_inches": {
                        "type": "number",
                        "description": "Screen size in inches if specified"
                    }
                },
                "required": ["use_case", "cpu_family", "ram_gb"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_configurations",
            "description": "Compare multiple laptop configurations to help the user choose. Use when the user is deciding between options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "configurations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "A descriptive name for this config"},
                                "use_case": {"type": "string"},
                                "cpu_family": {"type": "string"},
                                "gpu_type": {"type": "string"},
                                "ram_gb": {"type": "integer"},
                                "ssd_gb": {"type": "integer"}
                            }
                        },
                        "description": "List of configurations to compare"
                    }
                },
                "required": ["configurations"]
            }
        }
    }
]


def _build_prediction_inputs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tool parameters to backend API format."""
    inputs = {}

    # Map parameters
    if 'cpu_family' in params:
        inputs['cpu_family'] = params['cpu_family'].lower()

    if 'gpu_type' in params:
        gpu_type = params['gpu_type'].lower()
        if gpu_type == 'integrated' or gpu_type == 'none':
            inputs['gpu_is_integrated'] = True
            inputs['gpu_series'] = 'integrated'
        else:
            inputs['gpu_is_integrated'] = False
            inputs['gpu_series'] = gpu_type

    if 'ram_gb' in params:
        inputs['_ram_gb'] = params['ram_gb']

    if 'ssd_gb' in params:
        inputs['_ssd_gb'] = params['ssd_gb']

    if 'brand' in params:
        inputs['_brand'] = params['brand'].lower()

    if 'screen_inches' in params:
        inputs['_tamano_pantalla_pulgadas'] = params['screen_inches']

    return inputs


def _handle_get_price_prediction(params: Dict[str, Any]) -> str:
    """Handle the get_price_prediction tool call."""
    try:
        inputs = _build_prediction_inputs(params)
        use_case = params.get('use_case', 'general')

        result = explicar_prediccion(inputs, use_case=use_case)

        response = f"""
**Estimated Price: €{result['prediccion']:,.0f}**
(Range: €{result['prediccion_min']:,.0f} - €{result['prediccion_max']:,.0f})

**Configuration:**
- CPU: {params.get('cpu_family', 'Not specified').title()}
- GPU: {params.get('gpu_type', 'Integrated').title()}
- RAM: {params.get('ram_gb', 16)}GB
- Storage: {params.get('ssd_gb', 512)}GB
{f"- Brand: {params.get('brand', '').title()}" if params.get('brand') else ""}

**Top Price Factors:**
"""
        for feat in result['top_features'][:3]:
            response += f"- {feat['readable_name']}: {feat['importance_pct']:.1f}%\n"

        return response

    except Exception as e:
        return f"I encountered an error getting the price prediction: {str(e)}. Let me try with adjusted parameters."


def _handle_compare_configurations(params: Dict[str, Any]) -> str:
    """Handle the compare_configurations tool call."""
    try:
        configs = params.get('configurations', [])
        if not configs:
            return "No configurations provided for comparison."

        response = "**Configuration Comparison:**\n\n"

        for i, config in enumerate(configs, 1):
            inputs = _build_prediction_inputs(config)
            use_case = config.get('use_case', 'general')
            name = config.get('name', f'Option {i}')

            result = explicar_prediccion(inputs, use_case=use_case)

            response += f"**{name}**\n"
            response += f"- Price: €{result['prediccion']:,.0f} (€{result['prediccion_min']:,.0f} - €{result['prediccion_max']:,.0f})\n"
            response += f"- CPU: {config.get('cpu_family', 'Not specified').title()}\n"
            response += f"- GPU: {config.get('gpu_type', 'Integrated').title()}\n"
            response += f"- RAM: {config.get('ram_gb', 16)}GB\n\n"

        return response

    except Exception as e:
        return f"Error comparing configurations: {str(e)}"


def process_tool_calls(tool_calls: list) -> str:
    """Process tool calls and return results."""
    results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "get_price_prediction":
            result = _handle_get_price_prediction(arguments)
        elif function_name == "compare_configurations":
            result = _handle_compare_configurations(arguments)
        else:
            result = f"Unknown function: {function_name}"

        results.append(result)

    return "\n\n".join(results)


class ChatAdvisor:
    """Chat-based laptop advisor using OpenAI GPT-4."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the chat advisor.

        Parameters
        ----------
        api_key : str
            OpenAI API key
        model : str
            Model to use (default: gpt-4o)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.system_message = {"role": "system", "content": SYSTEM_PROMPT}

    def reset(self):
        """Reset the conversation history."""
        self.messages = []

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.

        Parameters
        ----------
        user_message : str
            The user's message

        Returns
        -------
        str
            The assistant's response
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[self.system_message] + self.messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Handle tool calls if any
        if assistant_message.tool_calls:
            # Process tool calls
            tool_results = process_tool_calls(assistant_message.tool_calls)

            # Add assistant's tool call message
            self.messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Add tool results
            for i, tool_call in enumerate(assistant_message.tool_calls):
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_results if i == 0 else ""  # Only add results once
                })

            # Get final response with tool results
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=[self.system_message] + self.messages,
            )

            final_message = final_response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": final_message})
            return final_message

        else:
            # No tool calls, just return the response
            content = assistant_message.content or ""
            self.messages.append({"role": "assistant", "content": content})
            return content

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Send a message and stream the response.

        Parameters
        ----------
        user_message : str
            The user's message

        Yields
        ------
        str
            Chunks of the assistant's response
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Make API call with streaming
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[self.system_message] + self.messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=True,
        )

        collected_content = ""
        collected_tool_calls = []
        current_tool_call = None

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Collect content
            if delta.content:
                collected_content += delta.content
                yield delta.content

            # Collect tool calls
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    if tc_chunk.index is not None:
                        while len(collected_tool_calls) <= tc_chunk.index:
                            collected_tool_calls.append({
                                "id": "",
                                "function": {"name": "", "arguments": ""}
                            })

                        if tc_chunk.id:
                            collected_tool_calls[tc_chunk.index]["id"] = tc_chunk.id
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                collected_tool_calls[tc_chunk.index]["function"]["name"] = tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                collected_tool_calls[tc_chunk.index]["function"]["arguments"] += tc_chunk.function.arguments

        # Process tool calls if any
        if collected_tool_calls and collected_tool_calls[0]["function"]["name"]:
            # Convert to proper format for processing
            class MockToolCall:
                def __init__(self, tc_dict):
                    self.id = tc_dict["id"]
                    self.function = type('obj', (object,), {
                        'name': tc_dict["function"]["name"],
                        'arguments': tc_dict["function"]["arguments"]
                    })()

            mock_calls = [MockToolCall(tc) for tc in collected_tool_calls]
            tool_results = process_tool_calls(mock_calls)

            # Add to messages and yield results
            self.messages.append({"role": "assistant", "content": collected_content})
            yield "\n\n"
            yield tool_results

            # Store final content
            self.messages.append({"role": "assistant", "content": tool_results})
        else:
            self.messages.append({"role": "assistant", "content": collected_content})

    def get_initial_greeting(self) -> str:
        """Get the initial greeting message."""
        return """Hi! I'm your laptop advisor. I'm here to help you find the perfect computer for your needs.

**What will you mainly use this laptop for?**

- 🎮 **Gaming** - Playing video games
- 💼 **Work** - Business and productivity
- 🎨 **Creative** - Video editing, design, 3D work
- 📚 **Student** - School and studying
- 🏠 **General** - Everyday tasks

Just tell me about your needs, and I'll help you find the right specs and price range!"""
