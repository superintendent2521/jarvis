#!/usr/bin/env python3
"""
OpenRouter Conversational AI Agent

A Python system that allows continuous conversation with AI models through OpenRouter,
with support for tool calling and dynamic system prompt modification.
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from openrouter_client import OpenRouterClient
from conversation import ConversationManager
from tool_manager import ToolManager


class OpenRouterAgent:
    """Main agent class that orchestrates the conversation with OpenRouter."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenRouterClient(api_key)
        self.conversation = ConversationManager()
        self.tool_manager = ToolManager()
        self.tool_instruction_message = self.tool_manager.get_tool_instruction_json()
        self.max_tool_iterations = 5
        self.log_file_path = Path(__file__).resolve().parent.parent / "conversation.log"
        self._initialize_log_file()

        print(f"Loaded {len(self.tool_manager.get_available_tools())} tools: {', '.join(self.tool_manager.get_available_tools())}")

    def _initialize_log_file(self):
        """Ensure session logging is set up."""
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        header = f"\n--- Session started {datetime.utcnow().isoformat()}Z ---\n"
        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(header)

    def _append_log(self, label: str, payload: str):
        """Append a timestamped entry to the log file."""
        timestamp = datetime.utcnow().isoformat()
        if not payload.endswith("\n"):
            payload = f"{payload}\n"
        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}Z] {label}\n")
            log_file.write(payload)



    def process_tool_calls(self, message) -> bool:
        '''Process any tool calls in the assistant's response. Returns True if any executed.'''
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return False

        executed_tool = False

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"\ndY\x15 Executing tool: {tool_name}")
            print(f"   Arguments: {tool_args}")

            log_record = {
                "tool_call_id": tool_call.id,
                "tool_name": tool_name,
                "arguments": tool_args
            }

            try:
                result = self.tool_manager.execute_tool(tool_name, tool_args)
                print(f"   Result: {result}")
                log_record["status"] = "success"
                log_record["result"] = result

                # Add tool result to conversation
                self.conversation.add_tool_result(tool_call.id, tool_name, result)
                executed_tool = True
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                print(f"   Error: {error_msg}")
                log_record["status"] = "error"
                log_record["error"] = error_msg
                self.conversation.add_tool_result(tool_call.id, tool_name, error_msg)
                executed_tool = True
            finally:
                self._append_log("Tool call", json.dumps(log_record, default=str, indent=2))

        return executed_tool

    def _tool_instruction_system_message(self):
        """Build the synthetic system message that explains tool JSON usage."""
        if not self.tool_instruction_message:
            return None

        return {
            "role": "system",
            "name": "tool_instructions",
            "content": (
                "When you decide a tool is required, respond with a tool call JSON payload "
                "that matches the following schema exactly:\n"
                f"{self.tool_instruction_message}"
            )
        }

    def _inject_tool_instructions(self, messages):
        """Ensure the model receives the JSON describing how to call each tool."""
        instruction_message = self._tool_instruction_system_message()
        if not instruction_message:
            return messages

        already_present = any(
            msg.get("role") == "system" and msg.get("name") == "tool_instructions"
            for msg in messages
        )
        if already_present:
            return messages

        injected_messages = []
        instruction_inserted = False

        for msg in messages:
            injected_messages.append(msg)
            if not instruction_inserted and msg.get("role") == "system":
                injected_messages.append(instruction_message)
                instruction_inserted = True

        if not instruction_inserted:
            injected_messages.insert(0, instruction_message)

        return injected_messages



    def _log_request(self, messages_to_send, tools, available_tool_count):
        '''Pretty-print the outgoing payload for debugging and capture it in the log.'''
        print(f"\n--> Sending to model ({self.client.default_model}) - prior messages only:")
        for i, msg in enumerate(messages_to_send):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  {i + 1}. {role}: {content}")

        log_payload = {
            "messages": messages_to_send,
            "available_tool_count": available_tool_count
        }
        self._append_log("Messages sent to model", json.dumps(log_payload, default=str, indent=2))
        print()

    def chat(self, user_input: str) -> str:
        """Process a single user message and return the assistant's response."""
        # Add user message to conversation
        self.conversation.add_user_message(user_input)

        # Get tool schemas for the request
        available_tools = self.tool_manager.get_available_tools()
        tools = self.tool_manager.get_tool_schemas() if available_tools else None

        iterations = 0
        final_response_text = ""

        try:
            while iterations < self.max_tool_iterations:
                iterations += 1
                messages_to_send = self._inject_tool_instructions(self.conversation.get_messages())
                self._log_request(messages_to_send, tools, len(available_tools))

                response = self.client.chat_completion(
                    messages=messages_to_send,
                    tools=tools
                )

                # Add assistant message to conversation
                self.conversation.add_assistant_message(response)

                if not getattr(response, "tool_calls", None):
                    final_response_text = response.content or ""
                    break

                tools_executed = self.process_tool_calls(response)

                if not tools_executed:
                    final_response_text = response.content or ""
                    break

            else:
                final_response_text = "Tool execution loop exceeded maximum iterations."

        except Exception as e:
            error_msg = f"Error communicating with OpenRouter: {str(e)}"
            print(f"❌ {error_msg}")
            final_response_text = error_msg

        return final_response_text

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.conversation.set_system_prompt(prompt)
        print(f"✅ System prompt updated: {prompt[:50]}...")

    def reset_conversation(self):
        """Reset the conversation while keeping the system prompt."""
        self.conversation.reset_conversation()
        print("🔄 Conversation reset")

    def show_help(self):
        """Display available commands."""
        print("\n📋 Available Commands:")
        print("  /help          - Show this help message")
        print("  /system <text> - Set new system prompt")
        print("  /reset         - Reset conversation")
        print("  /tools         - List available tools")
        print("  /model <name>  - Change model")
        print("  /quit          - Exit the program")
        print("  /status        - Show current status")

    def show_status(self):
        """Show current system status."""
        print(f"\n📊 Status:")
        print(f"  Model: {self.client.default_model}")
        print(f"  System Prompt: {self.conversation.get_system_prompt()}")
        print(f"  Tools: {len(self.tool_manager.get_available_tools())} loaded")
        print(f"  Conversation: {self.conversation.get_conversation_summary()}")

    def run_interactive_loop(self):
        """Run the main interactive conversation loop."""
        print("🤖 OpenRouter Conversational AI Agent")
        print("Type your message or use commands (type /help for commands)")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].split()[0].lower()
                    args = user_input[len(command) + 2:].strip()

                    if command == 'help':
                        self.show_help()
                    elif command == 'system':
                        if args:
                            self.set_system_prompt(args)
                        else:
                            print("❌ Please provide a system prompt: /system <prompt>")
                    elif command == 'reset':
                        self.reset_conversation()
                    elif command == 'tools':
                        tools = self.tool_manager.get_available_tools()
                        print(f"\n🔧 Available tools ({len(tools)}):")
                        for tool in tools:
                            print(f"  - {tool}")
                    elif command == 'model':
                        if args:
                            self.client.set_model(args)
                            print(f"✅ Model changed to: {args}")
                        else:
                            print("❌ Please provide a model name: /model <model_name>")
                    elif command == 'status':
                        self.show_status()
                    elif command == 'quit':
                        print("👋 Goodbye!")
                        break
                    else:
                        print(f"❌ Unknown command: {command}")
                else:
                    # Regular chat message
                    print("🤖 Assistant: ", end="", flush=True)
                    response = self.chat(user_input)
                    print(response)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main entry point."""
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Please set your OPENROUTER_API_KEY environment variable")
        print("   You can get an API key from: https://openrouter.ai/keys")
        sys.exit(1)

    try:
        agent = OpenRouterAgent(api_key)
        agent.run_interactive_loop()
    except Exception as e:
        print(f"❌ Failed to start agent: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
