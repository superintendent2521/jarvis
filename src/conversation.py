from typing import List, Dict, Any, Optional
from openai.types.chat import ChatCompletionMessage


class ConversationManager:
    """Manages conversation history and system prompts."""

    def __init__(self, system_prompt: str = "You are a helpful AI assistant with access to various tools."):
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []
        self.reset_conversation()

    def reset_conversation(self):
        """Reset the conversation with the current system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def set_system_prompt(self, prompt: str):
        """Update the system prompt and reset conversation."""
        self.system_prompt = prompt
        self.reset_conversation()

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, message: ChatCompletionMessage):
        """Add an assistant message to the conversation."""
        message_dict = {
            "role": message.role,
            "content": message.content,
        }

        # Add tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                }
                for tool_call in message.tool_calls
            ]

        self.messages.append(message_dict)

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: Any):
        """Add a tool execution result to the conversation."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": str(result)
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the current conversation messages."""
        return self.messages.copy()

    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self.system_prompt

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        total_messages = len(self.messages) - 1  # Exclude system message
        return f"Conversation with {total_messages} messages. System: {self.system_prompt[:50]}..."
