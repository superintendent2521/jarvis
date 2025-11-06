import os
import importlib.util
import inspect
import json
from typing import Dict, List, Any, Callable, Optional
from pathlib import Path


class ToolManager:
    """Manages loading and executing tools from the src/tools/ directory."""

    def __init__(self, tools_dir: str = None):
        if tools_dir is None:
            # Default to tools directory relative to this file's location
            current_file = Path(__file__).resolve()
            self.tools_dir = current_file.parent / "tools"
        else:
            self.tools_dir = Path(tools_dir)

        self.tools = {}
        self.tool_schemas = []
        self.load_tools()

    def load_tools(self):
        """Load all Python files from tools directory and extract tool functions."""
        print(f"Loading tools from directory: {self.tools_dir.absolute()}")
        if not self.tools_dir.exists():
            print(f"❌ Tools directory {self.tools_dir} does not exist")
            return

        py_files = list(self.tools_dir.glob("*.py"))
        print(f"Found {len(py_files)} Python files: {[f.name for f in py_files]}")

        for file_path in py_files:
            if file_path.name.startswith("__"):
                continue

            self._load_tool_from_file(file_path)

    def _load_tool_from_file(self, file_path: Path):
        """Load tools from a single Python file."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find functions with tool metadata
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, 'tool_schema'):
                    self.tools[name] = obj
                    self.tool_schemas.append(obj.tool_schema)
                    print(f"Loaded tool: {name}")

        except Exception as e:
            print(f"Error loading tool from {file_path}: {e}")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get the JSON schemas for all loaded tools."""
        # Return a deep copy so downstream mutations don't affect the source of truth
        return json.loads(json.dumps(self.tool_schemas))

    def get_tool_instruction_payload(self) -> List[Dict[str, Any]]:
        """Build a simplified payload describing each tool for system instructions."""
        payload = []
        for schema in self.tool_schemas:
            function_def = schema.get("function", {})
            params = function_def.get("parameters", {})
            payload.append({
                "name": function_def.get("name", ""),
                "description": function_def.get("description", ""),
                "arguments_schema": json.loads(json.dumps(params))
            })
        return payload

    def get_tool_instruction_json(self) -> Optional[str]:
        """Return a JSON string that explains how to call each tool."""
        payload = self.get_tool_instruction_payload()
        if not payload:
            return None

        instruction = {
            "tool_call_format": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "One of the tool names listed in tools[].name"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "JSON object that matches the selected tool's arguments_schema"
                    }
                },
                "required": ["tool_name", "arguments"]
            },
            "tools": payload
        }
        return json.dumps(instruction, indent=2)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")

        tool_func = self.tools[tool_name]
        try:
            return tool_func(**arguments)
        except Exception as e:
            return f"Error executing tool {tool_name}: {str(e)}"

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())


def tool(description: str = "", parameters: Dict[str, Any] = None):
    """
    Decorator to mark a function as a tool and generate its JSON schema.

    Usage:
        @tool(description="Calculate sum", parameters={"a": {"type": "number"}, "b": {"type": "number"}})
        def add(a: float, b: float) -> float:
            return a + b
    """
    def decorator(func):
        # Generate JSON schema
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters or {},
                    "required": list(parameters.keys()) if parameters else []
                }
            }
        }

        func.tool_schema = schema
        return func
    return decorator
