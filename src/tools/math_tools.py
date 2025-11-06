try:
    from src.tool_manager import tool
except ModuleNotFoundError:
    # Allows running when src isn't an importable package (e.g., python src/main.py)
    from tool_manager import tool


@tool(
    description="Add two numbers together",
    parameters={
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    }
)
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool(
    description="Multiply two numbers",
    parameters={
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    }
)
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@tool(
    description="Calculate the power of a number",
    parameters={
        "base": {"type": "number", "description": "Base number"},
        "exponent": {"type": "number", "description": "Exponent"}
    }
)
def power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return base ** exponent
