import datetime
import os
from dotenv import load_dotenv

try:
    from src.tool_manager import tool
except ModuleNotFoundError:
    # Allows running when src isn't an importable package (e.g., python src/main.py)
    from tool_manager import tool

# Load environment variables
load_dotenv()


@tool(
    description="Get the current date and time",
    parameters={}
)
def get_current_datetime() -> str:
    """Return the current date and time as a string."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool(
    description="Convert text to uppercase",
    parameters={
        "text": {"type": "string", "description": "Text to convert"}
    }
)
def to_uppercase(text: str) -> str:
    """Convert the given text to uppercase."""
    return text.upper()


@tool(
    description="Count the number of words in text",
    parameters={
        "text": {"type": "string", "description": "Text to analyze"}
    }
)
def count_words(text: str) -> int:
    """Count the number of words in the given text."""
    words = text.split()
    return len(words)


@tool(
    description="Get current weather information for a city using OpenWeatherMap API",
    parameters={
        "city": {"type": "string", "description": "City name (e.g., 'London', 'New York')"}
    }
)
def get_weather(city: str) -> str:
    """Get current weather information for the specified city."""
    try:
        from skyfall import SkyFall, SkyFallError
        
        # Get API key from environment variables
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            return "Error: OPENWEATHERMAP_API_KEY not found in environment variables. Please add it to your .env file."
        
        # Initialize SkyFall client
        client = SkyFall(api_key=api_key)
        
        # Get weather data
        weather_report = client.weather(city)
        
        # Format the response
        result = f"Weather in {weather_report.city}:\n"
        result += f"Description: {weather_report.description}\n"
        result += f"Temperature: {weather_report.temperature_c}°C (feels like {weather_report.feels_like_c}°C)\n"
        result += f"Humidity: {weather_report.humidity}%"
        
        return result
        
    except SkyFallError as e:
        return f"Weather API error: {str(e)}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"
