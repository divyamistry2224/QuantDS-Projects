import os

from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv()

print("Alpaca Key:", os.getenv("ALPACA_API_KEY"))
print("Polygon Key:", os.getenv("POLYGON_API_KEY"))
print("NewsAPI Key:", os.getenv("NEWS_API_KEY"))
