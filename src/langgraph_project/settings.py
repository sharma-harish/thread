import pydantic
from pydantic_settings import BaseSettings
from pathlib import Path

# Get the absolute path to project root (2 levels up from this file)
BASE_DIR = Path(__file__).resolve().parents[2]
KB_DIR = BASE_DIR / "knowledge-base"

# Use pydantic base settings for basic settings read from a .env file
class Settings(BaseSettings):
    openai_api_key: pydantic.SecretStr
    openai_model_name: str = "gpt-4o-mini"
    embeddings_model_name: str = "text-embedding-ada-002"
    knowledge_base_path: str = str(KB_DIR)
    google_api_key: str = pydantic.SecretStr
    google_model_name: str = "gemini-2.5-flash"
    google_embeddings_model_name: str = "gemini-embedding-001"
    use_google: bool = True
    wandb_api_key: str = pydantic.SecretStr

    class Config:
        env_file = "../../.env"
        env_file_encoding = "utf-8"


settings: Settings = Settings()