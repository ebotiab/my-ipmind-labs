from functools import cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_db: str
    postgres_password: SecretStr
    postgres_port: int
    postgres_server: str
    postgres_user: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"


@cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
