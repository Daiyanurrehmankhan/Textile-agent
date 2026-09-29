"""Connection helper for the Neon Postgres database."""
import psycopg2

from config.settings import settings


def get_connection():
    """Return a new psycopg2 connection using the configured DATABASE_URL."""
    return psycopg2.connect(settings.database_url)
