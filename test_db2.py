import asyncio
import asyncpg
from src.ipmind_labs.config import settings

async def main():
    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)
    tables = ['jobs', 'prism_results']
    for t in tables:
        rows = await conn.fetch("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = $1", t)
        print(f"Table {t}:")
        for r in rows:
            print(f"  {r['column_name']}: {r['data_type']}")
    await conn.close()

asyncio.run(main())
