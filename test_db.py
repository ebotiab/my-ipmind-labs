import asyncio

import asyncpg

from ipmind_labs.config import settings


async def test():
    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)

    # query enums
    q1 = """
    SELECT e.enumlabel 
    FROM pg_enum e 
    JOIN pg_type t ON e.enumtypid = t.oid 
    JOIN pg_attribute a ON a.atttypid = t.oid 
    JOIN pg_class c ON a.attrelid = c.oid 
    WHERE c.relname = 'prism_benchmarks' AND a.attname = 'standard'
    """
    res = await conn.fetch(q1)
    print("enums:", [r["enumlabel"] for r in res])

    q2 = """
    SELECT claim_id::text, expected_essentiality
    FROM prism_benchmarks
    """
    res2 = await conn.fetch(q2)
    print("first 3 rows:", [dict(r) for r in res2[:3]])

    await conn.close()


if __name__ == "__main__":
    asyncio.run(test())
