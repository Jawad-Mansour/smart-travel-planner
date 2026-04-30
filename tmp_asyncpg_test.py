import asyncio
import asyncpg


async def main():
    conn = await asyncpg.connect(
        "postgresql://postgres:postgres@127.0.0.1:5432/travel_planner", timeout=10
    )
    db = await conn.fetchval("select current_database()")
    print("OK", db)
    await conn.close()


asyncio.run(main())
