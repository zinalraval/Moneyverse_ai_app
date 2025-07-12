import asyncio
from app.db.postgres import get_db_session
from app.models.signal import Signal

async def print_active_signals():
    async with get_db_session() as db:
        result = await db.execute(
            Signal.__table__.select().where(
                (Signal.pair == 'BTC/USD') & (Signal.status == 'ACTIVE')
            )
        )
        signals = result.fetchall()
        if not signals:
            print("No active signals found for BTC/USD.")
        else:
            for row in signals:
                print(dict(row._mapping))

if __name__ == "__main__":
    asyncio.run(print_active_signals()) 