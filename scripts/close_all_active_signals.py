import asyncio
import sys
from app.db.postgres import async_session_factory
from app.models.signal import Signal, SignalStatus

async def close_all_active_signals():
    async with async_session_factory() as session:
        try:
            result = await session.execute(
                Signal.__table__.update()
                .where(Signal.status == SignalStatus.ACTIVE)
                .values(status=SignalStatus.COMPLETED)
                .returning(Signal.id)
            )
            await session.commit()
            updated = len(result.fetchall())
            print(f"Updated {updated} signals from ACTIVE to COMPLETED.")
        except Exception as e:
            print(f"Error: {e}")
            await session.rollback()
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(close_all_active_signals()) 