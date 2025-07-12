import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.db.postgres import get_db_session
from app.models.signal import Signal
from app.models.user import User
from app.models.license import License
from sqlalchemy import text

async def check_database():
    """Check database tables and data."""
    async with get_db_session() as db:
        print("=== DATABASE STATUS CHECK ===\n")
        
        # Check if tables exist
        print("1. Checking table existence...")
        result = await db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """))
        tables = result.fetchall()
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        print("\n2. Checking table row counts...")
        for table in tables:
            table_name = table[0]
            result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.fetchone()[0]
            print(f"  - {table_name}: {count} rows")
        
        print("\n3. Checking signals by status...")
        result = await db.execute(text("""
            SELECT status, COUNT(*) as count 
            FROM signals 
            GROUP BY status 
            ORDER BY count DESC;
        """))
        status_counts = result.fetchall()
        for status, count in status_counts:
            print(f"  - {status}: {count} signals")
        
        print("\n4. Checking signals by pair...")
        result = await db.execute(text("""
            SELECT pair, COUNT(*) as count 
            FROM signals 
            GROUP BY pair 
            ORDER BY count DESC 
            LIMIT 10;
        """))
        pair_counts = result.fetchall()
        for pair, count in pair_counts:
            print(f"  - {pair}: {count} signals")
        
        print("\n5. Sample active signals...")
        result = await db.execute(text("""
            SELECT id, pair, direction, entry_price, tp1, tp2, stop_loss, status, created_at
            FROM signals 
            WHERE status = 'ACTIVE'
            ORDER BY created_at DESC 
            LIMIT 5;
        """))
        active_signals = result.fetchall()
        if active_signals:
            for signal in active_signals:
                print(f"  - ID: {signal[0]}, {signal[1]} {signal[2]} @ {signal[3]}, TP1: {signal[4]}, TP2: {signal[5]}, SL: {signal[6]}, Status: {signal[7]}, Created: {signal[8]}")
        else:
            print("  - No active signals found")
        
        print("\n6. Recent signal activity...")
        result = await db.execute(text("""
            SELECT id, pair, status, created_at, updated_at
            FROM signals 
            ORDER BY created_at DESC 
            LIMIT 5;
        """))
        recent_signals = result.fetchall()
        for signal in recent_signals:
            print(f"  - ID: {signal[0]}, {signal[1]}, Status: {signal[2]}, Created: {signal[3]}, Updated: {signal[4]}")

if __name__ == "__main__":
    asyncio.run(check_database()) 