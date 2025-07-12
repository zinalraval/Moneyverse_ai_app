from sqlalchemy.orm import Session
from app.services.signal_generation import generate_signal, signal_dict_to_model, candles_to_df
from app.services.market_data import market_data_service, SIGNAL_TYPE_TO_INTERVAL
from app.models.signal import SignalType, SignalTimeframe, SignalStatus
from app.crud.signal_crud import get_active_signal_for_pair, create_new_signal, update_signal_status
from app.db.postgres import get_db_session
import asyncio
import logging
from typing import AsyncGenerator
from app.services.market_data_utils import SUPPORTED_PAIRS
import os
import random
from app.core.utils import safe_create_task

logger = logging.getLogger(__name__)

API_REQUESTS_PER_MINUTE = int(os.getenv("API_REQUESTS_PER_MINUTE", 8))
API_REQUEST_DELAY_SECONDS = float(os.getenv("API_REQUEST_DELAY_SECONDS", 8))

if len(SUPPORTED_PAIRS) > API_REQUESTS_PER_MINUTE:
    logger.warning(f"Number of pairs ({len(SUPPORTED_PAIRS)}) exceeds allowed API requests per minute ({API_REQUESTS_PER_MINUTE}). Some pairs may be skipped or you may hit rate limits.")
logger.info(f"API request delay per pair: {API_REQUEST_DELAY_SECONDS}s. Number of pairs: {len(SUPPORTED_PAIRS)}.")

async def check_and_update_signal_status(db, pair, current_price):
    active_signal = await get_active_signal_for_pair(db, pair)
    if not active_signal:
        return
    # Only update if still active
    if active_signal.status != SignalStatus.ACTIVE:
        return
    # BUY logic
    if active_signal.direction == "BUY":
        if current_price >= active_signal.tp1:
            await update_signal_status(db, active_signal.id, SignalStatus.TP1_HIT, exit_price=active_signal.tp1)
        elif current_price <= active_signal.stop_loss:
            await update_signal_status(db, active_signal.id, SignalStatus.SL_HIT, exit_price=active_signal.stop_loss)
    # SELL logic
    elif active_signal.direction == "SELL":
        if current_price <= active_signal.tp1:
            await update_signal_status(db, active_signal.id, SignalStatus.TP1_HIT, exit_price=active_signal.tp1)
        elif current_price >= active_signal.stop_loss:
            await update_signal_status(db, active_signal.id, SignalStatus.SL_HIT, exit_price=active_signal.stop_loss)

async def generate_signals_periodically():
    """Generates new signals periodically if no active signal exists for a pair."""
    while True:
        for pair in SUPPORTED_PAIRS:
            async with get_db_session() as db:
                try:
                    current_price = await market_data_service.get_live_price(pair)
                    if current_price is None:
                        logger.warning(f"Could not fetch live price for {pair}. Skipping signal generation.")
                        continue
                    # Check and update signal status before generating new signals
                    await check_and_update_signal_status(db, pair, current_price)
                    active_signal = await get_active_signal_for_pair(db, pair)
                    if active_signal:
                        logger.info(f"Skipping signal generation for {pair}: active signal already exists (ID: {active_signal.id})")
                        continue

                    # Use the correct SignalTimeframe enum value for database
                    timeframe_enum = SignalTimeframe.H1  # Use H1 for database
                    timeframe_string = "1h"  # Use "1h" for market data API
                    historical_candles = await market_data_service.get_candle_data(pair, timeframe_string)
                    if historical_candles is None or historical_candles.empty:
                        logger.warning(f"Could not fetch historical candle data for {pair} ({timeframe_string}). Skipping signal generation.")
                        continue

                    df = candles_to_df(historical_candles)

                    # Pass the desired trading style to the signal generation function
                    # For automated signals, we can use a default style like "Intraday".
                    # In a production system, this could be configurable per user or per pair.
                    new_signal_data = None
                    try:
                        new_signal_data = await generate_signal(
                            df=df,
                            pair=pair,
                            current_price=current_price,
                            signal_type=SignalType.INTRADAY,
                            user_id=1,
                            db=db,
                            timeframe=timeframe_enum
                        )
                    except Exception as e:
                        logger.error(f"Error generating signal for {pair}: {e}", exc_info=True)
                        continue # Skip to next pair if signal generation fails

                    if new_signal_data:
                        try:
                            signal_model = await create_new_signal(db, new_signal_data)
                            logger.info(f"Auto-generated new signal for {pair}: {signal_model.id}")
                        except Exception as e:
                            logger.error(f"Error saving new signal for {pair} to database: {e}", exc_info=True)
                            await db.rollback()
                            continue # Skip to next pair if saving fails
                    else:
                        logger.info(f"No signal generated for {pair} based on current market conditions.")

                    await db.commit()

                except Exception as e:
                    await db.rollback()
                    logger.error(f"Error during automated signal generation for {pair}: {str(e)}", exc_info=True)
            # Add delay to avoid API rate limit, with random jitter
            delay = API_REQUEST_DELAY_SECONDS + random.uniform(0, 1)
            logger.debug(f"Sleeping {delay:.2f}s before next pair to avoid rate limit.")
            await asyncio.sleep(delay)
        # Introduce a delay before the next iteration (outside the pair loop)
        await asyncio.sleep(120) # Generate signals every 120 seconds (2 minutes)

async def start_signal_automation():
    """Starts the automated signal generation background task."""
    logger.info("Starting automated signal generation task...")
    safe_create_task(generate_signals_periodically()) 