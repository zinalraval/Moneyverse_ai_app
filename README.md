# Trading Platform

A real-time trading platform with license-based access, built with FastAPI and Streamlit.

## Features

- Real-time trading signals and price updates
- License-based access control
- WebSocket-based real-time communication
- Modern, responsive dashboard with multi-language support and mobile mode
- AI-powered chart analysis and trading assistant chatbot
- News filtering with sentiment analysis and major economic event alerts
- Backtesting capabilities with equity curve visualization
- Docker-based deployment

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)
- PostgreSQL 15+ (for local development)

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-platform
```

2. Start the services:
```bash
docker-compose up --build
```

3. Access the applications:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/api/v1/docs

## Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend/streamlit
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Backend
cp .env.example .env
# Edit .env with your configuration

# Frontend
cd frontend/streamlit
cp .env.example .env
# Edit .env with your configuration
```

4. Run the services:
```bash
# Backend
uvicorn app.main:app --reload

# Frontend (in a separate terminal)
cd frontend/streamlit
streamlit run app.py
```

## Project Structure

```
.
├── app/                    # Backend FastAPI application
│   ├── api/               # API routes and WebSocket endpoints
│   ├── core/              # Core configuration and settings
│   ├── crud/              # Database operations
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   └── services/          # Business logic and services
├── frontend/              # Streamlit frontend
│   └── streamlit/         # Streamlit application
├── tests/                 # Test suite
├── alembic/               # Database migrations
├── docker-compose.yml     # Docker Compose configuration
└── README.md             # This file
```

## Testing

1. Run backend tests:
```bash
pytest
```

2. Run frontend tests:
```bash
cd frontend/streamlit
pytest
```

## Load Testing with Locust

You can simulate hundreds or thousands of concurrent users to test backend scalability using [Locust](https://locust.io/).

### 1. Install Locust
```bash
pip install locust
```

### 2. Configure Environment Variables
Set the following environment variables as needed:
- `API_BASE_URL` (default: http://localhost:8000/api/v1)
- `LOCUST_EMAIL` (default: testuser@example.com)
- `LOCUST_PASSWORD` (default: testpassword)

### 3. Run Locust
```bash
locust -f locustfile.py
```
Then open [http://localhost:8089](http://localhost:8089) in your browser to launch the Locust web UI.

### 4. Start a Test
- Enter the number of users (e.g., 1000) and spawn rate (e.g., 50 users/sec).
- Click "Start swarming".

### 5. Interpreting Results
- **Response time**: Should remain low under load.
- **Failure rate**: Should be near zero; investigate any errors.
- **Throughput**: Requests per second.
- **Bottlenecks**: If response times spike or errors increase, check backend logs, DB, and resource usage.

The default Locust script tests login, signal fetch, market data, news, and health endpoints. You can extend it for more scenarios as needed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## API Documentation

- **Swagger/OpenAPI docs** are available at `/docs` (and `/redoc`) when not in production mode.
- To enable docs, set `PRODUCTION=false` in your environment/config.
- Example: [http://localhost:8000/docs](http://localhost:8000/docs)
- The OpenAPI schema is available at `/api/v1/openapi.json`.

## Quickstart

### Backend (FastAPI)
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (see below)
export ENVIRONMENT=development
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost/dbname
# ...other variables as needed...

# Run the backend
uvicorn app.main:app --reload
```

### Frontend (Streamlit)
```bash
cd frontend/streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables & Configuration
- `ENVIRONMENT` (development/production)
- `DATABASE_URL` (Postgres connection string)
- `MONGODB_URL` (MongoDB connection string)
- `API_BASE_URL` (for frontend/backend communication)
- `TWELVEDATA_API_KEY`, `ALPHAVANTAGE_API_KEY`, `NEWS_API_KEY` (market/news providers)
- `SENTRY_DSN` (for error tracking)
- `MOCK_MARKET_DATA` (set to true to use mock data)
- See `.env` or `app/config.py` for more options.

## Codebase Overview

- `app/` — FastAPI backend (API, models, services, db, etc.)
- `frontend/streamlit/` — Streamlit frontend app
- `alembic/` — Database migrations
- `tests/` — Automated tests
- `scripts/` — Utility scripts (DB init, admin, etc.)
- `locustfile.py` — Load testing script
- `README.md` — This documentation

## Contributing

- Fork the repo and create a feature branch.
- Add tests for new features.
- Run `pytest` and `locust` to verify stability and performance.
- Submit a pull request with a clear description.
