from setuptools import setup, find_packages

setup(
    name="moneyverse-ai-backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "sqlalchemy>=1.4.0",
        "pydantic>=1.8.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-multipart>=0.0.5",
        "alembic>=1.7.0",
        "httpx>=0.23.0",
        "pandas>=1.3.0",
        "ccxt>=1.60.0",
        "yfinance>=0.1.70",
    ],
    python_requires=">=3.8",
) 