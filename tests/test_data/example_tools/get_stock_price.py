"""Stock price lookup tool for financial analysis."""

from datetime import datetime
from typing import Dict, Union

import yfinance as yf


def get_stock_price(ticker: str, days: str = "1d") -> Dict[str, Union[float, str, list, datetime]]:
    """Retrieve stock price data for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
        days (str): Number of days of historical data to fetch (default: 1d).
            Must be one of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

    Returns:
        Dict[str, Union[float, str, list, datetime]]: Dictionary containing:
            - prices (list): List of closing prices for requested days
            - currency (str): The currency of the price (e.g., 'USD')
            - timestamps (list): List of timestamps for each price

    Raises:
        ValueError: If the ticker symbol is invalid or data cannot be retrieved
        ValueError: If days parameter is not one of the allowed values

    Metadata:
        - version: 0.0.1
        - author: John Doe
        - requires_gpu: False
        - requires_api_key: False
    """
    # Validate days parameter
    valid_days = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    if days not in valid_days:
        raise ValueError(f"'days' must be one of: {valid_days}")

    try:
        # Get stock data
        stock = yf.Ticker(ticker)

        # Get historical data using valid period format
        hist = stock.history(period=days)

        if hist.empty:
            raise ValueError(f"No data returned for ticker {ticker}")

        return {
            "prices": hist["Close"].tolist(),
            "currency": stock.info.get("currency", "USD"),
            "timestamps": [ts.to_pydatetime() for ts in hist.index],
        }

    except Exception as e:
        raise ValueError(f"Failed to retrieve stock price for {ticker}: {str(e)}") from e
