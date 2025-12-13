"""
Alpaca API Client - EXECUTION ONLY
===================================
This client handles order execution and position management via Alpaca.
Data retrieval should use PolygonClient for better performance.

Author: OTREP-X Development Team
Lead Engineer: Gemini AI
"""

import os
import requests
from typing import Dict, List, Optional


class AlpacaClient:
    """
    Minimal Alpaca API client for EXECUTION operations only.
    
    Responsibilities:
    - Account information
    - Position management
    - Order execution
    
    Data retrieval should use PolygonClient.
    """
    
    def __init__(self, base_url: str = 'https://paper-api.alpaca.markets'):
        """
        Initialize Alpaca client with API credentials from environment.
        
        Args:
            base_url: Alpaca API base URL (paper or live)
        """
        self.base_url = base_url
        self.api_key = (
            os.getenv('APCA_API_KEY_ID')
            or os.getenv('ALPACA_API_KEY_ID')
            or os.getenv('ALPACA_API_KEY')
            or ''
        )
        self.secret_key = (
            os.getenv('APCA_API_SECRET_KEY')
            or os.getenv('ALPACA_API_SECRET_KEY')
            or os.getenv('ALPACA_SECRET_KEY')
            or os.getenv('ALPACA_API_SECRET')
            or ''
        )
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
    
    def get_account(self) -> Dict:
        """
        Fetch account information.
        
        Returns:
            Dict with account details including equity, cash, buying_power, etc.
        """
        response = requests.get(
            f'{self.base_url}/v2/account',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_positions(self) -> List[Dict]:
        """
        Fetch all current positions.
        
        Returns:
            List of position dictionaries with symbol, qty, entry price, P&L, etc.
        """
        response = requests.get(
            f'{self.base_url}/v2/positions',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Fetch position for a specific symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Position dict if exists, None if no position
        """
        try:
            response = requests.get(
                f'{self.base_url}/v2/positions/{symbol}',
                headers=self.headers
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError:
            return None
    
    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = 'market',
        time_in_force: str = 'day'
    ) -> Dict:
        """
        Submit an order to Alpaca.
        
        Args:
            symbol: Stock ticker symbol
            qty: Number of shares (absolute value)
            side: 'buy' or 'sell'
            order_type: Order type (market, limit, stop, etc.)
            time_in_force: 'day', 'gtc', 'ioc', etc.
            
        Returns:
            Order response dict with order ID, status, etc.
        """
        order = {
            'symbol': symbol,
            'qty': abs(qty),
            'side': side,
            'type': order_type,
            'time_in_force': time_in_force
        }
        response = requests.post(
            f'{self.base_url}/v2/orders',
            json=order,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def close_position(self, symbol: str) -> Dict:
        """
        Close an entire position for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Order response dict for the closing order
        """
        response = requests.delete(
            f'{self.base_url}/v2/positions/{symbol}',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_all_orders(self) -> List[Dict]:
        """
        Cancel all open orders.
        
        Returns:
            List of cancelled order responses
        """
        response = requests.delete(
            f'{self.base_url}/v2/orders',
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_orders(self, status: str = 'open') -> List[Dict]:
        """
        Fetch orders by status.
        
        Args:
            status: 'open', 'closed', 'all'
            
        Returns:
            List of order dictionaries
        """
        response = requests.get(
            f'{self.base_url}/v2/orders',
            headers=self.headers,
            params={'status': status}
        )
        response.raise_for_status()
        return response.json()
