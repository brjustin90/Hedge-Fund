import asyncio
import logging
import aiohttp
import yaml
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class TokenDiscovery:
    def __init__(self, config):
        self.config = config
        self.known_tokens = set()
        self.token_metadata = {}
        self.session = None
        
    async def initialize(self):
        """Initialize aiohttp session and load known tokens"""
        self.session = aiohttp.ClientSession()
        await self.load_known_tokens()
        
    async def load_known_tokens(self):
        """Load previously discovered tokens"""
        try:
            with open('config/known_tokens.json', 'r') as f:
                data = json.load(f)
                self.known_tokens = set(data.get('tokens', []))
                self.token_metadata = data.get('metadata', {})
        except FileNotFoundError:
            logger.info("No existing tokens file found, starting fresh")
            
    async def save_known_tokens(self):
        """Save discovered tokens"""
        data = {
            'tokens': list(self.known_tokens),
            'metadata': self.token_metadata,
            'last_updated': datetime.now().isoformat()
        }
        with open('config/known_tokens.json', 'w') as f:
            json.dump(data, f, indent=2)
            
    async def discover_new_tokens(self):
        """Discover new tokens from multiple sources"""
        new_tokens = set()
        
        # Discover from Jupiter aggregator
        jupiter_tokens = await self._discover_from_jupiter()
        new_tokens.update(jupiter_tokens)
        
        # Discover from Raydium
        raydium_tokens = await self._discover_from_raydium()
        new_tokens.update(raydium_tokens)
        
        # Discover from Solscan
        solscan_tokens = await self._discover_from_solscan()
        new_tokens.update(solscan_tokens)
        
        # Filter out known tokens
        truly_new = new_tokens - self.known_tokens
        
        # Get metadata for new tokens
        for token in truly_new:
            metadata = await self._get_token_metadata(token)
            if self._validate_token(metadata):
                self.known_tokens.add(token)
                self.token_metadata[token] = metadata
                
        await self.save_known_tokens()
        return truly_new
        
    async def _discover_from_jupiter(self):
        """Discover tokens from Jupiter aggregator"""
        try:
            async with self.session.get('https://token.jup.ag/all') as response:
                if response.status == 200:
                    data = await response.json()
                    return {token['address'] for token in data['tokens']}
        except Exception as e:
            logger.error(f"Error discovering tokens from Jupiter: {e}")
        return set()
        
    async def _discover_from_raydium(self):
        """Discover tokens from Raydium"""
        try:
            async with self.session.get('https://api.raydium.io/v2/sdk/token/raydium.mainnet.json') as response:
                if response.status == 200:
                    data = await response.json()
                    return {token['mint'] for token in data['tokens']}
        except Exception as e:
            logger.error(f"Error discovering tokens from Raydium: {e}")
        return set()
        
    async def _discover_from_solscan(self):
        """Discover tokens from Solscan"""
        try:
            # Get recent token transactions
            url = 'https://api.solscan.io/token/list'
            params = {
                'offset': 0,
                'limit': 100,
                'sortBy': 'created_date'
            }
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {token['mint'] for token in data.get('data', [])}
        except Exception as e:
            logger.error(f"Error discovering tokens from Solscan: {e}")
        return set()
        
    async def _get_token_metadata(self, token_address):
        """Get metadata for a token"""
        try:
            # Try Jupiter first
            async with self.session.get(f'https://token.jup.ag/token/{token_address}') as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'address': token_address,
                        'symbol': data.get('symbol'),
                        'name': data.get('name'),
                        'decimals': data.get('decimals'),
                        'created_at': data.get('created_at', datetime.now().isoformat()),
                        'total_supply': data.get('total_supply'),
                        'holder_count': data.get('holder_count'),
                        'volume_24h': data.get('volume_24h'),
                        'liquidity': data.get('liquidity'),
                        'price_usd': data.get('price_usd'),
                        'last_updated': datetime.now().isoformat()
                    }
                    
            # Fallback to Solscan
            async with self.session.get(f'https://api.solscan.io/token/meta?token={token_address}') as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'address': token_address,
                        'symbol': data.get('symbol'),
                        'name': data.get('name'),
                        'decimals': data.get('decimals'),
                        'created_at': data.get('created_at', datetime.now().isoformat()),
                        'total_supply': data.get('total_supply'),
                        'holder_count': data.get('holder_count'),
                        'volume_24h': 0,  # Need to get from another source
                        'liquidity': 0,  # Need to get from another source
                        'price_usd': 0,  # Need to get from another source
                        'last_updated': datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error getting metadata for token {token_address}: {e}")
        return None
        
    def _validate_token(self, metadata):
        """Validate token meets our criteria"""
        if not metadata:
            return False
            
        try:
            # Check token age
            created_at = datetime.fromisoformat(metadata['created_at'].replace('Z', '+00:00'))
            age_days = (datetime.now() - created_at).days
            
            if age_days > self.config['data']['max_token_age_days']:
                return False
                
            if age_days < self.config['data']['min_token_age_days']:
                return False
                
            # Check liquidity
            if metadata.get('liquidity', 0) < self.config['data']['min_liquidity_usd']:
                return False
                
            # Check volume
            if metadata.get('volume_24h', 0) < self.config['data']['min_volume_24h_usd']:
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False
            
    async def get_tradeable_tokens(self):
        """Get list of tokens that meet our trading criteria"""
        tradeable_tokens = []
        
        for address, metadata in self.token_metadata.items():
            # Update metadata if it's old
            last_updated = datetime.fromisoformat(metadata['last_updated'].replace('Z', '+00:00'))
            if datetime.now() - last_updated > timedelta(hours=1):
                updated_metadata = await self._get_token_metadata(address)
                if updated_metadata:
                    self.token_metadata[address] = updated_metadata
                    metadata = updated_metadata
                    
            if self._validate_token(metadata):
                tradeable_tokens.append(metadata)
                
        # Sort by age (newest first) and volume
        tradeable_tokens.sort(
            key=lambda x: (
                datetime.fromisoformat(x['created_at'].replace('Z', '+00:00')),
                -float(x.get('volume_24h', 0))
            )
        )
        
        return tradeable_tokens
        
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            
    async def run_discovery_loop(self):
        """Run continuous token discovery loop"""
        while True:
            try:
                logger.info("Starting token discovery cycle")
                new_tokens = await self.discover_new_tokens()
                
                if new_tokens:
                    logger.info(f"Discovered {len(new_tokens)} new tokens")
                    for token in new_tokens:
                        metadata = self.token_metadata[token]
                        logger.info(f"New token: {metadata['symbol']} ({metadata['name']})")
                        
                tradeable = await self.get_tradeable_tokens()
                logger.info(f"Current tradeable tokens: {len(tradeable)}")
                
                # Sleep for an hour before next discovery
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(300)  # Sleep for 5 minutes on error 