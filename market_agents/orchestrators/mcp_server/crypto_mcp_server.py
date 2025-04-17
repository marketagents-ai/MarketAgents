from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
import requests
import datetime as dt
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("DexScreenerServer")
BASE = "https://api.dexscreener.com"

# Map common chain aliases to DexScreener's canonical chainId strings
_CHAIN_ALIASES = {
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "arb": "arbitrum",
    "arbitrum": "arbitrum",
    "bsc": "bsc",
    "binance-smart-chain": "bsc",
    "polygon": "polygon",
    "matic": "polygon",
    "base": "base",
}

def _canonical_chain(chain: str) -> str:
    """Return DexScreener‑compatible chainId for common aliases."""
    return _CHAIN_ALIASES.get(chain.lower(), chain.lower())

def _resolve_pair(chain: str, query: str) -> Dict[str, Any]:
    """
    Return the first search‑match dict for *query* on the given *chain*
    using DexScreener's public search endpoint.

    Parameters
    ----------
    chain : str
        Chain identifier (e.g. 'ethereum', 'solana')
    query : str
        Search query string

    Returns
    -------
    dict
        Matching pair information
    """
    chain = _canonical_chain(chain)
    try:
        url = f"{BASE}/latest/dex/search?q={query}"
        resp = requests.get(url, timeout=8).json().get("pairs", [])
        for p in resp:
            if p.get("chainId") == chain:
                return p
        return resp[0] if resp else {}
    except Exception as e:
        logger.error(f"Error resolving pair: {str(e)}")
        return {}

@mcp.tool()
def get_dex_pair(
    chain: str,
    query: str,  # Only chain and query required
) -> Dict[str, Any]:
    """Return price/liquidity snapshot for a DEX pair."""
    chain = _canonical_chain(chain)
    try:
        match = _resolve_pair(chain, query)
        pair_address = match.get("pairAddress")
        if not pair_address:
            return {"error": "pair not found"}
        url = f"{BASE}/latest/dex/pairs/{chain}/{pair_address}"
        return requests.get(url, timeout=8).json()
    except Exception as e:
        logger.error(f"Error fetching DEX pair: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
def get_token_profile(
    chain: str,
    token_symbol: str,  # Only chain and symbol required
) -> Dict[str, Any]:
    """Get static token metadata (icon, website, socials)."""
    chain = _canonical_chain(chain)
    try:
        match = _resolve_pair(chain, token_symbol)
        token_address = (
            match.get("baseToken", {}).get("address")
            or match.get("tokenAddress")
        )
        if not token_address:
            return {"error": "token not found"}

        url = (
            f"{BASE}/token-profiles/latest/v1?"
            f"chainId={chain}&tokenAddress={token_address}"
        )
        return requests.get(url, timeout=8).json()
    except Exception as e:
        logger.error(f"Error fetching token profile: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
def get_boosted_tokens(
    chain: str,
    limit: int,
) -> Dict[str, Any]:
    """Get list of currently boosted tokens from DexScreener."""
    chain = _canonical_chain(chain)
    try:
        resp_text = requests.get(f"{BASE}/token-boosts/latest/v1", timeout=8).text
        tokens: list[dict] = []
        for line in resp_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, list):
                tokens.extend(parsed)
            else:
                tokens.append(parsed)

        tokens = [t for t in tokens if t.get("chainId") == chain]
        tokens = tokens[:limit]

        return {"boosted_tokens": tokens}
    except Exception as e:
        logger.error(f"Error fetching boosted tokens: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting DexScreener MCP Server...")
    mcp.run()