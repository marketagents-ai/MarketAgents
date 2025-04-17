from mcp.server.fastmcp import FastMCP
import requests, datetime as dt

mcp = FastMCP("DexScreenerServer")
BASE = "https://api.dexscreener.com"

# ---------------------------------------------------------------------------
# Helper: resolve pair / token via DexScreener fuzzy search
# ---------------------------------------------------------------------------
def _resolve_pair(chain: str, query: str):
    """
    Return the first search‑match dict for *query* on the given *chain*
    using DexScreener's public search endpoint.
    """
    url = f"{BASE}/latest/dex/search?q={query}"
    resp = requests.get(url, timeout=8).json().get("pairs", [])
    for p in resp:
        if p.get("chainId") == chain:
            return p
    return resp[0] if resp else {}

@mcp.tool()
def get_dex_pair(
    chain: str,
    query: str | None = None,
    pair_address: str | None = None,
):
    """
    Return price/liquidity snapshot for a DEX pair.

    Parameters
    ----------
    chain : str
        Chain slug (e.g. 'solana', 'ethereum', 'base').
    query : str, optional
        Human‑friendly string such as 'SOL/USDC' or 'JitoSOL'. The server
        will resolve it to a pair on the specified chain when *pair_address*
        is omitted.
    pair_address : str, optional
        Full on‑chain LP address.  Most agents can ignore this and rely on
        *query* resolution.

    Examples
    --------
    • get_dex_pair(chain='solana', query='SOL/USDC')
    • get_dex_pair(chain='solana', pair_address='G6drsaPCR3pxsEmS…')
    """
    if pair_address is None:
        if query is None:
            return {"error": "provide pair_address or query"}
        match = _resolve_pair(chain, query)
        pair_address = match.get("pairAddress")
        if not pair_address:
            return {"error": "pair not found"}
    url = f"{BASE}/latest/dex/pairs/{chain}/{pair_address}"
    return requests.get(url, timeout=8).json()

@mcp.tool()
def search_pairs(query: str):
    """Fuzzy search pairs or tokens (e.g., 'SOL/USDC'). This function still works but agents can now skip it."""
    return requests.get(f"{BASE}/latest/dex/search?q={query}", timeout=8).json()

@mcp.tool()
def get_token_profile(
    chain: str,
    token_symbol: str | None = None,
    token_address: str | None = None,
):
    """
    Static token metadata (icon, website, socials).

    Prefer giving a *token_symbol* (e.g. "SOL" or "JitoSOL").
    The helper will resolve it to an on‑chain address via DexScreener
    search.  Power users may still supply *token_address* directly.

    Parameters
    ----------
    chain : str
        Chain slug ('solana', 'base', 'arbitrum', etc.).
    token_symbol : str, optional
        Human‑friendly ticker or token name.  If provided, it is resolved
        to the correct contract address on the given chain.
    token_address : str, optional
        Full contract address.  Only needed when the symbol is ambiguous
        or not yet indexed by DexScreener.
    """
    # Resolve symbol first (safer for LLM agents)
    if token_symbol is not None:
        match = _resolve_pair(chain, token_symbol)
        token_address = (
            match.get("baseToken", {}).get("address")
            or match.get("tokenAddress")
        )
        if not token_address:
            return {"error": "token not found"}
    elif token_address is None:
        return {"error": "provide token_symbol or token_address"}

    url = (
        f"{BASE}/token-profiles/latest/v1?"
        f"chainId={chain}&tokenAddress={token_address}"
    )
    return requests.get(url, timeout=8).json()

@mcp.tool()
def get_boosted_tokens():
    url = f"{BASE}/token-boosts/latest/v1"
    return requests.get(url, timeout=8).json()

if __name__ == "__main__":
    mcp.run()