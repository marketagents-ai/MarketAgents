from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class SearchQueries(BaseModel):
    """Schema for search query generation"""
    queries: List[str] = Field(
        description="List of search queries generated from the base query",
        examples=[
            ["latest cryptocurrency trends December 2024", 
             "top performing crypto assets Q4 2024"]
        ]
    )

class AnalysisType(str, Enum):
    ASSET = "asset"
    SECTOR = "sector"
    MACRO = "macro"
    GENERAL = "general"

class AssetAnalysis(BaseModel):
    ticker: str = Field(..., description="Asset ticker (e.g., stock symbol or crypto token).")
    rating: Optional[str] = Field(None, description="Analyst rating (e.g., Buy/Hold/Sell) and brief rationale.")
    target_price: Optional[str] = Field(None, description="Target price or price range from sources.")
    sentiment: Optional[str] = Field(None, description="Overall sentiment (Bullish, Bearish, Neutral).")
    catalysts: List[str] = Field(default_factory=list, description="Key qualitative catalysts for the asset.")
    kpis: List[str] = Field(default_factory=list, description="Key performance indicators (quantitative) for the asset.")
    action: Optional[str] = Field(None, description="Recommended portfolio action (e.g., Add Long, Close Short).")
    sources: List[str] = Field(default_factory=list, description="Information sources (e.g., brokers, analysts).")

class SectorInfo(BaseModel):
    name: str = Field(..., description="Name of the sector or industry.")
    sentiment: Optional[str] = Field(None, description="Overall sentiment for the sector.")
    catalysts: List[str] = Field(default_factory=list, description="Sector-wide catalysts.")
    kpis: List[str] = Field(default_factory=list, description="Sector-level KPIs or metrics.")
    top_assets: List[str] = Field(default_factory=list, description="Representative assets with brief notes.")
    recommendation: Optional[str] = Field(None, description="Recommended sector exposure (e.g., Overweight/Underweight).")
    sources: List[str] = Field(default_factory=list, description="Key sector-level sources.")

class MacroTrends(BaseModel):
    indicators: List[str] = Field(default_factory=list, description="Macro indicators (e.g., GDP, inflation, interest rates).")
    interest_rates: Optional[str] = Field(None, description="Interest rate outlook.")
    global_factors: Optional[str] = Field(None, description="Global geopolitical or economic factors.")
    sentiment: Optional[str] = Field(None, description="Overall macro-level sentiment (risk-on/off).")

class MarketResearch(BaseModel):
    analysis_type: AnalysisType = Field(..., description="Type of analysis: asset, sector, macro, or general.")
    assets: List[AssetAnalysis] = Field(default_factory=list, description="Asset-level insights if 'asset' type.")
    sector: Optional[SectorInfo] = Field(None, description="Sector-level details if 'sector' type.")
    macro: Optional[MacroTrends] = Field(None, description="Macro-level insights if 'macro' type.")

class Severity(str, Enum):
    """Classification of recession severity based on GDP decline"""
    SEVERE = "severe"          # > 3% GDP decline
    MODERATE = "moderate"      # 1-3% GDP decline
    MILD = "mild"             # < 1% GDP decline

class RecoveryType(str, Enum):
    """Classification of economic recovery patterns"""
    V_SHAPED = "v_shaped"      # Sharp decline, rapid recovery
    U_SHAPED = "u_shaped"      # Prolonged bottom, gradual recovery
    W_SHAPED = "w_shaped"      # Double-dip recession pattern
    L_SHAPED = "l_shaped"      # Sharp decline, stagnant recovery

class GreatRecessionAnalysis(BaseModel):
    """Analysis of the 2007-2009 Financial Crisis/Great Recession"""
    severity: Severity = Field(
        ..., 
        description="Severity classification based on GDP decline: SEVERE (>3%), MODERATE (1-3%), MILD (<1%)"
    )
    recovery_pattern: RecoveryType = Field(
        ..., 
        description="Shape of recovery: V (quick), U (gradual), W (double-dip), L (stagnant)"
    )
    gdp_decline_pct: float = Field(
        ..., 
        ge=0, 
        le=20, 
        description="Maximum GDP decline from peak to trough as a percentage"
    )
    peak_unemployment: float = Field(
        ...,
        ge=0,
        le=15,
        description="Peak unemployment rate during the recession (%)"
    )

    fed_funds_peak: float = Field(
        ...,
        ge=0,
        le=10,
        description="Federal Funds Rate at the start of recession (%)"
    )
    fed_funds_trough: float = Field(
        ...,
        ge=0,
        le=10,
        description="Federal Funds Rate at the end of recession (%)"
    )
    total_rate_cuts: float = Field(
        ...,
        ge=0,
        le=10,
        description="Total percentage points of Fed rate cuts during recession"
    )
    qe_program_size: float = Field(
        ...,
        ge=0,
        le=5000,
        description="Size of QE program in billions USD during recession period"
    )

    class Config:
        schema_extra = {
            "example": {
                "severity": "severe",
                "recovery_pattern": "u_shaped",
                "gdp_decline_pct": 4.3,
                "peak_unemployment": 10.0,
                "fed_funds_peak": 5.25,
                "fed_funds_trough": 0.25,
                "total_rate_cuts": 5.0,
                "qe_program_size": 1750.0
            }
        }