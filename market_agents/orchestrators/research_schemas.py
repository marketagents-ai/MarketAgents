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

class CharacterAnalysis(BaseModel):
    """Schema for character analysis in literary works"""
    name: str = Field(..., description="Character name")
    role: str = Field(..., description="Character's role in the story")
    motivation: Optional[str] = Field(None, description="Character's primary motivations")
    key_quotes: List[str] = Field(default_factory=list, description="Notable quotes by or about the character")
    relationships: List[str] = Field(default_factory=list, description="Key relationships with other characters")
    character_arc: Optional[str] = Field(None, description="Character's development through the story")
    themes: List[str] = Field(default_factory=list, description="Themes associated with this character")
    interpretation: Optional[str] = Field(None, description="Critical interpretation of the character")

class ThematicElement(BaseModel):
    theme: str = Field(..., description="Name of the theme")
    description: str = Field(..., description="Explanation of the theme")
    examples: List[str] = Field(default_factory=list, description="Textual examples supporting this theme")
    significance: Optional[str] = Field(None, description="Broader significance of this theme")

class LiteraryAnalysis(BaseModel):
    """Schema for comprehensive literary analysis"""
    work_title: str = Field(default="Hamlet", description="Title of the literary work")
    author: str = Field(default="William Shakespeare", description="Author of the work")
    characters: List[CharacterAnalysis] = Field(default_factory=list, description="Analysis of key characters")
    themes: List[ThematicElement] = Field(default_factory=list, description="Major themes in the work")
    key_scenes: List[str] = Field(default_factory=list, description="Analysis of pivotal scenes")
    literary_devices: List[str] = Field(default_factory=list, description="Notable literary devices used")
    historical_context: Optional[str] = Field(None, description="Relevant historical background")
    interpretation: Optional[str] = Field(None, description="Overall interpretation or analysis")