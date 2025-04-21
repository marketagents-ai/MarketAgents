import json
import logging
import re
import traceback
import uuid
import yaml
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional, Union


# Configure logging
logger = logging.getLogger(__name__)
logger.handlers = []
logger.addHandler(logging.NullHandler())
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('web_search.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def load_prompts(prompt_path: Path = Path("market_agents/web_search/web_search_prompt.yaml")) -> dict:
    """Load prompts from yaml file."""
    try:
        with open(prompt_path, 'r') as file:
            prompts = yaml.safe_load(file)
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        raise


def load_config(config_path: str = "market_agents/web_search/web_search_config.yaml", 
                prompt_path: str = "./market_agents/web_search/web_search_prompt.yaml") -> tuple[Any, Dict]:
    """Load configuration and prompts."""
    try:
        # Load main config
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Debug print
        print("Loaded config data:", json.dumps(config_data, indent=2))
            
        # Load prompts
        with open(prompt_path, 'r') as f:
            prompts = yaml.safe_load(f)
            
        return config_data, prompts
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def clean_json_string(content: str) -> str:
    """Clean and prepare JSON string for parsing."""
    try:
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Remove any YAML or JSON markers
        content = re.sub(r'^---\s*', '', content)
        content = re.sub(r'---\s*$', '', content)
        
        # Remove markdown code block markers
        content = re.sub(r'^```.*\n', '', content)
        content = re.sub(r'\n```$', '', content)
        
        # Fix common JSON formatting issues
        content = re.sub(r'(\w+):', r'"\1":', content)  # Add quotes to keys
        content = re.sub(r':\s*"([^"]*)"(\s*[,}])', r': "\1"\2', content)  # Ensure string values are quoted properly
        content = re.sub(r':\s*([^"][^,}\n]*?)(\s*[,}])', r': "\1"\2', content)  # Quote unquoted values
        
        # Remove BOM or special characters
        content = content.encode('ascii', 'ignore').decode('ascii')
        
        return content.strip()
    except Exception as e:
        logger.error(f"Error cleaning JSON string: {str(e)}")
        return content


def convert_to_number(value: str) -> float:
    """Convert string with K/M/B suffixes to float."""
    try:
        if not isinstance(value, str):
            return float(value)
            
        value = value.strip().upper()
        if value.endswith('K'):
            return float(value[:-1]) * 1000
        elif value.endswith('M'):
            return float(value[:-1]) * 1000000
        elif value.endswith('B'):
            return float(value[:-1]) * 1000000000
        else:
            return float(value)
    except (ValueError, TypeError):
        return float('nan')


def get_default_summary(has_data: bool = False) -> Dict[str, Any]:
    """Get a default summary structure when analysis fails."""
    return {
        "NEWS_ANALYSIS": {
            "key_developments": "Analysis failed - using default summary",
            "market_impact": "Analysis failed - using default summary",
            "sector_implications": "Analysis failed - using default summary"
        },
        "MARKET_SENTIMENT": {
            "investor_reaction": "Analysis failed - using default summary",
            "sentiment_indicators": "Analysis failed - using default summary",
            "confidence_levels": "Analysis failed - using default summary"
        },
        "TRADING_CONSIDERATIONS": {
            "opportunities": "Analysis failed - using default summary",
            "risks": "Analysis failed - using default summary",
            "timeline": "Analysis failed - using default summary"
        },
        "BASIC_SUMMARY": {
            "overview": "Analysis failed - using default summary",
            "key_points": [],
            "conclusion": "Analysis failed - using default summary"
        },
        "KEY_POINTS_ANALYSIS": {
            "market_movements": [],
            "announcements": [],
            "policy_changes": [],
            "market_reactions": [],
            "notable_quotes": [],
            "significant_data": []
        },
        "MARKET_IMPACT_ANALYSIS": {
            "immediate_reactions": {
                "price_movements": [],
                "volume_activity": [],
                "market_changes": []
            },
            "sector_impacts": {
                "affected_sectors": [],
                "winners_losers": [],
                "ripple_effects": []
            },
            "macro_implications": {
                "interest_rates": "Analysis failed - using default summary",
                "inflation": "Analysis failed - using default summary",
                "currency": "Analysis failed - using default summary",
                "global_markets": "Analysis failed - using default summary"
            },
            "sentiment": {
                "market_sentiment": "Analysis failed - using default summary",
                "institutional_reaction": "Analysis failed - using default summary",
                "retail_reaction": "Analysis failed - using default summary"
            }
        },
        "TRADING_IMPLICATIONS_DETAILED": {
            "opportunities": {
                "stocks_to_watch": [],
                "entry_points": [],
                "risk_levels": [],
                "time_horizons": []
            },
            "risk_assessment": {
                "key_risks": [],
                "hedging_strategies": [],
                "stop_loss_levels": [],
                "volatility_outlook": "Analysis failed - using default summary"
            },
            "portfolio_adjustments": {
                "sector_rotations": [],
                "allocation_changes": [],
                "position_sizing": "Analysis failed - using default summary",
                "diversification": "Analysis failed - using default summary"
            },
            "timing": {
                "short_term": "Analysis failed - using default summary",
                "medium_term": "Analysis failed - using default summary",
                "long_term": "Analysis failed - using default summary",
                "key_dates": []
            },
            "actionable_recommendations": {
                "trading_ideas": [],
                "rationale": [],
                "price_levels": [],
                "risk_management": []
            },
            "quantitative_signals": {
                "technical_levels": [],
                "arbitrage_opportunities": [],
                "relative_value": "Analysis failed - using default summary",
                "risk_metrics": []
            }
        }
    }


def structure_text_response(content: str, has_data: bool = False) -> Dict[str, Any]:
    """Structure a non-JSON text response into a default format."""
    structured_response = get_default_summary(has_data)
    
    try:
        lines = content.split('\n')
        current_section = None
        current_subsection = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.isupper() or line.endswith(':'):
                # Section header
                current_section = line.rstrip(':').upper()
                structured_response[current_section] = {}
                current_subsection = None
                continue
            
            # If line starts with '-' or '*', treat as a list item
            if line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()
                if current_section:
                    if isinstance(structured_response[current_section], dict):
                        if current_subsection:
                            structured_response[current_section][current_subsection].append(line)
                        else:
                            structured_response[current_section]['points'] = [line]
                    else:
                        structured_response[current_section] = [line]
            else:
                # Treat as a subsection header
                current_subsection = line.lower().replace(' ', '_')
                if current_section:
                    structured_response[current_section][current_subsection] = []
        
        return structured_response
        
    except Exception as e:
        logger.error(f"Error structuring text response: {str(e)}")
        return structured_response


def create_analysis_structure(analysis_type: str) -> Dict[str, Any]:
    """Create the analysis structure based on the content type."""
    base_structure = {
        "OVERVIEW": {
            "main_points": [],
            "key_metrics": [],
            "critical_developments": []
        },
        "TECHNICAL_ANALYSIS": {
            "patterns": [],
            "indicators": [],
            "key_levels": [],
            "trend_analysis": ""
        },
        "FUNDAMENTAL_FACTORS": {
            "market_drivers": [],
            "sector_impacts": [],
            "risk_factors": []
        },
        "TRADING_IMPLICATIONS": {
            "opportunities": [],
            "entry_points": [],
            "risk_levels": [],
            "time_horizons": []
        },
        "MARKET_IMPACT": {
            "immediate_reactions": {
                "price_movements": [],
                "volume_changes": [],
                "sentiment_shifts": []
            },
            "sector_specific": {
                "affected_sectors": [],
                "potential_winners": [],
                "potential_losers": []
            },
            "macroeconomic": {
                "market_trends": [],
                "economic_indicators": [],
                "global_factors": []
            },
            "sentiment": {
                "retail_sentiment": "",
                "institutional_outlook": "",
                "market_psychology": ""
            }
        },
        "RECOMMENDATIONS": {
            "trading_ideas": [],
            "portfolio_adjustments": [],
            "risk_management": [],
            "timing_considerations": []
        }
    }

    if analysis_type in ["table_content", "mixed_content"]:
        base_structure["DATA_ANALYSIS"] = {
            "table_insights": [],
            "key_metrics": [],
            "trend_analysis": [],
            "comparative_analysis": []
        }

    if analysis_type in ["chart_content", "mixed_content"]:
        base_structure["CHART_ANALYSIS"] = {
            "pattern_recognition": [],
            "support_resistance": [],
            "trend_indicators": [],
            "volume_analysis": []
        }

    return base_structure


def analyze_sentiment_indicators(text: str, indicators: Dict[str, List[str]]) -> Dict[str, Any]:
    """Analyze text for sentiment indicators."""
    try:
        bullish_count = sum(text.lower().count(word) for word in indicators['bullish'])
        bearish_count = sum(text.lower().count(word) for word in indicators['bearish'])
        total_count = bullish_count + bearish_count
        
        if total_count > 0:
            sentiment_score = (bullish_count - bearish_count) / total_count
            return {
                "score": sentiment_score,
                "direction": "bullish" if sentiment_score > 0 else "bearish",
                "strength": abs(sentiment_score),
                "confidence": min(abs(sentiment_score) * 100, 100),
                "indicator_counts": {
                    "bullish": bullish_count,
                    "bearish": bearish_count
                }
            }
        return {
            "score": 0,
            "direction": "neutral",
            "strength": 0,
            "confidence": 0,
            "indicator_counts": {"bullish": 0, "bearish": 0}
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {"error": str(e)}


def detect_trend(data: pd.Series) -> Optional[Dict[str, Any]]:
    """Detect price trends in a time series."""
    try:
        returns = data.pct_change()
        recent_trend = returns.tail(5).mean()
        
        return {
            'direction': 'bullish' if recent_trend > 0 else 'bearish',
            'strength': float(abs(recent_trend)),
            'period': '5 periods',
            'confidence': float(min(abs(recent_trend) * 100, 100))
        }
    except Exception:
        return None


def calculate_sentiment(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calculate overall market sentiment."""
    try:
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # Check price movements
        for change in analysis.get('price_changes', []):
            if change['change'] > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            total_signals += 1

        # Check trends
        for trend in analysis.get('trend_analysis', []):
            if trend['direction'] == 'bullish':
                bullish_signals += 2
            else:
                bearish_signals += 2
            total_signals += 2

        if total_signals > 0:
            sentiment_score = (bullish_signals - bearish_signals) / total_signals
            return {
                'score': float(sentiment_score),
                'direction': 'bullish' if sentiment_score > 0 else 'bearish',
                'strength': float(abs(sentiment_score)),
                'confidence': float(min(abs(sentiment_score) * 100, 100))
            }
        return None
    except Exception:
        return None


def get_error_response(url: str, content_type: str, error: str) -> Dict[str, Any]:
    """Get a structured error response."""
    return {
        "error": str(error),
        "url": url,
        "content_type": content_type,
        "timestamp": datetime.now().isoformat(),
        "status": "error",
        "summary": "Error occurred during analysis",
        "details": {
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
    }


def determine_content_type(content: Dict[str, Any]) -> str:
    """Determine the type of content for analysis."""
    has_tables = bool(content.get('tables'))
    has_charts = bool(content.get('charts'))
    # has_text is not directly used, but can be helpful logic if needed
    # has_text = bool(content.get('text'))

    if has_tables and has_charts:
        return 'mixed_content_analysis'
    elif has_tables:
        return 'table_content_analysis'
    elif has_charts:
        return 'chart_content_analysis'
    else:
        return 'text_only_analysis'


def format_analysis_prompt(url: str, content_type: str, content_text: str, 
                           analysis_type: str, prompt_template: str, 
                           analysis_structure: Dict[str, Any]) -> str:
    """Format the analysis prompt with all necessary components."""
    return f"""
    Analyze this {analysis_type} content and provide professional trading insights:

    URL: {url}
    CONTENT TYPE: {content_type}
    
    CONTENT:
    {content_text}

    {prompt_template}

    Please structure your analysis according to the following JSON format:
    {json.dumps(analysis_structure, indent=2)}

    Requirements:
    1. Provide specific, actionable trading insights
    2. Include quantitative metrics where available
    3. Highlight key risk factors and market implications
    4. Focus on time-sensitive opportunities
    5. Include specific price levels and market conditions
    6. Provide clear entry/exit strategies
    """


def analyze_tables(tables: List[dict]) -> Dict[str, Any]:
    """Analyze numerical data from tables."""
    analysis = {
        'numerical_insights': [],
        'price_changes': [],
        'percentage_movements': [],
        'key_statistics': {}
    }

    try:
        for table in tables:
            try:
                df = pd.DataFrame(table)
                
                # Look for price-related columns
                price_cols = [col for col in df.columns if any(
                    term in col.lower() for term in ['price', 'value', 'close', 'open']
                )]
                
                # Look for percentage columns
                pct_cols = [col for col in df.columns if any(
                    term in col.lower() for term in ['%', 'percent', 'change']
                )]

                process_price_columns(df, price_cols, analysis)
                process_percentage_columns(df, pct_cols, analysis)

            except Exception as e:
                logger.warning(f"Error processing individual table: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in table analysis: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    return analysis


def process_price_columns(df: pd.DataFrame, price_cols: List[str], analysis: Dict[str, Any]):
    """Process price-related columns in a DataFrame."""
    for col in price_cols:
        try:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            valid_data = numeric_data.dropna()
            
            if not valid_data.empty:
                analysis['key_statistics'][col] = {
                    'mean': float(valid_data.mean()),
                    'max': float(valid_data.max()),
                    'min': float(valid_data.min()),
                    'std': float(valid_data.std())
                }
        except Exception as e:
            logger.warning(f"Error processing price column {col}: {str(e)}")


def process_percentage_columns(df: pd.DataFrame, pct_cols: List[str], analysis: Dict[str, Any]):
    """Process percentage-related columns in a DataFrame."""
    for col in pct_cols:
        try:
            pct_data = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100
            valid_pct = pct_data.dropna()
            
            if not valid_pct.empty:
                analysis['percentage_movements'].extend([
                    {
                        'column': col,
                        'value': float(val),
                        'index': str(idx)
                    } for idx, val in valid_pct.items() if abs(val) > 0.01
                ])
        except Exception as e:
            logger.warning(f"Error processing percentage column {col}: {str(e)}")


def get_analysis_template(prompts: Dict[str, Any], has_tables: bool, has_charts: bool) -> tuple[str, str]:
    """Get appropriate analysis template based on content type."""
    if has_tables and has_charts:
        return "mixed_content", prompts.get("mixed_content_analysis", prompts["text_only_analysis"])
    elif has_tables:
        return "table_content", prompts.get("table_content_analysis", prompts["text_only_analysis"])
    elif has_charts:
        return "chart_content", prompts.get("chart_content_analysis", prompts["text_only_analysis"])
    else:
        return "text_only", prompts["text_only_analysis"]


def clean_response_content(content: str) -> str:
    """Clean and prepare response content for parsing."""
    content = content.strip()
    content = re.sub(r'^```json\s*', '', content)
    content = re.sub(r'^```\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    return content


def get_fallback_response(analysis_type: str) -> Dict[str, Any]:
    """Get a structured fallback response."""
    return {
        "error": "Analysis failed after multiple attempts",
        "analysis_type": analysis_type,
        "timestamp": datetime.now().isoformat(),
        "structure": create_analysis_structure(analysis_type)
    }
