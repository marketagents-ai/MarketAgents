import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.market_agent_team import MarketAgentTeam
from market_agents.tools.callable_tool import CallableTool
from market_agents.tools.structured_tool import StructuredTool
from market_agents.workflows.workflow import Workflow, WorkflowStep
from minference.lite.models import LLMConfig, ResponseFormat

# Define structured output models
class CustomerProfile(BaseModel):
    """Customer profile with demographic and behavioral information."""
    
    customer_id: str = Field(..., description="Unique customer identifier")
    name: str = Field(..., description="Customer's full name")
    age: int = Field(..., description="Customer's age")
    location: str = Field(..., description="Customer's location (city, state)")
    occupation: str = Field(..., description="Customer's occupation")
    income_bracket: str = Field(..., description="Income bracket (Low, Medium, High)")
    
    # Shopping behavior
    purchase_frequency: str = Field(..., description="How often the customer makes purchases (Rarely, Occasionally, Frequently)")
    average_order_value: float = Field(..., description="Average order value in dollars")
    preferred_categories: List[str] = Field(..., description="Product categories the customer frequently purchases")
    shopping_channels: List[str] = Field(..., description="Preferred shopping channels (Online, In-store, Mobile app)")
    
    # Preferences and interests
    interests: List[str] = Field(..., description="Customer's interests and hobbies")
    brand_preferences: List[str] = Field(..., description="Preferred brands")
    price_sensitivity: str = Field(..., description="Price sensitivity level (Low, Medium, High)")
    
    # Engagement metrics
    email_engagement_rate: float = Field(..., description="Email open and click-through rate percentage")
    social_media_engagement: str = Field(..., description="Level of social media engagement (Low, Medium, High)")
    loyalty_program_status: str = Field(..., description="Status in loyalty program (None, Basic, Premium, VIP)")
    
    # Customer value
    customer_lifetime_value: float = Field(..., description="Estimated customer lifetime value in dollars")
    churn_risk: str = Field(..., description="Risk of customer churn (Low, Medium, High)")
    growth_potential: str = Field(..., description="Potential for increased spending (Low, Medium, High)")

class ProductRecommendation(BaseModel):
    """Product recommendation with rationale and expected outcomes."""
    
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    price: float = Field(..., description="Product price in dollars")
    
    # Recommendation details
    recommendation_strength: str = Field(..., description="Strength of recommendation (Low, Medium, High)")
    primary_benefit: str = Field(..., description="Primary benefit for the customer")
    fit_score: float = Field(..., description="How well the product fits the customer profile (0-100)")
    
    # Rationale
    customer_need_addressed: str = Field(..., description="Customer need or pain point addressed by this product")
    behavioral_evidence: str = Field(..., description="Evidence from customer behavior supporting this recommendation")
    preference_alignment: str = Field(..., description="How the product aligns with customer preferences")
    
    # Expected outcomes
    purchase_probability: float = Field(..., description="Estimated probability of purchase (0-100)")
    expected_satisfaction: float = Field(..., description="Expected customer satisfaction if purchased (0-100)")
    cross_sell_opportunities: List[str] = Field(..., description="Potential cross-sell opportunities if this product is purchased")

class MarketingCampaign(BaseModel):
    """Marketing campaign design with targeting, messaging, and metrics."""
    
    campaign_name: str = Field(..., description="Name of the marketing campaign")
    target_audience: str = Field(..., description="Description of the target audience")
    campaign_objective: str = Field(..., description="Primary objective of the campaign")
    
    # Campaign details
    channels: List[str] = Field(..., description="Marketing channels to be used")
    timing: str = Field(..., description="Timing and duration of the campaign")
    budget_allocation: Dict[str, float] = Field(..., description="Budget allocation across channels (percentages)")
    
    # Messaging
    primary_message: str = Field(..., description="Primary campaign message")
    key_selling_points: List[str] = Field(..., description="Key selling points to emphasize")
    call_to_action: str = Field(..., description="Primary call to action")
    
    # Creative approach
    creative_direction: str = Field(..., description="Overall creative direction and tone")
    visual_elements: List[str] = Field(..., description="Key visual elements to include")
    
    # Expected performance
    expected_reach: int = Field(..., description="Expected campaign reach (number of people)")
    expected_engagement_rate: float = Field(..., description="Expected engagement rate percentage")
    expected_conversion_rate: float = Field(..., description="Expected conversion rate percentage")
    expected_roi: float = Field(..., description="Expected return on investment (ROI) percentage")
    
    # Success metrics
    primary_kpis: List[str] = Field(..., description="Primary key performance indicators (KPIs)")
    success_criteria: str = Field(..., description="Criteria for determining campaign success")

# Define callable tools
def get_customer_data(customer_id: str) -> Dict[str, Any]:
    """
    Retrieve customer data for the specified customer ID.
    
    Args:
        customer_id: Unique customer identifier
        
    Returns:
        Dictionary containing customer data
    """
    # In a real implementation, this would call an API or database
    # For this example, we'll return mock data
    
    customer_data = {
        "C1001": {
            "customer_id": "C1001",
            "name": "Sarah Johnson",
            "age": 34,
            "location": "Seattle, WA",
            "occupation": "Software Engineer",
            "income_bracket": "High",
            "purchase_frequency": "Frequently",
            "average_order_value": 85.50,
            "preferred_categories": ["Electronics", "Books", "Home Office"],
            "shopping_channels": ["Online", "Mobile app"],
            "interests": ["Technology", "Reading", "Hiking", "Photography"],
            "brand_preferences": ["Apple", "Patagonia", "Allbirds"],
            "price_sensitivity": "Medium",
            "email_engagement_rate": 68.5,
            "social_media_engagement": "Medium",
            "loyalty_program_status": "Premium",
            "customer_lifetime_value": 3850.00,
            "churn_risk": "Low",
            "growth_potential": "High"
        },
        "C1002": {
            "customer_id": "C1002",
            "name": "Michael Rodriguez",
            "age": 42,
            "location": "Chicago, IL",
            "occupation": "Marketing Director",
            "income_bracket": "High",
            "purchase_frequency": "Occasionally",
            "average_order_value": 120.75,
            "preferred_categories": ["Men's Apparel", "Fitness", "Travel Gear"],
            "shopping_channels": ["Online", "In-store"],
            "interests": ["Fitness", "Travel", "Cooking", "Sports"],
            "brand_preferences": ["Nike", "Yeti", "Bose"],
            "price_sensitivity": "Low",
            "email_engagement_rate": 42.0,
            "social_media_engagement": "Low",
            "loyalty_program_status": "Basic",
            "customer_lifetime_value": 2950.00,
            "churn_risk": "Medium",
            "growth_potential": "Medium"
        },
        "C1003": {
            "customer_id": "C1003",
            "name": "Emily Chen",
            "age": 28,
            "location": "San Francisco, CA",
            "occupation": "UX Designer",
            "income_bracket": "Medium",
            "purchase_frequency": "Frequently",
            "average_order_value": 65.25,
            "preferred_categories": ["Beauty", "Women's Apparel", "Home Decor"],
            "shopping_channels": ["Mobile app", "Online", "In-store"],
            "interests": ["Art", "Fashion", "Yoga", "Interior Design"],
            "brand_preferences": ["Sephora", "Lululemon", "West Elm"],
            "price_sensitivity": "Medium",
            "email_engagement_rate": 82.0,
            "social_media_engagement": "High",
            "loyalty_program_status": "VIP",
            "customer_lifetime_value": 4200.00,
            "churn_risk": "Low",
            "growth_potential": "High"
        }
    }
    
    if customer_id in customer_data:
        return customer_data[customer_id]
    else:
        return {
            "customer_id": customer_id,
            "name": "Unknown Customer",
            "error": f"No data available for customer ID {customer_id}"
        }

def get_product_catalog(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve product catalog, optionally filtered by category.
    
    Args:
        category: Optional product category to filter by
        
    Returns:
        Dictionary containing product catalog data
    """
    # In a real implementation, this would call an API or database
    # For this example, we'll return mock data
    
    product_catalog = {
        "Electronics": [
            {
                "product_id": "P1001",
                "product_name": "Wireless Noise-Cancelling Headphones",
                "category": "Electronics",
                "price": 249.99,
                "features": ["Active noise cancellation", "30-hour battery life", "Voice assistant compatible"],
                "average_rating": 4.7,
                "in_stock": True
            },
            {
                "product_id": "P1002",
                "product_name": "Smart Home Hub",
                "category": "Electronics",
                "price": 129.99,
                "features": ["Voice control", "Compatible with major smart home devices", "Energy monitoring"],
                "average_rating": 4.3,
                "in_stock": True
            }
        ],
        "Books": [
            {
                "product_id": "P2001",
                "product_name": "The Future of Technology",
                "category": "Books",
                "price": 24.95,
                "features": ["Bestseller", "Written by industry expert", "Includes digital companion"],
                "average_rating": 4.8,
                "in_stock": True
            },
            {
                "product_id": "P2002",
                "product_name": "Productivity Masterclass",
                "category": "Books",
                "price": 19.99,
                "features": ["Practical techniques", "Case studies", "Workbook included"],
                "average_rating": 4.5,
                "in_stock": True
            }
        ],
        "Home Office": [
            {
                "product_id": "P3001",
                "product_name": "Ergonomic Desk Chair",
                "category": "Home Office",
                "price": 299.99,
                "features": ["Adjustable height", "Lumbar support", "Breathable mesh"],
                "average_rating": 4.6,
                "in_stock": True
            },
            {
                "product_id": "P3002",
                "product_name": "Adjustable Standing Desk",
                "category": "Home Office",
                "price": 449.99,
                "features": ["Electric height adjustment", "Memory settings", "Cable management"],
                "average_rating": 4.4,
                "in_stock": False
            }
        ],
        "Beauty": [
            {
                "product_id": "P4001",
                "product_name": "Premium Skincare Set",
                "category": "Beauty",
                "price": 89.99,
                "features": ["Natural ingredients", "Suitable for sensitive skin", "Cruelty-free"],
                "average_rating": 4.9,
                "in_stock": True
            },
            {
                "product_id": "P4002",
                "product_name": "Luxury Makeup Collection",
                "category": "Beauty",
                "price": 129.99,
                "features": ["Long-lasting", "Hypoallergenic", "Includes brush set"],
                "average_rating": 4.7,
                "in_stock": True
            }
        ]
    }
    
    if category and category in product_catalog:
        return {category: product_catalog[category]}
    elif category:
        return {"error": f"No products found in category {category}"}
    else:
        return product_catalog

def get_campaign_performance(channel: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve historical campaign performance data, optionally filtered by channel.
    
    Args:
        channel: Optional marketing channel to filter by
        
    Returns:
        Dictionary containing campaign performance data
    """
    # In a real implementation, this would call an API or database
    # For this example, we'll return mock data
    
    campaign_performance = {
        "Email": {
            "average_open_rate": 22.5,
            "average_click_rate": 3.8,
            "average_conversion_rate": 2.1,
            "average_roi": 420,
            "best_performing_segment": "Loyalty Program Members",
            "best_performing_time": "Tuesday mornings",
            "recent_trend": "Stable with slight improvement"
        },
        "Social Media": {
            "average_engagement_rate": 3.2,
            "average_click_rate": 1.5,
            "average_conversion_rate": 1.2,
            "average_roi": 380,
            "best_performing_platform": "Instagram",
            "best_performing_content": "User-generated content",
            "recent_trend": "Improving engagement but rising costs"
        },
        "Search": {
            "average_click_through_rate": 4.5,
            "average_cost_per_click": 1.85,
            "average_conversion_rate": 3.2,
            "average_roi": 520,
            "best_performing_keywords": "Brand terms and specific product searches",
            "recent_trend": "Stable performance with increasing competition"
        },
        "Display": {
            "average_click_through_rate": 0.8,
            "average_cost_per_click": 0.95,
            "average_conversion_rate": 0.7,
            "average_roi": 280,
            "best_performing_format": "Native ads",
            "recent_trend": "Declining performance, needs optimization"
        }
    }
    
    if channel and channel in campaign_performance:
        return {channel: campaign_performance[channel]}
    elif channel:
        return {"error": f"No performance data found for channel {channel}"}
    else:
        return campaign_performance

async def create_marketing_agent_with_workflow():
    """Create a marketing agent with a structured workflow."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Marketing Strategist",
        persona="I am a marketing strategist specializing in customer-centric marketing campaigns. I analyze customer data, develop personalized recommendations, and design targeted marketing campaigns to drive engagement and conversions.",
        objectives=[
            "Analyze customer profiles to identify needs and preferences",
            "Develop personalized product recommendations",
            "Design targeted marketing campaigns",
            "Optimize marketing performance and ROI"
        ],
        communication_style="Strategic, data-driven, and customer-focused",
        skills=[
            "Customer analysis",
            "Product recommendation",
            "Campaign design",
            "Marketing optimization"
        ]
    )
    
    # Create callable tools
    customer_data_tool = CallableTool(
        name="get_customer_data",
        description="Retrieve customer data for a specific customer ID",
        function=get_customer_data,
        parameters={"customer_id": "str"}
    )
    
    product_catalog_tool = CallableTool(
        name="get_product_catalog",
        description="Retrieve product catalog, optionally filtered by category",
        function=get_product_catalog,
        parameters={"category": "Optional[str]"}
    )
    
    campaign_performance_tool = CallableTool(
        name="get_campaign_performance",
        description="Retrieve historical campaign performance data, optionally filtered by channel",
        function=get_campaign_performance,
        parameters={"channel": "Optional[str]"}
    )
    
    # Create structured tools
    customer_profile_tool = StructuredTool(
        name="analyze_customer",
        description="Analyze customer data and create a comprehensive customer profile",
        output_schema=CustomerProfile
    )
    
    product_recommendation_tool = StructuredTool(
        name="recommend_products",
        description="Generate personalized product recommendations for a customer",
        output_schema=ProductRecommendation
    )
    
    marketing_campaign_tool = StructuredTool(
        name="design_campaign",
        description="Design a targeted marketing campaign for a customer",
        output_schema=MarketingCampaign
    )
    
    # Define workflow steps
    workflow_steps = [
        WorkflowStep(
            name="analyze_customer_profile",
            description="Analyze customer data to create a comprehensive customer profile",
            prompt_template="""
            You are a marketing strategist analyzing customer data to create a comprehensive customer profile.
            
            First, use the get_customer_data tool to retrieve data for customer ID {customer_id}.
            Then, analyze this data to create a detailed customer profile using the analyze_customer structured tool.
            
            Your analysis should include:
            - Demographic information
            - Shopping behavior and preferences
            - Engagement metrics and loyalty status
            - Customer value and growth potential
            
            Provide a comprehensive and accurate customer profile based on the available data.
            """,
            required_tools=["get_customer_data", "analyze_customer"],
            output_key="customer_profile"
        ),
        WorkflowStep(
            name="generate_product_recommendations",
            description="Generate personalized product recommendations based on customer profile",
            prompt_template="""
            You are a marketing strategist generating personalized product recommendations.
            
            Review the customer profile from the previous step:
            {customer_profile}
            
            Use the get_product_catalog tool to retrieve relevant product categories based on the customer's preferred categories.
            Then, use the recommend_products structured tool to generate personalized product recommendations.
            
            Your recommendations should:
            - Align with the customer's preferences and interests
            - Address specific customer needs
            - Have a strong rationale based on customer behavior
            - Include expected outcomes and benefits
            
            Generate the top product recommendation for this customer.
            """,
            required_tools=["get_product_catalog", "recommend_products"],
            output_key="product_recommendation"
        ),
        WorkflowStep(
            name="design_marketing_campaign",
            description="Design a targeted marketing campaign based on customer profile and product recommendations",
            prompt_template="""
            You are a marketing strategist designing a targeted marketing campaign.
            
            Review the customer profile and product recommendation from previous steps:
            Customer Profile: {customer_profile}
            Product Recommendation: {product_recommendation}
            
            Use the get_campaign_performance tool to retrieve historical performance data for relevant marketing channels.
            Then, use the design_campaign structured tool to create a targeted marketing campaign.
            
            Your campaign design should:
            - Target the specific customer effectively
            - Promote the recommended product appropriately
            - Use channels that align with customer preferences
            - Include compelling messaging and creative direction
            - Set realistic performance expectations
            
            Design a personalized marketing campaign for this customer and product recommendation.
            """,
            required_tools=["get_campaign_performance", "design_campaign"],
            output_key="marketing_campaign"
        )
    ]
    
    # Create workflow
    marketing_workflow = Workflow(
        name="personalized_marketing_workflow",
        description="A workflow for creating personalized marketing campaigns",
        steps=workflow_steps
    )
    
    # Create agent with tools and workflow
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="marketing_strategist",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7
        ),
        persona=persona,
        tools=[
            customer_data_tool,
            product_catalog_tool,
            campaign_performance_tool,
            customer_profile_tool,
            product_recommendation_tool,
            marketing_campaign_tool
        ],
        workflows=[marketing_workflow]
    )
    
    print(f"Created Marketing Strategist with ID: {agent.id}")
    print(f"Available tools: {[tool.name for tool in agent.tools]}")
    print(f"Available workflows: {[workflow.name for workflow in agent.workflows]}")
    
    return agent

async def run_marketing_workflow(agent, customer_id):
    """Run the personalized marketing workflow for a specific customer."""
    
    print(f"\nRunning personalized marketing workflow for customer ID: {customer_id}")
    print("-" * 80)
    
    # Execute the workflow
    workflow_result = await agent.execute_workflow(
        workflow_name="personalized_marketing_workflow",
        inputs={"customer_id": customer_id}
    )
    
    print("\nWorkflow Execution Complete")
    print("-" * 80)
    
    # Extract and display results
    customer_profile = workflow_result.get("customer_profile")
    product_recommendation = workflow_result.get("product_recommendation")
    marketing_campaign = workflow_result.get("marketing_campaign")
    
    print("\nCustomer Profile:")
    print(f"Name: {customer_profile.name}")
    print(f"Age: {customer_profile.age}")
    print(f"Location: {customer_profile.location}")
    print(f"Occupation: {customer_profile.occupation}")
    print(f"Income Bracket: {customer_profile.income_bracket}")
    print(f"Preferred Categories: {', '.join(customer_profile.preferred_categories)}")
    print(f"Interests: {', '.join(customer_profile.interests)}")
    print(f"Customer Lifetime Value: ${customer_profile.customer_lifetime_value:.2f}")
    print(f"Churn Risk: {customer_profile.churn_risk}")
    print(f"Growth Potential: {customer_profile.growth_potential}")
    
    print("\nProduct Recommendation:")
    print(f"Product: {product_recommendation.product_name}")
    print(f"Category: {product_recommendation.category}")
    print(f"Price: ${product_recommendation.price:.2f}")
    print(f"Recommendation Strength: {product_recommendation.recommendation_strength}")
    print(f"Fit Score: {product_recommendation.fit_score}/100")
    print(f"Primary Benefit: {product_recommendation.primary_benefit}")
    print(f"Customer Need Addressed: {product_recommendation.customer_need_addressed}")
    print(f"Purchase Probability: {product_recommendation.purchase_probability}%")
    
    print("\nMarketing Campaign:")
    print(f"Campaign Name: {marketing_campaign.campaign_name}")
    print(f"Objective: {marketing_campaign.campaign_objective}")
    print(f"Channels: {', '.join(marketing_campaign.channels)}")
    print(f"Primary Message: {marketing_campaign.primary_message}")
    print(f"Call to Action: {marketing_campaign.call_to_action}")
    print(f"Expected Conversion Rate: {marketing_campaign.expected_conversion_rate}%")
    print(f"Expected ROI: {marketing_campaign.expected_roi}%")
    print(f"Primary KPIs: {', '.join(marketing_campaign.primary_kpis)}")
    
    return workflow_result

async def main():
    # Create a marketing agent with workflow
    agent = await create_marketing_agent_with_workflow()
    
    # Run the workflow for different customers
    customer_ids = ["C1001", "C1002", "C1003"]
    
    for customer_id in customer_ids:
        await run_marketing_workflow(agent, customer_id)

if __name__ == "__main__":
    import json
    asyncio.run(main())
