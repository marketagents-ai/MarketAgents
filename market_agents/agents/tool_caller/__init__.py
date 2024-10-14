from market_agents.agents.tool_caller.engine import Engine
from market_agents.environments.mechanisms.information_board import add_post, get_all_posts, upvote_post, downvote_post

# Initialize the tool caller with the available tools
TOOL_CALLER = Engine()
TOOL_CALLER.add_tools([add_post, get_all_posts, upvote_post, downvote_post])