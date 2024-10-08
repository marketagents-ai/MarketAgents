from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import asyncio

# Initialize FastAPI app
app = FastAPI()

class PostType(str, Enum):
    """
    Enum for the type of post. Currrently

    - Cooperative: A post that is cooperative in nature, meaning it is 
    could be a proposal or suggestion to other agents to cooperate with the agent.
    - Informative: A post that is informative in nature, meaning it 
    provides information that is beneficial to other agents.
    - Deceptive: A post that is deceptive in nature, meaning the agent
    aims to gain something at the expense of other agents.
    """
    COOPERATIVE = "Cooperative"
    INFORMATIVE = "Informative"
    DECEPTIVE = "Deceptive"

class User(BaseModel):
    """
    User class for the information board.

    - id: The id of the user.
    - name: The name of the user.
    - karma: The karma of the user.
    """
    id: int
    name: str
    karma: int = 0

class Post(BaseModel):
    """
    Post class for the information board.

    - id: The id of the post.
    - title: The title of the post.
    - content: The content of the post.
    - user_id: The id of the poster agent.
    - karma: The karma of the post.
    - date_posted: The date the post was posted.
    - categories: The categories the post is filed under.
    - post_type: The type of post.
    """
    id: int
    title: str
    content: str
    user_id: int
    date_posted: datetime = Field(default_factory=datetime.now)
    categories: List[str] = []
    post_type: PostType

class Category(BaseModel):
    """
    Category class for the information board.

    - name: The name of the category.
    """
    name: str

# Mock database
users = [
    User(id=1, name="Alice", karma=10),
    User(id=2, name="Bob", karma=-5)
]
posts = [
    Post(id=1, title="Good Post", content="This is a good post", user_id=1, karma=5, categories=["Test Category 1", "Test Category 2"], post_type=PostType.INFORMATIVE),
    Post(id=2, title="Bad Post", content="This is a bad post", user_id=1, karma=-3, categories=["Test Category 2"], post_type=PostType.DECEPTIVE),
    Post(id=3, title="Another Good Post", content="This is another good post", user_id=2, karma=2, categories=["Test Category 3"], post_type=PostType.COOPERATIVE),
    Post(id=4, title="Another Bad Post", content="This is another bad post", user_id=2, karma=-1, categories=["Test Category 1"], post_type=PostType.DECEPTIVE)
]
categories = [Category(name="Test Category 1"), Category(name="Test Category 2"), Category(name="Test Category 3")]

@app.post("/posts/")
async def add_post(post: Post):
    """
    Add a post to the information board.

    Args:
        post: The post to add.
    """
    # Mock implementation on mock db
    post.id = len(posts) + 1
    posts.append(post)
    return {"message": "Post added successfully", "post_id": post.id}

@app.get("/posts/")
async def get_all_posts(
    count: Optional[int] = Query(None, description="Number of posts to return"),
    title: Optional[str] = Query(None, description="Filter posts by title"),
    categories: Optional[List[str]] = Query(None, description="Filter posts by categories"),
    order_by: str = Query("karma", description="Order posts by 'karma' or 'date'")
):
    """
    Get all posts from the information board
    with various filters and sorting options.

    Args:
        count: The number of posts to return.
        title: The title of the posts to return.
        categories: The categories of the posts to return.
        order_by: The order of the posts to return.
    """
    filtered_posts = posts

    if title:
        filtered_posts = [post for post in filtered_posts if title.lower() in post.title.lower()]

    if categories:
        filtered_posts = [post for post in filtered_posts if any(cat in post.categories for cat in categories)]

    if order_by == "karma":
        filtered_posts.sort(key=lambda x: x.karma, reverse=True)
    elif order_by == "date":
        filtered_posts.sort(key=lambda x: x.date_posted, reverse=True)
    else:
        raise HTTPException(status_code=400, detail="Invalid order_by parameter")

    if count:
        filtered_posts = filtered_posts[:count]

    return filtered_posts

@app.put("/posts/{post_id}/upvote")
async def upvote_post(post_id: int):
    """
    Upvote a post on the information board.

    Args:
        post_id: The id of the post to upvote.
    """
    post = next((post for post in posts if post.id == post_id), None)
    if post:
        post.karma += 1
        return {"message": "Post upvoted successfully", "new_karma": post.karma}
    raise HTTPException(status_code=404, detail="Post not found")

@app.put("/posts/{post_id}/downvote")
async def downvote_post(post_id: int):
    """
    Downvote a post on the information board.

    Args:
        post_id: The id of the post to downvote.
    """
    post = next((post for post in posts if post.id == post_id), None)
    if post:
        post.karma -= 1
        return {"message": "Post downvoted successfully", "new_karma": post.karma}
    raise HTTPException(status_code=404, detail="Post not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
