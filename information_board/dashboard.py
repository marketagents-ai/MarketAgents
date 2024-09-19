# Install required packages
# You may need to install dash-bootstrap-components
# pip install dash-bootstrap-components

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import dash_bootstrap_components as dbc

# Sample JSON data for users and posts
users_json = '''
[
    {"user_id": 1, "username": "Alice"},
    {"user_id": 2, "username": "Bob"},
    {"user_id": 3, "username": "Charlie"},
    {"user_id": 4, "username": "David"},
    {"user_id": 5, "username": "Eve"}
]
'''

posts_json = '''
[
    {"post_id": 1, "user_id": 1, "content": "Post content 1", "post_type": "informative", "upvotes": 10, "downvotes": 2},
    {"post_id": 2, "user_id": 2, "content": "Post content 2", "post_type": "cooperative", "upvotes": 15, "downvotes": 3},
    {"post_id": 3, "user_id": 3, "content": "Post content 3", "post_type": "deceptive", "upvotes": 5, "downvotes": 8},
    {"post_id": 4, "user_id": 4, "content": "Post content 4", "post_type": "informative", "upvotes": 20, "downvotes": 1},
    {"post_id": 5, "user_id": 5, "content": "Post content 5", "post_type": "cooperative", "upvotes": 7, "downvotes": 2},
    {"post_id": 6, "user_id": 1, "content": "Post content 6", "post_type": "deceptive", "upvotes": 3, "downvotes": 12},
    {"post_id": 7, "user_id": 2, "content": "Post content 7", "post_type": "informative", "upvotes": 13, "downvotes": 4},
    {"post_id": 8, "user_id": 3, "content": "Post content 8", "post_type": "cooperative", "upvotes": 8, "downvotes": 3},
    {"post_id": 9, "user_id": 4, "content": "Post content 9", "post_type": "deceptive", "upvotes": 2, "downvotes": 10},
    {"post_id": 10, "user_id": 5, "content": "Post content 10", "post_type": "informative", "upvotes": 18, "downvotes": 0}
]
'''

users_df = pd.read_json(users_json)
posts_df = pd.read_json(posts_json)

comments_df = pd.DataFrame({
    'comment_id': range(1, 21),
    'post_id': np.random.randint(1, 11, size=20),
    'user_id': np.random.randint(1, 6, size=20),
    'content': ['Comment content {}'.format(i) for i in range(1, 21)],
    'upvotes': np.random.randint(0, 10, size=20),
    'downvotes': np.random.randint(0, 5, size=20)
})

posts_df['karma'] = posts_df['upvotes'] - posts_df['downvotes']
comments_df['karma'] = comments_df['upvotes'] - comments_df['downvotes']

def calculate_user_karma(posts_df, comments_df, users_df):
    post_karma = posts_df.groupby('user_id')['karma'].sum().reset_index()
    comment_karma = comments_df.groupby('user_id')['karma'].sum().reset_index()
    user_karma = pd.merge(post_karma, comment_karma, on='user_id', how='outer', suffixes=('_post', '_comment'))
    user_karma = user_karma.fillna(0)
    user_karma['total_karma'] = user_karma['karma_post'] + user_karma['karma_comment']
    user_karma = pd.merge(user_karma, users_df, on='user_id', how='left')
    return user_karma

user_karma_df = calculate_user_karma(posts_df, comments_df, users_df)

# Merge Posts with User Karma
posts_with_karma_df = pd.merge(posts_df, user_karma_df[['user_id', 'total_karma']], on='user_id', how='left')
posts_with_karma_df = posts_with_karma_df.fillna({'total_karma': 0})

# Sorting Functions
def sort_posts_by_user_karma(posts_df):
    return posts_df.sort_values(by='total_karma', ascending=False)

def sort_posts_by_post_karma(posts_df):
    return posts_df.sort_values(by='karma', ascending=False)

def sort_posts_by_combined_karma(posts_df, weight_user=0.5, weight_post=0.5):
    posts_df['combined_karma'] = weight_user * posts_df['total_karma'] + weight_post * posts_df['karma']
    return posts_df.sort_values(by='combined_karma', ascending=False)

# Initialize Dash app with Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]  # You can choose other themes from dbc.themes
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Customizing the layout
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Community Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        className='mb-4'
    ),
    dbc.Row([
        dbc.Col([
            html.Label('Sort Posts By:', className='h5'),
            dcc.Dropdown(
                id='sort-option',
                options=[
                    {'label': 'User Karma', 'value': 'user_karma'},
                    {'label': 'Post Karma', 'value': 'post_karma'},
                    {'label': 'Combined Karma', 'value': 'combined_karma'}
                ],
                value='user_karma',
                className='mb-4'
            ),
        ], width=4),
        dbc.Col([
            dbc.Alert(
                "Earn more karma by contributing valuable posts and comments!",
                color="info",
                dismissable=True,
                className='mt-2'
            )
        ], width=8),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='posts-graph')
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Users Karma', className='mt-4'),
            dcc.Graph(id='users-graph')
        ], width=12),
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output('posts-graph', 'figure'),
    Output('users-graph', 'figure'),
    Input('sort-option', 'value')
)
def update_graph(sort_option):
    if sort_option == 'user_karma':
        sorted_posts = sort_posts_by_user_karma(posts_with_karma_df)
    elif sort_option == 'post_karma':
        sorted_posts = sort_posts_by_post_karma(posts_with_karma_df)
    else:
        sorted_posts = sort_posts_by_combined_karma(posts_with_karma_df)
    
    # Posts Graph
    fig_posts = px.bar(
        sorted_posts,
        x='post_id',
        y='karma',
        color='post_type',
        hover_data=['content', 'total_karma'],
        title='Posts Sorted by {}'.format(sort_option.replace('_', ' ').title()),
        labels={'karma': 'Post Karma', 'post_id': 'Post ID'},
        color_discrete_map={
            'informative': 'green',
            'cooperative': 'blue',
            'deceptive': 'red'
        }
    )
    fig_posts.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='LightGray')
    )

    # Users Graph
    user_karma_df_sorted = user_karma_df.sort_values(by='total_karma', ascending=False)
    fig_users = px.bar(
        user_karma_df_sorted,
        x='username',
        y='total_karma',
        title='User Karma',
        labels={'total_karma': 'Total Karma', 'username': 'Username'},
        color='total_karma',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_users.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='LightGray'),
        coloraxis_showscale=False
    )

    return fig_posts, fig_users

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)