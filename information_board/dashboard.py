# dashboard.py

# Ensure you have the required packages installed:
# pip install pandas numpy plotly dash dash-bootstrap-components

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import json
from io import StringIO  # Import StringIO for reading JSON strings
import dash

# Initialize the Dash app with a Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]  # You can choose other themes from dbc.themes
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Community Dashboard"

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

# Load data from JSON using StringIO to avoid FutureWarning
users_df = pd.read_json(StringIO(users_json))
posts_df = pd.read_json(StringIO(posts_json))

# Simulate Comments Data (you can adjust this to accept JSON as well)
np.random.seed(42)  # For reproducible results
comments_df = pd.DataFrame({
    'comment_id': range(1, 21),
    'post_id': np.random.randint(1, 11, size=20),
    'user_id': np.random.randint(1, 6, size=20),
    'content': ['Comment content {}'.format(i) for i in range(1, 21)],
    'upvotes': np.random.randint(0, 10, size=20),
    'downvotes': np.random.randint(0, 5, size=20)
})

# Calculate Karma
posts_df['karma'] = posts_df['upvotes'] - posts_df['downvotes']
comments_df['karma'] = comments_df['upvotes'] - comments_df['downvotes']

# Function to calculate user karma
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

# Add number of comments per post
comments_count = comments_df.groupby('post_id').size().reset_index(name='num_comments')
posts_with_karma_df = pd.merge(posts_with_karma_df, comments_count, on='post_id', how='left')
posts_with_karma_df = posts_with_karma_df.fillna({'num_comments': 0})

# Add a simulated date for posts (since we don't have actual dates)
# For demonstration, assume post_id correlates with time
posts_with_karma_df['date'] = pd.date_range(start='2023-01-01', periods=posts_with_karma_df.shape[0], freq='D')

# Define the Navbar
navbar = dbc.NavbarSimple(
    brand="Community Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
    fluid=True,
    className='mb-4'
)

# Define the Sidebar Navigation
sidebar = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Users Karma", href="/users-karma", active="exact"),
                dbc.NavLink("Posts Insights", href="/posts-insights", active="exact"),
                dbc.NavLink("Timeline", href="/timeline", active="exact"),
                dbc.NavLink("Comments", href="/comments", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 70,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
        "overflow-y": "auto"
    },
)

# Define the Content Area
content = html.Div(id="page-content", style={"margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem"})

# App Layout with navbar, sidebar, and content
app.layout = html.Div([
    dcc.Location(id="url"),
    navbar,
    sidebar,
    content
])

# Home Page Layout
home_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Welcome to the Community Dashboard", className='text-center mb-4')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Total Users", className="card-title"),
                            html.P(len(users_df), className="card-text")
                        ]),
                        color="primary", inverse=True
                    ),
                    width=3,
                    className='mb-4'
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Total Posts", className="card-title"),
                            html.P(len(posts_df), className="card-text")
                        ]),
                        color="success", inverse=True
                    ),
                    width=3,
                    className='mb-4'
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Total Comments", className="card-title"),
                            html.P(len(comments_df), className="card-text")
                        ]),
                        color="info", inverse=True
                    ),
                    width=3,
                    className='mb-4'
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Overall Karma", className="card-title"),
                            html.P(int(user_karma_df['total_karma'].sum()), className="card-text")
                        ]),
                        color="warning", inverse=True
                    ),
                    width=3,
                    className='mb-4'
                ),
            ])
        ], width=12)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='post-type-distribution',
                figure=px.pie(
                    posts_df,
                    names='post_type',
                    title='Distribution of Post Types',
                    color_discrete_map={
                        'informative': 'green',
                        'cooperative': 'blue',
                        'deceptive': 'red'
                    }
                ).update_traces(textposition='inside', textinfo='percent+label')
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='overall-karma-distribution',
                figure=px.bar(
                    pd.DataFrame({
                        'Category': ['Post Karma', 'Comment Karma'],
                        'Karma': [posts_df['karma'].sum(), comments_df['karma'].sum()]
                    }),
                    x='Category',
                    y='Karma',
                    title='Overall Karma Distribution',
                    color='Category',
                    barmode='group',
                    color_discrete_map={'Post Karma': 'green', 'Comment Karma': 'blue'}
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridcolor='LightGray')
                )
            )
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='monthly-karma-trend',
                figure=px.line(
                    posts_with_karma_df.resample('M', on='date').agg({'karma': 'sum', 'num_comments': 'sum'}),
                    x=posts_with_karma_df.resample('M', on='date').agg({'karma': 'sum'}).index,
                    y=['karma', 'num_comments'],
                    title='Monthly Total Karma and Comments',
                    labels={'value': 'Total', 'date': 'Month'},
                    markers=True
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridcolor='LightGray')
                )
            )
        ], width=12),
    ]),
], fluid=True)

# Users Karma Page Layout
users_karma_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Users Karma", className='text-center mb-4')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='user-karma-bar',
                figure=px.bar(
                    user_karma_df.sort_values(by='total_karma', ascending=False),
                    x='username',
                    y='total_karma',
                    title='User Karma Distribution',
                    labels={'total_karma': 'Total Karma', 'username': 'Username'},
                    color='total_karma',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    hover_data=['karma_post', 'karma_comment']
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='LightGray')
                )
            )
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Top Users", className='mt-4'),
            dbc.Table.from_dataframe(
                user_karma_df.sort_values(by='total_karma', ascending=False).head(10),
                striped=True,
                bordered=True,
                hover=True,
                className='text-center'
            )
        ], width=12)
    ])
], fluid=True)

# Posts Insights Page Layout
posts_insights_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Posts Insights", className='text-center mb-4')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Average Upvotes", className="card-title"),
                            html.P(f"{posts_df['upvotes'].mean():.2f}", className="card-text")
                        ]),
                        color="success", inverse=True
                    ),
                    width=4,
                    className='mb-4'
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Average Downvotes", className="card-title"),
                            html.P(f"{posts_df['downvotes'].mean():.2f}", className="card-text")
                        ]),
                        color="danger", inverse=True
                    ),
                    width=4,
                    className='mb-4'
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Average Comments", className="card-title"),
                            html.P(f"{posts_with_karma_df['num_comments'].mean():.2f}", className="card-text")
                        ]),
                        color="info", inverse=True
                    ),
                    width=4,
                    className='mb-4'
                ),
            ])
        ], width=12)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
id='post-type-pie',
                figure=px.pie(
                    posts_df,
                    names='post_type',
                    values='karma',
                    title='Post Types Contribution to Total Karma',
                    color='post_type',
                    color_discrete_map={
                        'informative': 'green',
                        'cooperative': 'blue',
                        'deceptive': 'red'
                    }
                ).update_traces(textposition='inside', textinfo='percent+label')
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id='karma-vs-comments-scatter',
                figure=px.scatter(
                    posts_with_karma_df,
                    x='karma',
                    y='num_comments',
                    color='post_type',
                    size='upvotes',
                    hover_data=['content', 'downvotes'],
                    title='Karma vs. Number of Comments',
                    labels={'karma': 'Karma', 'num_comments': 'Number of Comments'},
                    color_discrete_map={
                        'informative': 'green',
                        'cooperative': 'blue',
                        'deceptive': 'red'
                    }
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridcolor='LightGray')
                )
            )
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Top Performing Posts", className='mt-4'),
            dbc.Table.from_dataframe(
                posts_with_karma_df.sort_values(by='karma', ascending=False).head(10),
                striped=True,
                bordered=True,
                hover=True,
                className='text-center'
            )
        ], width=12)
    ])
], fluid=True)

# Timeline Page Layout
timeline_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Posts Timeline", className='text-center mb-4')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select Metric:', className='h5'),
            dcc.Dropdown(
                id='timeline-metric',
                options=[
                    {'label': 'Karma', 'value': 'karma'},
                    {'label': 'Number of Comments', 'value': 'num_comments'},
                    {'label': 'Upvotes', 'value': 'upvotes'},
                    {'label': 'Downvotes', 'value': 'downvotes'}
                ],
                value='karma',
                multi=False,
                className='mb-4'
            ),
        ], width=4),
        dbc.Col([
            html.Label('Post Type:', className='h5'),
            dcc.Dropdown(
                id='timeline-post-type',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Informative', 'value': 'informative'},
                    {'label': 'Cooperative', 'value': 'cooperative'},
                    {'label': 'Deceptive', 'value': 'deceptive'}
                ],
                value=['All'],
                multi=True,
                className='mb-4',
                placeholder="Select post types..."
            ),
        ], width=4),
        dbc.Col([
            html.Label('Date Range:', className='h5'),
            dcc.DatePickerRange(
                id='timeline-date-range',
                min_date_allowed=posts_with_karma_df['date'].min(),
                max_date_allowed=posts_with_karma_df['date'].max(),
                start_date=posts_with_karma_df['date'].min(),
                end_date=posts_with_karma_df['date'].max(),
                display_format='YYYY-MM-DD',
                className='mb-4'
            ),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='timeline-graph')
        ], width=12),
    ]),
], fluid=True)

# Comments Page Layout
comments_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Posts and Their Comments", className='text-center mb-4')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Search Posts"),
                dbc.Input(id='search-posts', type='text', placeholder='Enter post ID or keyword...', debounce=True),
            ], className='mb-4')
        ], width=6),
        dbc.Col([
            dbc.Button("Reset Filters", id='reset-filters', color='secondary', className='mb-4')
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='posts-comments-scatter')
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Select a Post to View Comments", className='mt-4'),
            dash_table.DataTable(
                id='posts-table',
                columns=[
                    {"name": i, "id": i} for i in ['post_id', 'user_id', 'content', 'post_type', 'upvotes', 'downvotes', 'karma', 'num_comments']
                ],
                data=posts_with_karma_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                page_size=10,
                sort_action='native',
                filter_action='native',
                row_selectable='single'
            )
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Comments for Selected Post", className='mt-4'),
            html.Div(id='selected-post-comments')
        ], width=12),
    ]),
], fluid=True)

# Callback to handle page content based on URL
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return home_layout
    elif pathname == "/users-karma":
        return users_karma_layout
    elif pathname == "/posts-insights":
        return posts_insights_layout
    elif pathname == "/timeline":
        return timeline_layout
    elif pathname == "/comments":
        return comments_layout
    # If the user tries to reach a different page, return a 404 message
    return dbc.Container([
        html.H1("404: Not found", className="text-danger"),
        html.Hr(),
        html.P(f"Page '{pathname}' was not found."),
    ], style={"textAlign": "center", "marginTop": "2rem"})

# Callback to update the timeline graph
@app.callback(
    Output('timeline-graph', 'figure'),
    [Input('timeline-metric', 'value'),
     Input('timeline-post-type', 'value'),
     Input('timeline-date-range', 'start_date'),
     Input('timeline-date-range', 'end_date')]
)
def update_timeline(metric, post_types, start_date, end_date):
    # Filter by date range
    mask = (posts_with_karma_df['date'] >= pd.to_datetime(start_date)) & (posts_with_karma_df['date'] <= pd.to_datetime(end_date))
    filtered_posts = posts_with_karma_df.loc[mask]

    # Filter by post type
    if post_types and 'All' not in post_types:
        filtered_posts = filtered_posts[filtered_posts['post_type'].isin(post_types)]

    # Aggregate by date
    aggregated = filtered_posts.groupby('date').agg({metric: 'sum'}).reset_index()
    aggregated = aggregated.sort_values('date')

    # Create the timeline graph
    fig = px.line(
        aggregated,
        x='date',
        y=metric,
        title=f'Timeline of {metric.replace("_", " ").title()}',
        labels={'date': 'Date', metric: metric.replace('_', ' ').title()},
        markers=True
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray')
    )
    return fig

# Combined callback for filtering posts and displaying comments
@app.callback(
    [Output('posts-comments-scatter', 'figure'),
     Output('posts-table', 'data'),
     Output('selected-post-comments', 'children')],
    [Input('search-posts', 'value'),
     Input('reset-filters', 'n_clicks'),
     Input('posts-table', 'active_cell')],
    [State('posts-table', 'data')]
)
def update_posts_and_comments(search_value, n_clicks, active_cell, table_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize with current state
    filtered_posts = posts_with_karma_df
    comments_section = dash.no_update

    # Filter posts based on search or reset
    if triggered_id in ['search-posts', 'reset-filters']:
        if triggered_id == 'reset-filters':
            search_value = ''

        if search_value and search_value.strip() != '':
            search_value = search_value.lower()
            if search_value.isdigit():
                filtered_posts = filtered_posts[filtered_posts['post_id'] == int(search_value)]
            else:
                filtered_posts = filtered_posts[filtered_posts['content'].str.lower().str.contains(search_value)]

    # Update scatter plot
    fig_scatter = px.scatter(
        filtered_posts,
        x='karma',
        y='num_comments',
        text='post_id',
        color='post_type',
        hover_data=['content', 'upvotes', 'downvotes'],
        title='Posts Karma vs. Number of Comments',
        labels={'karma': 'Karma', 'num_comments': 'Number of Comments'},
        color_discrete_map={
            'informative': 'green',
            'cooperative': 'blue',
            'deceptive': 'red'
        }
    )
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray')
    )

    # Update table data
    table_data = filtered_posts[['post_id', 'user_id', 'content', 'post_type', 'upvotes', 'downvotes', 'karma', 'num_comments']].to_dict('records')

    # Display comments for selected post
    if triggered_id == 'posts-table' and active_cell:
        row = active_cell['row']
        post_id = table_data[row]['post_id']

        selected_comments = comments_df[comments_df['post_id'] == post_id]
        selected_comments = selected_comments.sort_values(by='comment_id')
        selected_comments = pd.merge(selected_comments, users_df, on='user_id', how='left')

        comments_list = []
        for _, comment in selected_comments.iterrows():
            comments_list.append(
                dbc.ListGroupItem([
                    html.H6(f"Comment {comment['comment_id']} by {comment['username']}", className='mb-1'),
                    html.P(comment['content'], className='mb-1'),
                    html.Span(f"Upvotes: {comment['upvotes']} | Downvotes: {comment['downvotes']}", className='text-muted', style={'fontSize': '0.9em'})
                ])
            )

        if selected_comments.empty:
            comments_section = dbc.Alert("No comments for this post.", color="warning")
        else:
            comments_section = dbc.ListGroup(comments_list, flush=True)

        post_details = filtered_posts[filtered_posts['post_id'] == post_id].iloc[0]
        username = users_df[users_df['user_id'] == post_details['user_id']]['username'].values[0]
        post_dict = {
            'post_id': post_details['post_id'],
            'user_id': post_details['user_id'],
            'username': username,
            'content': post_details['content'],
            'post_type': post_details['post_type'],
            'upvotes': post_details['upvotes'],
            'downvotes': post_details['downvotes'],
            'karma': post_details['karma'],
            'num_comments': int(post_details['num_comments']),
            'comments': selected_comments.to_dict(orient='records')
        }
        post_json = json.dumps(post_dict, indent=4)

        post_json_component = dbc.Card(
            dbc.CardBody(
                dcc.Markdown(f"```json\n{post_json}\n```")
            ),
            className='mb-4'
        )

        comments_section = [
            html.H5(f"Post {post_id} Details", className='mt-4'),
            dbc.Card([
                dbc.CardBody([
                    html.P(post_details['content'], style={'fontStyle': 'italic'}),
                    html.H6("Comments:", className='mt-3'),
                    comments_section,
                    html.H6("Post JSON:", className='mt-3'),
                    post_json_component
                ])
            ])
        ]
    elif triggered_id in ['search-posts', 'reset-filters']:
        comments_section = dbc.Alert("Select a post from the table to view its comments.", color="info")

    return fig_scatter, table_data, comments_section

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)