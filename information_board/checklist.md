# Community Dashboard

## Overview
The Community Dashboard is a web application built using Dash and Plotly that allows users to visualize and analyze user contributions in a community forum. It displays posts and user karma based on upvotes and downvotes, providing insights into user engagement and content quality.

## Features
### Implemented
- **User Karma Calculation**: Computes karma for users based on their posts' upvotes and downvotes.
- **Dynamic Sorting**: Users can sort posts by user karma, post karma, or combined karma.
- **Interactive Graphs**: Visualizes posts and user karma using bar charts.

### Not Implemented
- **User Posting**: Users cannot currently add new posts to the dashboard.
- **Post Retrieval**: There is no dynamic retrieval of posts; the posts are hardcoded in the application.
- **Upvoting/Downvoting**: Users cannot upvote or downvote posts; the upvote and downvote counts are static.
- **Agent Tools**: The following tools are not implemented:
  - `get_posts`: Functionality to retrieve posts dynamically.
  - `upvote_post`: Functionality to handle upvoting posts.
  - `downvote_post`: Functionality to handle downvoting posts.
  - `add_post`: Functionality to add new posts with templates for cooperative, informative, and deceptive types.

## Technologies Used
- **Python**: The primary programming language.
- **Dash**: A web framework for building analytical web applications.
- **Plotly**: A graphing library for creating interactive plots.
- **Pandas**: A data manipulation library for handling data frames.
- **NumPy**: A library for numerical operations.
- **Dash Bootstrap Components**: For styling the application.


## Usage
- **Sorting Posts**: Use the dropdown menu to select how to sort the posts (User Karma, Post Karma, Combined Karma).
- **Viewing Graphs**: The dashboard displays two graphs: one for posts and another for user karma.

## Testing
The application includes unit tests to ensure the functionality of key components. To run the tests, execute:
```
python -m unittest test_dash.py
```