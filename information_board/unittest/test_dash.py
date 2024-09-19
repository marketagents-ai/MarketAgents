import unittest
import pandas as pd
import numpy as np
from dashboard import calculate_user_karma, sort_posts_by_user_karma, sort_posts_by_post_karma, sort_posts_by_combined_karma

class TestDashboardFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.users_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'username': ['Alice', 'Bob', 'Charlie']
        })
        self.posts_df = pd.DataFrame({
            'post_id': [1, 2, 3],
            'user_id': [1, 2, 3],
            'upvotes': [10, 15, 5],
            'downvotes': [2, 3, 8]
        })
        self.comments_df = pd.DataFrame({
            'comment_id': [1, 2],
            'post_id': [1, 2],
            'user_id': [1, 2],
            'upvotes': [3, 4],
            'downvotes': [1, 0]
        })

    def test_calculate_user_karma(self):
        user_karma_df = calculate_user_karma(self.posts_df, self.comments_df, self.users_df)
        expected_karma = {
            'user_id': [1, 2, 3],
            'total_karma': [10 - 2 + 3 - 1, 15 - 3 + 4 - 0, 5 - 8 + 0][0:3]
        }
        expected_df = pd.DataFrame(expected_karma)
        pd.testing.assert_frame_equal(user_karma_df[['user_id', 'total_karma']], expected_df)

    def test_sort_posts_by_user_karma(self):
        sorted_posts = sort_posts_by_user_karma(self.posts_df)
        self.assertTrue(sorted_posts['upvotes'].iloc[0] >= sorted_posts['upvotes'].iloc[1])

    def test_sort_posts_by_post_karma(self):
        self.posts_df['karma'] = self.posts_df['upvotes'] - self.posts_df['downvotes']
        sorted_posts = sort_posts_by_post_karma(self.posts_df)
        self.assertTrue(sorted_posts['karma'].iloc[0] >= sorted_posts['karma'].iloc[1])

    def test_sort_posts_by_combined_karma(self):
        self.posts_df['karma'] = self.posts_df['upvotes'] - self.posts_df['downvotes']
        sorted_posts = sort_posts_by_combined_karma(self.posts_df, weight_user=0.5, weight_post=0.5)
        self.assertTrue(sorted_posts['combined_karma'].iloc[0] >= sorted_posts['combined_karma'].iloc[1])

if __name__ == '__main__':
    unittest.main()