import unittest
import asyncio
import tempfile
import os
from reptarINDEX import Graph, GraphNode, chunk_text, load_yaml_config, generate_summary

class TestReptarINDEX(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.test_text = "This is a test sentence. This is another test sentence. And here's a third one."

    def test_graph_node_creation(self):
        node = GraphNode("test_node", text="test text", is_folder=False, is_cluster=False, path="/test/path", start_pos=0, end_pos=10, chunk_id=1)
        self.assertEqual(node.name, "test_node")
        self.assertEqual(node.text, "test text")
        self.assertFalse(node.is_folder)
        self.assertFalse(node.is_cluster)
        self.assertEqual(node.path, "/test/path")
        self.assertEqual(node.start_pos, 0)
        self.assertEqual(node.end_pos, 10)
        self.assertEqual(node.chunk_id, 1)

    def test_graph_add_node(self):
        node = GraphNode("test_node")
        added_node = self.graph.add_node(node)
        self.assertIn(node, self.graph.nodes)
        self.assertIn(node, self.graph.root.children)
        self.assertEqual(added_node, node)

    def test_chunk_text(self):
        chunks = chunk_text(self.test_text, target_chunk_size=50, overlap=2)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.split()), 50)

    def test_load_yaml_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("""
            summary_system_prompt: "Test system prompt"
            summary_user_prompt: "Test user prompt {text}"
            """)
            temp_file_path = temp_file.name

        config = load_yaml_config(temp_file_path)
        self.assertEqual(config['summary_system_prompt'], "Test system prompt")
        self.assertEqual(config['summary_user_prompt'], "Test user prompt {text}")

        os.unlink(temp_file_path)

    @unittest.mock.patch('reptarINDEX.OpenAI')
    def test_generate_summary(self, mock_openai):
        mock_client = unittest.mock.Mock()
        mock_openai.return_value = mock_client
        mock_response = unittest.mock.Mock()
        mock_response.choices[0].message.content = "Test summary"
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "summary_system_prompt": "Test system prompt",
            "summary_user_prompt": "Summarize: {text}"
        }
        summary_model = "test_model"

        summary = generate_summary("Test text", config, summary_model)
        self.assertEqual(summary, "Test summary")

    @unittest.mock.patch('reptarINDEX.get_embeddings')
    async def test_graph_search(self, mock_get_embeddings):
        mock_get_embeddings.return_value = [[1.0, 0.0, 0.0]]
        
        node1 = GraphNode("node1", text="test document one", embedding=[1.0, 0.0, 0.0])
        node2 = GraphNode("node2", text="test document two", embedding=[0.0, 1.0, 0.0])
        self.graph.add_node(node1)
        self.graph.add_node(node2)

        embeddings = {'model': 'test_model', 'data': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
        results = await self.graph.search("test document", embeddings)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], node1)
        self.assertGreater(results[0][1], results[1][1])  # Check if node1 has a higher score

if __name__ == '__main__':
    unittest.main()