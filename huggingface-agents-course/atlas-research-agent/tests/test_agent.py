import os
import unittest
from unittest.mock import patch
from agent import AtlasAgent, is_local_endpoint

class AgentTests(unittest.TestCase):
    def test_local_mode_does_not_forward_cloud_secret(self):
        with patch.dict(os.environ,{'ATLAS_ENDPOINT':'http://127.0.0.1:1234/v1/chat/completions','HF_TOKEN':'test-only-secret'}):
            self.assertEqual(AtlasAgent(token='another-test-secret').token,'')
        self.assertFalse(is_local_endpoint('http://127.0.0.1.example.com:1234/'))
        self.assertTrue(is_local_endpoint('http://localhost:1234/'))

    def test_cloud_requires_token(self):
        with patch.dict(os.environ,{'ATLAS_ENDPOINT':'https://router.huggingface.co/v1/chat/completions','HF_TOKEN':''}):
            with self.assertRaises(ValueError):AtlasAgent()

    def test_tool_observation_reaches_next_model_call(self):
        with patch.dict(os.environ,{'ATLAS_ENDPOINT':'http://127.0.0.1:1234/v1/chat/completions'}):
            agent=AtlasAgent()
        replies=[{'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':'a','type':'function','function':{'name':'calculate','arguments':'{"expression":"11*13"}'}}]}}]},
                 {'choices':[{'message':{'role':'assistant','content':'143'}}]}]
        with patch.object(agent,'complete',side_effect=replies) as complete:
            result=agent.solve('Multiply 11 by 13')
        self.assertEqual(result['answer'],'143')
        self.assertEqual(result['events'][0]['result'],'143')
        self.assertEqual(complete.call_args_list[1].args[0][-1]['role'],'tool')

if __name__=='__main__':unittest.main()
