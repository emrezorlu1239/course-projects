import unittest
from atlas_tools import calculate, public_url, reverse_text

class ToolTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculate('round(sum([12.5*4, 7.25*2]),2)'),64.5)
    def test_reject_code(self):
        for value in ["__import__('os').getcwd()",'(1).__class__','[1]*999999999','2**999999']:
            with self.assertRaises(ValueError):calculate(value)
    def test_no_local_urls(self):
        for url in ['file:///etc/passwd','http://127.0.0.1/','http://[::1]/','http://169.254.169.254/']:
            with self.assertRaises(ValueError):public_url(url)
    def test_reverse(self):
        self.assertEqual(reverse_text('abc'),'cba')

if __name__=='__main__':unittest.main()
