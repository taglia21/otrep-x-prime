"""
OTREP-X PRIME Basic Test Cases
"""

import unittest
from src import app

class BasicTestCase(unittest.TestCase):
    """Fundamental test cases"""

    def setUp(self):
        """Create test client"""
        self.app = app.app.test_client()
        self.app.testing = True

    def test_health_check(self):
        """Verify service health endpoint"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Operational', response.data)

    def test_config_loading(self):
        """Verify configuration loading"""
        self.assertEqual(app.app.config['DEBUG'], False)

if __name__ == '__main__':
    unittest.main()
