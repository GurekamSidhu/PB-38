from flask_testing import TestCase
from flaskr import create_app

class BaseTestCase(TestCase):
	""" Base Tests """
	def create_app(self):
		app = create_app("test")
		return app
