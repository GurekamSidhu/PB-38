import json
from tests.base import BaseTestCase

USER_TOKEN='6b93ccba414ac1d0ae1e77f3fac560c748a6701ed6946735a49d463351518e16'
ADMIN_TOKEN='240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9'


RETRAIN_URL="api/retrain"


class TestRetrain(BaseTestCase):
	def testUserAccess(self):
		headers = dict(
			Authorication="Bearer {}".format(USER_TOKEN)
		)
		response = self.client.get(
			RETRAIN_URL,
			headers=headers)
		data = json.loads(response.data.decode())
		self.assertEquals(data['status'], 'Failure')
	
	def testInvalidAccess(self):
		headers = dict(
			Authorization = "Bearer {}".format("invalid__token")
		)
		response = self.client.get(
			RETRAIN_URL,
			headers=headers)
		data = json.loads(response.data.decode())
		self.assertEquals(data['status'], 'Failure')

	def testValidAccess(self):
		headers = dict(
			Authorization = "Bearer {}".format(ADMIN_TOKEN)
		)
		response = self.client.get(
			RETRAIN_URL,
			headers=headers)
		data = json.loads(response.data.decode())
		self.assertEquals(data['status'], 'Success')
