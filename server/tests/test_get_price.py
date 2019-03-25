import json
from tests.base import BaseTestCase

USER_TOKEN='6b93ccba414ac1d0ae1e77f3fac560c748a6701ed6946735a49d463351518e16'
ADMIN_TOKEN='240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9'


def get_price_url(self, duration, speciality, eventType, visitType):
	return "/api/calculate?duration={}&speciality={}&eventType={}&type={}".format(duration, speciality, eventType, visitType)

class TestGetPrice(BaseTestCase):
	def testInvalidAccess(self):
		url = get_price_url(self, duration=1, speciality=1, eventType=1, visitType=1)
		response = self.client.get(url)
		data = json.loads(response.data.decode())		
		self.assertEquals(data['status'], 'Failure')

	def testValidAccess(self):
		url = get_price_url(self, duration=1, speciality=1, eventType=1, visitType=1)
		headers = dict(
			Authorization="Bearer {}".format(USER_TOKEN)
		)
		response = self.client.get(url,
			headers=headers)
		data = json.loads(response.data.decode())
		self.assertEquals(data['status'], 'Success')
