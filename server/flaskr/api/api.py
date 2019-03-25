from flask import Flask, Blueprint, current_app, request, render_template
from flask_restful import Resource, Api
from webargs.flaskparser import parser, abort						# To parse arguments
from webargs import fields, validate							# To validate arguments

from .price_model import PriceModel

import runpy
import datetime
import numpy as np

# Register blueprint
api_blueprint = Blueprint('api', 'api', url_prefix='/api')
api = Api(api_blueprint)


users = {
	'guest': '6b93ccba414ac1d0ae1e77f3fac560c748a6701ed6946735a49d463351518e16',
	'admin': '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9'
}

"""
API ENDPOINTS
"""
class Retrain(Resource):
	def get(self):
		''' Retrains the model'''
		auth_header = request.headers.get('Authorization')
		if auth_header is None:
			response = {
				'status':'Failure',
				'message': 'Invalid Access. Please provide token.'
			}
			return response, 401
		
		if authenticate(auth_header, admin=True) == False:
			response = {
				'status':'Failure',
				'message':'Invalid Token.  Admin access only.'
			}
			return response, 401
		PriceModel.retrain_model()
		current_app.model = PriceModel.load_model()
		if current_app.model is not None:
			return {
				'status':'Success',
				'message':'Successfully retrained model.'
			}, 200
		else:
			return {
				'status':'Failure',
				'message':'Failed to retrain model.'
			}, 500


class Price(Resource):
	def get(self):
		''' Get predicted price for given parameters '''
		''' Returns 400 error if invalid parameters '''
		auth_header = request.headers.get('Authorization')
		if auth_header is None:
			response = {
				'status':'Failure',
				'message':'Invalid Access. Please provide token.'
			}
			return response, 401	
		if authenticate(auth_header) == False:
			response = {
				'status': 'Failure',
				'message': 'Invalid token.'
			}
			return response, 401
		args = self.get_request_parameters(request)
		price = self.get_price(args['duration'], args['speciality'], args['eventType'], args['type'])
		response = {
			'status':'Success',
			'message':'Successful price generation.',
			'data': {
				'price': price[0]
			}
		}
		return response, 200

	def get_request_parameters(self, data):
		''' Parse and validate request parameters '''
		request_args = {
			"duration" : fields.Int(validate=lambda val: val > 0),
			"speciality" : fields.Int(validate=lambda val: val > 0),
			"eventType" : fields.Int(validate=lambda val: val > 0),
			"type" : fields.Int(validate=lambda val: val > 0),
		}
		return parser.parse(request_args, data)
		
	def format_features(self, duration, speciality, eventType, typ):
		''' Produces a features vector '''
		speciality_length = 12
		event_type_length = 3
		type_length = 6

		features = np.zeros((1,22))

		duration_vector = np.asarray([duration])
		
		speciality_vector = np.zeros(speciality_length, dtype=int)
		speciality_vector[speciality] = 1

		event_type_vector = np.zeros(event_type_length, dtype=int)
		event_type_vector[speciality] = 1
		
		type_vector = np.zeros(type_length, dtype=int)
		type_vector[typ] = 1

		vector = np.concatenate((duration_vector, speciality_vector, event_type_vector, type_vector))
		features[0] = vector
		return features

	def get_price(self, duration, speciality, eventType, typ):
		''' Return price given set of features '''
		features = self.format_features(duration, speciality, eventType, typ)
		return current_app.model.predict(features)


api.add_resource(Retrain, '/retrain')
api.add_resource(Price, '/calculate')


@parser.error_handler
def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(400, errors=err.messages)


def authenticate(auth_header, admin=False):
	''' Given the authentication header, determine if the correct token has been passed. '''
	auth_token = ''
	try:
		auth_token = auth_header.split(" ")[1]
	except IndexError:
		return False
	
	if admin:
		return auth_token == users['admin']
	for user, token in users.items():
		if auth_token == token:
			return True
	return False

if __name__ == '__main__':
    app.run(debug=True)
