from flask import Flask, Blueprint, request, render_template
from flask_restful import Resource, Api
from webargs.flaskparser import parser, abort						# To parse arguments
from webargs import fields, validate							# To validate arguments

import pickle as pkl
import os
import runpy
import datetime
import numpy as np

# Register blueprint
api_blueprint = Blueprint('api', 'api', url_prefix='/api')
api = Api(api_blueprint)

home = os.getenv("HOME")
# top_level = '/dynprice'
top_level = '/capstone/PB-38'

script_file = home + top_level + '/script/receipts_model.py'
model_file = home + top_level + '/bin/receipts-model.pkl'
runpy.run_path(script_file)
model = pkl.load(open(model_file, 'rb'))
print("API file running")

speciality_length = 12
event_type_length = 3
type_length = 6


"""
API ENDPOINTS
"""
class Retrain(Resource):
	def get(self):
		''' Retrains the model'''
		runpy.run_path(script_file)
		model = pkl.load(open(model_file, 'rb'))
		if model is not None:
			return 1
		else:
			return 0

api.add_resource(Retrain, '/retrain')

class Price(Resource):
	def get(self):
		''' Get predicted price for given parameters '''
		''' Returns 400 error if invalid parameters '''
		args = self.get_request_parameters(request)
		features = format_features(args['duration'], args['speciality'], args['eventType'], args['type'])
		price = model.predict(features)
		return {'price': price[0]}

	def get_request_parameters(self, data):
		''' Parse and validate request parameters '''
		request_args = {
			"duration" : fields.Int(validate=lambda val: val > 0),
			"speciality" : fields.Int(validate=lambda val: val > 0),
			"eventType" : fields.Int(validate=lambda val: val > 0),
			"type" : fields.Int(validate=lambda val: val > 0),
		}
		return parser.parse(request_args, data)
		
api.add_resource(Price, '/calculate')

def format_features(duration, speciality, eventType, typ):
	features = np.zeros((1,22))
	duration_vector = np.asarray([duration])
	speciality_vector = np.zeros(speciality_length, dtype=int)
	speciality_vector[speciality] = 1
	event_type_vector = np.zeros(event_type_length, dtype=int)
	event_type_vector[eventType] = 1
	type_vector = np.zeros(type_length, dtype=int)
	type_vector[typ] = 1
	vector = np.concatenate((duration_vector, speciality_vector, event_type_vector, type_vector))
	print("Duration: " + str(len(duration_vector)))
	print("Speciality: " + str(len(speciality_vector)))
	print("Event type: " + str(len(event_type_vector)))
	print("Type: " + str(len(type_vector)))
	features[0] = vector
	return features

@parser.error_handler
def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(400, errors=err.messages)


if __name__ == '__main__':
    app.run(debug=True)
