from flask import Flask, Blueprint, request, render_template
from flask_restful import Resource, Api
from webargs.flaskparser import parser, abort						# To parse arguments
from webargs import fields, validate							# To validate arguments

from ..model.receiptsfakemodel import ReceiptsModel 					# Model to query

import datetime

# Register blueprint
api_blueprint = Blueprint('api', 'api', url_prefix='/api')
api = Api(api_blueprint)


model = ReceiptsModel()

"""
API ENDPOINTS
"""
class Retrain(Resource):
	def get(self):
		''' Retrains the model'''
		model = ReceiptsModel()
		return {'trainedAt': str(datetime.datetime.now())}

api.add_resource(Retrain, '/retrain')

class Price(Resource):
	def get(self):
		''' Get predicted price for given parameters '''
		''' Returns 400 error if invalid parameters '''
		args = self.get_request_parameters(request)
		price = model.predict_price(args['duration'], args['speciality'], args['eventType'], args['type'])
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

@parser.error_handler
def handle_request_parsing_error(err, req, schema, error_status_code, error_headers):
    """webargs error handler that uses Flask-RESTful's abort function to return
    a JSON error response to the client.
    """
    abort(400, errors=err.messages)


if __name__ == '__main__':
    app.run(debug=True)
