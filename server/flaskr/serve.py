import click
import functools

from flask import Flask, Blueprint, flash, request, render_template
from flask.cli import with_appcontext

# For forms
from flask_wtf import Form
from wtforms import IntegerField, SelectField, SubmitField
from wtforms.validators import NumberRange, InputRequired
import requests
import json

# Add blueprint
app_blueprint = Blueprint('app', __name__, url_prefix='')

class TraitsForm(Form):
	''' Event parameters form '''
	speciality_types = ['Cardiology', 'Neurology', 'Dermatology', 'Clinical Genetics and Genomics', 'Pulmonology', 'Anesthesiology', 'Allergy and Immunology', 'Aerospace Medicine', 'Clinical Counselling', 'Family Medicine/General Practitioner', 'Plastic Surgery', 'Pediatrics']
	event_types = ['Video', 'Report', 'Voice']
	visit_types = ['Counselling Video', 'Consultation Video', 'Consultation Report', 'Consultation Email', 'Counselling Email', 'Counselling Voice']
	# visit_types = ['Counselling Video', 'Consultation Video', 'Consultation Report', 'Consultation Email', 'Counselling Email', 'Counselling Voice', 'Consultation Voice', 'Counselling Report']

	specialities = []
	events = []
	visits = []
	for i in range(0, len(speciality_types)):
		specialities.append((i + 1, speciality_types[i]))

	for i in range(0, len(event_types)):
		events.append((i + 1, event_types[i]))

	for i in range (0, len(visit_types)):
		visits.append((i + 1, visit_types[i]))

	duration 	= IntegerField('Duration(s)', validators=[InputRequired(),NumberRange(min=1)])				# Must be greater than 0
	speciality 	= SelectField('Speciality',coerce=int, choices=specialities)	
	event_type 	= SelectField('Event Type',coerce=int, choices=events)
	visit_type 	= SelectField('Visit Type',coerce=int, choices=visits)
	submit		= SubmitField('Get Price')

"""
FRONT-END ENDPOINTS
"""
@app_blueprint.route('/', methods=('GET', 'POST'))
def getPrice():
	form=TraitsForm(csrf_enabled=False)
	if request.method == 'GET':
		'''Render form to input visit details'''
		return render_template('index.html', form=form)
	
	if request.method == 'POST':
		'''Request price for given details'''
		form_data = request.form
		if form.validate_on_submit() == False:
			flash("Invalid Input")
			return render_template('index.html', form=form)
		data = {'duration': form_data['duration'], 'speciality' : form_data['speciality'], 'eventType' : form_data['event_type'], 'type': form_data['visit_type']}
		headers = {
			'Authorization': 'Basic 6b93ccba414ac1d0ae1e77f3fac560c748a6701ed6946735a49d463351518e16'
		}
		r = requests.get(request.base_url + 'api/calculate', headers=headers, params=data)
		if r.status_code == requests.codes.ok:
			price = json.loads(r.text)['data']['price']
			error = json.loads(r.text)['data']['error']
			return render_template('index.html', price=price, error=error)
		else:
			return 'error'
