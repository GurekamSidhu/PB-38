import click
import functools

from flask import Flask, Blueprint, request, render_template
from flask import current_app as app
from flask.cli import with_appcontext

# For forms
from flask_wtf import Form
from wtforms import IntegerField,SelectField
import requests

# Add blueprint
app_blueprint = Blueprint('app', __name__, url_prefix='')

class TraitsForm(Form):
	''' Event parameters form '''
	speciality_types = ['Neurology', 'Pulmonology', 'Family Medicine', 'Clinical Genetics and Genomics', 'Clinical Counselling', 'Plastic Surgery', 'Anesthesiology', 'Pediatrics', 'Allergy and Immunology', 'Aerospace Medicine', 'Dermatology', 'Cardiology', 'None']
	event_types = ['Video', 'Report', 'Voice']
	visit_types = ['Counseling Video', 'Consulatation Video', 'Consulatation Report', 'Consultation Email', 'Conseling Email', 'Conseling Voice', 'Consultation Voice', 'Conseling Report']
	
	specialities = []
	events = []
	visits = []
	for i in range(0, len(speciality_types)):
		specialities.append((i + 1, speciality_types[i]))

	for i in range(0, len(event_types)):
		events.append((i + 1, event_types[i]))

	for i in range (0, len(visit_types)):
		visits.append((i + 1, visit_types[i]))

	duration 	= IntegerField('Duration')				# Must be greater than 0
	speciality 	= SelectField('Specialities', choices=specialities)	
	event_type 	= SelectField('Event Type', choices=events)
	visit_type 	= SelectField('Visit Type', choices=visits)


"""
FRONT-END ENDPOINTS
"""
@app_blueprint.route('/', methods=('GET', 'POST'))
def getPrice():
	if request.method == 'GET':
		'''Render form to input visit details'''
		form = TraitsForm(csrf_enabled=False)	
		return render_template('entertraits.html', form=form)
	
	if request.method == 'POST':
		'''Request price for given details'''
		form_data = request.form	
		data = {'duration': form_data['duration'], 'speciality' : form_data['speciality'], 'eventType' : form_data['event_type'], 'type': form_data['visit_type']}
		r = requests.get(request.base_url + 'api/calculate', params=data)
		print(r.text)
		return r.text
