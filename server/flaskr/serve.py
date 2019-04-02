import os
import click
import functools

from flask import Flask, Blueprint, flash, request, render_template
from flask.cli import with_appcontext

from flask_wtf import Form
from wtforms import IntegerField, SelectField, SubmitField
from wtforms.validators import NumberRange, InputRequired
import requests
import json
import bson

SFILE_PATH = os.getenv("HOME") + '/dump/porton/services.bson'
FEATURES_PATH = 'bin/receipts_dict.json'

# Add blueprint
app_blueprint = Blueprint('app', __name__, url_prefix='')

class TraitsForm(Form):
	''' Event parameters form '''
	with open(SFILE_PATH, 'rb') as sfile:
		services = bson.decode_all(sfile.read())

	with open(FEATURES_PATH, 'r') as rfile:
		featureslist = json.loads(rfile.read())
	
	speciality_types = featureslist['specialty']
	event_types = featureslist['eventType']
	visit_types = featureslist['type']
	
	specialities = []
	events = []
	visits = []
	
	for key, value in speciality_types.items():
		specialities.append((value, key))

	for key, value in event_types.items():
		events.append((value, key))

	for service in services:
		service_id = service['_id']
		if service_id in visit_types.keys():
			visits.append((visit_types[service_id], service['name']))		

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
