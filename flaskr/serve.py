import click
import functools

from flask import (Blueprint, render_template, flash, request, redirect, url_for)
from flask import current_app as app
from flask.cli import with_appcontext
from .model.receiptsmodel import ReceiptsModel

def init_model():
    model = ReceiptsModel()
    return model

bp = Blueprint('serve', __name__, url_prefix='')
model = init_model()

@click.command('init-model')
@with_appcontext
def init_model_command():
    init_model()
    click.echo('Initialized the model.')

def init_app(app):
    app.cli.add_command(init_model_command)

@bp.route('/', methods=('GET', 'POST'))
def entertraits():
    if request.method == 'POST':
        error = None
        duration = None
        try:
            duration = float(request.form['duration'])
        except ValueError:
            error = 'Duration must be a number'
        speciality = int(request.form['speciality'])
        event_type = int(request.form['eventType'])
        typ = int(request.form['type'])

        if duration == None and error is None:
            error = 'Duration is required'

        if error is None:
            predicted_price = model.predict_price(duration, speciality, event_type, typ)
            return redirect(url_for('serve.showprice', price=predicted_price))
        
        flash(error)
        
    return render_template('entertraits.html')

@bp.route('/showprice/<price>')
def showprice(price):
    return render_template('showprice.html', price=price)