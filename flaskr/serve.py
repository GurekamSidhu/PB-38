import click
import functools

from flask import (Blueprint, render_template, flash, request, redirect, url_for)
from flask import current_app as app
from flask.cli import with_appcontext
from .model.nnscikit import Perceptron

def init_model():
    percep = Perceptron()
    percep.run_regress((8,5))
    return percep

bp = Blueprint('serve', __name__, url_prefix='')
percep = init_model()

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
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        smoker = float(request.form['smoker'])
        ne = float(request.form['ne'])
        nw = float(request.form['nw'])
        se = float(request.form['se'])
        sw = float(request.form['sw'])
        error = None

        if age == None:
            error = 'Age is required'
        elif sex == None:
            error = 'Sex is required'
        elif bmi == None:
            error = 'BMI is required'
        elif children == None:
            error = 'Children is required'
        elif smoker == None:
            error = 'Smoker is required'
        elif ne == None:
            error = 'NE is required'
        elif nw == None:
            error = 'NW is required'
        elif se == None:
            error = 'SE is required'
        elif sw == None:
            error = 'SW is required'

        if error is None:
            predicted_price = percep.predict_price(age, sex, bmi, children, smoker, ne, nw, se, sw)
            return redirect(url_for('serve.showprice', price=predicted_price))
        
        flash(error)
        
    return render_template('entertraits.html')

@bp.route('/showprice/<price>')
def showprice(price):
    return render_template('showprice.html', price=price)