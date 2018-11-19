import click

from flask import (Blueprint, render_template)
from flask.cli import with_appcontext
from .model.nnscikit import Perceptron

def init_model():
    percep = Perceptron()
    percep.run_regress((8,5))

@click.command('init-model')
@with_appcontext
def init_model_command():
    init_model()
    click.echo('Initialized the model.')

def init_app(app):
    app.cli.add_command(init_model_command)

