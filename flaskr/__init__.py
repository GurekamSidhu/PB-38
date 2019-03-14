import os

from flask import Flask
from flask_bootstrap import Bootstrap
from .api.api import api_blueprint
from .serve import app_blueprint

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    # default configuration
    app.config.from_mapping(
        SECRET_KEY='dev'
    )

    app.static_folder = 'static'
    bootstrap = Bootstrap(app)

    if test_config is None:
        # overrides default configuration if config.py exists
        app.config.from_pyfile('config.py', silent=True)
    else:
        # test_config can be passed and used instead of instance config
        app.config.from_mapping(test_config)

    # Register blueprints 
    app.register_blueprint(api_blueprint)
    app.register_blueprint(app_blueprint)

    return app
