import os

from flask import Flask
from flask_bootstrap import Bootstrap

from .config import config_by_name
from .api.price_model import PriceModel
from .api.api import api_blueprint
from .serve import app_blueprint

def create_app(config_name):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_by_name[config_name])    

    obj = PriceModel.load_model()
    app.model = obj.get('model', None)
    app.score = obj.get('score', None)
    app.error = obj.get('error', None)

    bootstrap = Bootstrap(app)

    # Register blueprints 
    app.register_blueprint(api_blueprint)
    app.register_blueprint(app_blueprint)

    return app
