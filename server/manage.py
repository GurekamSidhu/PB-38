import os
import unittest

from flask_script import Manager
from flaskr import create_app

app = create_app(os.environ.get("FLASK_ENV", default="dev"))
app.app_context().push()

manager = Manager(app)

@manager.command
def run():
    "Run the server"
    # get ssl cert and key
    CERT_FILE = os.environ.get("PATH_TO_CERTIFICATE", default=None)
    KEY_FILE = os.environ.get("PATH_TO_KEY", default=None)
    
    if CERT_FILE is not None and KEY_FILE is not None:
    	app.run(ssl_context=('cert.pem', 'key.pem'))
    else:
        app.run()

@manager.command
def test():
    """Runs the unit tests."""
    tests = unittest.TestLoader().discover('tests', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    return 1

if __name__ == '__main__':
    manager.run()
