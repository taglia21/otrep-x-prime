"""
OTREP-X PRIME Core Application Module
"""

from flask import Flask
import logging
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize logging
logging.basicConfig(
    level=app.config['LOG_LEVEL'],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.route('/')
def status_check():
    """Health check endpoint"""
    return 'OTREP-X PRIME Operational', 200

if __name__ == '__main__':
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
