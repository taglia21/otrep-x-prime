"""
OTREP-X PRIME Command Line Interface
"""

import click
from src import app, utils

@click.group()
def cli():
    """OTREP-X PRIME Management Interface"""
    pass

@cli.command()
@click.option('--host', default='localhost', help='Interface to bind')
@click.option('--port', default=5000, help='Port to listen')
def run(host, port):
    """Start the application server"""
    utils.configure_logging()
    click.echo(f"Starting OTREP-X PRIME service on {host}:{port}")
    app.run(host=host, port=port)

@cli.command()
def migrate():
    """Run database migrations"""
    click.echo("Executing database migrations...")
    # Add migration logic here

@cli.command()
def seed():
    """Load initial dataset"""
    click.echo("Seeding initial data...")
    # Add data seeding logic here

if __name__ == '__main__':
    cli()
