"""
OTREP-X PRIME Configuration Management
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class Config:
    """Base configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-insecure-key')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://localhost/otrep')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB upload limit

class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    HOST = 'localhost'
    PORT = 5000

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 8000

def get_config(env: str = 'development') -> Config:
    """Retrieve environment-specific configuration"""
    config_mapping = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': DevelopmentConfig  # Use dev config for testing
    }
    return config_mapping.get(env.lower(), DevelopmentConfig)()
