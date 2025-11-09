from core.data_feed import DataFeed
from strategies.stoichiometric import StoichiometricStrategy
from utils.logger import Logger

print("Initializing OTREP-X PRIME core runtime...")

if __name__ == '__main__':
    logger = Logger()
    data = DataFeed(logger)
    strategy = StoichiometricStrategy(data, logger)
    strategy.run()
