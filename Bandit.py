"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass

#--------------------------------------#



class Visualization():

    def plot1(self):
        # Visualize the performance of each bandit: linear and log
        pass

    def plot2(self):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        pass



#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    """
    pass

#--------------------------------------#

class ThompsonSampling(Bandit):
    "logg"
    pass




def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
