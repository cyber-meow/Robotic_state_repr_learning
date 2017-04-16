
from Interaction import Interaction
from ExploringBot import ExploreBot
from NavEnvSimp import NavEnv


navEnv = NavEnv()
bot = ExploreBot()
inter = Interaction(navEnv, bot)
inter.observationSerie()
