from coinrun.coinrunenv import make, register_coinrun_envs

__all__ =  [
    'make',
    'register_coinrun_envs',
]

# register the gymnasium environments
register_coinrun_envs()
