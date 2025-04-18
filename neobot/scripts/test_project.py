import os
realpath=os.path.realpath(__file__)
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(realpath)
print(LEGGED_GYM_ROOT_DIR)