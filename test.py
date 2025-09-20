import numpy as np
_next_idx=0
_maxsize=2e6
_next_idx = int((_next_idx + 1) %_maxsize)
print(type(_next_idx),_next_idx)
print(type(_maxsize))