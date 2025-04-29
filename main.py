import torch, numpy as np
import matplotlib
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("NumPy:", np.__version__)
print(matplotlib.__version__)