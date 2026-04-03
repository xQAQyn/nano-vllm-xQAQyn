import torch
import os

# Disable torch.compile for tests (inductor doesn't work well in CPU test mode)
torch._dynamo.config.suppress_errors = True
os.environ["TORCHDYNAMO_DISABLE"] = "1"
