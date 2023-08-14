"""
Type aliases for certain objects.
"""


from typing import Any, Optional, Union

import torch


Context = Any
Device = Optional[Union[torch.device, str]]
