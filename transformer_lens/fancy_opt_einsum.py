import os

import fancy_einsum
import torch


def fancy_opt_einsum(equation: str, *operands):
    """
    Variation on fancy opt einsum that uses opt_einsum for the contraction.
    
    Evaluates the Einstein summation convention on the operands.
    
    See: 
      https://pytorch.org/docs/stable/generated/torch.einsum.html
      https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
      https://optimized-einsum.readthedocs.io/en/stable/index.html
    """
    assert torch.backends.opt_einsum.enabled
    if "USE_OPT_FOR_FANCY_EINSUM" in os.environ:
        from opt_einsum import contract
        new_equation = fancy_einsum.convert_equation(equation)
        return contract(new_equation, *operands)
    else:
        return fancy_einsum.einsum(equation, *operands)
