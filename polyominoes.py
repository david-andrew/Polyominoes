import torch
# import numpy as np
from math import prod

from itertools import permutations, product



import pdb

"""
#TODO: do both numpy and torch versions, and compare which is faster
# - torch first
# - consider making dimensionally agonistic

#canonical orientations:
- sort by bbox face areas
- sort by location of center of mass relative to bbox faces


canonical hash is 
- bit value with lowest value out of all possible rotations
- dimensions of bounding box

"""

#TODO: compare with uint8/int8/uint64/int64
dtype = torch.bool

def main(dim:int):

    #each element is the element's canonical hash (minimum hash out of all rotations), followed by bounds along each axis
    #initial element is just a single cube in the center with bounds 1x1x...x1 and hash 1
    #e.g. 2D elements are (has, x, y), 3D elements are (hash, x, y, z)
    levels: list[set[tuple[int, ...]]] = [{(1,)*(dim+1)}]

    while True:
        n = len(levels) + 1 #number of cubes in a shape at current level
        prev_level = levels[-1]

        level = set() # just the canonical IDs
        all_rotations = set() # IDs of shapes over all rotations so we don't need rotate when checking for duplicates. Only do all rotations when adding a new shape that wasn't a duplicate

        prev_shapes = build_shapes_tensor(prev_level, pad=1)

        successors = generate_successors(prev_shapes, dim)
        successors = successors.reshape(prod(successors.shape[:2]), *successors.shape[2:])


        #for each successor shape:
        # - generate shape ID
        # - if ID not in all_rotations:
        #    - compute all rotations and IDs of that shape
        #    - add all IDs to all_rotations
        #    - add canonical ID to level
        
        #TBD if a parallelized version, that does all the rotations all at once, without knowing if it's a duplicate yet, will be better...
        pdb.set_trace()
        ...


def generate_successors(prev_shapes: torch.Tensor, dim:int) -> torch.Tensor:
    """generate all the possible size n+1 polyominoes given a polyomino of size n"""

    open_spots = ~prev_shapes
    has_neighbor = torch.zeros_like(prev_shapes, dtype=dtype)
    for i in range(dim):
        has_neighbor += torch.roll(prev_shapes, 1, dims=i+1)
        has_neighbor += torch.roll(prev_shapes, -1, dims=i+1)
    has_neighbor = has_neighbor > 0

    successor_spots = open_spots & has_neighbor

    # Use advanced indexing to place the ones at the correct places
    successors = torch.zeros((successor_spots.sum(),) + successor_spots.shape, dtype=dtype)
    indices = torch.where(successor_spots)
    successors[(torch.arange(indices[0].size(0)),) + indices] = 1

    # merge in the previous shapes with the successor spots to get all successors
    successors = successors | prev_shapes

    return successors




def build_shapes_tensor(shape_ids: set[tuple[int,...]], pad:int=0) -> torch.Tensor:
    bounds = torch.tensor([shape[:-1] for shape in shape_ids])
    maxes = torch.max(bounds, dim=0).values

    shapes = torch.zeros((len(shape_ids), *maxes+2*pad), dtype=dtype)

    for i, shape in enumerate(shape_ids):
        H, *dims = shape
        shapes[[i] + [slice(pad, x+pad) for x in dims]] = shape_from_hash(H, dims)

    return shapes

def shape_from_hash(H: int, dims:tuple[int]) -> torch.Tensor:
    # convert hash to binary array, padding beginning with zeros up to length x*y*z
    digits = [int(i) for i in  bin(H)[2:].zfill(prod(dims))]

    # construct shape by reshaping array to x,y,z dimensions
    shape = torch.tensor(digits, dtype=dtype).reshape(*dims)

    return shape




def get_rotations(n:int, *, dtype=torch.int, include_identity:bool=False) -> torch.Tensor:
    """
    Generate all axis aligned orientations for an n-dimensional space.
    E.g. in 2D this is the 4 rotations of a square (0, 90, 180, 270 degrees)
    E.g. in 3D this is the 24 rotations of a cube, etc.

    Args:
        n (int): number of dimensions
        dtype (torch.dtype, optional): data type of output tensor. Defaults to torch.int.
        include_identity (bool, optional): whether to include the identity matrix in the output. Defaults to False.
    
    Returns:
        torch.Tensor: tensor of shape (n_rotations, n, n) containing all rotations
    """

    # Process for generating all rotations in N dimensions:
    # 1. Generate all permutations of columns (or rows) of the identity matrix
    # 2. Apply all combinations of negations to each permutation
    #
    # All under the constraint that the determinant of the final matrix is 1
    # This can be achieved efficiently by ensuring that swaps plus negations is even

    #generate all permutations of columns in the identity matrix
    swaps = torch.tensor([*permutations(range(n))])

    # determine which permutations have an even number of swaps vs which are odd
    parity = torch.stack(
        [(swaps[:, :-i] > swaps[:, i:]).sum(dim=1) for i in range(1, n)]
    ).sum(dim=0) % 2
    
    # generate all combinations of negations
    negations = torch.tensor([*product([1,-1], repeat=n)])

    # separate out even parity negations from odd parity negations
    negations = torch.stack([
        negations[negations.prod(dim=1) == 1],
        negations[negations.prod(dim=1) == -1]
    ])
    

    # get all permutations of the identity matrix
    I = torch.eye(n, dtype=dtype)
    perms = I[swaps]

    # apply all combinations of negations to each permutation
    rotations = perms[:, None] * negations[parity][..., None]
    rotations = rotations.reshape(-1, n, n)

    if not include_identity:
        rotations = rotations[1:]

    return rotations















def bounding_box():
    ...
    #sums along xy, yz, zx, then find first/last non-zero index

def hash_to_grid():
    ...

def grid_to_hash():
    ...

def rotations():
    ...

def successions():
    
    ...
















if __name__ == "__main__":
    main(2) #2D
    # main(3) #3D
    # main(4) #4D