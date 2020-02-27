[//]: # (author: samtenka)
[//]: # (change: 2020-02-11)
[//]: # (create: 2020-02-11)
[//]: # (descrp: top-level documentation for Looking Glass, a program inductor
                 based on syntax-semantics duality and intended for tasks as 
                 conceptually rich and core-cognitively grounded as ARC)
[//]: # (to use: Open and read this document in a web browser.)

# looking-glass

## Representations: Galois Duality and Programming with Holes

## A Rich Toy Domain and its Specific Language

We consider a task as sampled from a distribution over (N x M, n x m)
input-output pairs of grids.  For now, we fix all sidelengths as
N=M=n=m=8.

The primitive types are `int`, `bool`, `color`, and `grid`, and in addition to
usual arithmetic, comparison, and literal constructors, we have: 

        lock: int -> int -> Maybe pos 
        plot: pos -> (int x int)
        view: grid -> pos -> color
        mark: grid -> pos -> color -> grid 

