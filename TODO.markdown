THIS WEEK TODO

```all on branch TREE```

sample programs from dsl
 >< train currently hacky fit_weights.py on one program 
 >< automate and expand the sigs_by_name for resources.py
     >< decorate methods in resources.py to know their types
     >< establish with kevin: no polymorphism
     [] introduce further abstractions
        >< replace imperative iteration by map, filter, fold, len, count, argmax
        >< uniformize manipulation and representation of polymorphics
        >< introduce named (instead of built in) product types to lg_types.py
            >< implement projections and pairings for each such type
            >< pair<grid,cell>, pair<grid,grid>, pair<shape,color>much used 
            [] pair<cell, color>, pair<pair<shape,color>, int> also used
     >< annotate all types appropriately 
     >< noise as arguments should be reflected in types
 [] let a probabilistic dsl direct generate_script.py.  Inspired by AI Coq,
        e.g. the 'root' token, when generated, performs a coq 'intro'!
     >< sample based on 'root', 'resources' notions of tree type
     >< sample based also on weight learner! 
     [] clarify whether resources is set or multiset of types (relevant to
        logistic regression in fit_weights.py and also to sampling in
        generate_script.py)

functionalize 10 easy tasks 
 >< regard top levels in terms of pair<grid,grid> outer return type
 >< write chaining in terms of split<> helper so that heads can be useful
        for learning weights
 [] parameterize shape gen
     [] shapes: plus, times, and square 
     [] new type: object schema?  itself with a prior and from which objects
        can be generated 
     [] pen type??
 [] produce 10 trees, each feed-able into fit_weights.py; put in separate file  
     >< 0
     [] 1
     [] 2
     [] 3
     [] 4
     [] 5
     [] 6
     [] 7
     [] 8
     [] 9

make sure programs are evaluatable
 >< parse programs into runnable lambdas (in parse.py)
 >< populate resources.py to provide implementations used by
        interpret_script.py
     [] clarify interface with shape.py, grid.py, etc

use l1 regularized logistic regression to train on sample trees
 [] implement probabilistic model and logistic regression 
 [] sensitize to further conditions? 
     [] condition bigram generation on index of argument in arglist? 
     [] condition on order in resources?
     [] condition on depth?

sample and run programs!
 [] send to cathy
 [] send to arc email list

obtain further trees while introducing new primitives
 [] functionalize 10 easy tasks each in a new way, e.g. rotations
 [] compress code based on 20 solutions to 10 easy tasks
 [] functionalize 10 medium tasks

injectivity analysis
 [] plan out

