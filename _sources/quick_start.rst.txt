Quickstart
==========
A quick and dirty guide to using the CSAM codebase

Requirements
^^^^^^^^^^^^
To run the code, make sure the following packages are installed, preferably in a virtual environment.

.. literalinclude:: ../../requirements.txt

.. note::
    The Sphinx dependencies can be found in ``docs/conf.py``.

Overview
^^^^^^^^
The CSAM codebase is structured modularly, see :numref:`structure` for a graphical overview.

The package :mod:`wrappers` provides interfaces to the core code components in :mod:`src` and :mod:`vis`. For example, it defines the First and Second Approximation steps in the CSAM algorithm and applies the tapering of the physical data. Refer to the :doc:`APIs <api>` for more details.

Helper functions and data structures are provided for the processing of user-defined topographies (:mod:`src.var.topo`), grids (:mod:`src.var.grid`), and input parameters (:mod:`src.var.params`).

These *building blocks* are the assembled for different kinds of experiments in the user-defined run scripts. Some examples can be found in the subpackage :mod:`runs`.

.. graphviz::
    :align: center
    :name: structure
    :alt: CSAM program structure
    :caption: CSAM program structure

    digraph {    
        graph [
            fontname="Verdana", 
            fontsize="12",
            ];

        node [
            fontname="Verdana", 
            fontsize="12", 
            color=transparent, 
            shape=record
            ];

        edge [
            fontname="Sans", 
            fontsize="9"
            ];

        //-----------------------------------
        topo [
            label="external\ntopography data"
            ];

        ncdata [
            label="src.io.ncdata", 
            fontcolor=red, 
            URL="modules/src.io.html#src.io.ncdata", 
            target="_top"
            ];

        vartopo [
            label="src.var.topo", 
            fontcolor=red, 
            URL="modules/src.var.html#src.var.topo", 
            target="_top"
            ];
    
        subgraph cluster_topo {
            margin=0
            label = <<B>load the<br/>topography</B>>;
            topo -> ncdata -> vartopo [weight=99];
        };
    
        //-----------------------------------
        extgrid [
            label="external ICON\ngrid"
            ];

        readdat [
            label="src.io.ncdata.read_dat",
            fontcolor=red, 
            URL="modules/src.io.html#src.io.ncdata.read_dat", target="_top"
            ];

        vargrid [
            label="src.var.grid",
            fontcolor=red, 
            URL="modules/src.var.html#src.var.grid"
            target="_top"
            ];

        delaunay [
            label=<regional Delaunay<br/>triangulation:<br/><font color="red">src.delaunay</font>>, 
            URL="modules/src.delaunay.html#src.delaunay.get_decomposition", 
            target="_top"
            ];

        isosceles [
            label=<idealised:<br/><font color="red">src.utils.isosceles<br/>src.utils.delaunay</font>>, 
            URL="modules/src.utils.html#src.utils.isosceles", 
            target="_top"
            ];
        
        subgraph cluster_grid {
            margin=0;
            label = <<B>define the grid</B>>;
            extgrid -> readdat;
            readdat -> vargrid;
            delaunay -> vargrid [weight=1];
            isosceles -> vargrid;
        };
        
        //-----------------------------------
        inputs [
            label=<user-defined<br/>inputs:<br/><font color="red">inputs</font>>,
            URL="modules/inputs.html",
            target="_top"
            ];

        params [
            label=<<font color="red">src.var.params</font>>,
            URL="modules/src.var.html#src.var.params",
            target="_top"
            ];
        
        subgraph cluster_input {
            margin=0;
            label = <<B>define run<br/>parameters</B>>;
            inputs -> params;
        }

        //-----------------------------------

        runs [
            label=<assemble the components<br/>in a run script:<br/><font color="red">runs</font>>, 
            color=black,
            URL="modules/runs.html",
            target="_top"
            ];
        
        vartopo:s -> runs:w [ltail=cluster_topo];
        params:s -> runs:e [ltail=cluster_input];
        vargrid:s -> runs:n [ltail=cluster_grid];
       
        nodepoint [shape=point, color=black, width=0.02];
        runs:s -> nodepoint:n [style=invis];
        nodepoint:n -> runs:s [weight=999];

        //-----------------------------------
        
        wrappers [
            label=<interface modules:<br/><font color="red">wrappers</font>>, 
            color=black,
            URL="modules/wrappers.html",
            target="_top"
            ];

        nodepoint:e -> wrappers [style=invis,weight=0];
        wrappers:n -> nodepoint:s [arrowhead=none];
        
        exp [
            label="use the wrapper components as\nbuilding blocks to interface\nwith the core components"
            ];
        
        {rank=same; exp ; nodepoint};

        //-----------------------------------

        src [
            label=<core modules:<br/><font color="red">src</font>>, 
            color=black,
            URL="modules/src.html"
            target="_top"
            ];

        vis [
            label=<visualisation modules:<br/><font color="red">vis</font>>, 
            color=black,
            URL="modules/vis.html"
            target="_top"
            ];
    
        nodepoint1 [shape=point, style=invis, width=0.01];
        
        wrappers:s -> nodepoint1:n [style=invis];
        {rank=same; src; nodepoint1; vis};
        
        src:n -> wrappers:w [weight=-10];
        vis:n -> wrappers:e;
    }

A first run
^^^^^^^^^^^

To reproduce the coarse grid study (*Coarse Delaunay triangulation (approximately R2B4)* in the manuscript):

1. Make the changes in the user-defined input file, :mod:`inputs.lam_run`. Specifically, enable the switch:

.. code-block:: python

    run_case = "R2B4"

2. Make sure to import the correct user-defined input file. Then execute the run script :mod:`runs.delaunay_runs`:

.. code-block:: console

    python3 ./runs/delaunay_runs.py

Alternatively, the run script could be executed via ``ipython``.

.. note::

    The development of the CSAM codebase frontend is currently ongoing. The current design approach of the program structure aims to simplify debugging and diagnostics using an ``ipython`` environment.