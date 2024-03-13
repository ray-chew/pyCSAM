Quickstart
==========


Introduction
------------

.. graphviz::
    :align: center

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

Requirements
^^^^^^^^^^^^
.. literalinclude:: ../../requirements.txt



Example
-------
.. note::

    to be completed
