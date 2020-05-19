# Global Routing Explain

Credits: The pictures and illustration examples are taken from some of the literature on global routing

## Global Routing Problem
The global routing problem can be modeled as a grid graph `G(V, E)`, where each vertex v<sub>i</sub> represents a rectangular region of the chip, so called a global routing cell (Gcell) or global routing tile, and an edge e<sub>ij</sub> represents the boundary between v<sub>i</sub> and v<sub>j</sub> with a given maximum routing resource m<sub>ij</sub>.  Figure below shows how the chip can be abstracted into a grid graph where m<sub>AB</sub> = 3. A global routing is to find paths that connect the pins inside the Gcells through `G(V, E)` for every net.
![alt text](gcell.png)

## Benchmarks
The benchmarks (present in ./benchmarks) are taken from a subset of the ones in ISPD global routing contests. The core idea is to formulate the routing problem outside the Virtuoso environment to facilitate algorithm development. The optimization objective function will be congestion and wirelength:
- **Congestion**: Minimum total overflow for all benchmarks. Each benchmark will have a number of edges between global routing tiles; each of these edges will have a capacity. All nets must be routed within the global routing graph, ideally with no graph edge exceeding the specified capacity. Mathematically, `congestion = max(0,utilization-capacity)`.
- **Minimum wirelength**: If two routing solutions achieve the same capacity constraint, the one with smaller wirelength will be a better solution. Wirelength will be measured based on number of tile-to-tile connections.

## Input Format
The input file format is a variation of *Labyrinth* format. 

| Source line | Example `./benchmarks/small.gr` | Explanation |
| :---------- | :---------: | ----------: |
| grid <> <> <> | grid 3 3 2 | x grids (#gcells), y grids(#gcells), number of routing layers | 
| vertical capacity <> <> ... | vertical capacity 0 1 | max allowed vertical capacity for a gcell on each layer (in length units) **See note 1** | 
| horizontal capacity <> <> ... | horizontal capacity 1 0 | max allowed horizontal capacity for a gcell on each layer (in length units) | 
| minimum width <> <> ...  | minimum width 1 1 | minimum width of track on each layer (in length units) |
| minimum spacing <> <> ... | minimum spacing 0 0 | minimum spacing between tracks on each layer (in length units) |  
| via spacing <> <> ... | via spacing 0 0 | via spacing on each layer (in length units) | 
| <> <> <> <> | 0 0 10 10 | Origin coordinates on lower left followed by (grid/)tile width and height (in length units) | 
| *newline* | *newline* | Optional *newline* |
| num net <> | num_net 1 | number of nets in the netlist of the design | 
| <> <> <> <> | A 0 2 1 | net_name, net_id, number_of_pins, minimum_width (override layer specification). Iterate over nets |
| <> <> <> | 5 5 1 | pin coordinates in terms of x, y, layer. **See note 4** |
| <> <> <> | 25 5 1 | another example of pin coordinates. Iterate over pins |
| <> | 4 | # adjustments depicting reduced capacity (again overrides layer specification) **See note 6** |
| <> <> <> <> <> <> <> | 1 0 1   2 0 1   0 | coordinates of the gcells that form this edge (x,y,layer), along with the adjusted capacity (in length units) |
  

### Notes:
1. `Max tracks possible = capacity/(width+spacing)`

2. Each layer may have a unique capacity in each direction per global routing tile, and it may be different for horizontal and vertical directions. Preferred routing directions will be given by having a zero capacity in the non-preferred direction. In the vertical and horizontal capacity lines, the first number indicates the capacity on the first layer, the second number the second layer, and so on. Minimum wire widths and minimum wire spacings are also specified; this impacts capacity calculations as shown in the earlier note.

3. The lower left corner (minimum X and Y) of the global routing region is specified, as well as the width (X dimension) and height (Y dimension) of each tile.

4. Additionally, pin positions are given in terms of absolute length, rather than tile/grid coordinates. Conversion from pin positions to tile numbers can be done with `floor(pin_x - lower_left_x)/tile_width` and `floor (pin_y - lower_left_y)/tile_height`. Pins will not be on the borders of global routing cells, so there should be no ambiguity. All pins will be within the specified global routing region.

5. Each net will have a minimum routed width; this width will span all layers. When routing, compute the utilization of a global routing graph edge by adding the widths of all nets crossing and edge, plus the minimum spacing multiplied by the number of nets. Each wire will require spacing between it's neighbors; think of this as having one-half of the minimum spacing reserved on each side of a wire.

6. Congestion is modeled by including capacity adjustments. In the global routing benchmarks, there may be obstacles, or pre-routed wires. To communicate this to the global router, pairs of (adjacent) global routing tiles may have a capacity that is different from the default values specified at the start of a benchmark file.

7. Calculation of capacity is more complex than is done in typical academic global routing tools. Each global routing tile will have a capacity; this is a measure of the available space, not the number of global routing tracks. If the minimum wire width is 20, the minimum spacing 10, and the capacity of a tile is given as 450, this corresponds to 15 minimum width tracks (15 * (20 + 10)) . Therefore, capacity here will have the dimensions of length. The capacity specified as the default value may be different than the width or height of a tile. In general, it is desirable to have routing utilization of a tile be below the capacity, as higher utilization can be more difficult for detail routers to complete. 

## Output Format
The output format is a variation of the `BoxRouter format`. See `./benchmarks/small.solution` for an example. 

| Source line | Example `./benchmarks/small.solutions` | Explanation |
| :---------- | :---------: | ----------: |
| <> <> <> | A 0 14 | netname, net id, number of tracks. Iterate over nets |
| (<>,<>,<>)-(<>,<>,<>) | (5,5,1)-(15,5,1) | track/connection from coordinates of one gcell to the other. Iterate over tracks |


## Example
An illustration on a small example benchmark (./benchmarks/small.gr) is provided below.  
![alt text](small.png?raw=true "Example")  

## Evaluation
A script to evaluate the routing solution in terms of capacity overflow and wirelength is present at ./evaluation/eval2008.pl

## Ideas for Solution
Routing for a two-pin net can be done with a general purpose algorithm such as A\*-search. For multi-pin nets (more than two pins), one approach could be decompose the net into two-pin nets and then route those two-pin nets. One decomposition could be done by an MST algorithm like Prim (polynomial time), but this may lead to suboptimal solutions. A rectilinear steiner minimum tree (NP-hard), on the other hand may yield a better solution. As an example, a 4-pin net and its Minimum Spanning Tree routing is shown below.  
![alt text](./mst.png?raw=true "4-pin net and MST")  

Rectilinear Steiner Minimum tree routing of the net is shown below  
![alt text](rsmt.png?raw=true "RSMT")  
