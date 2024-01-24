from graphviz import Digraph

# edges = [
#     ("INPUT", "Distortion Correction"),
#     ("Distortion Correction", "ROI"),
#     ("ROI", "Background Subtraction"),
#     ("Background Subtraction", "Ceiling at 90/255"),
#     ("Ceiling at 90/255", "Rescaling"),
#     ("Rescaling", "Opening by 5x5 Disk"),
#     ("Opening by 5x5 Disk", "Median Blur 5px"),
#     ("Median Blur 5px", "OUTPUT"),
# ]

edges = [
    ("INPUT", "Distortion Correction"),
    ("Distortion Correction", "ROI"),
    ("ROI", "Background Subtraction"),
    ("Background Subtraction", "Ceiling at 90/255"),
    ("Ceiling at 90/255", "Rescaling"),
    ("Rescaling", "KILOBOT FOUND "),
    ("KILOBOT FOUND", "Cropping  to KB_SIZExKB_SIZE"),
    ("Cropping  to KB_SIZExKB_SIZE", "Ajust Median to 30/255"),
    ("Ajust Median to 30/255", "Rescale"),
    ("Rescale", "Invert"),
    ("Invert", r"Binary Threshold\n150"),
    ("Invert", r"Adaptative Binary Threshold\nGaussian\nBlock size=21\nC=-14 + softness"),
    (r"Adaptative Binary Threshold\nGaussian\nBlock size=21\nC=-14 + softness", "Bitwise AND"),
    (r"Binary Threshold\n150", "Bitwise AND"),
    ("Bitwise AND", r"Mass Filter\n > 30 - softness\n < 400"),
    (r"Mass Filter\n > 30 - softness\n < 400", r"Out of torus erosion\n in_radius=60px\n out_radius=KB_SIZE-5"),
    (r"Out of torus erosion\n in_radius=60px\n out_radius=KB_SIZE-5", "OUTPUT"),
]

shapes = {
    "INPUT": "diamond",
    "KILOBOT FOUND ": "diamond",
    "KILOBOT FOUND": "diamond",
    "OUTPUT": "diamond",
}


g = Digraph('G', filename='Pipeline_mask.gv')

# NOTE: the subgraph name needs to begin with 'cluster' (all lowercase)
#       so that Graphviz recognizes it as a special cluster subgraph
g.attr('node', shape="box")

for key in shapes.keys():
    g.node(key, shape=shapes[key])


g.edges(edges)
g.attr(label=r'\n\nKilobot Identification Pipeline')


g.view()
