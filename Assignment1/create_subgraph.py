import sys
import os

# Create output directory
outdir = 'subgraphs'
outpath = os.path.join(os.getcwd(),outdir)
if not os.path.isdir(outpath):
    os.mkdir(outpath)


for graph in ['facebook', 'amazon']:

    if graph == 'facebook':
        in_graph_filename = graph + '_combined.txt'
    elif graph == 'amazon':
        in_graph_filename = 'com-' + graph + '.ungraph.txt'

    out_graph_filename = graph + '.elist'
    out_graph_path = os.path.join(outpath, out_graph_filename)

    fd_in = open(in_graph_filename, 'r')
    fd_out = open(out_graph_path, 'w')

    for line in fd_in:
        t = line.split()
        if graph == 'facebook':
            if int(t[0])%5 != 0 and int(t[1])%5 != 0:
                fd_out.write(t[0] + ' ' + t[1] + '\n')
        
        if graph == 'amazon':
            if int(t[0])%4 == 0 and int(t[1])%4 == 0:
                fd_out.write(t[0] + ' ' + t[1] + '\n')

    fd_in.close()
    fd_out.close()