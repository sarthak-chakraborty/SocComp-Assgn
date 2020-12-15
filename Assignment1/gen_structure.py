import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import snap
import statistics

Rnd = snap.TRnd(42)
Rnd.Randomize()

graph_filename = sys.argv[1]

dirname = 'subgraphs'
graph_path = os.path.join(os.getcwd(), dirname, graph_filename)
fd_in = open(graph_path, 'r')

# Plot Directory
plotdir = 'plots'
plotpath = os.path.join(os.getcwd(),plotdir)
if not os.path.isdir(plotpath):
    os.mkdir(plotpath)


# Initialize a new graph
G = snap.TUNGraph.New()


# [1] Size of the network
for line in fd_in:
    t = line.split()
    node1 = int(t[0])
    node2 = int(t[1])
    try:
        G.AddNode(node1)
    except:
        pass
    try:
        G.AddNode(node2)
    except:
        pass
    G.AddEdge(node1, node2)

fd_in.close()

# Output Sentences
print("Number of nodes: {}".format(G.GetNodes()))
print("Number of edges: {}".format(G.GetEdges()))




# [2] Degree of nodes in the network 
DegToCnt = snap.TIntPrV()
snap.GetOutDegCnt(G, DegToCnt)
degree_count = {}
for item in DegToCnt:
    degree_count[item.GetVal1()] = item.GetVal2()

OutDeg = snap.TIntPrV()
snap.GetNodeOutDegV(G, OutDeg)
node_deg = {}
for item in OutDeg:
    node_deg[item.GetVal1()] = item.GetVal2()

max_deg_nodes = [k for k, v in node_deg.items() if v == max(node_deg.values())]

# Output sentences
print("Number of nodes with degree=7: {}".format(snap.CntOutDegNodes(G, 7)))
print("Node id(s) with highest degree: ", end=" ")
for node in max_deg_nodes:
    if node == max_deg_nodes[-1]:
        print(node)
    else:
        print(str(node) + ", ", end=" ")

# Plot Degree Distribution
plot_filename = 'deg_dist_' + graph_filename[:-6] + '.png'
plot_filedir = os.path.join(plotpath, plot_filename)
plt.figure()
plt.scatter(list(degree_count.keys()), list(degree_count.values()), s=10)
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.title("Degree Distribution ({})".format(graph_filename[:-6]))
plt.savefig(plot_filedir)




# [3] Paths in the network
full_diam = []
for num_test_nodes in [10, 100, 1000]:
    d = snap.GetBfsFullDiam(G, num_test_nodes, False)
    full_diam.append(d)
    print("Approximate full diameter by sampling {} nodes: {}".format(num_test_nodes, d))

print("Approximate full diameter (mean and variance): {}, {}".format(round(statistics.mean(full_diam), 4), round(statistics.variance(full_diam), 4)))

eff_diam = []
for num_test_nodes in [10, 100, 1000]:
    d = snap.GetBfsEffDiam(G, num_test_nodes, False)
    eff_diam.append(d)
    print("Approximate effective diameter by sampling {} nodes: {}".format(num_test_nodes, round(d, 4)))

print("Approximate effective diameter (mean and variance): {}, {}".format(round(statistics.mean(eff_diam), 4), round(statistics.variance(eff_diam), 4)))

# Get Shortest Path Distribution
shortest_path_dist = {}
for NI in G.Nodes():
    NIdToDist = snap.TIntH()
    shortestPath = snap.GetShortPath(G, NI.GetId(), NIdToDist)
    for item in NIdToDist:
        if NIdToDist[item] in shortest_path_dist:
            shortest_path_dist[NIdToDist[item]] += 1
        else:
            shortest_path_dist[NIdToDist[item]] = 1

# Plot shortest path
plot_filename = 'shortest_path_' + graph_filename[:-6] + '.png'
plot_filedir = os.path.join(plotpath, plot_filename)
plt.figure()
plt.scatter(list(shortest_path_dist.keys()), list(shortest_path_dist.values()), s=10)
plt.xlabel("Shortest Path Length")
plt.ylabel("Frequency")
plt.title("Shortest Path Distribution ({})".format(graph_filename[:-6]))
plt.savefig(plot_filedir)

"""
FOR FASTER COMPUTATION, UNCOMMENT THE FOLLOWING LINE AND COMMENT OUT LINE 107-125
"""
# snap.PlotShortPathDistr(G, "shortest_path_{}".format(graph_filename[:-6]), "Shortest Path Distribution ({})".format(graph_filename[:-6]))



# [4] Components of the network
SCC = snap.GetMxScc(G)
print("Fraction of nodes in largest connected component: {}".format(round(SCC.GetNodes()/G.GetNodes(), 4)))

Edge_Bridge = snap.TIntPrV()
snap.GetEdgeBridges(G, Edge_Bridge)
print("Number of edge bridges: {}".format(len(Edge_Bridge)))

ArticulationPoint = snap.TIntV()
snap.GetArtPoints(G, ArticulationPoint)
print("Number of articulation points: {}".format(len(ArticulationPoint)))

CComp = snap.TIntPrV()
snap.GetSccSzCnt(G, CComp)
connected_component = {}
for comp in CComp:
    connected_component[comp.GetVal1()] = comp.GetVal2()

# Plot Degree Distribution
plot_filename = 'connected comp_' + graph_filename[:-6] + '.png'
plot_filedir = os.path.join(plotpath, plot_filename)
plt.figure()
plt.scatter(list(connected_component.keys()), list(connected_component.values()), s=15)
plt.xlabel("Size of Connected Components")
plt.ylabel("Number of components")
plt.title("Connected Component Distribution ({})".format(graph_filename[:-6]))
plt.savefig(plot_filedir)




# [5] Connectivity and Clustering in the Network
cluster_coeff = snap.GetClustCf(G, -1)
print("Average clustering coefficient: {}".format(round(cluster_coeff, 4)))

num_triads = snap.GetTriads(G, -1)
print("Number of triads: {}".format(num_triads))

node_id = G.GetRndNId(Rnd)
node_cluster_coeff = snap.GetNodeClustCf(G, node_id)
print("Clustering coefficient of random node {}: {}".format(node_id, round(node_cluster_coeff, 4)))

node_id = G.GetRndNId(Rnd)
node_num_triads = snap.GetNodeTriads(G, node_id)
print("Number of triads random node {} participates: {}".format(node_id, node_num_triads))

triad_edge = snap.GetTriadEdges(G)
print("Number of edges that participate in at least one triad: {}".format(triad_edge))

cf_dist = snap.TFltPrV()
coeff = snap.GetClustCf(G, cf_dist, -1)
degree_coeff = {}
for pair in cf_dist:
    degree_coeff[pair.GetVal1()] = pair.GetVal2()

# Plot Degree Distribution
plot_filename = 'clustering_coeff_' + graph_filename[:-6] + '.png'
plot_filedir = os.path.join(plotpath, plot_filename)
plt.figure()
plt.scatter(list(degree_coeff.keys()), list(degree_coeff.values()), s=10)
plt.xlabel("Degree")
plt.ylabel("Clustering Coefficient")
plt.title("Clustering Coefficient Distribution ({})".format(graph_filename[:-6]))
plt.savefig(plot_filedir)