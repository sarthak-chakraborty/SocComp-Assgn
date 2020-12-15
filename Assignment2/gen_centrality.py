import numpy as np
import snap
import os
import sys
import time
import operator
from collections import deque

INF = sys.maxsize


def ReadGraph(filename):
	"""
	Reads the graph in edgelist format from the filename
	and stores it as TUNGraph using snap library
	"""
	G = snap.TUNGraph.New()

	fd_in = open(filename, 'r')

	# Read and store the graph
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
	return G


def CreateOutDir(out_dir):
	"""
	Creates an output directory where the files will be stored
	"""
	out_path = os.path.join(os.getcwd(),out_dir)
	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	return out_path


def SaveCentrality(centrality, path, name):
	"""
	Saves the centrality file as txt in `path` directory
	"""
	file_name = name + ".txt"
	file_path = os.path.join(path, file_name)
	fd = open(file_path, 'w')

	for key, value in centrality:
		fd.write(str(key) + " " + str(value) + "\n")


def sort_dict(_dict):
	return sorted(_dict.items(), key=operator.itemgetter(1), reverse=True)	


def CalcShortestPath(G):
	"""
	Returns a 2D numpy array for all pair shortest paths
	"""
	shortest_path = [-1] * G.GetNodes()
	dist = np.array([[INF for j in range(G.GetNodes())] for i in range(G.GetNodes())])

	for NI in G.Nodes():
		NIdToDist = snap.TIntH()
		dist[NI.GetId()][NI.GetId()] = 0
		sp = snap.GetShortPath(G, NI.GetId(), NIdToDist)	# Shortest path to all nodes from node NI
		shortest_path[NI.GetId()] = sp

		for item in NIdToDist:
			dist[NI.GetId()][item] = NIdToDist[item]

	return dist


def ClosenessCentrality(G):
	"""
	Calculates Closeness Centrality of all nodes and 
	returns a dictionary where  `key`=node_id, `value`=closeness centrality of that node
	"""
	n_nodes = G.GetNodes()
	centrality = {}
	dist = CalcShortestPath(G)	# Stores all pair shortest path

	for NI in G.Nodes():
		node = NI.GetId()
		tot_shortest_path = sum(dist[node])

		if tot_shortest_path > 0 and n_nodes > 1:
			centr = (n_nodes - 1) / float(tot_shortest_path)
			centrality[node] = round(centr, 6)
		else:
			centrality[node] = 0.0
		
	return centrality


def BetweennessCentrality(G):
	"""
	Calculates Betweenness Centrality using Brandes' Algorithm and
	returns a dictionary where  `key`=node_id, `value`=betweenness centrality of that node
	"""
	node_list = [node_id for node_id in range(G.GetNodes())]

	# Find the neighbours of each node
	neighbour_list = {}
	for NI in G.Nodes():
		edgelist = [e for e in NI.GetOutEdges()]
		neighbour_list[NI.GetId()] = edgelist

	centrality = {node : 0 for node in node_list}

	# Brandes' Algorithm
	for s in node_list:
		stack = []
		path = {node : [] for node in node_list}
		g = {node: 0 for node in node_list}
		d = {node : -1 for node in node_list}
		Q = deque([])

		g[s] = 1
		d[s] = 0
		Q.append(s)
		while Q:
			# Get the path information for each shortest path from one noe to the other
			node = Q.popleft()
			stack.append(node)
			for nbr_node in neighbour_list[node]:
				if d[nbr_node] < 0:
					Q.append(nbr_node)
					d[nbr_node] = d[node] + 1
				if d[nbr_node] == d[node] + 1:
					g[nbr_node] = g[nbr_node] + g[node]
					path[nbr_node].append(node)

		delta = {node : 0 for node in node_list}
		while stack:
			node = stack.pop()
			# Calculate Betweenness Centrality for `node`
			for node_i in path[node]:
				delta[node_i] = delta[node_i] + (g[node_i]/g[node]) * (1 + delta[node])
			if node != s:
				centrality[node] = centrality[node] + delta[node]

	return centrality


def PageRank(G, alpha=0.8, TOL=1e-5, max_iter=200):
	"""
	Calculates Biased Page Rank for all nodes and
	returns a dictionary where  `key`=node_id, `value`=Page Rank of that node
	"""
	node_list = [node_id for node_id in range(G.GetNodes())]

	# Adjacency Matrix P
	P = np.array([[0. for j in node_list] for i in node_list])
	for EI in G.Edges():
		P[EI.GetSrcNId()][EI.GetDstNId()] = 1.
		P[EI.GetDstNId()][EI.GetSrcNId()] = 1.
	
	# Adjust values of P to make it Markovian
	for i in range(len(P)):
		s = sum(P[i])
		if s == 0.:
			new_row = [ 1. / G.GetNodes() for node in node_list]
			P[i] = new_row
		else:
			P[i] = P[i] / s

	# Create teleportation matrix (multiply Vector of all 1's to transpose of preference vector)
	pref_vec = np.array([0. for node_id in node_list])
	for node in node_list:
		if node % 4 == 0:
			pref_vec[node] = 4. / G.GetNodes()
	pref_vec = pref_vec.reshape(1,-1)
	teleport = np.matmul( np.ones(len(node_list)).reshape(1,-1).T , pref_vec )

	P = alpha * P + (1 - alpha) * teleport	# Transition Matrix

	# Power Iteration
	pr = np.random.rand(P.shape[1]).reshape(1,-1)
	for _ in range(max_iter):
		pr_prev = pr.copy()
		pr = np.dot(pr, P)
		pr = pr / np.sum(pr)

		# Check if maximum change in PR for a node is less than tolerance
		if np.max(np.abs(pr - pr_prev)) < TOL:
			break
	
	pr = pr.reshape(-1)
	centrality = {i : pr[i] for i in range(len(pr))}

	return centrality


def CalcCentrality(G, name):
	"""
	Wrapper Function for centrality calculations
	"""
	centrality = {}
	if name == "closeness":
		centrality = ClosenessCentrality(G)
	elif name == "betweenness":
		centrality = BetweennessCentrality(G)
	elif name == "pagerank":
		centrality = PageRank(G, alpha=0.8)

	return sort_dict(centrality)


def main():
	graph_filename = "facebook_combined.txt"
	graph_path = os.path.join(os.getcwd(), graph_filename)

	G = ReadGraph(graph_path)

	centralities = ["closeness", "betweenness", "pagerank"]

	# Create "centralities folder"
	out_path = CreateOutDir('centralities')

	for centr in centralities:
		centrality = CalcCentrality(G, name=centr)
		SaveCentrality(centrality, out_path, name=centr)


if __name__ == "__main__":
	main()