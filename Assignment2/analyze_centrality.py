import numpy as np
import snap
import os
import sys
import time
import operator

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


def ReadCentrality(path, name):
	"""
	Read centrality file and store it in dictionary
	"""
	file_name = name + ".txt"
	file_path = os.path.join(path, file_name)
	fd = open(file_path, 'r')

	centrality = {}
	for line in fd:
		t = line.split()
		centrality[int(t[0])] = float(t[1])

	return centrality


def sort_dict(_dict):
	return sorted(_dict.items(), key=operator.itemgetter(1), reverse=True)


def ClosenessCentrality(G):
	centrality = {}
	for NI in G.Nodes():
		centr = snap.GetClosenessCentr(G, NI.GetId())
		centrality[NI.GetId()] = round(centr, 6)
		
	return centrality


def BetweennessCentrality(G):
	centrality = {}
	Nodes = snap.TIntFltH()
	Edges = snap.TIntPrFltH()
	snap.GetBetweennessCentr(G, Nodes, Edges, 0.8)

	for node in Nodes:
		centrality[node] = round(Nodes[node], 6)

	return centrality


def PageRank(G):
	centrality = {}
	PRank = snap.TIntFltH()
	snap.GetPageRank(G, PRank)

	for item in PRank:
		centrality[item] = PRank[item]

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
		centrality = PageRank(G)

	return sort_dict(centrality)


def Compare(centr1, centr2):
	"""
	Compares two sets of centality lists and returns the number of overlaps
	"""
	num = 0
	node1 = [item[0] for item in centr1]
	node2 = [item[0] for item in centr2]

	for key in node1:
		if key in node2:
			num += 1
	
	return num


def main():
	graph_filename = "facebook_combined.txt"
	graph_path = os.path.join(os.getcwd(), graph_filename)
	centrality_path = os.path.join(os.getcwd(), 'centralities')

	G = ReadGraph(graph_path)

	centralities = ["closeness", "betweenness", "pagerank"]

	for centr in centralities:
		centrality_snap = CalcCentrality(G, name=centr)
		top_hundred_snap = centrality_snap[:100]	# Top 100 ranked nodes

		centrality_user = ReadCentrality(centrality_path, name=centr)
		centrality_user = sort_dict(centrality_user)
		top_hundred_user = centrality_user[:100]	# Top 100 ranked nodes
		num = Compare(top_hundred_snap, top_hundred_user)

		if centr == "closeness":
			print("#overlaps for Closeness Centrality: {}".format(num))
		elif centr == "betweenness":
			print("#overlaps for Betweenness Centrality: {}".format(num))
		elif centr == "pagerank":
			print("#overlaps for PageRank Centrality: {}".format(num))


if __name__ == "__main__":
	main()