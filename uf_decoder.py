"""Union-Find and peeling decoder for surface codes
This script builds a graph needed for UF from the pymatching graph
Runs Union Find decoder to obtain erasures for the peeling decoder
peeling decoder the creates a tree using BFS and peels to estimate required error corrections 
"""

#from __future__ import annotations

from collections import deque

import numpy as np



def build_uf_graph(graph):
    """Converts a `matching.to_networkx()` graph to a UF-ready structure.

    Returns (adj, edge_faults, boundary):
      adj: represents which vertices are connected to said vertex (dictionary)
      edge_faults: represents the fault ids for each edge (data qubit)
      boundary:    set of boundary vertices

      Just simplifies data to what is needed for UF and Peeling
    """
    boundary = {n for n, d in graph.nodes(data=True) if d.get("is_boundary")} # n = node id, d = attributes
  

    adj = {v: [] for v in graph.nodes} # inittiisese dictionary with subject a list of all neighbouring nodes
    edge_faults = {} # all the faults detected by staibalisers
    for u, v, data in graph.edges(data=True):
        edge_id = tuple(sorted((u, v))) # each indidual detection is represented by and edge_id 
        edge_faults[edge_id] = set(data.get("fault_ids", set()))
        adj[u].append(v) 
        adj[v].append(u) # allows you to traverse graph from either end

    return adj, edge_faults, boundary


def lit_vertices(detection_events_row):
    """Detector indices that fired in a single shot -> the UF cluster seeds."""
    return {i for i, fired in enumerate(detection_events_row) if fired}

# -------------------------------------------------------------------------------------
# Used for UF

class Node:
    """
    Represents a vertex, and is used in Union Find Algorithm
    """

    def __init__(self, vertex):
        self.id = vertex
        self.parent = self          # initialises node as a root node
        self.size = 1
        self.parity = False         # True == odd number of lit vertices
        self.touches_boundary = False
        self.fringe = set()         # frontier edge ids of this cluster, used when growing cluster

    def is_valid(self):
        """A cluster stops growing once it is even-parity or hits the boundary."""
        return (not self.parity) or self.touches_boundary

    def __repr__(self): # defines how the object looks when you print it
        return f"Node({self.id}, size={self.size}, parity={int(self.parity)})"


def find(node):
    """Return the root of node's cluster, with path compression."""
    root = node
    while root.parent is not root:
        root = root.parent
    # compress: point every node on the path straight at the root, mentioned in paper
    while node.parent is not root:
        node.parent, node = root, node.parent
    return root


def union(a, b):
    """Merge two roots by size and combine their cluster metadata.

    Returns the surviving root.
    """
    if a is b:
        return a
    if a.size < b.size:               # keep the larger tree as the survivor
        a, b = b, a
    b.parent = a
    a.size += b.size
    a.parity ^= b.parity
    a.touches_boundary = a.touches_boundary or b.touches_boundary
    a.fringe |= b.fringe
    return a
# --------------------------------------------------------------------------------
# Decoder

class UnionFindDecoder:
    """Classic grow-all-odd Union-Find syndrome validation."""

    def __init__(self, adj, edge_faults, boundary, num_observables=None):
        self.adj = adj
        self.edge_faults = edge_faults
        self.boundary = set(boundary)
        # length of the predicted-flip array decode returns. Pass
        # matching.num_fault_ids to guarantee the same shape as
        # matching.decode; otherwise infer from the fault ids present.

        # This is needed for the output to get the same format as for decoder as actual output
        if num_observables is None:
            all_fids = [fid for faults in edge_faults.values() for fid in faults]
            num_observables = (max(all_fids) + 1) if all_fids else 0
        self.num_observables = num_observables
        # precompute each vertex's incident edge ids
        self.incident = {v: set() for v in adj}
        for (u, w) in edge_faults: # goes through all edges to their respective vertex
            self.incident[u].add((u, w))
            self.incident[w].add((u, w))
        # populated by grow()
        self.nodes = {}
        self.support = {}
        self.roots = set()

    def grow(self, syndrome):
        """Grow clusters until all are valid; return the erasure edge set.

        syndrome is the set of lit vertex ids 
         stores self.nodes, self.roots, self.support for
        inspection and for the peeling step.
        """
        syndrome = set(syndrome)

        # --- reset per shot: every vertex is its own singleton cluster ---
        self.nodes = {v: Node(v) for v in self.adj}
        for v, n in self.nodes.items():
            n.parity = v in syndrome
            n.touches_boundary = v in self.boundary
            n.fringe = set(self.incident[v])  # ALL edges touching v, regardless of support
        self.support = {e: 0 for e in self.edge_faults}
        roots = set(self.nodes.values())

        # --- grow-all-odd loop --- 
        while True:
            invalid = [r for r in roots if not r.is_valid()] # creates a list of clusters with odd parity
            if not invalid:
                break

            # (a) grow: each invalid cluster pushes half an edge along its fringe
            # Increases support dictionary, uses this to grow custer in (b) 
            newly_full = []
            grew = False
            for r in invalid:
                for e in r.fringe:
                    if self.support[e] < 2:
                        self.support[e] += 1
                        grew = True
                        if self.support[e] == 2:
                            newly_full.append(e)
            if not grew:
                break

            # (b) fuse: a fully-grown edge merges the clusters at its endpoints
            for (u, v) in newly_full: # if a boundary is unlit, it still has a root node, so it is added to the clluster and fringes is updated
                ru, rv = find(self.nodes[u]), find(self.nodes[v])
                if ru is not rv:
                    merged = union(ru, rv)
                    roots.discard(ru)
                    roots.discard(rv)
                    roots.add(merged)

            # (c) drop fully-grown edges from the fringe they now sit inside
            for e in newly_full:
                find(self.nodes[e[0]]).fringe.discard(e)

        self.roots = roots
        return {e for e, s in self.support.items() if s == 2} # represnts erasures between stabalisers that have been expaned through UF

    def cluster_of(self, vertex):
        """Root node of the cluster containing vertex, post-decode."""
        return find(self.nodes[vertex])

    # -----------------------------------------------------------------------
    # Peeling
  
    def peel(self, erasure, syndrome):
        """Reduce an erasure to a correction edge set.

        Spanning forest created using Breadth first search, this is then flipped so all 
        children nodes are interactied with before parents. peeling then occurs flipping erasures with defects and 
        flipping the parent nodes (as that will be given a defect once the child has been flipped) explained better in paper
        """
        # (1) adjacency of the erased subgraph
        # convering erasure edges back into linked nodes 
        erased_adj = {}
        for (u, v) in erasure:
            erased_adj.setdefault(u, []).append(v)
            erased_adj.setdefault(v, []).append(u)

        # (2) rooted spanning forest via BFS; seed boundary vertices first so
        #     their components root at the boundary.
        parent = {}          # child -> (parent_vertex, edge_id)
        order = []           # BFS visitation order (roots first)
        visited = set()
        seeds = [v for v in erased_adj if v in self.boundary]
        seeds += [v for v in erased_adj if v not in self.boundary]
        for s in seeds:
            if s in visited:
                continue
            visited.add(s)
            order.append(s)
            queue = deque([s]) # used in BFS examples, is quicker than using a normal list
            while queue:
                x = queue.popleft()
                for y in erased_adj[x]:
                    if y not in visited:
                        visited.add(y)
                        parent[y] = (x, tuple(sorted((x, y))))
                        order.append(y)
                        queue.append(y)

        # (3) peel leaves-inward: reverse BFS order => children before parents.
        #     Roots (no parent) are the absorbers and are never peeled.
        defects = set(syndrome)
        correction = set()
        for u in reversed(order):
            if u not in parent:
                continue
            p, edge_id = parent[u]
            if u in defects:
                # This edge is part of the correction. Applying it flips the
                # defect state of both its endpoints: u is now resolved, and
                # the defect is pushed up to the parent p.
                correction.add(edge_id)
                defects.discard(u)          # u's defect is corrected by this edge
                if p in defects:            # the edge also toggles the parent:
                    defects.discard(p)      #   parent already had a defect -> cancels
                else:
                    defects.add(p)          #   parent had none -> inherits it
        return correction

    # -----------------------------------------------------------------------
   # Decode function
    def decode(self, shot):
        """
        Predict logical-observable flips for a single shot.
        """
        syndrome = shot if isinstance(shot, set) else lit_vertices(shot)
        erasure = self.grow(syndrome)
        correction = self.peel(erasure, syndrome)
        return self._correction_to_observables(correction)

    def _correction_to_observables(self, correction):
        """XOR the fault ids of the correction edges into a flip array."""
        flipped = set()
        for e in correction:
            flipped ^= self.edge_faults[e]
        out = np.zeros(self.num_observables, dtype=np.uint8)
        for fid in flipped:
            out[fid] = 1
        return out
