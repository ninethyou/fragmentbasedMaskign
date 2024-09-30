import random
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds

from rdkit.Chem import BRICS

def get_fragments(mol):
    Chem.SanitizeMol(mol)
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges


import numpy as np
def remove_by_motif_strict(Graph, motif_batch, num_to_delete):
    G = Graph.copy()
    total_nodes = len(G.nodes())
    motif_nodes = {i: np.where(motif_batch == i)[0].tolist() for i in np.unique(motif_batch)}
    removed_nodes = set()

    while len(removed_nodes) < num_to_delete:
        if len(G.nodes()) <= (total_nodes - num_to_delete):
            break  # Stop if the graph has fewer nodes left than needed to delete.

        available_motifs = [m for m in motif_nodes if len(set(motif_nodes[m]) - removed_nodes) > 0]
        if not available_motifs:
            break

        chosen_motif = random.choice(available_motifs)
        nodes_to_remove = set(motif_nodes[chosen_motif]) - removed_nodes

        # Check node existence in the graph before removal
        nodes_to_remove = {node for node in nodes_to_remove if node in G.nodes()}

        # Adjust the nodes to remove to not exceed num_to_delete
        if len(removed_nodes) + len(nodes_to_remove) > num_to_delete:
            excess = (len(removed_nodes) + len(nodes_to_remove)) - num_to_delete
            nodes_to_remove = set(random.sample(nodes_to_remove, len(nodes_to_remove) - excess))

        # Remove nodes from the graph and update the removed nodes set
        G.remove_nodes_from(nodes_to_remove)
        removed_nodes.update(nodes_to_remove)


    removed_edges = [(u, v) for u, v in Graph.edges() if u in removed_nodes or v in removed_nodes]

    return list(removed_nodes), removed_edges