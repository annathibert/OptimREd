#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Bio import SeqIO, AlignIO
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.SubsMat.MatrixInfo import * 
from ete3 import Tree
import pdb
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
import numpy as np
import copy

def translate_aln(aln_cod, transl_table="Standard"):
	"""Translate cod aln to aa aln. 
	"""
	seq_list=[]
	for record in aln_cod:
		tmp_record = record[:]
		# Convert to Seq if mutable seq
		try:
			new_seq = tmp_record.seq.toseq().translate(gap="-", table=transl_table)
		except AttributeError:
			new_seq = tmp_record.seq.translate(gap="-", table=transl_table)
		seq_list+=[SeqRecord(new_seq,id=record.id)]
	aln_aa = MultipleSeqAlignment(seq_list)
	return aln_aa

def simulate_editions(aln_cod, columns, species, params_ed, return_nbed=False):
	"""Change all C to T for specified species and columns. Returns new aln.
	return_nbed -- True if return the number of C modified to T
	params_ed -- Looks like {src:"C", tgt:"T"}
	"""
	copy_aln = copy.deepcopy(aln_cod)
	nb_ed = 0
	for record in copy_aln:
		if record.id in species:
			tmp_seq = str(record.seq)
			# tmp_new_seq = str(record.seq)
			# Seq has to be MutableSeq to directly modify the sequence
			# Should probably do differently
			try:
				record.seq = record.seq.tomutable()
			except:
				pass
			for edp in columns:
				if tmp_seq[edp]==params_ed["src"]:
					record.seq[edp] = params_ed["tgt"]
					nb_ed+=1
	if return_nbed:
		return copy_aln, nb_ed
	return copy_aln



def get_SPscores(aln_aa, subst_mat=blosum62, subset=[], old_SP=None, gap_score=-4):
	"""Return a vector of Sum-of-Pair scores for all positions in the alignment
	Keyword arguments:
	aln_aa -- amino acid alignment
	subst_mat -- substitution matrix 
	subset -- subset of columns to be computed (all others will be put to 0) if provided
	"""
	# weird way of adding new SP scores when there is a subset 
	# should do the same way as sankoff_aln
	if old_SP==None:
		SPscore = [0]*aln_aa.get_alignment_length()
	else:
		SPscore = list(old_SP)
	def __get_SPscore_col(col):
		"""Compute Sum-Of-Pair score for one column
		"""
		col_score = 0 
		col_seq = aln_aa[:,c]
		for i in range(len(col_seq)-1):
			for j in range(i+1, len(col_seq)):
				if col_seq[i]=='-' or col_seq[j]=='-' or col_seq[i]=='*' or col_seq[j]=='*':
					col_score+=gap_score
				else:
					(si, sj)=(col_seq[i], col_seq[j])
					if (si,sj) in subst_mat:
						col_score += subst_mat[(si,sj)]
					else:
						col_score += subst_mat[(sj,si)]
		return col_score
	# If no subset of columns is provided, all the SP scores for all positions have to be recomputed
	if subset==[]:
		for c in range(aln_aa.get_alignment_length()):
			SPscore[c] = __get_SPscore_col(c)
	# Otherwise, only the provided columns are recomputed
	else:
		for c in subset:
			SPscore[c] = __get_SPscore_col(c)
	return SPscore

def get_confusion_matrix(truth_set, predicted_set, all_set):
	"""Return number of True Positive (TP), TN, FP, FN
	"""
	TP = len(list(set(truth_set) & set(predicted_set)))
	FP = len(list(set(predicted_set)-set(truth_set)))
	FN = len(list(set(truth_set)-set(predicted_set)))
	TN = len(list(set(all_set)-(set(truth_set)|set(predicted_set))))
	return [TP, TN, FP, FN]

def get_accuracy(truth_set, predicted_set, all_set):
	"""Return sensitivity and specificity
	"""
	[TP, TN, FP, FN] = get_confusion_matrix(truth_set, predicted_set, all_set)
	if (TP+FN)==0:
		sens=0
	else:
		sens = TP / (TP+FN)
	if (TN+FP)==0:
		spec = 0
	else:
		spec = TN / (TN+FP)
	return [sens, spec]


def read_to_list(filename):
	""" Reads a file with list of edited species or columns and returns a list element
	"""
	res_list = []
	with open(filename) as file:
		for line in file:
			line = line.rstrip()
			try:
				line = int(line)
			except:
				pass
			res_list += [line]
	return res_list

def sankoff(t, seq_dict, dist_dict, init_dict, penalty=0, ed_list=[], init_value_ifed=0, init_dict_value=0):
	"""Return custom Sankoff parsimony scores (for each character) and tree of a single MSA column 
	t -- tree (not modified in place)
	seq_dict -- {"genome1":'A', "genome2":'T'}
	init_dict -- key: possible character value
	penalty -- value to add in edited nodes 
	ed_list -- list of edited genomes 
	init_value_ifed -- penalty value added in each single leaf
	init_dict_value -- value tu put to initial character
	"""
	inf = float('Inf')
	tree = t.copy()
	i = 0
	# Get ready
	for node in tree.traverse():
		score_dict = dict(init_dict)
		if node.is_leaf():
			# IDEA 1 : add penalty for editing in leaf
			if node.name in ed_list:
				score_dict[seq_dict[node.name]]=init_value_ifed
			else:
				score_dict[seq_dict[node.name]]=init_dict_value
			node.add_features(sequence=seq_dict[node.name], scores=score_dict)
		else:
			node.add_features(name="n"+str(i), sequence="", scores=score_dict, origin={})
			i+=1
	# Get edited node
	if ed_list!=[]:
		edited_node = tree.get_common_ancestor(ed_list)
	# Traverse 
	for node in tree.traverse("postorder"):
		if not node.is_leaf():
			# Compute sA 
			[left_node, right_node] = node.get_children()
			for key in node.scores:
				# Left
				minL = inf
				for keyL, valueL in left_node.scores.items():
					if key=="*" or keyL=="*":
						scoreL = valueL + max(dist_dict.values())
					else:
						try:
							scoreL = valueL + dist_dict[(key, keyL)]
						except:
							scoreL = valueL + dist_dict[(keyL, key)]
					if scoreL < minL:
						minL = scoreL
						chosen_left = keyL
				# Right
				minR = inf
				for keyR, valueR in right_node.scores.items():
					if key=="*" or keyR=="*":
						scoreR = valueR + max(dist_dict.values())
					else:
						try:
							scoreR = valueR + dist_dict[(key, keyR)]
						except:
							scoreR = valueR + dist_dict[(keyR, key)]
					if scoreR < minR:
						minR = scoreR
						chosen_right = keyR
				# IDEA 2 : add penalty for editing in node 
				if ed_list==[] or node != edited_node:
					sc_total = minR + minL 
				else:
					sc_total = (minR + minL) +  penalty
				node.scores[key] = sc_total
				node.origin[key] = (chosen_left, chosen_right)
	# Backtracking to get good labels
	for node in tree.traverse("preorder"):
		if not node.is_leaf():
			if node.is_root():
				node.sequence = min(node.scores, key=node.scores.get)
			[left_node, right_node] = node.get_children()
			if not left_node.is_leaf():
				left_node.sequence = node.origin[node.sequence][0]
			if not right_node.is_leaf():
				right_node.sequence = node.origin[node.sequence][1]
	return tree.scores, tree

def get_matrix(filename):
	""" Read distance matrix from file
	""" 
	with open(filename) as f:
		matrix_dict = {}
		for line in f:
			line=line.rstrip()
			if line.startswith("\t"):
				aas = line.split("\t")[1:]
			else:
				row = line.split("\t")
				tmp_char = row[0]
				for i in range(len(row[1:])):
					try:
						matrix_dict[(tmp_char, aas[i])]=int(row[i+1])
					except ValueError:
						matrix_dict[(tmp_char, aas[i])]=float(row[i+1])
	return matrix_dict


def sankoff_aln(t, aln, dist_dict, penalty=0, ed_init=0, ed_spe=[], ed_cols=[], subset=[], type="nt"):
	""" Calls sankoff parsimony custom score of each columns (or a subset of columns) of the specified alignment. can be codon or aa MSA 
	t -- tree
	aln -- MSA (can be nt or aa)
	dist_dict -- distance matrix 
	penalty -- penalty value to add to all characters of a node for concerned edited columns
	ed_init -- penalty value to add to leaves characters at the beginning of the Sankoff algorithm
	ed_spe, ed_cols -- lists of edited species and columns
	subset -- subset of columns to compute
	type -- nt if nt aln, aa if aa aln
	"""
	inf = float('Inf')
	# Initialize possibilities with proper characters
	init_dict = {}
	for aa_couple in dist_dict:
		if aa_couple[0] not in init_dict:
			init_dict[aa_couple[0]]=inf
	# If subset of columns is not specified, compute Sankoff scores for every column
	if subset==[]:
		sankoff_scores = []
		diff_ed_ned = []
		for j in range(aln.get_alignment_length()):
			column = aln[:,j:j+1]
			# Sequence dict look like {'a':"A", 'b':"C", 'c':"T", 'd':"G"}
			seq_dict = {}
			ed_list = []
			for record in column:
				seq_dict[record.id] = str(record.seq)
				# Get edited list of species only for concerned columns (else ed_list is empty and the column score will be computed with basic Sankoff score)	
				if type=="aa" and j in [int(x/3) for x in ed_cols]:
					if record.id in ed_spe:
						ed_list.append(record.id)
			# Compute sankoff score on the column
			res_sankoff = sankoff(t, seq_dict, dist_dict, init_dict, penalty=penalty, init_dict_value=0, ed_list=ed_list, init_value_ifed=ed_init)
			sankoff_scores.append(min(res_sankoff[0].values()))
	# If subset of columns is specified, compute Sankoff scores for only these columns and everything else will be inf 
	else:
		sankoff_scores = [inf] * aln.get_alignment_length()
		diff_ed_ned = [inf] * aln.get_alignment_length()
		for s in subset:
			column = aln[:,s:s+1]
			seq_dict = {}
			ed_list=[]
			for record in column:
				seq_dict[record.id] = str(record.seq)
				if type=="aa" and s in [int(x/3) for x in ed_cols]:
					if record.id in ed_spe:
						ed_list.append(record.id)			
			res_sankoff = sankoff(t, seq_dict, dist_dict, init_dict, penalty=penalty, init_dict_value=0, ed_list=ed_list, init_value_ifed=ed_init)
			sankoff_scores[s] = min(res_sankoff[0].values())
	return sankoff_scores

if __name__ == "__main__":
	# Test Sankoff simple
	print("Testing simple Sankoff on a column...")
	sequence_dict={'a':"A", 'b':"C", 'c':"T", 'd':"G"}
	t = Tree("((a, b), (c, d));")
	dist_dict = {('A','C'):9, ('A', 'T'):3, ('A', 'G'):4, ('A', 'A'):0, ('T','G'):2, ('T','C'):4, ('G', 'C'):4, ('T', 'T'):0, ('C', 'C'):0, ('G', 'G'):0}
	inf = float('Inf')
	score_init = {"A":inf, "T":inf, "C":inf, "G":inf}
	res_sankoff = sankoff(t, sequence_dict, dist_dict, score_init)
	print(res_sankoff[0])
	print(res_sankoff[1].get_ascii(attributes=["name", "sequence"], show_internal=True))

	# Test Sankoff align
	print("Testing Sankoff on nt alignment...")
	a = SeqRecord(Seq("ATG", generic_dna), id="a")
	b = SeqRecord(Seq("CTG", generic_dna), id="b")
	c = SeqRecord(Seq("TTC", generic_dna), id="c")
	d = SeqRecord(Seq("GTC", generic_dna), id="d")
	aln_cod = MultipleSeqAlignment([a,b,c,d])
	res = sankoff_aln(t, aln_cod, dist_dict, type="nt")
	print(aln_cod)
	print("Results",res)

	# Test grantham matrix
	g_file = "grantham_matrix.txt"
	dist_mat_aa = get_matrix(g_file)
	aln_aa = translate_aln(aln_cod)
	res = sankoff_aln(t, aln_aa, dist_mat_aa, type="aa")
	print(aln_aa)
	print("Results",res)