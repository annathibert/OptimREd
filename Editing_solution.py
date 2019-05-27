from utils import *
import json
import copy
import random

class Editing_solution:
	def __init__(self, aln_cod, tree, params):
		self.tree = tree.copy()
		self.aln_cod = copy.deepcopy(aln_cod)
		self.init_aln_cod = copy.deepcopy(aln_cod)
		self.params = dict(params)
		# to do: check if number of species in aln / tree are the same 
		self.all_species = self.tree.get_leaf_names()
		self.aln_aa = translate_aln(aln_cod, params["General"]["transl_table"])
		self.__init_editable_columns()
		# Initialize solution !col before species
		self.__init_predicted_columns()
		self.__init_predicted_species()	
		# Create first solution 
		self.aln_cod, self.nb_sed = simulate_editions(self.init_aln_cod, self.predicted_columns, self.predicted_species, self.params["Editing"], return_nbed=True)
		self.aln_aa = translate_aln(self.aln_cod, self.params["General"]["transl_table"])	
		self.__init_scores()
		# Keep init SP scores for heuristics
		# WARNING modify heuristics init
		self.__init_heuristics()
		# Keep trace of moving order
		# useful if heuristics or order specified in parameters
		self.col_order = 0
		self.tree_order = 0
	
	def __init_heuristics(self):
		"""Initialise heuristics 
		"""
		# 
		if "Sum-of-Pairs" in self.params["Score"]:
			# Give editable columns in order of SP
			editable_columns_aa = [int(x/3) for x in self.editable_columns]
			all_columns_aa = [i for i in range(self.aln_aa.get_alignment_length())]
			def norm_range(list_v):
				min_v = np.min(list_v)
				max_v = np.max(list_v)
				return [(list_v[i]-min_v)/(max_v-min_v) for i in range(len(list_v))]
			normSP = norm_range(self.SP_scores)
			order_aa = [x for _,x in sorted(zip(normSP,all_columns_aa))]
			order_aa = [x for x in order_aa if x in editable_columns_aa]
			order_nt = []
			for aa in order_aa:
				cs = range(aa*3, aa*3+3)
				for c in cs:
					if c in self.editable_columns:
						order_nt+=[c]
			self.order_nt = order_nt
		else:
			self.order_nt = self.editable_columns
	
	def __init_editable_columns(self):
		"""Initialize possible editable columns depending on the filter in params
		"""
		self.editable_columns = []
		params=self.params["Initialisation"]["Filter_editable_columns"]
		if params["Method"]=="thresholdCT":
			for i in range(self.aln_cod.get_alignment_length()):
				col = self.aln_cod[:,i]
				total_tc = col.count('C') + col.count('T') 
				pourc_ct = total_tc*100/len(col)
				if pourc_ct > params["Value"]:
					self.editable_columns.append(i)
		elif params["Method"]=="singleCT":
			for i in range(self.aln_cod.get_alignment_length()):
				col = self.aln_cod[:,i]
				if col.count('C')>=1 and col.count('T')>=1:
					self.editable_columns.append(i)
		elif params["Method"]=="list":
			for ec in params["Values"]:
				self.editable_columns.append(ec)
		else:
			print("Error - Param init editable columns not found.")
	
	def __init_predicted_species(self):
		""" Initialize E0 set of initial predicted edited species
		""" 
		params = self.params["Initialisation"]["Species"]	
		if params["Method"] == "all": 
			self.predicted_species = self.all_species
		elif params["Method"] == "list":
			self.predicted_species = params["Values"]
		elif params["Method"] == "random":
			binary_tab = np.random.choice([0, 1], size=(len(self.all_species),), p=[1-params["Probability"], params["Probability"]])
			self.predicted_species = [self.all_species[i] for i, x in enumerate(binary_tab) if x==1]
		elif params["Method"] == "none":
			self.predicted_species=[]
		elif params["Method"] == "random_node":
			rd_node = self.__get_random_node()
			self.predicted_species = list(rd_node.get_leaf_names())
		else:
			print("Error - Param init predicted species not found.")
	
	def __get_random_node(self, out_node=None):
		""" Return a random node from tree
		"""
		node_list = []
		for node in self.tree.traverse():
			if not node.is_leaf() and node!=self.tree.get_tree_root():
				if out_node==None: # No node avoided -- not sure why
					node_list.append(node)
				elif node!=out_node:
					node_list.append(node)
		rd_node = np.random.choice(node_list)
		return rd_node
	
	def __move_neighbouring_node(self, node):
		"""Tree move by choosing a neighbouring node from the current one in the tree
		"""
		neigh_nodes = []
		# Get parent node 		
		tmp_common = []
		for s in node.get_sisters():
			tmp_common.extend(s.get_leaf_names())
		tmp_common.extend(node.get_leaf_names())
		# Root cannot be considered as we want both edited and non-edited species
		if self.tree.get_common_ancestor(tmp_common) != self.tree.get_tree_root():
			neigh_nodes.append(self.tree.get_common_ancestor(tmp_common))
		# Get sister nodes
		neigh_nodes.extend(list(node.get_sisters()))
		# Get children nodes 
		node.get_children()
		neigh_nodes.extend(list(node.get_children()))
		# Return a random node among neighbouring nodes
		rd_n = np.random.choice(neigh_nodes)
		return rd_n
	

	def __init_predicted_columns(self):
		"""Initialise P0 set of predicted edited columns
		"""
		params = self.params["Initialisation"]["Columns"]
		if params["Method"] == "all": 
			self.predicted_columns = self.editable_columns
		elif params["Method"] == "list":
			self.predicted_columns = params["Values"]
		elif params["Method"] == "random":
			binary_tab = np.random.choice([0, 1], size=(len(self.editable_columns),), p=[1-params["Probability"], params["Probability"]])
			self.predicted_columns = [self.editable_columns[i] for i, x in enumerate(binary_tab) if x==1]
		elif params["Method"] == "none":
			self.predicted_columns = []
		# not sure
		elif params["Method"] == "heuristics":
			def norm_range(list_v):
				min_v = np.min(list_v)
				max_v = np.max(list_v)
				return [(list_v[i]-min_v)/(max_v-min_v) for i in range(len(list_v))]
			initSP = get_SPscores(self.aln_aa)
			editable_SP = norm_range([initSP[int(c/3)] for c in self.editable_columns])
			edset=[]
			for i in range(len(editable_SP)):
				rd = random.uniform(0, 1)
				# Si rd < proba norm de index editableSP keep
				if rd < editable_SP[i] and rd < random.gauss(params["Probability"],0.1):
					edset +=[self.editable_columns[i]]
			print(len(edset),"Columns edited")
			self.predicted_columns = edset
		else:
			print("Error - Param init predicted columns not found.")

	def __init_scores(self):
		"""Initialize scores attributes - computationally demanding scores like SP or Sankoff for each columns
		"""
		params = self.params["Score"]
		# Check for SP and other computationally long scores
		if "Sum-of-Pairs" in params:
			params_SP = params["Sum-of-Pairs"]
			global subst_mat
			# subst_mat={}
			# WARNING Dangerous exec, should probablywrite blosum62 in a dict directly
			exec("subst_mat="+params_SP["subst_mat"], globals())
			self.SP_scores = get_SPscores(self.aln_aa, subst_mat=subst_mat, gap_score=params_SP["gap_score"])
		if "AA_Sankoff" in params:
			dist_dict_aa = get_matrix(params["AA_Sankoff"]["g_file"])
			# IDEA 2 (penalty in nodes)
			self.aa_sankoff = sankoff_aln(self.tree, self.aln_aa, dist_dict_aa,ed_spe=self.predicted_species, ed_cols=self.predicted_columns, type="aa", penalty=params["AA_Sankoff"]["init_value"])
			# IDEA 1 (penalty in leaves)
			# self.aa_sankoff = sankoff_aln(self.tree, self.aln_aa, dist_dict_aa,ed_spe=self.predicted_species, ed_cols=self.predicted_columns, type="aa", ed_init=params["AA_Sankoff"]["init_value"])			
		# Compute whole score
		self.__compute_score()
	

	def __compute_score(self):
		"""Compute the whole score with all terms in parameters
		"""
		params = self.params["Score"]
		score = 0 
		if "AA_Sankoff" in params:
			score += sum(self.aa_sankoff)
		if "Sum-of-Pairs" in params:
			score += sum(self.SP_scores) * (params["Sum-of-Pairs"]["lambda"] if "lambda" in params["Sum-of-Pairs"] else 1)
		if "NbEditedColumns" in params:
			score += len(self.predicted_columns) * (params["NbEditedColumns"]["lambda"] if "lambda" in params["NbEditedColumns"] else -1)
		self.score = score
	
	def __update(self, subset=[]):
		"""Update aln from Pi and Ei, translate it and update scores 
		"""
		# Need a subset for columns scores that are hard to compute and that we only want to update
		# Update whole alns from initial aln
		self.aln_cod, self.nb_sed = simulate_editions(self.init_aln_cod, self.predicted_columns, self.predicted_species, self.params["Editing"], return_nbed=True)
		self.aln_aa = translate_aln(self.aln_cod, self.params["General"]["transl_table"])
		# Update score (SP or whatever) 
		self.__update_scores(subset_cols=subset)

	def __update_scores(self,subset_cols=[]):
		"""Update score with possible subset of columns
		"""
		params = self.params["Score"]
		if "AA_Sankoff" in params:
			dist_dict_aa = get_matrix(params["AA_Sankoff"]["g_file"])
			if subset_cols == []:
				# IDEA 2 with node
				self.aa_sankoff = sankoff_aln(self.tree, self.aln_aa, dist_dict_aa,ed_spe=self.predicted_species, ed_cols=self.predicted_columns, type="aa", penalty=params["AA_Sankoff"]["init_value"])
				# IDEA 1 with leaves
				#self.aa_sankoff = sankoff_aln(self.tree, self.aln_aa, dist_dict_aa,ed_spe=self.predicted_species, ed_cols=self.predicted_columns, type="aa", ed_init=params["AA_Sankoff"]["init_value"])
			else:
				# IDEA 2 with node
				tmp_sankoff = sankoff_aln(self.tree, self.aln_aa, dist_dict_aa, subset=[int(x/3) for x in subset_cols], ed_spe=self.predicted_species, ed_cols=self.predicted_columns, type="aa", penalty=params["AA_Sankoff"]["init_value"])
				# IDEA 1 with leaves
				# tmp_sankoff = sankoff_aln(self.tree, self.aln_aa, dist_dict_aa, subset=[int(x/3) for x in subset_cols], ed_spe=self.predicted_species, ed_cols=self.predicted_columns, type="aa", ed_init=params["AA_Sankoff"]["init_value"])
				for ts in range(len(tmp_sankoff)):
					if tmp_sankoff[ts] != float('Inf'):
						self.aa_sankoff[ts] = tmp_sankoff[ts]
		if "Sum-of-Pairs" in params:
			# params = self.params["Score"]["Sum-of-Pairs"]
			# WARNING dangerous exec
			exec("subst_mat="+params["Sum-of-Pairs"]["subst_mat"])
			# Subset_cols is nt position, convert it to aa position
			subset_cols_aa = [int(sc/3) for sc in subset_cols]
			self.SP_scores = get_SPscores(self.aln_aa, subst_mat=subst_mat, gap_score=params["Sum-of-Pairs"]["gap_score"], subset=subset_cols_aa, old_SP=self.SP_scores)
		# Compute whole score
		self.__compute_score()
	
	def next_state(self, movetype):
		"""Call tree move or columns move from movetype and pass subset of columns to recompute 
		"""
		if movetype=="Columns":
			cols_bef = list(self.predicted_columns)
			# Change columns
			self.__move_columns()
			# Update only changed columns 
			cols_aft = list(self.predicted_columns)
			subset = list((set(cols_bef)|set(cols_aft))-(set(cols_bef)&set(cols_aft)))
			self.__update(subset=subset)
		elif movetype=="Species":
			cols_bef = list(self.predicted_columns)
			# Change species 
			self.__move_species()
			# Update only columns considered as edited before and after new move 
			cols_aft = list(self.predicted_columns)
			subset = list((set(cols_bef)|set(cols_aft)))
			self.__update(subset=subset)
		else:
			print("Bad movetype")
	
	def __move_columns(self):
		"""Move set of columns in Pi
		"""
		params = self.params["Move"]["Columns"]
		if params["Method"]=="random":
			# Get indexes of edited positions
			ind_edited= [self.editable_columns.index(self.predicted_columns[i]) for i in range(0,len(self.predicted_columns))]
			# Get corresponding binary tab
			binary_tab = [1 if i in ind_edited else 0 for i in range(self.aln_cod.get_alignment_length())]
			# Switch random columns (0->1 or 1->0)
			rds = random.sample(range(len(self.editable_columns)), params["number"])
			for rd in rds:
				binary_tab[rd]=1 if binary_tab[rd]==0 else 0
			# Put in predicted columns attribute 
			self.predicted_columns=[self.editable_columns[i] for i, x in enumerate(binary_tab) if x==1]
		elif params["Method"]=="heuristics":
			# Switch
			for i in range(params["number"]):
				self.col_order=self.col_order%len(self.order_nt)
				col_to_switch = self.order_nt[self.col_order]
				if col_to_switch in self.predicted_columns:
					self.predicted_columns = list(set(self.predicted_columns)-set([col_to_switch]))
				else:
					self.predicted_columns+=[col_to_switch]				
		else:
			print("Error - move param not found.")

	def __move_species(self):
		"""Move set of species in Ei
		"""
		params = self.params["Move"]["Species"]
		if params["Method"]=="random":
			binary_tab=[0]*len(self.aln_cod)
			ind_edited =[self.all_species.index(self.predicted_species[i]) for i in range(0,len(self.predicted_species))]
			for ind in ind_edited:
				binary_tab[ind]=1
			rds = random.sample(range(len(self.all_species)), params["number"])
			for rd in rds:
				binary_tab[rd]=1 if binary_tab[rd]==0 else 0
			self.predicted_species=[self.all_species[i] for i, x in enumerate(binary_tab) if x==1]
		elif params["Method"]=="heuristics":
			for i in range(params["number"]):
				self.tree_order=self.tree_order%len(self.order_species)
				sp_to_switch = self.order_species[self.tree_order]
				if sp_to_switch in self.predicted_species:
					self.predicted_species = list(set(self.predicted_species)-set([sp_to_switch]))
				else:
					self.predicted_species+=[sp_to_switch]
		elif params["Method"]=="random_node":
			node = self.tree.get_common_ancestor(self.predicted_species)
			self.predicted_species = self.__move_neighbouring_node(node).get_leaf_names()	
		else:
			print("Error - move param not found.")
	
	def print_infos(self):
		st = "Edited Columns: " + str(len(self.predicted_columns)) + "\n"
		#st += str(self.predicted_columns) + "\n"
		st += "Edited Species: " + str(len(self.predicted_species)) + "\n"
		#st += str(self.predicted_species) + "\n"
		st += "Score: " + str(self.score) + "\n"
		print(st)

		

if __name__=="__main__":
	print("Reading files...")
	aln_cod_file = "./Simul_data/1/dna_1.fasta"
	tree_file = "./Simul_data/tree.nw"
	aln_cod = AlignIO.read(aln_cod_file, "fasta")
	tree = Tree(tree_file)
	with open("./Params/params_solution.json") as f:
		params = json.load(f)
	print("Initialize solution...")
	init_sol = Editing_solution(aln_cod, tree, params)
	print("Done.")
	init_sol.print_infos()
	print("Computing next state...")
	init_sol.next_state("Species")
	init_sol.print_infos()
	print("Computing next state...")
	init_sol.next_state("Columns")
	init_sol.print_infos()
	pdb.set_trace() 