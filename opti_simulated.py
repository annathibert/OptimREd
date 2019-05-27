from Editing_solution import *
import progressbar
from time import sleep
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Optimize
def get_data(solution, truth_species, truth_columns):
	# ["i", "sens_species", "spec_species", "sens_columns", "spec_columns", "score"]
	[sens_species, spec_species] = get_accuracy(truth_species, solution.predicted_species, solution.all_species)
	[sens_columns, spec_columns] = get_accuracy(truth_columns, solution.predicted_columns, solution.editable_columns)
	[TP, TN, FP, FN] = get_confusion_matrix(truth_columns, solution.predicted_columns, solution.editable_columns)
	acc_columns = (TP+TN)/(TP+TN+FP+FN)
	return [sens_species, spec_species, sens_columns, spec_columns, acc_columns, solution.score]

def generate_results(df, filename=None, true_score=None):
	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	sns.set(style="whitegrid")
	df_columns = df[['sens_columns', 'spec_columns']]
	df_species = df[['sens_species', 'spec_species']]
	df_score = df[['score','optimal','no_ed']]
	sns.lineplot(data=df_score, linewidth=2.5, ax=ax1)
	ax2.set(ylim=(0,1))
	sns.lineplot(data=df_columns, linewidth=2.5, ax=ax2)
	sns.lineplot(data=df_species, linewidth=2.5, ax=ax3)
	# Show plot if no filename, save otherwise
	if filename==None:
		plt.show()
	else:
		plt.savefig(filename)
		plt.gcf().clear()
	plt.close(fig)



def optimed(init_sol, params, truth_columns, truth_species, filename_image=None, stop_criterion=0):
	"""Optimization by simulated annealing
	init_sol -- initial solution x0
	params -- SA parameters
	truth_columns -- true positives edited columns
	truth_species -- true positives edited species
	filename_image -- None if display only, otherwise will save into png result img
	stop_criterion -- 0 if itermax only, otherwise stop after a number of iterations if the score has not changed
	""" 
	# Get true solution and its score 
	true_params = dict(init_sol.params)
	true_params["Initialisation"]["Columns"]["Method"]="list"
	true_params["Initialisation"]["Columns"]["Values"]=truth_columns
	true_params["Initialisation"]["Species"]["Method"]="list"
	true_params["Initialisation"]["Species"]["Values"]=truth_species
	true_solution = Editing_solution(copy.deepcopy(init_sol.init_aln_cod), init_sol.tree, true_params)
	# Get observed solution (with no edits) and its score 
	copy_params = dict(init_sol.params)
	copy_params["Initialisation"]["Columns"]={"Method":"list", "Values":[]}
	copy_params["Initialisation"]["Species"]={"Method":"list", "Values":[]}
	no_ed_sol = Editing_solution(copy.deepcopy(init_sol.init_aln_cod), init_sol.tree, copy_params)

	# hill climbing True if no cooling schedule (only accepting good moves)
	hill_climbing = (params["hill_climbing"]==1)
	
	# Determine sd with 200 simulated moves to get T0 initial temp and tau cooling factor
	if hill_climbing==False:
		print("Determining sd...")
		current_solution = copy.deepcopy(init_sol)
		keep_scores = []
		for i in range(200):
			# print(i, "/200")
			next_solution = copy.deepcopy(current_solution)
			if i%2==0:
				next_solution.col_order+=1
				current_solution.col_order+=1
				movetype="Columns"
			else:
				next_solution.tree_order+=1
				current_solution.tree_order+=1
				movetype="Species"
			movetype="Columns"
			next_solution.next_state(movetype)
			keep_scores.append(next_solution.score)
			current_solution = copy.deepcopy(next_solution)
		sd_scores = np.std(keep_scores)
		# print(sd_scores)
		T0 = -sd_scores/np.log2(0.9)
		tau = np.exp(np.log2(10e-5/T0)/params["itermax"])
		Ti = T0
		# print("Done. T0",T0, "tau", tau, "sd", sd_scores)

	# Initialize states, parameters, moves
	current_solution = copy.deepcopy(init_sol)
	#cooling_factor = np.exp(np.log2(params["tf"]/params["tinit"])/params["itermax"])
	#temp = params["tinit"]
	moves = np.random.choice(["Columns", "Species"], params["itermax"], p=[1-params["tmp"], params["tmp"]])
	
	# Start progressbar
	bar = progressbar.ProgressBar(maxval=params["itermax"])
	bar.start()
	
	results = []
	i=0
	stop=False
	# Record scores and number of same score for stopping criterion
	record_score = init_sol.score
	cpt_score = 0

	# Start iterating
	while(i<params["itermax"] and stop==False):
		# Update progress bar
		bar.update(i+1)
		movetype = moves[i]

		# Move to next solution
		next_solution = copy.deepcopy(current_solution)
		# Order (used only for heuristics..)
		if movetype == "Columns":
			next_solution.col_order+=1
			current_solution.col_order+=1
		else:
			next_solution.tree_order+=1
			current_solution.tree_order+=1
		next_solution.next_state(movetype)
		
		# Compute delta score 
		deltaScore = next_solution.score - current_solution.score

		# Decide to accept or reject solution 
		# <0 for minimization problem
		# Accept good move
		if deltaScore < 0:
			current_solution = copy.deepcopy(next_solution)
		# If simulated annealing, accept bad move under some conditions
		elif hill_climbing==False:
			rd = np.random.random_sample()
			proba = np.exp(-(np.absolute(deltaScore)/Ti))
			if rd < proba:
				current_solution = copy.deepcopy(next_solution)

		# Cool down system
		if hill_climbing==False:
			Ti = Ti*tau

		# Add data to results
		all_data = get_data(current_solution, truth_species, truth_columns)		
		all_data.extend([true_solution.score, no_ed_sol.score, i+1])
		results+=[all_data]

		# Update stopping criterion
		i+=1
		if stop_criterion!=0: 
			if current_solution.score == record_score:
				cpt_score+=1
				if cpt_score > stop_criterion:
					stop=True
			else:
				record_score = current_solution.score
				cpt_score = 0

	# End of optimization
	bar.finish()
	# Write results in df 
	df_index = [it+1 for it in range(i)]
	df_results = pd.DataFrame(results, index=df_index, columns=["sens_species", "spec_species", "sens_columns", "spec_columns","acc_columns", "score", "optimal", "no_ed", "index"])	

	current_data = get_data(current_solution, truth_species, truth_columns)
	print(current_data)
	generate_results(df_results, filename=filename_image)
	return [current_solution, current_data]
	


if __name__=="__main__":
	# Read files 
	print("Reading files...")
	aln_cod_file = "./Simul_data/1/dna_1.fasta"
	tree_file = "./Simul_data/tree.nw"
	init_aln_cod = AlignIO.read(aln_cod_file, "fasta")
	tree = Tree(tree_file)
	with open("./Params/params_solution.json") as f:
		params = json.load(f)
	with open("./Params/params_simul_sa.json") as f:
		params_SA = json.load(f)

	# Read truth edited columns and species 
	init_edited_columns=read_to_list("./Simul_data/1/dna_1.plist")
	edited_species=read_to_list("./Simul_data/1/dna_1.slist")

	cheat_spe = True
	cheat_none = True

	# Split dataset in sub_aln 
	nb_dataset = 5
	interv = 3*int(int(init_aln_cod.get_alignment_length()/3) / nb_dataset)
	for ds in range(nb_dataset):
		# Get sub_aln and edited columns new indexes
		print("---------------- NEW DATASET %s / %s" %(ds, nb_dataset) )
		start=ds*interv
		end=(ds+1)*interv
		aln_cod = init_aln_cod[:,start:end]
		edited_columns = [x-ds*interv for x in init_edited_columns if x in range(start, end)]
		
		# Test different matrices
		for mat_file in ["mat_grantham.txt"]: # ["mat_epstein.txt","mat_grantham.txt", "mat_sneath.txt"]:
			params["Score"]["AA_Sankoff"]["g_file"]=mat_file
			# pdb.set_trace()
			mean_dist = np.mean(list(get_matrix(mat_file).values()))
			# Try different penalty values (in percentage of mean distance from distance matrix)
			for init_value in [x*mean_dist for x in [0,0.1,0.5,1]]:
				params["Score"]["AA_Sankoff"]["init_value"]=init_value
				now = datetime.datetime.now()

				# CHEAT SPECIES: only column moves, random columns init
				if cheat_spe:
					params["Initialisation"]["Columns"] = {"Method":"random", "Probability":0.2}
					params["Initialisation"]["Species"] = {"Method":"list", "Values":edited_species}
					params_SA["tmp"]=0
					annot = "%s_%s_dataset%s_penalty%.2f_cheatSpe" % (now.isoformat(),mat_file,ds,init_value)
					print("\nInitialize solution...%s"%annot)
					init_sol = Editing_solution(copy.deepcopy(aln_cod), tree, params)
					init_sol.print_infos()				
					solution = optimed(init_sol, params_SA, truth_columns=edited_columns, truth_species=edited_species, filename_image="Results/Res_%s.png"%annot)
					solution[0].print_infos()

				# tree and column moves, random init
				if cheat_none:
					params["Initialisation"]["Species"] = {"Method":"random_node"}
					params["Initialisation"]["Columns"] = {"Method":"random", "Probability":0.2}
					params_SA["tmp"]=0.2
					annot = "%s_%s_dataset%s_penalty%.2f_cheatNone" % (now.isoformat(),mat_file,ds,init_value)
					print("\nInitialize solution...%s"%annot)
					init_sol = Editing_solution(copy.deepcopy(aln_cod), tree, params)
					init_sol.print_infos()
					solution = optimed(init_sol, params_SA, truth_columns=edited_columns, truth_species=edited_species, filename_image="Results/Res_%s.png"%annot)
					solution[0].print_infos()
