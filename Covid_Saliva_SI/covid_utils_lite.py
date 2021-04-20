"""
This script takes mrn+date or acct_no and outputs all info in L for that sample, plus the time since the presentation (initial vs. repeat).
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from collections import defaultdict
from datetime import datetime, timedelta
from getpass import getuser
from getpass import getuser
from matplotlib.patches import Rectangle
from numpy.random import randint
from os.path import exists
from os import getcwd
from pickle import load, dump
from scipy.stats import theilslopes as ts, linregress, wilcoxon, gmean, mannwhitneyu as mwu
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.cluster import contingency_matrix
from textwrap import wrap


version = "0.63"
# print("version:", version)


# Helper functions ----------------------------------------------

# for ct2vl
def load_traces(
	Cts_passed = False,
	intensity_threshold_for_positive = 50,
	cycle_threshold_for_intensity_threshold = 40,
	quantitation_threshold = 4.,
	delta_Ct_max = 1.41, # from LOD Matters paper; use as default
	debug = False,
	quiet = True,
	infile = False,
	):
	"""
	User will need:

	1. A csv where each row is a trace and each column is the intensity at a cycle (could generalize to have a measurements-per-cycle value). We provide one in combined_arrays.csv. This is infile
	2. The quantitation threshold of their reader (which we estimated empirically above; get this from Abbott)
	3. The Ct value at some known amount of virus (e.g. Ct at LOD)
	4. The delta between Ct value and cycle at max efficiency (we can calculate it if given Ct values; otherwise we default to 1.41±0.93)
	"""
	#
	# Read in file
	traces = np.genfromtxt(infile, delimiter=',')
	#
	# Get shape of matrix
	no_traces, no_cycles = traces.shape
	#
	# In production, read in Ct values, or better, require the first column to be Ct values for each run, and split these off (or even use them to decide which are positive)
	# Absent Ct values, we use an intensity threshold and cycle at which that threshold must be reached here.
	if not Cts_passed:
		positive_traces = traces[traces[:,cycle_threshold_for_intensity_threshold] > intensity_threshold_for_positive]
	if debug:
		print("No. presumed positives: %i" % len(positive_traces) )
		# inspect --- make sure to configure matplotlib backend correctly or else you get segmentation fault
		for i in positive_traces:
			_ = plt.plot(i, linewidth=0.5)
		_ = plt.xlabel("Cycle no.")
		_ = plt.ylabel("Intensity")
		_ = plt.xticks(range(0, 45, 5))
		_ = plt.show()
	#
	# Threshold to just get positive traces
	thresholded_positive_traces = np.copy(positive_traces)
	thresholded_positive_traces[positive_traces < quantitation_threshold] = quantitation_threshold
	rhos = thresholded_positive_traces[:,1:] / thresholded_positive_traces[:,:-1]
	efficiencies = rhos - 1.
	#
	# Sanity check: assert that max_efficiencies are between 0.95 and 1.6 (reasonable for an RT-PCR assay based on literature)
	max_efficiencies = np.max(efficiencies, axis=1)
	# print(no,min(max_efficiencies))
	try: assert(np.max(max_efficiencies) < 1.6)
	except AssertionError:
		warning = "warning: the maximum amplification efficiency is > 160%% (%i%%). The quantitation_threshold you passed to ct2vl may be too low, or there may be some problem with your input data (perhaps scaled up? intensity_threshold too low?). (The quantitation threshold is the intensity value above which the intensity reliably reflects the amount of amplified product. Below this threshold, noise is too high for intensity to be reliable.)"
		warning = "\n".join(wrap(warning, 100))
		if not quiet:
			print(warning % (100*np.max(max_efficiencies)))
	try: assert(np.min(max_efficiencies) > 0.95)
	except AssertionError:
		warning = "warning: the maximum amplification efficiency is < 95%% (%i%%). The quantitation_threshold you passed to ct2vl may be too high, or there may be some problem with your input data (perhaps scaled down?). (The quantitation threshold is the intensity value above which the intensity reliably reflects the amount of amplified product. Below this threshold, noise is too high for intensity to be reliable.)" 
		warning = "\n".join(wrap(warning, 100))
		if not quiet:
			print(warning % (100*np.min(max_efficiencies)))
	#
	# Get slope and intercept for a Thiel-Sen (linear) fit of the max efficiency against cycle no
	max_efficiencies = np.max(efficiencies, axis=1)
	cycles_at_max_efficiency = np.argmax(efficiencies, axis=1)
	# here is where, if we have the Ct values, we can calculate delta_Ct_max, which is the number of cycles before maximum efficiency is reached, that the Ct is called
	Ct_at_max_efficiency = cycles_at_max_efficiency - delta_Ct_max
	m_ts, b_ts, m_ts_lo, m_ts_hi = ts(max_efficiencies, Ct_at_max_efficiency)
	#
	# Eq. 6 from the LOD Matters paper's Supplementary Information; taken directly from ct2vl
	m = m_ts
	b = m*delta_Ct_max + (b_ts + 1) # = rho_0 = efficiency + 1
	#
	return m, b


def make_platform_parameter_hash(infile=False):
	vL_alinity = 100.
	CtL_alinity = 37.96
	#
	# m2000 (from github ct2vl library)
	m_m2000 = -0.0283
	b_m2000 = m_m2000*1.41 + (1.379 + 1) # = rho_0 = efficiency + 1
	vL_m2000 = 100 # viral load at LoD, in copies/mL
	CtL_m2000 = 26.06
	#
	m_alinity, b_alinity = load_traces(infile=infile)
	platform_parameter_hash = {
		"alinity": {
			"m": m_alinity,
			"b": b_alinity,
			"vL": vL_alinity,
			"CtL": CtL_alinity,
			"neg": 47.,
			},
		"m2000": {
			"m": m_m2000,
			"b": b_m2000,		
			"vL": vL_m2000,
			"CtL": CtL_m2000,
			"neg": 37.
			}
		}
	return platform_parameter_hash
try:
	platform_parameter_hash = make_platform_parameter_hash(infile="combined_arrays.csv")
except:
	print("import failure:")
	print("please run the following command, with the infile option filled out in the first line (infile is where the traces live):")
	print("platform_parameter_hash = make_platform_parameter_hash(infile=  )")
	platform_parameter_hash = {} # dummy value, so ct2vl will load (running the above will populate it)


platform_neg_val_hash = {
	"alinity": platform_parameter_hash["alinity"]["CtL"], 
	"m2000": platform_parameter_hash["m2000"]["CtL"]
	}


def str2datetime(s):
	return datetime.strptime(s, "%y-%m-%d %H:%M") # e.g. 20-12-13 22:13
s2d = str2datetime # convenience


def label2filebase(label):
	# helper function for making the outfile for when we want to save figures
	try: f1, f2 = label.split(":")
	except: 
		f1 = label
		f2 = ""
	f1 = f1.strip()
	f2 = f2.strip()
	f1 = f1.replace(" ", "-")
	f2 = f2.replace("\n", " ")
	f2 = f2.replace("  ", " ")
	f2 = f2.replace(" ", "_")
	f2 = f2.replace(",", "")
	f2 = f2.replace(";", "")
	filebase = "/%s_%s" % (f1, f2)
	return filebase



# User functions -------------------------------------------


def plot_contingency_table(contingency_table, color="#cccccc", reflect_across_y_equals_x=True, reflect_across_y_equals_negative_x=True, file="", save=False):
	"""
	Note that contingency_matrix from sklearn.metrics.cluster returns values with (0,0) in the upper left 
	and (1, 1) in the lower right, but our desired table puts the +ses first, meaning (1,1) in the upper 
	left. Therefore we set the parameter "flip=True" by default.
	"""
	# check that the shape is 2 x 2 (if not, something's missing, the contingency table will be a single column, and we can't plot it)
	if contingency_table.shape != (2,2):
		print("error: plot_contingency_table: shape not (2,2)")
		return
	# flip contingency table
	if reflect_across_y_equals_x:
		contingency_table = np.flip(contingency_table)
	# flip to make gold standard go across top vs. down left
	if reflect_across_y_equals_negative_x:
		contingency_table = contingency_table.T
	#
	# initialize figure
	plt.figure(figsize=(1.75, 1.75))
	# plot horizontal lines
	plt.plot((1, 3), (3, 3), linewidth=1, c="black")
	plt.plot((1, 3), (2, 2), linewidth=1, c="black")
	plt.plot((1, 4), (1, 1), linewidth=1, c="black")
	# plot vertical lines
	plt.plot((1, 1), (1, 3), linewidth=1, c="black")
	plt.plot((2, 2), (1, 3), linewidth=1, c="black")
	plt.plot((3, 3), (0, 3), linewidth=1, c="black")
	# get axes
	ax = plt.gca()    
	# add 2x2 table numbers
	plt.text(1.5, 2.5, str(contingency_table[0][0]), transform=ax.transData, ha="center", va="center")
	plt.text(2.5, 2.5, str(contingency_table[0][1]), transform=ax.transData, ha="center", va="center")
	plt.text(1.5, 1.5, str(contingency_table[1][0]), transform=ax.transData, ha="center", va="center")
	plt.text(2.5, 1.5, str(contingency_table[1][1]), transform=ax.transData, ha="center", va="center")
	# add totals
	plt.text(3.5, 2.5, str(np.sum(contingency_table, axis=1)[0]), transform=ax.transData, ha="center", va="center")
	plt.text(3.5, 1.5, str(np.sum(contingency_table, axis=1)[1]), transform=ax.transData, ha="center", va="center")
	plt.text(1.5, 0.5, str(np.sum(contingency_table, axis=0)[0]), transform=ax.transData, ha="center", va="center")
	plt.text(2.5, 0.5, str(np.sum(contingency_table, axis=0)[1]), transform=ax.transData, ha="center", va="center")
	plt.text(3.5, 0.5, str(np.sum(contingency_table)), transform=ax.transData, ha="center", va="center")
	#
	# add plusses and minuses
	plt.text(1.5, 3.5, "+", transform=ax.transData, ha="center", va="center", fontsize=16)
	plt.text(2.5, 3.5, "–", transform=ax.transData, ha="center", va="center", fontsize=16)
	plt.text(0.5, 2.5, "+", transform=ax.transData, ha="center", va="center", fontsize=16)
	plt.text(0.5, 1.5, "–", transform=ax.transData, ha="center", va="center", fontsize=16)
	#
	# shade off-diagonals
	ax.add_patch(Rectangle( (1, 1), 1, 1, color=color, zorder=-100 ))
	ax.add_patch(Rectangle( (2, 2), 1, 1, color=color, zorder=-100 ))
	#
	plt.xlim((0,4))
	plt.ylim((0,4))
	# remove border
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	# remove ticks and labels
	plt.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
	#
	if save:
		plt.savefig(file, transparent=True)
	return

# Test it out
# contingency_table = np.array(((9, 11), (1, 26)))
# plot_contingency_table(contingency_table)


# Our own version of contingency_matrix(), since scipy's version fails when a category is entirely missing from the data
def make_contingency_table(y1, y2):
	# y1 = gold standard
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for i, j in zip(y1, y2):
		if i == j:
			if i == 1: TP += 1
			else: TN += 1
		elif i == 1: FN += 1
		else: FP += 1
	contingency_table = np.array(  ((TN, FP), (FN, TP))  ) # note we adopt the convention of the falses first
	return contingency_table


def get_wilcoxon(df, col1="Ctrl_Ct", col2="Test_Ct", platform="", neg_val=100, vl=True):
	# note: be careful not to combine data from different platforms, if vl=False
	try: 
		if vl:
			if not platform:
				raise TypeError("you must specify platform if vl=True (default)")
			y1, y2 = list(zip(*[(ct2vl(ii, p), ct2vl(jj, p)) for _, (ii, jj, p) in df[[col1, col2, platform]].iterrows() if ct2vl(ii, p) >= neg_val and ct2vl(jj, p) >= neg_val]))
			mean1 = gmean(y1)
			mean2 = gmean(y2)
		else:
			y1, y2 = zip(*[(ii, jj) for ii, jj in zip( list(df[col1]), list(df[col2]) ) if ii < neg_val and jj < neg_val])
			mean1 = np.mean(y1)
			mean2 = np.mean(y2)
	except ValueError: # no positive values (all values 37.)
		return(1.)
	_, p_ct = wilcoxon(y1, y2)
	return p_ct, mean1, mean2


def get_kappa(df, col1="Ctrl_Ct", col2="Test_Ct", neg_val=37.):
	y1 = [1 if y < neg_val else 0 for y in list(df[col1])] # y1 = control (NP)
	y2 = [1 if y < neg_val else 0 for y in list(df[col2])] # y2 = test (nasal)
	kappa = cohen_kappa_score(y1, y2)
	return kappa


def get_kappa_p_value(df_master, n, kappa, n_rpts=10000):
	sample_kappas = [get_kappa(df_master.sample(n=n), "Ctrl_Ct", "Test_Ct") for ii in range(n_rpts)]
	sample_kappas.sort()
	uncorrected_p = bisect(sample_kappas, kappa)/n_rpts
	p_kappa = min(uncorrected_p, 1-uncorrected_p) # if uncorrected_p is very close to 1, that just means it's unusually high. We want our p-value to capture this unusualness, so it's the remaining density to whichever is the smaller end of the distribution
	return p_kappa


# function for making figure
figsize = (4,4.3)
def make_figure(df, label, save=False, vl=False, size=20, alpha=0.7, fontsize=20, col1="Ctrl_Ct", col2="Test_Ct", platform="test_platform", plot_folder="", xlabel="control", ylabel="test", axis_max=False, lod=100):
	# filter any NaNs and standardize col types
	df = df[[col1, col2, platform]]
	df = df.dropna()
	df = df.astype({col1: float, col2: float})
	#
	# plot figure
	plt.figure(figsize=figsize)
	if vl:
		x = df[[col1, platform]]
		y = df[[col2, platform]]
		x = [ct2vl(i, platform=j) for _, (i, j) in x.iterrows()] # Ctrl_Ct
		y = [ct2vl(i, platform=j) for _, (i, j) in y.iterrows()] # Test_Ct
	else:
		x = df[col1]
		y = df[col2]
	plt.scatter(x, y, alpha=0.5, label=label, s=size, c="black") # note, control Ct on x-axis (~independent variable)
	#
	# 1:1 diagonal line
	plt.plot((1e-1, 1e14), (1e-1, 1e14), zorder=-1000, linewidth=0.5, c="black") # xlim and ylim will control this
	#
	# get kappa
	kappa = get_kappa(df, col1=col1, col2=col2)
	#    
	# plot cosmetics 
	label += "; κ=%.2f" % kappa
	plt.title(label)
	if vl:
		xlabel = "Viral load, " + xlabel
		ylabel = "Viral load, " + ylabel
		if not axis_max:
			axis_max = 10**np.ceil(np.log10(max(x + y)))
		plt.fill_between(np.linspace(0, lod, 2), np.linspace(0, lod, 2)*1e20, color="#cfcfcf", zorder=-1000, linewidth=0.) # fill to the left of lod
		plt.fill_between(np.linspace(0, 1e20, 2), np.linspace(lod, lod, 2), color="#cfcfcf", zorder=-1000, linewidth=0.)# plt.fill_between(x, y1, y2=0) # fill under lod
		# plt.axvline(100, linestyle=":", linewidth=1., color="black", zorder=1000)
		# plt.axhline(100, linestyle=":", linewidth=1., color="black", zorder=1000)
		plt.xlabel(xlabel, fontsize=fontsize)
		plt.ylabel(ylabel, fontsize=fontsize)
		plt.xlim((0.1, axis_max))
		plt.ylim((0.1, axis_max))
		plt.xscale('log')
		plt.yscale('log')
		plt.xticks([10**i for i in range(0, 10, 3)])
		plt.yticks([10**i for i in range(0, 10, 3)])
	else:
		xlabel = "Ct value, " + xlabel
		ylabel = "Ct value, " + ylabel
		plt.xlabel(xlabel, fontsize=fontsize)
		plt.ylabel(ylabel, fontsize=fontsize)
		if not axis_max:
			axis_max = max(list(platform_neg_val_hash.values()))
			axis_max = 10*np.ceil(axis_max/10)# round up
		plt.xlim((0, axis_max))
		plt.ylim((0, axis_max))
		plt.xticks(range(0, 41, 10))
		plt.yticks(range(0, 41, 10))
	plt.grid(linewidth=0.5, alpha=0.5, zorder=-1000)
	ax = plt.gca()
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	plt.tight_layout()
	if save:
		if plot_folder == "":
			plot_folder = getcwd()
		filebase = plot_folder + label2filebase(label)
		if vl:
			filebase += "_vl"
		plt.savefig(filebase + ".pdf", transparent=True)
	else: 
		plt.show()
		filebase = ""
	#
	return


def make_2x2table(df, label, col1="Ctrl_Ct", col2="Test_Ct", platform="test_platform", plot_folder="", platform_neg_val_hash=platform_neg_val_hash, save=False, return_data=True, vl=False, categorical=False):
	# filter any NaNs and standardize col types
	df = df[[col1, col2, platform]]
	df = df.dropna()
	df = df.astype({col1: float, col2: float})
	#
	# get values and cutoffs
	col1_vals = list(df[col1])
	col2_vals = list(df[col2])
	platforms = list(df[platform])
	cutoffs = [platform_neg_val_hash[i] for i in platforms]
	#
	# make 2x2 table
	if categorical:
		y1 = col1_vals
		y2 = col2_vals
	else:
		if vl:
			y1 = [1 if ct2vl(y, platform=p) >= ct2vl(cutoff, platform=p) else 0 for y, cutoff, p in zip(col1_vals, cutoffs, platforms)]
			y2 = [1 if ct2vl(y, platform=p) >= ct2vl(cutoff, platform=p) else 0 for y, cutoff, p in zip(col2_vals, cutoffs, platforms)]
		else:
			y1 = [1 if y < cutoff else 0 for y, cutoff in zip(col1_vals, cutoffs)]
			y2 = [1 if y < cutoff else 0 for y, cutoff in zip(col2_vals, cutoffs)]
	#
	contingency_table = contingency_matrix(y1, y2)
	if contingency_table.size != (2, 2): # this means one of the categories is missing; we have to construct it ourselves
		contingency_table = make_contingency_table(y1, y2) # my version of contingency_matrix() for when this happens
	if save: 
		if plot_folder == "":
			plot_folder = getcwd()
		filebase = plot_folder + label2filebase(label)
	else: filebase = ""
	plot_contingency_table(contingency_table, file=filebase + "_2x2.pdf", save=save)
	#
	# get kappa
	kappa = get_kappa(df, col1=col1, col2=col2)
	#
	# return data
	if return_data:
		try: (TN, FP), (FN, TP) = contingency_table
		except: 
			make_contingency_table(y1, y2)
		n = len(df)
		return (n, TP, FN, FP, TN, kappa)
	else:
		return


def ct2vl(Ct, platform, H=platform_parameter_hash):
	"""
	Converts Ct to vl
	"""
	# get params
	m =   H[platform]["m"]
	b =   H[platform]["b"]
	vL =  H[platform]["vL"]
	CtL = H[platform]["CtL"]
	# perform calculation
	log_v = np.log(vL) + (CtL-1+b/m)*np.log(m*(CtL-1)+b) - (Ct-1+b/m)*np.log(m*(Ct-1)+b)  + Ct-CtL
	#
	return np.exp( log_v )

# unit test
try:
	assert(np.isclose(ct2vl(37.96, "alinity"), 100))
	assert(np.isclose(ct2vl(26.06, "m2000"), 100))
except KeyError: # most likely means you haven't loaded platform_parameter_hash
	print("""Please run the following (you should see no errors):
assert(np.isclose(ct2vl(37.96, "alinity"), 100))
assert(np.isclose(ct2vl(26.06, "m2000"), 100))
""")
