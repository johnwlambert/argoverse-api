
import collections
from collections import defaultdict
import copy
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pdb
import seaborn as sns
import scipy
from typing import Any, List, Mapping

from mseg.utils.csv_utils import read_csv



def make_forecasting_plots():
	""" """
	sns.set_style({'font.family': 'monospace'}) #'Times New Roman'})
	plt.style.use('ggplot')

	result_dict = get_forecasting_results()

	# labels = [
	# 	'minADE (K=1)',
	# 	'minADE (K=6)'
	# ]

	# labels = [
	# 	'DAC (K=1)',
	# 	'DAC (K=6)'
	# ]

	# labels = [
	# 	'MR (K=1)',
	# 	'MR (K=6)'
	# ]

	# labels = [
	# 	'minFDE (K=1)',
	# 	'minFDE (K=6)'
	# ]

	# labels = [
	# 	'p-minADE (K=6)',
	# 	'p-minFDE (K=6)'
	# ]
	make_plot(result_dict, labels)


def make_tracking_plots():
	""" """
	sns.set_style({'font.family': 'monospace'}) #'Times New Roman'})
	plt.style.use('ggplot')

	result_dict = get_tracking_results()
	
	# labels = ['C:MOTA', 'P:MOTA', 'C:IDF1', 'P:IDF1']
	labels = ['C:MT', 'P:MT', 'C:ML', 'P:ML']

	# # FPs and FNs decreased
	# labels = ['C:FP','P:FP','C:FN','P:FN']

	# # Effect of Speed
	# labels = [
	# 	'C:MT',
	# 	'C:MT-FST',
	# ]

	# # Effect of Distance
	# labels = [
	# 	'C:MT', 
	# 	'C:MT-FAR',
	# ]

	# Effect of Occlusion
	# labels = [
	# 	'C:MT', 
	# 	'C:MT-OCC',
	# 	'P:MT',
	# 	'P:MT-OCC',
	# ]

	# labels = [
	# 	'C:FRG',
	# 	'P:FRG',
	# ]

	# labels = [
	# 	'C:FRG-FAR',
	# 	'P:FRG-FAR',
	# 	'C:SW-FAR',
	# 	'P:SW-FAR',
	# 	'C:FRG-OCC',
	# 	'P:FRG-OCC',
	# 	'C:SW-OCC',
	# 	'P:SW-OCC',
	# 	'C:FRG-FST',
	# 	'C:SW-FST',
	# ]

	# Effect on MOTP
	# labels = [
	# 	'C:MOTPD',
	# 	'P:MOTPD',
	# 	'C:MOTPI',
	# 	'P:MOTPI',
	# ]
	# labels = [
	# 	'C:MOTPO',
	# 	'P:MOTPO',
	# ]
	make_plot(result_dict, labels)


def get_tracking_results():
	""" """
	fpath = '/Users/johnlamb/Downloads/cvpr-argoverse-tracking-winners.csv'
	rows = read_csv(fpath, delimiter=',')
	result_dict = defaultdict(list)
	for i,row in enumerate(rows):
		print(row['Team name'])
		# 'Submission ID,Submitted at,AVG-RANK,
		result_dict['Team name'] += [row['Team name']]
		result_dict['C:MOTA']  += [float(row['C:MOTA'])]
		result_dict['P:MOTA']  += [float(row['P:MOTA'])]
		result_dict['C:MOTPD'] += [float(row['C:MOTPD'])]
		result_dict['P:MOTPD'] += [float(row['P:MOTPD'])]
		result_dict['C:MOTPO'] += [float(row['C:MOTPO'])]
		result_dict['P:MOTPO'] += [float(row['P:MOTPO'])]
		result_dict['C:MOTPI'] += [float(row['C:MOTPI'])]
		result_dict['P:MOTPI'] += [float(row['P:MOTPI'])]
		result_dict['C:IDF1']  += [100 * float(row['C:IDF1'])]
		result_dict['P:IDF1']  += [100 * float(row['P:IDF1'])]
		result_dict['C:MT']    += [100 * float(row['C:MT'])]
		result_dict['P:MT']    += [100 * float(row['P:MT'])]
		result_dict['C:ML']    += [100 * float(row['C:ML'])]
		result_dict['P:ML']    += [100 * float(row['P:ML'])]
		result_dict['C:FP']    += [int(row['C:FP'])]
		result_dict['P:FP']    += [int(row['P:FP'])]
		result_dict['C:FN']    += [int(row['C:FN'])]
		result_dict['P:FN']    += [int(row['P:FN'])]
		result_dict['C:SW']    += [int(row['C:SW'])]
		result_dict['P:SW']    += [int(row['P:SW'])]
		result_dict['C:FRG']   += [int(row['C:FRG'])]
		result_dict['P:FRG']   += [int(row['P:FRG'])]
		result_dict['C:MT-OCC']  += [100 * float(row['C:MT-OCC'])]
		result_dict['C:MT-FAR']  += [100 * float(row['C:MT-FAR'])]
		result_dict['C:ML-OCC']  += [100 * float(row['C:ML-OCC'])]
		result_dict['C:ML-FAR']  += [100 * float(row['C:ML-FAR'])]
		result_dict['C:FRG-OCC'] += [int(row['C:FRG-OCC'])]
		result_dict['C:FRG-FAR'] += [int(row['C:FRG-FAR'])]
		result_dict['C:SW-OCC']  += [int(row['C:SW-OCC'])]
		result_dict['C:SW-FAR']  += [int(row['C:SW-FAR'])]
		result_dict['C:MT-FST']  += [100 * float(row['C:MT-FST'])]
		result_dict['C:ML-FST']  += [100 * float(row['C:ML-FST'])]
		result_dict['C:FRG-FST'] += [int(row['C:FRG-FST'])]
		result_dict['C:SW-FST']  += [int(row['C:SW-FST'])]
		result_dict['P:MT-OCC']  += [100 * float(row['P:MT-OCC'])]
		result_dict['P:MT-FAR']  += [100 * float(row['P:MT-FAR'])]
		result_dict['P:ML-OCC']  += [100 * float(row['P:ML-OCC'])]
		result_dict['P:ML-FAR']  += [100 * float(row['P:ML-FAR'])]
		result_dict['P:FRG-OCC'] += [int(row['P:FRG-OCC'])]
		result_dict['P:FRG-FAR'] += [int(row['P:FRG-FAR'])]
		result_dict['P:SW-OCC']  += [int(row['P:SW-OCC'])]
		result_dict['P:SW-FAR']  += [int(row['P:SW-FAR'])]
	return result_dict


def get_forecasting_results():
	""" """
	fpath = '/Users/johnlamb/Downloads/cvpr-argoverse-forecasting-winners.csv'
	rows = read_csv(fpath, delimiter=',')
	result_dict = defaultdict(list)
	for i,row in enumerate(rows):

		print(row['Team name'])
		result_dict['Team name'] += [row['Team name']]
		result_dict['minADE (K=1)'] 			+= [float(row['minADE (K=1)'])]
		result_dict['minFDE (K=1)'] 			+= [float(row['minFDE (K=1)'])]
		result_dict['DAC (K=1)'] 				+= [float(row['DAC (K=1)'])]
		result_dict['MR (K=1)']					+= [float(row['MR (K=1)'])]
		result_dict['minADE (K=6)'] 			+= [float(row['minADE (K=6)'])]
		result_dict['minFDE (K=6)'] 			+= [float(row['minFDE (K=6)'])]
		result_dict['DAC (K=6)'] 				+= [float(row['DAC (K=6)'])]
		result_dict['MR (K=6)'] 				+= [float(row['MR (K=6)'])]
		result_dict['p-minADE (K=6)'] 			+= [float(row['p-minADE (K=6)'])]
		result_dict['p-minFDE (K=6)'] 			+= [float(row['p-minFDE (K=6)'])]

	return result_dict


def make_plot(result_dict, labels):
	""" """
	x = np.arange(len(result_dict['Team name']))  # the label locations

	if len(labels) == 2:
		centers = [-0.2,0.2]
		width=0.4
	else:
		centers = np.linspace(-0.3, 0.3, len(labels) )
		width = centers[1] - centers[0]

	fig, ax = plt.subplots()

	all_rects = []

	colors = [ "#ECA154", "#007672", "#245464", "#78909c"] # "#595959"]# "#212121"] # "#d3e8ef" ]
	for label, offset,color in zip(labels, centers, colors):
		rects = ax.bar(x=x + offset, height=result_dict[label], width=width, label=label, color=color)
		all_rects += [rects]

		# colors = [ "#ECA154", "#007672", "#245464", "#d3e8ef" ]
		# for rect,color in zip(rects, colors):
		# 	rect.set_color(color)

	# Add some text for labels, title and custom x-axis tick labels, etc.
	# ax.set_ylabel('Error (degrees)')
	
	#ax.set_title('Scores by group and gender')
	ax.set_xticks(x)
	# # rotate the labels with proper anchoring.
	ha = 'right'
	ax.set_xticklabels(result_dict['Team name'], rotation=45, ha=ha)

	ax.legend(
		loc='upper center',
		# bbox_to_anchor=(1, 0.5)
		bbox_to_anchor=(0.5, 1.3),
		# bbox_to_anchor=(0.5, 1.0), 
		# shadow=True, 
		ncol=4, 
	)

	def autolabel(rects):
	    """Attach a text label above each bar in *rects*, displaying its height."""
	    for rect in rects:
	        height = rect.get_height()
	        ax.annotate(f'{height:.2f}',
	                    xy=(rect.get_x() + rect.get_width() / 2, height),
	                    xytext=(0, 3),  # 3 points vertical offset
	                    textcoords="offset points",
	                    ha='center', va='bottom')

	for rects in all_rects:
		autolabel(rects)

	fig.tight_layout()
	plt.show()




if __name__ == '__main__':
	""" """
	#make_tracking_plots()
	make_forecasting_plots()


