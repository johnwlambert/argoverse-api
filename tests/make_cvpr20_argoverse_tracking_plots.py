
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


def main():
	""" """
	fpath = '/Users/johnlamb/Downloads/cvpr-argoverse-tracking-winners.csv'
	rows = read_csv(fpath, delimiter=',')
	sns.set_style({'font.family': 'monospace'}) #'Times New Roman'})
	plt.style.use('ggplot')

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

	x = np.arange(len(result_dict['Team name']))  # the label locations
	
	# labels = ['C:MOTA', 'P:MOTA', 'C:IDF1', 'P:IDF1']
	# labels = ['C:MT', 'P:MT', 'C:ML', 'P:ML']
	# labels = ['C:FP','P:FP','C:FN','P:FN']

	# labels = [
	# 	'C:MT',
	# 	'C:MT-FST',
	# ]

	# labels = [
	# 	'C:MT', 
	# 	'C:MT-OCC',
	# 	'P:MT',
	# 	'P:MT-OCC',
	# ]

	# labels = [
	# 	'C:MT', 
	# 	'C:MT-FAR',
	# 	# 'C:ML',
	# 	# 'C:ML-FAR',
	# 	# 'P:MT-FAR',
	# 	# 'P:ML-FAR',
	# ]

	labels = [
		'C:MT', 
		'C:MT-OCC',
	# 	'C:ML-OCC',
		'P:MT',
		'P:MT-OCC',
	# 	'P:ML-OCC',
	# 	'C:MT-FAR',
	# 	'C:ML-FAR',
	# 	'P:MT-FAR',
	# 	'P:ML-FAR',
	# 	'C:MT-FST',
	# 	'C:ML-FST'
	]

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

	if len(labels) == 2:
		centers = [-0.2,0.2]
		width=0.4
	else:
		centers = np.linspace(-0.3, 0.3, len(labels) )
		width = centers[1] - centers[0]

	fig, ax = plt.subplots()

	all_rects = []
	for label, offset in zip(labels, centers):
		rects = ax.bar(x=x + offset, height=result_dict[label], width=width, label=label)
		all_rects += [rects]

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
	        ax.annotate(f'{height:.0f}',
	                    xy=(rect.get_x() + rect.get_width() / 2, height),
	                    xytext=(0, 3),  # 3 points vertical offset
	                    textcoords="offset points",
	                    ha='center', va='bottom')

	for rects in all_rects:
		autolabel(rects)

	fig.tight_layout()
	plt.show()







if __name__ == '__main__':
	main()