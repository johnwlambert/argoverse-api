#!/usr/bin/python3

from collections import namedtuple, defaultdict
import numpy as np
import os
from pathlib import Path
import pdb
from scipy.spatial.transform import Rotation
import shutil
from typing import Tuple

from argoverse.utils.json_utils import save_json_dict
from argoverse.evaluation.eval_tracking import eval_tracks

_ROOT = Path(__file__).resolve().parent


def check_mkdir(dirpath):
	""" """
	if not Path(dirpath).exists():
		os.makedirs(dirpath, exist_ok=True)

def yaw_to_quaternion3d(yaw: float) -> Tuple[float,float,float,float]:
	"""
		Args:
		-   yaw: rotation about the z-axis
		Returns:
		-   qx,qy,qz,qw: quaternion coefficients
	"""
	qx,qy,qz,qw = Rotation.from_euler('z', yaw).as_quat()
	return qx,qy,qz,qw


fields = ('l', 'w', 'h', 'qx', 'qy', 'qz', 'qw', 'cx', 'cy', 'cz', 'track_id', 'label_class')
TrackedObjRec = namedtuple('TrackedObjRec', fields, defaults=(None,) * len(fields))


class TrackedObjects:
	def __init__(self, log_id: str, is_gt: bool):
		""" """
		self.ts_to_trackedlabels_dict = defaultdict(list)
		self.log_id = log_id

		tracks_type = 'gt' if is_gt else 'pred'
		self.log_dir = f'{_ROOT}/test_data/'
		self.log_dir += f'eval_tracking_dummy_logs_{tracks_type}/{self.log_id}'

	def add_obj(self, o: TrackedObjRec, ts_ns: int):
		"""
			Args:
			-	ts_ns: timestamp in nanoseconds
		"""
		self.ts_to_trackedlabels_dict[ts_ns] += [
			{
				"center": {"x": o.cx, "y": o.cy, "z": o.cz},
				"rotation": {"x": o.qx , "y": o.qy, "z": o.qz , "w": o.qw},
				"length": o.l,
				"width": o.w,
				"height": o.h,
				"track_label_uuid": o.track_id,
				"timestamp": ts_ns, # 1522688014970187
				"label_class": o.label_class,
			}]

	def save_to_disk(self):
		"""
		Labels and predictions should be saved in JSON e.g.
			`tracked_object_labels_315969629019741000.json`
		"""
		for ts_ns, ts_trackedlabels in self.ts_to_trackedlabels_dict.items():
			json_fpath = f'{self.log_dir}/per_sweep_annotations_amodal/'
			check_mkdir(json_fpath)
			json_fpath += f'tracked_object_labels_{ts_ns}.json'
			save_json_dict(json_fpath, ts_trackedlabels)


def dump_scenario_json(centers, yaw_angles, log_id, is_gt, run_eval=True):
	"""
	Egovehicle stationary (represented by `o`).
	Seqeuence of 4-nanosecond timestamps.
	"""
	t_objs = TrackedObjects(log_id=log_id, is_gt=is_gt)

	l = 2
	w = 2
	h = 1
	track_id = 'obj_a'
	label_class = 'VEHICLE'

	cx, cy, cz = centers[0]
	qx,qy,qz,qw = yaw_to_quaternion3d(yaw=yaw_angles[0])
	tor = TrackedObjRec(l,w,h,qx,qy,qz,qw,cx,cy,cz,track_id,label_class)
	t_objs.add_obj(tor, ts_ns=0)
	
	cx, cy, cz = centers[1]
	qx,qy,qz,qw = yaw_to_quaternion3d(yaw=yaw_angles[1])
	tor = TrackedObjRec(l,w,h,qx,qy,qz,qw,cx,cy,cz,track_id,label_class)
	t_objs.add_obj(tor, ts_ns=1)

	cx, cy, cz = centers[2]
	qx,qy,qz,qw = yaw_to_quaternion3d(yaw=yaw_angles[2])
	tor = TrackedObjRec(l,w,h,qx,qy,qz,qw,cx,cy,cz,track_id,label_class)
	t_objs.add_obj(tor, ts_ns=2)
	
	cx, cy, cz = centers[3]
	qx,qy,qz,qw = yaw_to_quaternion3d(yaw=yaw_angles[3])
	tor = TrackedObjRec(l,w,h,qx,qy,qz,qw,cx,cy,cz,track_id,label_class)
	t_objs.add_obj(tor, ts_ns=3)
	
	t_objs.save_to_disk()

	if not run_eval:
		return None

	pred_log_dir = f'{_ROOT}/test_data/eval_tracking_dummy_logs_pred'
	gt_log_dir = f'{_ROOT}/test_data/eval_tracking_dummy_logs_gt'
	
	out_fpath = f'{_ROOT}/test_data/{log_id}.txt'
	out_file = open(out_fpath, 'w')
	eval_tracks(
		path_tracker_output_root=pred_log_dir,
		path_dataset_root=gt_log_dir,
		d_min=0,
		d_max=100,
		out_file=out_file,
		centroid_method="average",
		diffatt=None,
		category='VEHICLE'
	)
	out_file.close()

	with open(out_fpath, 'r') as f:
		result_lines = f.readlines()
		result_vals = result_lines[0].strip().split(' ')

		fn, num_frames, mota, motp_c, motp_o, motp_i, idf1 = result_vals[:7]
		most_track, most_lost, num_fp, num_miss, num_sw, num_frag = result_vals[7:]

		# Todo: change `num_flag` to `num_frag`
		result_dict = {
			'filename': fn,
			'num_frames': int(num_frames),
			'mota': float(mota),
			'motp_c': float(motp_c),
			'motp_o': float(motp_o),
			'motp_i': float(motp_i),
			'idf1': float(idf1),
			'most_track': float(most_track),
			'most_lost': float(most_lost),
			'num_fp': int(num_fp),
			'num_miss': int(num_miss),
			'num_sw': int(num_sw),
			'num_frag': int(num_frag),
		}
	shutil.rmtree(pred_log_dir)
	shutil.rmtree(gt_log_dir)
	return result_dict


def get_1obj_gt_scenario():
	"""
	Egovehicle stationary (represented by `o`).
	Seqeuence of 4-nanosecond timestamps.

	|-|
	| |
	|-|

	|-|
	| |
	|-|
			o (x,y,z) = (0,0,0)
	|-|
	| |
	|-|

	|-|
	| | (x,y,z)=(-3,2,0)
	|-|
	"""
	centers = []
	# timestamp 0
	cx = -3
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 1
	cx = -1
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 2
	cx = 1
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 3
	cx = 3
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	yaw_angles = [0,0,0,0]
	return centers, yaw_angles


def test_1obj_perfect():
	""" """
	log_id = '1obj_perfect'
	gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

	centers = gt_centers
	yaw_angles = gt_yaw_angles

	# dump the ground truth first
	_ = dump_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True, run_eval=False)
	result_dict = dump_scenario_json(centers, yaw_angles, log_id, is_gt=False, )

	assert result_dict['num_frames'] == 4
	assert result_dict['mota'] == 100.0
	assert result_dict['motp_c'] == 0.0
	assert result_dict['motp_o'] == 0.0
	assert result_dict['motp_i'] == 0.0
	assert result_dict['idf1'] == 1.0
	assert result_dict['most_track'] == 1.0
	assert result_dict['most_lost'] == 0.0
	assert result_dict['num_fp'] == 0
	assert result_dict['num_miss'] == 0
	assert result_dict['num_sw'] == 0
	assert result_dict['num_frag'] == 0


def test_1obj_offset_translation():
	""" """
	log_id = '1obj_offset_translation'
	
	centers = []

	# timestamp 0
	cx = -4
	cy = 3
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 1
	cx = -2
	cy = 3
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 2
	cx = 0
	cy = 3
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 3
	cx = 2
	cy = 3
	cz = 0
	centers += [(cx,cy,cz)]

	yaw_angles = [0,0,0,0]

	# dump the ground truth first
	gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

	# dump the ground truth first
	_ = dump_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True, run_eval=False)
	result_dict = dump_scenario_json(centers, yaw_angles, log_id, is_gt=False)

	assert result_dict['num_frames'] == 4
	assert result_dict['mota'] == 100.0
	assert np.allclose( result_dict['motp_c'], np.sqrt(2), atol=0.01) # (1,1) away each time
	assert result_dict['motp_o'] == 0.0
	assert result_dict['motp_i'] == 0.0
	assert result_dict['idf1'] == 1.0
	assert result_dict['most_track'] == 1.0
	assert result_dict['most_lost'] == 0.0
	assert result_dict['num_fp'] == 0
	assert result_dict['num_miss'] == 0
	assert result_dict['num_sw'] == 0
	assert result_dict['num_frag'] == 0


def test_1obj_poor_translation():
	"""
	Miss in 1st frame, TP in 2nd frame,
	lost in 3rd frame, retrack as TP in 4th frame

	Yields 1 fragmentation. Prec=0.5, recall=0.5, F1=0.5

	mostly tracked if it is successfully tracked
	for at least 80% of its life span

	If a track is only recovered for less than 20% of its
	total length, it is said to be mostly lost (ML)
	"""
	log_id = '1obj_poor_translation'
	
	centers = []

	# timestamp 0
	cx = -5
	cy = 4
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 1
	cx = -2
	cy = 3
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 2
	cx = 1
	cy = 4
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 3
	cx = 4
	cy = 3
	cz = 0
	centers += [(cx,cy,cz)]

	yaw_angles = [0,0,0,0]

	# dump the ground truth first
	gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

	# dump the ground truth first
	_ = dump_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True, run_eval=False)
	result_dict = dump_scenario_json(centers, yaw_angles, log_id, is_gt=False)

	assert result_dict['num_frames'] == 4
	sw = 0
	mota = 1 - ((2 + 2 + 0) / 4) # 1 - (FN+FP+SW)/#GT
	assert mota == 0.0
	assert result_dict['mota'] == 0.0
	assert np.allclose( result_dict['motp_c'], np.sqrt(2), atol=0.01) # (1,1) away each time
	assert result_dict['motp_o'] == 0.0
	assert result_dict['motp_i'] == 0.0
	prec = 0.5
	recall = 0.5
	f1 = 2 * prec * recall / (prec + recall)
	assert f1 == 0.5
	assert result_dict['idf1'] == 0.5
	assert result_dict['most_track'] == 0.0
	assert result_dict['most_lost'] == 0.0
	assert result_dict['num_fp'] == 2
	assert result_dict['num_miss'] == 2 # false-negatives
	assert result_dict['num_sw'] == 0
	assert result_dict['num_frag'] == 1


def test_1obj_poor_orientation():
	""" """
	log_id = '1obj_poor_orientation'
	
	centers = []
	# timestamp 0
	cx = -3
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 1
	cx = -1
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 2
	cx = 1
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 3
	cx = 3
	cy = 2
	cz = 0
	centers += [(cx,cy,cz)]

	yaw_angles = [0.25,-0.25,0.25,-0.25]

	# dump the ground truth first
	gt_centers, gt_yaw_angles = get_1obj_gt_scenario()

	# dump the ground truth first
	_ = dump_scenario_json(gt_centers, gt_yaw_angles, log_id, is_gt=True, run_eval=False)
	
	pdb.set_trace()
	result_dict = dump_scenario_json(centers, yaw_angles, log_id, is_gt=False)

	
	assert result_dict['num_frames'] == 4
	assert result_dict['mota'] == 100.0
	assert result_dict['motp_c'] == 0
	assert result_dict['motp_o'] == 0.0 # ?????
	assert result_dict['motp_i'] == 0.0
	assert result_dict['idf1'] == 1.0
	assert result_dict['most_track'] == 1.0
	assert result_dict['most_lost'] == 0.0
	assert result_dict['num_fp'] == 0
	assert result_dict['num_miss'] == 0
	assert result_dict['num_sw'] == 0
	assert result_dict['num_frag'] == 0


"""
Additional examples are here: https://arxiv.org/pdf/1603.00831.pdf
"""


def get_orientation_error_deg(yaw1: float, yaw2: float):
	"""
	smallest difference between 2 angles
	https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles

		Args:
		-	yaw1: angle around unit circle, in radians in [-pi,pi]
		-	yaw2: angle around unit circle, in radians in [-pi,pi]

		Returns:
		-	error: smallest difference between 2 angles, in degrees
	"""
	assert -np.pi < yaw1 and yaw1 < np.pi
	assert -np.pi < yaw2 and yaw2 < np.pi

	error = np.rad2deg(yaw1 - yaw2)
	if error > 180:
		error -= 360
	if error < -180:
		error += 360
	return np.abs(error)


def test_orientation_error1():
	""" """
	yaw1 = np.deg2rad(179)
	yaw2 = np.deg2rad(-179)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	assert np.allclose(error_deg, 2.0, atol=1e-2)

def test_orientation_error2():
	""" """
	yaw1 = np.deg2rad(-179)
	yaw2 = np.deg2rad(179)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	print(error_deg)
	assert np.allclose(error_deg, 2.0, atol=1e-2)

def test_orientation_error3():
	""" """
	yaw1 = np.deg2rad(179)
	yaw2 = np.deg2rad(178)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	assert np.allclose(error_deg, 1.0, atol=1e-2)

def test_orientation_error4():
	""" """
	yaw1 = np.deg2rad(178)
	yaw2 = np.deg2rad(179)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	assert np.allclose(error_deg, 1.0, atol=1e-2)

def test_orientation_error5():
	""" """
	yaw1 = np.deg2rad(3)
	yaw2 = np.deg2rad(-3)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	assert np.allclose(error_deg, 6.0, atol=1e-2)

def test_orientation_error6():
	""" """
	yaw1 = np.deg2rad(-3)
	yaw2 = np.deg2rad(3)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	assert np.allclose(error_deg, 6.0, atol=1e-2)

def test_orientation_error7():
	""" """
	yaw1 = np.deg2rad(-177)
	yaw2 = np.deg2rad(-179)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	assert np.allclose(error_deg, 2.0, atol=1e-2)

def test_orientation_error8():
	""" """
	yaw1 = np.deg2rad(-179)
	yaw2 = np.deg2rad(-177)

	error_deg = get_orientation_error_deg(yaw1, yaw2)
	assert np.allclose(error_deg, 2.0, atol=1e-2)



if __name__ == '__main__':
	""" """
	# test_1obj_perfect()
	# test_1obj_offset_translation()
	# test_1obj_poor_translation()
	# test_1obj_poor_orientation()

	test_orientation_error1()
	test_orientation_error2()
	test_orientation_error3()
	test_orientation_error4()
	test_orientation_error5()
	test_orientation_error6()
	test_orientation_error7()
	test_orientation_error8()


