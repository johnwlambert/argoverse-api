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

			# until deprecated pose reading removed from eval file
			pose_fpath = f'{self.log_dir}/poses/'
			check_mkdir(pose_fpath)
			pose_fpath += f'city_SE3_egovehicle_{ts_ns}.json'
			save_json_dict(pose_fpath,
				{
					'rotation': [0,0,0,1],
					'translation': [0,0,0]
				})

		save_json_dict(f'{self.log_dir}/city_info.json', {"city_name": 'fake'})


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
	""" """
	log_id = '1obj_offset_translation'
	
	centers = []

	# timestamp 0
	cx = -5
	cy = 4
	cz = 0
	centers += [(cx,cy,cz)]

	# timestamp 1
	cx = -2
	cy = 4
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

	pdb.set_trace()
	# assert result_dict['num_frames'] == 4
	# assert result_dict['mota'] == 100.0
	# assert np.allclose( result_dict['motp_c'], np.sqrt(2), atol=0.01) # (1,1) away each time
	# assert result_dict['motp_o'] == 0.0
	# assert result_dict['motp_i'] == 0.0
	# assert result_dict['idf1'] == 1.0
	# assert result_dict['most_track'] == 1.0
	# assert result_dict['most_lost'] == 0.0
	# assert result_dict['num_fp'] == 0
	# assert result_dict['num_miss'] == 0
	# assert result_dict['num_sw'] == 0
	# assert result_dict['num_frag'] == 0




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



if __name__ == '__main__':
	""" """
	# test_1obj_perfect()
	# test_1obj_offset_translation()
	test_1obj_poor_translation()
	# test_1obj_poor_orientation()





