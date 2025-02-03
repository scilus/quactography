import numpy as np 
import nibabel as nib
import quactography.image.utils as utils
import pytest
import os
from quactography.adj_matrix.reconst import *


test_data = [
    (nib.load('/home/kevin-da/LibrairiesQuack/Quackto/quactography/data/simplePhantoms/fanning_2d_3bundles/wm_vf.nii.gz'),
     (np.array([1320, 1321, 1322, 1323, 1324, 1325, 1374, 1375, 1376, 1377, 1378, 1379, 1428, 1429, 1430, 1431, 1432, 1433, 1482, 1483, 1484, 1485, 1486, 1487, 1536, 1537, 1538, 1539, 1540, 1541, 1590, 1591, 1592, 1593, 1594, 1595]),
      np.array([1, 6,    7,   36,   38,   42,   43,   44,   73,   75,   79,   80,   81,  110,
  112,  116,  117,  118,  147,  149,  153,  154,  155,  184,  190,  191,  216,  217,
  223,  228,  229,  252,  253,  254,  258,  260,  264,  265,  266,  289,  290,  291,
  295,  297,  301,  302,  303,  326,  327,  328,  332,  334,  338,  339,  340,  363,
  364,  365,  369,  371,  375,  376,  377,  400,  401,  406,  412,  413,  438,  439,
  445,  450,  451,  474,  475,  476,  480,  482,  486,  487,  488,  511,  512,  513,
  517,  519,  523,  524,  525,  548,  549,  550,  554,  556,  560,  561,  562,  585,
  586,  587,  591,  593,  597,  598,  599,  622,  623,  628,  634,  635,  660,  661,
  667,  672,  673,  696,  697,  698,  702,  704,  708,  709,  710,  733,  734,  735,
  739,  741,  745,  746,  747,  770,  771,  772,  776,  778,  782,  783,  784,  807,
  808,  809,  813,  815,  819,  820,  821,  844,  845,  850,  856,  857,  882,  883,
  889,  894,  895,  918,  919,  920,  924,  926,  930,  931,  932,  955,  956,  957,
  961,  963,  967,  968,  969,  992,  993,  994,  998, 1000, 1004, 1005, 1006, 1029,
 1030, 1031, 1035, 1037, 1041, 1042, 1043, 1066, 1067, 1072, 1078, 1079, 1104, 1105,
 1111, 1140, 1141, 1142, 1146, 1148, 1177, 1178, 1179, 1183, 1185, 1214, 1215, 1216,
 1220, 1222, 1251, 1252, 1253, 1257, 1259, 1288, 1289, 1294]))
     )
    ]

@pytest.mark.parametrize("node_mask, expected", test_data)    
def test_build_adjacency_matrix(node_mask,expected):
    nodes_mask_im = utils.slice_along_axis(
        node_mask.get_fdata().astype(bool), 'coronal', None  
    )
    adj_matrix, nodes_indice = build_adjacency_matrix(nodes_mask_im)
    assert np.array_equal(nodes_indice, expected[0])
    assert np.array_equal(np.flatnonzero(adj_matrix), expected[1])