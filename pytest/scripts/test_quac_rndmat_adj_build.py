import subprocess
import os
import numpy as np
import pytest

def test_quac_rndmat_adj_build():
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../QUAC_RANDMATRIX_ADJ_BUILD.py'))
    output_file = 'test_output_graph.npz'
    
    # Test case 1: edges_matter is True
    result = subprocess.run(['python3', script_path, '5', '10', 'True', output_file], capture_output=True, text=True)
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert 'arr_0' in data
    assert data['arr_0'].shape[0] == 5

    # Test case 2: edges_matter is False
    result = subprocess.run(['python3', script_path, '5', '10', 'False', output_file], capture_output=True, text=True)
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert 'arr_0' in data
    assert data['arr_0'].shape[0] == 5

    # Clean up
    os.remove(output_file)