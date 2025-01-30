import subprocess
import os
import numpy as np
import pytest

def test_quac_rndmat_adj_build():
    script_path = os.path.abspath('/home/kevin-da/LibrairiesQuack/Quackto/quactography/scripts/quac_randmatrix_adj_build.py')
    output_file = 'test_output_graph.npz'
    
    # Test case 1: edges_matter is True
    result = subprocess.run(['python3', script_path, '5', '10', 'True', output_file], capture_output=True, text=True)
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    print(data.files)
    assert 'adjacency_matrix' in data
    assert data['adjacency_matrix'].shape[0] == 5

    # Clean up
    os.remove(output_file)

    # Test case 2: edges_matter is False
    result = subprocess.run(['python3', script_path, '5', '10', 'False', output_file], capture_output=True, text=True)
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert 'adjacency_matrix' in data
    assert data['adjacency_matrix'].shape[0] == 5

    # Clean up
    os.remove(output_file)

def test_quac_rndmat_adj_build_NOT():
    script_path = os.path.abspath('/home/kevin-da/LibrairiesQuack/Quackto/quactography/scripts/quac_randmatrix_adj_build.py')
    output_file = 'test_output_graph.npz'
    
    # Test case 1: edges_matter is True
    result = subprocess.run(['python3', script_path, '6', '10', 'True', output_file], capture_output=True, text=True)
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    print(data.files)
    assert 'adjacency_matrix' in data
    assert not data['adjacency_matrix'].shape[0] == 6

    # Clean up
    os.remove(output_file)

    # Test case 2: edges_matter is False
    result = subprocess.run(['python3', script_path, '6', '10', 'False', output_file], capture_output=True, text=True)
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert 'adjacency_matrix' in data
    assert not data['adjacency_matrix'].shape[0] == 6

    # Clean up
    os.remove(output_file)
    