import os
import numpy as np

output_file = "test_output_graph.npz"


def test_help_option(script_runner):
    ret = script_runner.run('quac_randmatrix_adj_build.py', '-h')
    assert ret.success


def test_quac_rndmat_adj_build(script_runner):
    # Test case 1: edges_matter is True
    result = script_runner.run(
        'quac_randmatrix_adj_build.py', "7", "10", "True", output_file
        )
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert "adjacency_matrix" in data
    assert data["adjacency_matrix"].shape[0] == 5

    # Clean up
    os.remove(output_file)

    # Test case 2: edges_matter is False
    result = script_runner.run(
        'quac_randmatrix_adj_build.py', "5", "6", "False", output_file
    )
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert "adjacency_matrix" in data
    assert data["adjacency_matrix"].shape[0] == 5
    # Clean up
    os.remove(output_file)


def test_quac_rndmat_adj_build_NOT(script_runner):
    # Test case 1: edges_matter is True
    result = script_runner.run(
        'quac_randmatrix_adj_build.py', "6", "11", "True", output_file
    )
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert "adjacency_matrix" in data
    assert data["adjacency_matrix"].shape[0] == 6

    # Clean up
    os.remove(output_file)


def test_missing_args(script_runner):
    # Test case : missing args
    result = script_runner.run(
        'quac_randmatrix_adj_build.py', "6", "True", output_file
    )
    assert not result.returncode == 0
    assert not os.path.exists(output_file)
