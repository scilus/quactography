import pathlib
import os
import numpy as np

DIR_PATH = str(pathlib.Path(__file__).parent.parent.parent /
               'data/simplePhantoms/fanning_2d_5bundles')


def test_help_option(script_runner):
    ret = script_runner.run('quac_randmatrix_adj_build.py', '-h')
    assert ret.success


def test_quac_mat_adj_build(script_runner):
    wm_path = DIR_PATH + "/wm_vf.nii.gz"
    fods_path = DIR_PATH + "/fods.nii.gz"
    output_file = "test_output_graph.npz"

    # Test case 1: edges_matter is True
    result = script_runner.run(
        'quac_matrix_adj_build.py',
        wm_path,
        fods_path,
        "--threshold",
        "0.02",
        output_file,
        "--save_only"
    )
    assert result.returncode == 0
    assert os.path.exists(output_file)
    data = np.load(output_file)
    assert "adjacency_matrix" in data
    assert data["adjacency_matrix"].shape[0] == 582

    # Clean up
    os.remove(output_file)


def test_missing_args(script_runner):
    output_file = "test_output_graph.npz"
    # Test case : missing args
    result = script_runner.run(
        'quac_matrix_adj_build.py', output_file
    )
    assert not result.returncode == 0
    assert not os.path.exists(output_file)
