import numpy as np 
import pytest
from quactography.adj_matrix.filter import *

test_data = [
    (
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([0, 1, 2]),
        (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
    ),
    (
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        np.array([0, 1, 2]),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 1, 2])),
    )]

test_KEEP = [
    (
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([0, 1, 2]),
        (np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), np.array([0, 1, 2])),
    ),
    (
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        np.array([0, 1, 2]),
        (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 1, 2])),
    )]

@pytest.mark.parametrize("graph, node_indices, expected", test_data)
def test_remove_orphan_nodes(graph, node_indices, expected):

    out_graph,out_ind = remove_orphan_nodes(graph, node_indices)
    assert np.array_equal(out_graph, expected[0])
    assert np.array_equal(out_ind, expected[1])

@pytest.mark.parametrize("graph, node_indices, expected", test_KEEP)
def test_remove_orphan_nodes_KEEP(graph, node_indices, expected):
    out_graph,out_ind = remove_orphan_nodes(graph, node_indices, keep_indices=np.array([2]))
    assert np.array_equal(out_graph, expected[0])
    assert np.array_equal(out_ind, expected[1])

test_data_intermediate = [
    (
    np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]]),
    np.array([0, 1, 2]),
    np.array([[0, 0, 3], [0, 0, 0], [3, 0, 0]]),
    )
    ]

@pytest.mark.parametrize("graph, node_indices, expected", test_data_intermediate)
def test_remove_intermediate_connections(graph, node_indices, expected):
    
    out_graph = remove_intermediate_connections(graph)
    assert np.array_equal(out_graph, expected)
    assert np.array_equal(remove_intermediate_connections(graph,node_indices,keep_indices=[1]),graph)

def test_remove_zero_column_row():
    graph = np.array([[0, 0, 3], [0, 0, 0], [3, 0, 0]])
    out_graph = remove_zero_columns_rows(graph)
    assert np.array_equal(out_graph, np.array([[0, 3], [3, 0]]))

