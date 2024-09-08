python build_adj_matrix.py ../data/wm.nii.gz ../data/fodf.nii.gz graph

python draw_adj_matrix.py graph.npz

python build_adj_matrix.py ../data/wm.nii.gz ../data/fodf.nii.gz graph --threshold 0.4

python draw_adj_matrix.py graph.npz

python build_random_adj_matrix.py 5 6 True rand_graph

python draw_random_adj_matrix.py rand_graph rand_graph_visu

python find_max_intensity_diffusion_path.py rand_graph 1 0 qaoa_solver_infos --alphas 1 2 3 --reps 1  -npr 2

python plot_distribution_probabilities.py qaoa_solver_infos_alpha_1.npz qaoa_solver_infos_alpha_2.npz qaoa_solver_infos_alpha_3.npz visu_total_dist visu_selected_dist

python plot_optimal_paths.py graph qaoa_solver_infos_alpha_1.npz qaoa_solver_infos_alpha_2.npz qaoa_solver_infos_alpha_3.npz opt_paths
