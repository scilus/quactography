# VISU:
# # Loop through the results and print the minimum cost for each value of alpha:
# for i, alpha in enumerate(alphas):
#     print("Alpha : ", alpha, " ({})".format(i))
#     print(results[i][0])
#     print(f"Minimum cost: {results[i][1]}")

#     print()

#     alpha_min_costs.append(results[i][2])
#     visualize(
#         starting_nodes,
#         ending_nodes,
#         mat_adj,
#         list(map(int, (alpha_min_costs[i][2]))),
#         alpha,
#         results[i][1],
#         starting_node,
#         ending_node,
#         reps,
#         all_weights_sum,
#     )
#     print(str(alpha_min_costs[i][2]))

# # OUTPUT :
# # Save the minimum cost for different values of alpha to a file with the corresponding binary path in txt file:
# alpha_min_costs = np.array(alpha_min_costs, dtype="str")
# np.savetxt(
#     r"output\alpha_min_cost_classical_read_leftq0.txt",
#     alpha_min_costs,
#     delimiter=",",
#     fmt="%s",
# )
