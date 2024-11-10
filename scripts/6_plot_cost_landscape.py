# # Code from:https://el-openqaoa.readthedocs.io/en/main/notebooks/07_cost_landscapes_w_manual_mode.html

# import numpy as np
# import matplotlib.pyplot as plt
# from openqaoa.qaoa_components import QAOADescriptor, create_qaoa_variational_params
# from openqaoa.backends import create_device
# from openqaoa.backends.qaoa_backend import get_qaoa_backend
# from openqaoa.optimizers import get_optimizer
# from openqaoa.utilities import random_classical_hamiltonian
# from qiskit.quantum_info import SparsePauliOp

# # generate the mixer Hamiltonian
# from openqaoa.qaoa_components import PauliOp, Hamiltonian
# import numpy as np

# paul_term = [  # type: ignore
#     PauliOp("Z", (0,)),
#     PauliOp("Z", (1,)),
#     PauliOp("Z", (2,)),
#     PauliOp("ZZ", (0, 2)),  # type: ignore
#     PauliOp("ZZ", (0, 1)),  # type: ignore
#     PauliOp("ZZ", (1, 2)),  # type: ignore
# ]

# coef_terms = [1, 0.5, 0.5, 1, 1, -4]

# cost_hamiltonian = Hamiltonian(paul_term, coef_terms, constant=4.5)
# n_qubits = 3
# qubit_register = [0, 1, 2]


# pauli_terms = [PauliOp("X", (i,)) for i in qubit_register]
# pauli_coeffs = [1] * n_qubits
# mixer_hamiltonian = Hamiltonian(pauli_terms, pauli_coeffs, constant=0)  # type: ignore

# qaoa_descriptor_p1 = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p=1)
# qaoa_descriptor_p2 = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p=1)
# variate_params_std = create_qaoa_variational_params(
#     qaoa_descriptor_p1, "standard", "ramp"
# )
# variate_params_fourier = create_qaoa_variational_params(
#     qaoa_descriptor_p2, "fourier", "ramp", q=1
# )

# # call the device to use
# device_qiskit = create_device(location="local", name="qiskit.qasm_simulator")

# # initialize the backend with the device and circuit_params
# backend_qiskit_p1 = get_qaoa_backend(qaoa_descriptor_p1, device_qiskit, n_shots=500)
# backend_qiskit_p2 = get_qaoa_backend(qaoa_descriptor_p2, device_qiskit, n_shots=500)


# # helper function to produce the cost landscape
# def plot_cost_landscape(mixer_angle_iter, cost_angle_iter, variational_params, backend):
#     """
#     This function constructs a 2-D array containing cost values for different pairs
#     of parameter values.

#     Parameters
#     ----------

#     """
#     cost_landscape = np.zeros(
#         (mixer_angle_iter.size, mixer_angle_iter.size), dtype=float
#     )

#     for i, mixer_angle in enumerate(mixer_angle_iter):
#         for j, cost_angle in enumerate(cost_angle_iter):
#             variational_params.update_from_raw([mixer_angle, cost_angle])
#             cost_landscape[i, j] = backend.expectation(variational_params)

#     return cost_landscape


# # cost landscape for standard parameterization
# gammas = np.linspace(-np.pi, np.pi, 25)
# betas = np.linspace(-np.pi / 2, np.pi / 2, 25)

# cost_landscape_std = plot_cost_landscape(
#     betas, gammas, variate_params_std, backend_qiskit_p1
# )

# # cost landscape for Fourier parameterization:
# us = np.linspace(0, 2 * np.pi, 50)
# vs = np.linspace(0, np.pi, 50)

# cost_landscape_fourier = plot_cost_landscape(
#     vs, us, variate_params_fourier, backend_qiskit_p2
# )


# fig, axes = plt.subplots(
#     1, 2, sharex=False, sharey=False, figsize=(15, 7), gridspec_kw={"wspace": 0.3}
# )
# cmap = "viridis"

# im_0 = axes[0].contourf(
#     cost_landscape_std,
#     cmap=cmap,
#     levels=100,
#     extent=(gammas[0], gammas[-1], betas[0], betas[-1]),
# )
# axes[0].set_title("Cost Landscape for Standard Parameterization")
# axes[0].set_xlabel("Gammas")
# axes[0].set_ylabel("Betas")

# cbar_ax = fig.add_axes([0.472, 0.15, 0.01, 0.7])
# fig.colorbar(im_0, cax=cbar_ax)
# im_1 = axes[1].contourf(
#     cost_landscape_fourier,
#     cmap=cmap,
#     levels=100,
#     extent=(gammas[0], gammas[-1], betas[0], betas[-1]),
# )
# axes[1].set_title("Cost Landscape for Fourier Parameterization")
# axes[1].set_xlabel("$u$'s (Fourier Params)")
# axes[1].set_ylabel("$v$'s (Fourier Params)")

# cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
# fig.colorbar(im_1, cax=cbar_ax)
# plt.show()


# paul_term = [  # type: ignore
#     PauliOp("Z", (0,)),
#     PauliOp("Z", (3,)),
#     PauliOp("Z", (4,)),
#     PauliOp("Z", (5,)),
#     PauliOp("ZZ", (3, 4)),  # type: ignore
#     PauliOp("ZZ", (0, 3)),  # type: ignore
#     PauliOp("ZZ", (0, 4)),  # type: ignore
#     PauliOp("ZZ", (0, 1)),  # type: ignore
#     PauliOp("ZZ", (0, 2)),  # type: ignore
#     PauliOp("ZZ", (1, 2)),  # type: ignore
#     # PauliOp("ZZZ", (1, 3, 5)),
#     # PauliOp("ZZZ", (2, 4, 5)),
# ]

# coef_terms = [
#     -1.0,
#     -0.5,
#     -0.5,
#     1.0,
#     1.0,
#     1.0,
#     1.0,
#     1.0,
#     1.0,
#     1.0,
#     # -4.0,
#     # -4.0,
# ]


# cost_hamiltonian = Hamiltonian(paul_term, coef_terms, constant=7)  # type: ignore
# n_qubits = 3
# qubit_register = [0, 1, 2]


# pauli_terms = [PauliOp("X", (i,)) for i in qubit_register]
# pauli_coeffs = [1] * n_qubits
# mixer_hamiltonian = Hamiltonian(pauli_terms, pauli_coeffs, constant=0)  # type: ignore

# qaoa_descriptor_p1 = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p=1)
# qaoa_descriptor_p2 = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p=1)
# variate_params_std = create_qaoa_variational_params(
#     qaoa_descriptor_p1, "standard", "ramp"
# )
# variate_params_fourier = create_qaoa_variational_params(
#     qaoa_descriptor_p2, "fourier", "ramp", q=1
# )

# # call the device to use
# device_qiskit = create_device(location="local", name="qiskit.qasm_simulator")

# # initialize the backend with the device and circuit_params
# backend_qiskit_p1 = get_qaoa_backend(qaoa_descriptor_p1, device_qiskit, n_shots=500)
# backend_qiskit_p2 = get_qaoa_backend(qaoa_descriptor_p2, device_qiskit, n_shots=500)


# # cost landscape for standard parameterization
# gammas = np.linspace(-np.pi, np.pi, 25)
# betas = np.linspace(-np.pi / 2, np.pi / 2, 25)

# cost_landscape_std = plot_cost_landscape(
#     betas, gammas, variate_params_std, backend_qiskit_p1
# )

# # cost landscape for Fourier parameterization:
# us = np.linspace(0, 2 * np.pi, 50)
# vs = np.linspace(0, np.pi, 50)

# cost_landscape_fourier = plot_cost_landscape(
#     vs, us, variate_params_fourier, backend_qiskit_p2
# )


# fig, axes = plt.subplots(
#     1, 2, sharex=False, sharey=False, figsize=(15, 7), gridspec_kw={"wspace": 0.3}
# )
# cmap = "viridis"

# im_0 = axes[0].contourf(
#     cost_landscape_std,
#     cmap=cmap,
#     levels=100,
#     extent=(gammas[0], gammas[-1], betas[0], betas[-1]),
# )
# axes[0].set_title("Cost Landscape for Standard Parameterization")
# axes[0].set_xlabel("Gammas")
# axes[0].set_ylabel("Betas")

# cbar_ax = fig.add_axes([0.472, 0.15, 0.01, 0.7])
# fig.colorbar(im_0, cax=cbar_ax)
# im_1 = axes[1].contourf(
#     cost_landscape_fourier,
#     cmap=cmap,
#     levels=100,
#     extent=(gammas[0], gammas[-1], betas[0], betas[-1]),
# )
# axes[1].set_title("Cost Landscape for Fourier Parameterization")
# axes[1].set_xlabel("$u$'s (Fourier Params)")
# axes[1].set_ylabel("$v$'s (Fourier Params)")

# cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
# fig.colorbar(im_1, cax=cbar_ax)
# plt.show()
