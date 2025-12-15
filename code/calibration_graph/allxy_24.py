from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Optional, List, Literal


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 1024
    use_state_discrimination: bool = True
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 60
    load_data_id: Optional[int] = None
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"


description = "Single-qubit ALLXY calibration experiment"

node = QualibrationNode(
    name="24_allxy",
    description=description,
    parameters=Parameters(),
)

u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()

if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Determine qubits
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

n_avg = node.parameters.num_averages
reset_type = node.parameters.reset_type_thermal_or_active
flux_point = node.parameters.flux_point_joint_or_independent

# ALLXY sequences (21)
allxy_sequences = [
    ("I",  "I"),
    ("X",  "X"),
    ("Y",  "Y"),
    ("X",  "Y"),
    ("Y",  "X"),
    ("X/2", "I"),
    ("Y/2", "I"),
    ("X/2", "Y/2"),
    ("Y/2", "X/2"),
    ("X/2", "Y"),
    ("Y/2", "X"),
    ("X",  "Y/2"),
    ("Y",  "X/2"),
    ("X/2", "X"),
    ("X",  "X/2"),
    ("Y/2", "Y"),
    ("Y",  "Y/2"),
    ("X",  "I"),
    ("Y",  "I"),
    ("X/2","X/2"),
    ("Y/2","Y/2"),
]
num_seqs = len(allxy_sequences)


def play_allxy_gate(qubit: Transmon, label: str):
    if label == "I":
        qubit.xy.wait(qubit.xy.operations["x90"].length) 
    elif label == "X":
        qubit.xy.play("x180")
    elif label == "Y":
        qubit.xy.play("y180")
    elif label == "X/2":
        qubit.xy.play("x90")
    elif label == "Y/2":
        qubit.xy.play("y90")
    else:
        qubit.xy.wait(qubit.xy.operations["x90"].length)


def play_allxy_sequence(qubit: Transmon, seq_idx: int):
    with switch_(seq_idx):
        for idx, (g1, g2) in enumerate(allxy_sequences):
            with case_(idx):
                play_allxy_gate(qubit, g1)
                play_allxy_gate(qubit, g2)


with program() as allxy_prog:
    seq_idx = declare(int)
    n = declare(int)

    I, I_st, Q, Q_st, n_dummy, n_st = qua_declaration(num_qubits=num_qubits)

    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):
        align()
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(seq_idx, 0, seq_idx < num_seqs, seq_idx + 1):
            with for_(n, 0, n < n_avg, n + 1):
                # Reset
                if reset_type == "active":
                    active_reset(qubit, "readout")
                else:
                    qubit.resonator.wait(qubit.thermalization_time * u.ns)

                qubit.align()
                play_allxy_sequence(qubit, seq_idx)
                qubit.align()

                # Single-shot readout classification
                readout_state(qubit, state[i])

                # Save one classified shot
                save(state[i], state_st[i])

    with stream_processing():
        for i in range(num_qubits):
            (
                state_st[i] 
                .buffer(n_avg) 
                .map(FUNCTIONS.average()) 
                .buffer(num_seqs) 
                .save(f"state{i + 1}") 
             )

if node.parameters.simulate:
    from qm import SimulationConfig

    sim_cfg = SimulationConfig(duration=node.parameters.simulation_duration_ns)
    job = qmm.simulate(config, allxy_prog, sim_cfg)
    samples = job.get_simulated_samples()
    for name, s in samples.items():
        plt.figure()
        s.plot()
        plt.title(name)
    plt.show()

    node.machine = machine
    node.results["figure"] = plt.gcf()
    node.save()

elif node.parameters.load_data_id is None:
    node.results = {}
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(allxy_prog)

    ds = fetch_results_as_xarray(
        job.result_handles,
        qubits,
        {"sequence": np.arange(num_seqs)},
    )
    node.results = {"ds": ds}

else:
    ds = node.results.get("ds", None)

fig, axes = plt.subplots(1, num_qubits, figsize=(4 * num_qubits, 4), squeeze=False)

labels = [f"{g1}-{g2}" for (g1, g2) in allxy_sequences]
x = np.arange(num_seqs)

for j, q in enumerate(qubits):
    ax = axes[0, j]

    # Pull probability and compute binomial 1σ errors
    p = ds["state"].sel(qubit=q.name).values.astype(float)
    
    # Binomial 1-sigma error bars
    N = n_avg
    yerr = np.sqrt(p * (1.0 - p) / N)
    
    ax.errorbar(x, p, yerr=yerr, fmt="o", markersize=3, capsize=3, label="data ±1σ")

    # Ideal stairwell: first 5 is 0, next 12 is 0.5, and final 4 is 1
    ideal = np.zeros(num_seqs)
    ideal[5:17] = 0.5      
    ideal[17:] = 1.0       
    ax.step(x, ideal, where="mid", linestyle="--", linewidth=1.5, label="ideal")

    # Piecewise best fits
    segments = [(0, 5), (5, 17), (17, num_seqs)]
    first_seg = True
    for (start, end) in segments:
        x_seg = x[start:end]
        y_seg = p[start:end]

        if len(x_seg) < 2:
            continue

        m_seg, b_seg = np.polyfit(x_seg, y_seg, 1)
        y_fit_seg = m_seg * x_seg + b_seg

        label_fit = "best fit" if first_seg else None
        ax.plot(x_seg, y_fit_seg, "-", color="tab:green", label=label_fit)
        first_seg = False

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)

    ax.set_ylim(-0.05, 1.05)
    ax.set_title(q.name)
    ax.set_xlabel("ALLXY gate pair")
    ax.set_ylabel("P(e)")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best")

plt.tight_layout()
plt.show()

# Save to node
node.results["figure"] = fig
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()
