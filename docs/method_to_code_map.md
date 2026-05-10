# Method To Code Map

This map connects the CAST paper concepts to implementation files.

| Paper concept | Code location |
|---|---|
| CAST public API | `models/cast.py` |
| CAST configuration and forward pass | `models/cast_core.py` |
| causal Transformer backbone | `layer_perm_mixed_ablation.py`, `models/cast_shared.py` |
| empirical successor retrieval | `models/cast_shared.py::SuccessorRetrievalMemory` |
| persistence-successor anchor | `models/cast_core.py::BoundedAnchorMixer` |
| bounded local transport | `models/cast_core.py::ConservativeRadiusOneTransport` |
| structural transport regularization | `models/cast_core.py::CASTModel.forward` |
| model registry and public ablations | `models/registry.py` |
| tensor benchmark task | `tasks/tensor_dir.py` |
| queue benchmark task | `tasks/queue_num_in_sys.py` |
| offline metrics | `evaluation/offline.py`, `evaluation/metrics.py` |
| rollout metrics | `evaluation/rollout.py`, `evaluation/metrics.py` |

Public CAST ablations are implemented in `models/cast_core.py`:

- `full`
- `without_structural_reg`
- `anchor_only`
- `single_head_retrieval`
- `fixed_local_kernel`
- `without_persistence_mix`
