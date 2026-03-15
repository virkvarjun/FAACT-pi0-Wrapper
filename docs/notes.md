# FAACT Implementation Notes

## Changelog

### v0.1.0 (MVP)

- Backbone: `Pi05PolicyWrapper` (LeRobot), `StubBackboneWrapper` (fallback)
- Data: `RolloutLogger`, `build_chunk_dataset`, chunk-level labels (failure_within_k)
- FAACT: `FactMLP`, `FactTemporal` (temporal needs seq input)
- Governor: threshold-based execute/reject
- Scripts: collect, build_dataset, train, calibrate, eval_offline, eval_online, run_wrapped
- Configs: default.yaml

## Integration assumptions

- LeRobot PI05: `predict_action_chunk(batch)` returns (B, chunk_size, action_dim)
- Preprocessor expects batch with `observation.images.<key>`, `observation.state`, `task`
- Postprocessor receives single action tensor, returns unnormalized action

## Label semantics

- `y_fail_within_k_chunks`: 1 if episode fails and failure occurs within next K chunks from this chunk
- `y_episode_fail`: 1 if episode fails (anytime)
- `y_intervention_good`: reserved for future (intervening here would have prevented failure)
