# Inference benchmark

Stand-alone comparison of the network (`MultiModelFramework`) and the TRF baseline: wall-clock time,
throughput, FLOPs and memory of the network at several batch sizes; time, iterations and errors of
TRF with baseline and noisy initialisation.

```bash
PIDU_USECASE=1 python research/timings/benchmark.py
```

Settings (batch sizes, number of TRF samples, iterations) are in the `__main__` block. Results go to
`outputs/timings/`; `outputs/` here holds an earlier run. The paper's Fig. 2 timings come from
`scripts/benchmark_timings.py` instead (see [docs/reproduce.md](../../docs/reproduce.md)).
