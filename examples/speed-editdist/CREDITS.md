# Credits & prior art

**No external datasets.** Test vectors are generated deterministically by
`gen_vectors.py` (seed 20260619); cost is measured as wasmi/Wasmtime *fuel*.

## Edit distance / bit-parallelism
- Wagner, R. A. & Fischer, M. J. (1974). *The string-to-string correction
  problem.* JACM 21(1):168-173. — the naive DP seed.
- Myers, G. (1999). *A fast bit-vector algorithm for approximate string matching
  based on dynamic programming.* JACM 46(3):395-415. — the bit-parallel target.
- Hyyrö, H. (2003). *Explaining and extending the bit-parallel approximate string
  matching algorithm of Myers.* Tech. report, Univ. of Tampere.

## Framing: minimize instruction count under a correctness gate
- Mankowitz, D. J. et al. (2023). *Faster sorting algorithms discovered using
  deep reinforcement learning* (AlphaDev). Nature 618:257-263. — the closest
  analog: minimize instruction count for fixed kernels under a byte-exact gate;
  results merged into LLVM libc++.
- Romera-Paredes, B. et al. (2024). *Mathematical discoveries from program search
  with large language models* (FunSearch). Nature 625:468-475.
- AlphaEvolve (DeepMind, 2025). — evolutionary code optimization under
  programmatic verifiers.

This example replicates that loop shape with a deterministic, platform-independent
fuel meter in place of measured CPU latency.
