# Historical research records

The official EA algorithm is implemented on `main`. The following immutable
source snapshots are retained only to audit and reproduce earlier research
records; they are not supported EA algorithms or runtime choices:

- [`paper-reference-20261022`](https://github.com/AutoChengh/evasive-acceleration/tree/paper-reference-20261022):
  the historical numerical estimator and its research/reproducibility records.
- [`legacy-public-20261022`](https://github.com/AutoChengh/evasive-acceleration/tree/legacy-public-20261022):
  the public main branch immediately before this update.

The paper's empirical risk analyses and numerical-resolution sensitivity
results were generated with the historical numerical estimator. Its updated
computational-efficiency experiment uses the EA algorithm provided on `main`.
The mathematical definition, four CV/CTRV hypotheses and equal-weight
aggregation are unchanged; historical numerical estimates are not assumed
identical to current EA outputs frame by frame.

For auditing historical results, check out the appropriate tag in a separate
directory and environment and follow its recorded instructions. Preserve the
input yaw rates, numerical settings and source revision with your results.
