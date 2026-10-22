# Historical reproduction

The main branch provides one supported EA implementation. Earlier solvers are
archived, not selectable runtime options:

- [`paper-reference-20261022`](https://github.com/AutoChengh/evasive-acceleration/tree/paper-reference-20261022):
  the preserved reference estimator and its numerical/reproducibility records.
- [`legacy-public-20261022`](https://github.com/AutoChengh/evasive-acceleration/tree/legacy-public-20261022):
  the public main branch immediately before this update.

The paper's empirical risk analyses and numerical-resolution sensitivity
results use the reference estimator. The updated computational-efficiency
experiment uses the continuous-time implementation now provided on main.
The EA definition, four CV/CTRV hypotheses and equal-weight aggregation are
shared; numerical outputs are not assumed identical frame by frame.

For historical reproduction, check out the appropriate tag in a separate
directory and environment and follow its recorded instructions. Preserve the
input yaw rates, numerical settings and source revision with your results.
