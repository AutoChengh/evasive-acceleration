# Strictly causal yaw-rate estimation

EA can consume yaw rate supplied by a sensor or upstream tracker.  When it is
unavailable, the installed `evasive_acceleration` API provides the causal
estimator implemented in `src/yaw_rate.py`. The estimator uses no observation
after the frame being evaluated. Both input paths share the same EA solver.

## Definition

For timestamped headings `(t_k, theta_k)`, first wrap the backward angular
difference to `[-pi, pi)`:

```text
delta_theta_k = ((theta_k - theta_(k-1) + pi) mod (2*pi)) - pi
raw_rate_k = delta_theta_k / (t_k - t_(k-1))
```

The full-precision filter state is updated using the actual elapsed time:

```text
alpha_k = 1 - exp(-(t_k - t_(k-1)) / tau)
z_k = alpha_k * raw_rate_k + (1 - alpha_k) * z_(k-1)
```

The release defaults are:

| Parameter | Default |
|---|---:|
| EWMA time constant `tau` | 0.6 s |
| Output deadband | 0.002 rad/s |
| Output decimal places | 4 |

These are explicit preprocessing parameters, not part of the physical EA
definition.  They must be reported with an experiment.

## Initialization

- The first observation outputs `0`, because one heading cannot identify a
  derivative.  Diagnostics mark this value as an initialization placeholder.
- The first backward difference directly initializes `z` on the second
  observation; the placeholder zero is not fed into the filter.
- Later observations use the EWMA recurrence.
- Rounding and the deadband affect only the public output.  They are never fed
  back into the full-precision state.

## State and failure rules

Maintain a separate estimator for every continuously tracked object.  Reset it
after an object-ID change or a genuine trajectory restart.  Invalid headings,
invalid timestamps, and duplicate or decreasing timestamps raise `ValueError`
without changing the existing state.

Irregular sampling and occasional missing observations are handled through the
actual elapsed time.  A long gap still leaves an identifiability limitation:
the two endpoint headings cannot reveal unobserved full rotations.  An
application may therefore impose a dataset-specific maximum-gap reset rule.

When an evaluation window is cut from a longer observed trajectory, compute
yaw rate on the full available causal history first and then select the
window.  Restarting at the window boundary changes the initial estimates.

## APIs

Install from the checkout with `python -m pip install -e .`. Imports then work
from any working directory, without adding `src` to `sys.path`.

Online estimation (one independent estimator per object):

```python
from evasive_acceleration import CausalYawRateEWMA

estimator = CausalYawRateEWMA(tau_s=0.6)
estimator.update(0.0, 0.0)  # Zero initialization placeholder.
yaw_rate = estimator.update(0.1, 0.01)  # 0.1 rad/s.
diagnostics = estimator.last_estimate
print(yaw_rate, diagnostics.initialized)  # 0.1, True
```

Offline application of the identical recursion:

```python
from evasive_acceleration import causal_yaw_rate

yaw_rates = causal_yaw_rate([0.0, 0.1, 0.2], [0.0, 0.01, 0.02])
# array([0.0, 0.1, 0.1]); the first value is a placeholder.
```

Batch EA without yaw-rate columns:

```bash
ea batch --input examples/data/without_yaw_rate.csv --yaw-rate-source estimated
```

For CSV processing, install `python -m pip install -e ".[batch]"`. Batch defaults
to `auto`: use each finite supplied yaw-rate value and estimate individual
missing values. `Time (s)` is required only when estimation is needed;
`--time-column` selects another timestamp column. Where valid timestamps are
present, causal histories are updated even on rows whose yaw rates are supplied.
This allows later missing values to use the preceding observations. `estimated`
ignores all supplied yaw-rate values, while `provided` never substitutes an
estimate. Neither the direct state API nor `provided` requires heading history.

Keep each CSV in increasing time order with one continuously tracked interaction
pair. Interleaved pairs require upstream separation into independent streams.
The batch implementation resets an estimator when its corresponding
`Vehicle Number` or `2_Vehicle Number` changes. It does not sort using future
rows or apply centered smoothing.

Output columns preserve the yaw rate used, its source, initialization status,
and causal sample count. Row errors include specific diagnostic messages.
Default result directories include the policy name to separate comparison
runs, and the adjacent `.run.json` stores preprocessing settings. Existing
results require `--overwrite`. See the [README](../README.md#reading-the-results)
for output meanings and [from_history.py](../examples/from_history.py) for a
complete two-agent example.

## Interpretation and limitations

Inputs are radians and seconds. Positive yaw rate means increasing heading
counterclockwise from the positive x axis in a Cartesian frame. Body yaw is not
always the direction of displacement, especially for pedestrians or at low speed.

The filter suppresses high-frequency fluctuations but introduces response lag
when yaw rate changes. Its output accuracy depends on heading quality, sampling
frequency, and gaps. It does not repair 180-degree box-heading flips, guarantee
physical plausibility, or infer unobserved motion between widely separated
samples. A four-decimal output does not imply four-decimal measurement accuracy.
Do not silently replace invalid observations with zero: invalid updates preserve
the estimator state and report an error. A first-frame placeholder of zero is
distinct from a later initialized estimate of zero.

## Required validation properties

The automated tests cover:

- the documented numerical example;
- constant-turn and irregular-time sequences;
- wrapping from `179 deg` to `-179 deg`;
- prefix consistency, demonstrating that future observations do not affect
  earlier outputs;
- state preservation after invalid updates;
- initialization diagnostics;
- end-to-end batch estimation without yaw-rate columns;
- automatic fallback for individual missing supplied values.
