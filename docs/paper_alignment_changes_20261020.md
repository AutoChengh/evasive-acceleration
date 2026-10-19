# EA 论文计算方法修改清单

正文的 EA 定义、二维相对加速度公式、CV/CTRV 传播、四个标量等权平均及有限时域推导均保留。本清单只列与修正后 v2 实现有关的必要修改，给出可直接替换的英文。投稿 Word、PDF 和 LaTeX 原件尚未修改。

本地 v2 的 `0.3.0.dev1` 保留四份冻结参考源码，运行时另用 `verified_solver.py` 和 `verified_calculator.py`。所有局部方向在细网格求解；有限候选必须通过实际 SAT 碰撞复核；不返回未经细网格验证的粗值。当前接触或重叠仍按数值约定返回正无穷。历史实验统计不能因代码修正而自动视为已经重新验证。

## 核对版本与位置

正文来源为 `COMMTR_正文_Word_设定2_严格3s_零宽限.docx`，SHA-256 为 `d18d68473a992ca0904044eb3af5311460b55bef84c345bbbc7da0d02de04df3`。

补充材料来源为 `COMMTR_补充材料_设定2_严格3s_零宽限.pdf`，SHA-256 为 `87bf7c808079652b7d76fa9a5de2b01097858c3914910d1b5c81fa0a6c32921d`。下文页码均为该 PDF 的印刷页码，不是 Word 页码。正文用章节和原句定位，避免 Word 与配套 PDF 分页不同。

## 正文修改

### 正文方法 2.2 当前接触的输出约定

定位：默认预测时域为 10 s 的段落，原句：

> Current OBB interior overlap is assigned +∞ because no acceleration applied at the current instant can remove an existing collision.

替换为：

> Current OBB interior overlap is assigned +∞ because no acceleration applied at the current instant can remove an existing collision. For the numerical estimator, current OBB contact is also conservatively assigned +∞; this is an output convention distinct from the tangential-contact boundary allowed in the ideal definition.

同段中有关搜索上限的原句可以保留：

> If no feasible magnitude is found within a_max, the component and hence the aggregate are reported as +∞ rather than clipped to the bound.

这样只补充数值估计的当前接触约定，不改变正文式 (5)–(8) 的连续理想定义及证明。不要把此处的数值约定全局改写成“理想 EA 对任何相切均为无穷”。

### 正文方法 2.2 搜索与最终复核

定位：式 (15) 前的段落，原文：

> For each sampled direction, we use an interval search based on the Separating Axis Theorem (SAT) to find the smallest acceleration magnitude verified as collision-free on the numerical time grid. Repeating this calculation over a finite direction set Φ gives

本段在代码修正后可以保留。若希望明确最终采用细网格，只替换其中 `on the numerical time grid` 为 `on the fine temporal grid` 即可；不需要重写定义或式 (15)。

## 补充材料修改

### 第 2 页 1a.2 当前碰撞与零值

定位：式 (7) 后，以 `Collision means overlap of OBB interiors` 开头的整段。

原文：

> Collision means overlap of OBB interiors; tangential contact belongs to the boundary of the closed collision-avoiding set. If the OBB interiors already overlap at s = 0, EA is +∞ by definition because an acceleration applied at the current instant cannot alter that configuration. If 0 ∈ S_m,T, EA is zero. The numerical implementation uses closed contact-or-overlap intervals to locate this boundary and then verifies a candidate immediately on its collision-free side. Thus, the conceptual contact convention and the finite-precision collision test are stated separately.

整段替换为：

> In the ideal definition, collision means overlap of OBB interiors, while tangential contact belongs to the boundary of the closed collision-avoiding set. Existing interior overlap at s = 0 gives EA = +∞; otherwise, the ideal EA is zero when the zero vector belongs to S_m,T. The numerical estimator conservatively reports +∞ for current OBB contact or overlap. For states without current contact, it uses closed contact-or-overlap intervals to locate future exclusion boundaries and accepts a finite candidate only after a collision-free SAT check on the fine temporal grid. The ideal boundary convention and the numerical current-contact rule are therefore distinguished explicitly.

排版时保留原稿的原生数学对象 `s = 0`、`S_m,T` 等，不要把本清单的纯文本下标直接当成最终公式排版。

### 第 3 页 1a.3 射线边界候选的验证

定位：式 (13) 后，从 `The candidate ... is the next representable floating-point number` 到 `A direction is infeasible ...` 的三句。

原文：

> The candidate ρ̂_m(ϕ) is the next representable floating-point number greater than b_m,ϕ and is accepted only after a collision-free SAT check. If that check fails on the fine temporal grid, the next connected exclusion boundary is examined. A direction is infeasible when no verified candidate exists in [0, a_max].

替换为：

> A candidate is first placed at the next representable floating-point value above the outgoing exclusion boundary and checked against the translated OBBs on the fine temporal grid. If boundary rounding still produces contact, the candidate is advanced by a floating-point-scaled increment and checked again. The search proceeds through subsequent exclusion components when necessary. A direction is reported as unresolved if no verified candidate is found within [0, a_max].

该表述对应修正后的实际程序。浮点尺度步进不是额外的车辆安全距离，也不是把加速度整体增加 10⁻³ m/s²。原来的区间连通容差解释保留。

### 第 3 页 1a.3 粗到细流程与备用搜索

定位：以 `Algorithm 1 summarises the estimator` 开头、以 `No candidate obtained only on the coarse temporal grid is reported` 结尾的整段。

原文：

> Algorithm 1 summarises the estimator. A full-circle coarse search identifies the best angular neighbourhood. The candidate at the best coarse direction and all local directions are then recomputed on the fine temporal grid. If local refinement returns no finite candidate, the verified coarse-direction candidate is retained. No candidate obtained only on the coarse temporal grid is reported.

替换为：

> Algorithm 1 summarises the estimator. A full-circle coarse search identifies a local angular neighbourhood. The best coarse direction and all local directions are solved again on the fine temporal grid, with a final SAT check for every finite candidate. Baseline collisions found only on the fine grid are localised using a full-circle search on that grid. If the coarse search or local fine search produces no feasible candidate, a full-circle fine-grid search provides a verified fallback. No candidate obtained only on the coarse temporal grid is reported.

这一修改没有弱化细网格验证要求，只说明新增的备用搜索。`fine_refine_ratio = 1.15` 不应再被写成运行条件；它只作为兼容参数保留在 Python 接口中。

### 第 3 页 1b.1 当前状态相切的判定

定位：标题 `Collision detection` 下，含式 (17) 的整段。原开头为：

> Observed collision filtering follows the interior-overlap convention in Eq. (7).

原结尾为：

> Equality is tangential contact and is not interior overlap. This observed-state test is distinct from the closed contact-or-overlap intervals used only to locate a future collision-avoidance boundary.

建议用下面的段落替换，并保留式 (17) 的 `≥` 作为理想内部重叠的几何性质：

> For the ideal interior-overlap convention in Eq. (7), the boxes have disjoint interiors if at least one SAT axis satisfies Eq. (17). Equality denotes tangential contact. The numerical current-state test is more conservative: contact or overlap is flagged unless an axis satisfies |dᵀn| > R(n) + ε_geom, where ε_geom = 10⁻¹² m is a floating-point tolerance rather than a physical safety margin. A flagged current state receives a numerical EA of +∞. Future exclusion boundaries are handled by the interval search and candidate verification described above.

如果不希望增加新符号，也可把 `ε_geom` 的半句写成 `with a 10⁻¹² m floating-point tolerance`；二者只选其一，无需重复。

### 第 4 页 Algorithm 1

第 1 行将：

> if the observed OBB interiors overlap at s = 0 then

替换为：

> if current OBB contact or overlap is detected at s = 0 then

为覆盖粗搜索失败的细网格救援，原第 12–18 行需小幅调整。推荐完整伪代码如下，避免只改某一行留下前后矛盾：

```text
Input: states of A and B; T, Δt_c, Δt_f, Φ_c, δϕ, N_l, a_max
Output: computed EA

if current OBB contact or overlap is detected:
    return +∞
for m in {CV, CTRV}²:
    propagate the baseline OBBs on the coarse and fine temporal grids
    if zero added acceleration is collision-free on the fine grid:
        e_m ← 0; continue
    search Φ_c on the coarse grid for a localisation direction
    if the coarse baseline is collision-free or no finite coarse candidate is found:
        search Φ_c on the fine grid for a localisation direction
    if no finite localisation candidate is found:
        e_m ← +∞; continue
    form N_l local directions around the localisation direction
    solve the localisation direction and every local direction on the fine grid
    retain only candidates that pass the SAT collision check
    if no verified candidate is found:
        search Φ_c again on the fine grid and retain verified candidates
    e_m ← the smallest retained magnitude, or +∞ if none is found
return the arithmetic mean of the four e_m values
```

默认细时间步长小于粗时间步长。代码允许自定义参数，并使用 `min(Δt_c, Δt_f)` 作为最终验证步长，避免误设参数使最终网格更稀；默认和文中现有敏感性设置不受影响，不必为此扩充正文。

### 第 5 页 1c 输出约定

定位：Supplementary Table 1 下第一段末部：

> Existing interior overlap is represented by +∞; a search failure within a_max is also represented by +∞ rather than clipped to the cap.

替换为：

> Current contact or overlap is represented by +∞ in the numerical estimator; an unresolved search within a_max is also represented by +∞ rather than clipped to the cap.

同页下一段的原句：

> Positive infinity means either current interior overlap or no feasible candidate within the stated magnitude domain.

替换为：

> Positive infinity means either current contact or overlap, or failure to find a verified candidate within the prescribed search domain.

“零值是在细网格上无基线碰撞”“有限正值是搜索得到并验证的候选”两句可以保留：修正后的实现已执行对应检查。

## 保持不变的内容

- 二维恒定相对加速度及二次平移，不改变预测的基线朝向。
- CV/CTRV 传播及接近零 yawrate 时的直线分支。
- 四个有序模态的标量等权平均，不是四个最优向量求平均。
- 10 s 预测时域、0.10/0.02 s 网格、72/101 个方向、±5° 局部范围和 100 m/s² 搜索上限。
- 理想可行集、单调时域收敛证明及尾部界。
- 最新投稿版已经统一的 yawrate 输入说明。
- 论文实验图表和统计数值。本次不直接覆盖，须另行核查修正对真实实验数据的影响。
