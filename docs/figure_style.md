# Fig. 1-aligned style for the new examples

The original vector Fig. 1 remains unchanged. Inspection of its embedded text
and vector fill colors found predominantly Arial, with Cambria Math for some
formulas, small bold labels, and one Calibri character. For the new scientific
plots, we use its dominant Arial face at regular weight throughout, consistent
with the project's no-bold figure convention. This is a harmonized plotting
style, not a claim that every original PDF glyph uses one font.

The shared implementation is `src/evasive_acceleration/plotting.py`.
Matplotlib is an optional dependency, loaded only on use; importing the EA
calculation API still does not load the plotting stack.

| Role | Hex color |
|---|---|
| EA accent / stronger coral | `#D66A5B` |
| Light coral fill | `#E89F8C` |
| Muted green / actor A | `#81B199` |
| Light green fill | `#A5C9B5` |
| Muted gold / actor B | `#F1CB8D` |
| Auxiliary slate | `#7A8B8C` |
| Body text | `#404040` |
| Muted borders | `#A6A6A6` |
| Warm panel background | `#FCF9F4` |
| Grid | `#D9D9D9` |

Colors are rounded from the vector RGB values to 8-bit hex. New charts avoid
gradients, bold headings, and mixed-unit overlays. EA and TTC use separate
vertical axes/panels; colors identify quantities, not calibrated risk levels.

```python
from evasive_acceleration.plotting import FIG1_COLORS, figure_style

with figure_style() as font_provenance:
    # Create and save a Matplotlib figure here.
    print(font_provenance)
```

The style is a temporary rcParams context, not a persistent global override.
Font provenance identifies the requested/actual family, regular font file,
and whether substitution occurred.

## Arial availability

Regular Arial is required by default. The example export on the development
machine uses its installed Arial; fonts are not copied into this repository.
On another machine, set `EA_ARIAL_FONT_PATH` to an existing regular Arial TTF
that you have permission to use. Invalid paths and non-Arial files are errors.

There is no automated font download, license acceptance, or disguised fallback.
`EA_ALLOW_FONT_FALLBACK=1` explicitly enables a warned preview-only substitute
(Liberation Sans, otherwise DejaVu Sans). CI uses this mode because hosted Linux
runners do not normally include Arial. A preview passing CI does not establish
publication-font compliance; final figure export must pass strict Arial mode.

Notebook prose/table CSS follows the resolved family and regular weight.
Code editors and GitHub Markdown retain the viewer's own code/UI typography.
The shipped comparison image and locally executed notebook use actual Arial.
