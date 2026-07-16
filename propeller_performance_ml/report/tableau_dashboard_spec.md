# Tableau Dashboard Specification — UAV Propeller Performance

**Data source:** `data/processed/tableau_extract.csv` (27,495 rows, 22 fields).
Connect Tableau directly to this file (Text file connection) and, optionally,
create a `.hyper` extract for speed.

A rendered reference of the finished dashboard is
`outputs/figures/10_dashboard.png` (built by `src/06_tableau_dashboard.py`).

## Business question the dashboard answers
> *Which propellers deliver the most efficient thrust for a UAV, and what blade
> geometry and operating point drive that efficiency?*

## Key fields to use
| Field | Role | Notes |
|-------|------|-------|
| `propeller_name`, `propeller_brand`, `blade_class` | Dimensions | filtering / colour |
| `advanced_ratio_input` (J) | Continuous | primary x-axis of the physics story |
| `efficiency_clipped` | Measure | efficiency capped to the physical band |
| `peak_efficiency`, `j_at_peak_eff` | Measure | one value per propeller |
| `thrust_coefficient_output`, `power_coefficient_output` | Measure | trade-off |
| `solidity`, `pitch_to_diameter` | Measure | geometric drivers |
| `working_state` | Dimension | Working vs Windmill/brake filter |

## Recommended worksheets → dashboard tiles
1. **KPI band** (5 text/BAN tiles): `COUNTD(propeller_name)`, `COUNT()` rows,
   `MAX(peak_efficiency)`, `MEDIAN(peak_efficiency)`, `MEDIAN(solidity)`.
2. **Efficiency curve** — line: `efficiency_clipped` vs `J`, detail =
   `propeller_name`, filter `working_state = Working`. *Story: efficiency
   rises, peaks near J≈0.6–0.8, then collapses.*
3. **Brand ranking** — horizontal bar: `AVG(peak_efficiency)` by
   `propeller_brand`, sorted descending, Top-12 filter.
4. **Geometry driver** — scatter: `peak_efficiency` vs `pitch_to_diameter`,
   colour = `blade_class`. *Story: higher pitch/diameter → higher peak
   efficiency.*
5. **Solidity view** — scatter: `peak_efficiency` vs `solidity`, colour =
   `blade_class`. *Story: within a blade class, solidity is a weak driver;
   blade count matters more.*
6. **Thrust–power trade-off** — density/heatmap of `C_T` vs `C_P`.
7. **Blade-count box plot** — `peak_efficiency` by `blade_class`.

## Interactivity (data storytelling)
- Dashboard **filter actions**: selecting a brand (tile 3) filters tiles 2, 4,
  5, 7.
- **Highlight action** on `blade_class` across all tiles.
- A `working_state` quick filter (defaults to *Working*) so the audience never
  sees the misleading windmill-state efficiency spikes.
- Add a **Story** with three points: (1) *What we measured*, (2) *What makes a
  propeller efficient* (pitch/diameter + operating at the right J), (3) *The
  2-blade sweet spot* for these small-UAV propellers.

## Colour / formatting
- Blade class palette: 2-blade `#4C72B0`, 3-blade `#DD8452`, 4-blade `#55A868`.
- Keep gridlines light; label KPI tiles boldly; title every sheet with the
  *insight*, not the field names.
