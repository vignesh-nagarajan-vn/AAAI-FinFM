# AAAI-FinFM

- **Target Venue:** [FinFM: Foundation Models and Generative AI for Finance](https://finfm.finhorizons.org/), a workshop at [AAAI 2027](https://aaai.org/conference/aaai/aaai-27/), Montréal, Feb 22 or 23, 2027
  - **Deadlines:** Paper due Nov 20, 2026 (AoE). Notification Dec 2, 2026.
  - **Format:** Short paper, research track: 4 pages of technical content in the AAAI-27 template, double-blind on OpenReview. References and appendices are unlimited.
  - **Status:** Non-archival, with no proceedings or DOI. One co-author must present in person.
- **Source Project:** [World-Models](https://github.com/vignesh-nagarajan-vn/World-Models), the full work, which targets ICML 2027. ICML allows prior presentation at non-archival workshops.

This repo holds a 4-page theory paper drawn from the World-Models project. The project builds world models for credit: learned simulators of how a lending system evolves, with decentralized (on-chain) lending as the testbed. The paper asks when such a model's predictions can be trusted, and answers with the math alone. Every result is proved, and each has a numerical check in the source repo.

## Paper Scope

- **Cascade prediction error (main result).** Liquidation-map error acts as shock error, so cascade error is at most $L_\varphi\eta/(1-\bar\lambda)$. The bound is sharp and fails at liquidation walls; breach shocks stay stable. Aggregates have an error floor.
- **Evaluating interventions.** Composition and behavioral effects of a governance change, a placebo test, and a χ² overlap bound on when model error can pass for behavior change. All of it is cited technique, framed for this setting.
- **Constants from the model layers.** The outcome error from smoothing the liquidation threshold, and the price impact implied by a profit-gated liquidator. Only the proved bounds are included.

Left out: all experiments, since results so far are synthetic and real chain data is not loaded yet. Also left out are the two smoothing and price-impact predictions that did not hold on learned models.

## Repo Roadmap

```
AAAI-FinFM/
└── README.md      venue details, source project, paper scope, this roadmap
```
