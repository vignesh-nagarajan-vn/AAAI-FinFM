# AAAI-FinFM

- **Target Venue:** [FinFM: Foundation Models and Generative AI for Finance](https://finfm.finhorizons.org/), a workshop at [AAAI 2027](https://aaai.org/conference/aaai/aaai-27/) in Montréal
  - **Dates:** Paper due Nov 20, 2026 AoE (Nov 21, 11:59 UTC). Notification Dec 2. Workshop Feb 22 or 23, 2027.
  - **Format:** Short paper, research track: 4 pages of technical content in the AAAI-27 template, double-blind on OpenReview. References and the appendix do not count toward it.
  - **Status:** Non-archival, with no proceedings or DOI. One co-author must present in person.
- **Source Project:** [World-Models](https://github.com/vignesh-nagarajan-vn/World-Models), the full work for ICML 2027, which permits non-archival workshop papers.
- **Paper:** [`paper/main.pdf`](paper/main.pdf), anonymized. Main text ends on page 4, then references and the proof appendix.

This repo holds a 4-page theory paper drawn from the World-Models project. The project builds world models for credit: learned simulators of how a lending system evolves, with decentralized (on-chain) lending as the testbed. The paper asks how far a learned simulator can miss a liquidation cascade, given its error in the liquidation map, and answers with the math alone. Every result is proved in the appendix and checked numerically.

## Paper Scope

- **Cascade prediction error (main result).** Map error acts as shock error, so cascade error is at most L<sub>φ</sub>η/(1−λ̄). The bound is sharp and fails at liquidation walls; breach shocks stay stable. Aggregates have an error floor.
- **Constants and interventions.** The price impact implied by a liquidator selling on a pool, the smoothing tolerance, and a χ² bound on when model error can pass for behavior change. Only proved bounds are included.
- **Figures.** Figure 1 draws how map error becomes cascade error. Figure 2 checks the bounds on 320 random pairs of books, and Figure 3 checks the impact constant against simulated arbitrage. None uses market data.

Left out: all model experiments, since results so far are synthetic and real chain data is not loaded yet. Also left out are the two smoothing and price-impact predictions that did not hold on learned models.

## Build

```bash
python scripts/make_figures.py       # writes paper/figures/*.pdf (numpy, matplotlib, LaTeX)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The 40 numerical checks the paper cites live in the source project's `math-derivations/` folder.

## Repo Roadmap

```
AAAI-FinFM/
├── README.md                       venue details, scope, build
├── scripts/make_figures.py         seeded script for Figures 1 to 3
└── paper/
    ├── main.tex                    the paper, AAAI-27 anonymous submission format
    ├── main.pdf                    compiled anonymized PDF
    ├── references.bib              cited entries from the source project's bibliography
    ├── aaai2027.sty, aaai2027.bst  AAAI-27 author kit style files
    └── figures/                    the three figure PDFs
```
