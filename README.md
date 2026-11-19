# AAAI-FinFM

- **Target Venue:** [FinFM: Foundation Models and Generative AI for Finance](https://finfm.finhorizons.org/), a workshop at [AAAI 2027](https://aaai.org/conference/aaai/aaai-27/), Montréal, Feb 22 or 23, 2027
  - **Deadlines:** Paper due Nov 20, 2026 (AoE), which is Nov 21 at 11:59 UTC. Notification Dec 2, 2026.
  - **Format:** Short paper, research track: 4 pages of technical content in the AAAI-27 template, double-blind on OpenReview. References and appendices are unlimited.
  - **Status:** Non-archival, with no proceedings or DOI. One co-author must present in person.
- **Source Project:** [World-Models](https://github.com/vignesh-nagarajan-vn/World-Models), the full work, which targets ICML 2027. ICML allows prior presentation at non-archival workshops.
- **Paper:** [`paper/main.pdf`](paper/main.pdf), anonymized. Main text ends at the top of page 4, followed by references and the proof appendix.

This repo holds a 4-page theory paper drawn from the World-Models project. The project builds world models for credit: learned simulators of how a lending system evolves, with decentralized (on-chain) lending as the testbed. The paper asks how far a learned simulator can miss a liquidation cascade, given its error in the liquidation map, and answers with the math alone. Every result is proved in the appendix and checked numerically.

## Paper Scope

- **Cascade prediction error (main result).** Liquidation-map error acts as shock error, so cascade error is at most $L_\varphi\eta/(1-\bar\lambda)$. The bound is sharp and fails at liquidation walls; breach shocks stay stable. Aggregates have an error floor.
- **Constants and interventions.** The price impact implied by a liquidator selling on a pool, the smoothing tolerance, and a χ² bound on when model error can pass for behavior change. Only proved bounds are included.
- **Figure 1.** Size error against breach-shock error on 320 random pairs of liquidation books. It is a numerical check of the bounds, not market data.

Left out: all model experiments, since results so far are synthetic and real chain data is not loaded yet. Also left out are the two smoothing and price-impact predictions that did not hold on learned models.

## Build

```bash
python scripts/make_figure.py        # writes paper/figures/cascade_error.pdf (numpy, matplotlib)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The 40 numerical checks the paper cites live in the source project's `math-derivations/` folder.

## Before Submitting

- Upload `paper/main.pdf` to the FinFM venue on OpenReview, with every author's OpenReview profile attached.
- Code is optional supplementary material. If attached, strip names and repo links first, since review is double-blind.
- Do not mention ICML anywhere in the submission or the poster.

## Repo Roadmap

```
AAAI-FinFM/
├── README.md                     venue details, scope, build, submission notes
├── scripts/make_figure.py        seeded script for Figure 1
└── paper/
    ├── main.tex                  the paper, AAAI-27 anonymous submission format
    ├── main.pdf                  compiled anonymized PDF
    ├── references.bib            cited entries, copied from the source project's verified bibliography
    ├── aaai2027.sty, aaai2027.bst  AAAI-27 author kit style files
    └── figures/cascade_error.pdf Figure 1
```
