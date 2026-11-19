# AAAI-FinFM

- **Target Venue:** [FinFM: Foundation Models and Generative AI for Finance](https://finfm.finhorizons.org/), a workshop at [AAAI 2027](https://aaai.org/conference/aaai/aaai-27/), Montréal, Feb 22 or 23, 2027
  - **Deadlines:** Paper due Nov 20, 2026 (AoE). Notification Dec 2, 2026.
  - **Format:** AAAI-27 template, double-blind on OpenReview. Short papers up to 4 pages, long papers up to 7. References and appendices are unlimited.
  - **Status:** Non-archival, with no proceedings or DOI. One co-author must present in person.

This repo holds a short workshop version of the World-Models project. The project builds world models for credit: learned simulators of how a lending system evolves given its state and the external factors acting on it. Decentralized (on-chain) lending is the testbed, because every loan, liquidation, and rule change is public and timestamped. The full version targets ICML 2027, which allows prior presentation at non-archival workshops.

The world model has three layers: an exact re-implementation of the protocol's rules, learned policies for borrowers, liquidators, and arbitrageurs, and a sparse learned interface for external factors. All experiments so far run on a synthetic world; no real chain data has been loaded yet.

## Research Questions

- **RQ1: Interventional validity.** Can a world model trained on observational on-chain lending data correctly predict the effects of interventions it never saw in training?
- **RQ2: Robustness under regime shift.** Does a sparse, learned interface for external factors help world models generalize to unseen crisis regimes better than models with no external inputs or with all inputs fed in naively?
- **RQ3: Tail dynamics from agent structure.** Can a multi-agent world model of learned borrower, liquidator, and arbitrageur policies reproduce the tail statistics of liquidation cascades that aggregate, single-agent models miss?

## Repo Roadmap

```
AAAI-FinFM/
└── README.md      venue details, project summary, RQs, this roadmap
```
