# Commercial Projects

The open-source `mlconfgen` libraries and weights are also the foundation of several commercial products. This page lists projects that use the same family of models and APIs.

---

## Quantori MLConfGen

**[Commercial Inference Server](https://quantori.com/mlconfgen)**

Quantori offers a commercial, production-ready deployment of MLConfGen for spatially aware small-molecule generation in drug-discovery workflows: SaaS, on-prem, and cloud (including GCP and AWS), with REST and MCP integration.

Capabilities highlighted on the product site include:

- Spatially conditioned generation without heavy shape descriptors
- Fixed-fragment / scaffold-anchored generation
- Objective-guided fine-tuning against custom scoring functions
- Lightweight models suitable for modest hardware and pipeline integration (docking, geometry optimisation, free-energy methods)
- Integrated tools to generate pocket shapes and 3D Editor

---

## Trust Me, It Binds

**[View on Steam](https://store.steampowered.com/app/5119610/Trust_Me_It_Binds/)**

*Trust Me, It Binds* is an indie procedural deckbuilder that embeds on-device molecule generation during play. The molecules in the game are generated using MLConfGen; the game framing turns real chemistry into a satirical AI drug-design startup simulator . See the [Steam store page](https://store.steampowered.com/app/5119610/Trust_Me_It_Binds/) for more details. Purchasing the game is a nice way to support the project.

---

## Relationship to this repository

| | Open-source repo | Commercial products |
|---|---|---|
| Code | Apache 2.0 on GitHub | Product-specific packaging and infrastructure |
| Weights | Hugging Face (see model card) | hosted as part of the product |
| APIs | Python (`mlconfgen`) and JavaScript (`mlconfgen` on npm) | REST, MCP, cloud UIs (Quantori); embedded ONNX / local inference (Steam game) |

Using the open-source libraries does not require or include access to the commercial offerings above. Conversely, the commercial products ship additional tooling, hosting, and support beyond what is documented here.
