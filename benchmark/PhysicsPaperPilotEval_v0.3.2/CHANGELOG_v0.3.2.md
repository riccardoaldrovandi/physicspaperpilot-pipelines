# PhysicsPaperPilot v0.3.2 — changelog rispetto a v0.3.1

Data: 2026-11-25. Base: v0.3.1. Cambia solo l'evidenza delle 15 domande a scelta multipla (MCQ).

Le domande MCQ chiedono quale metodo o compito tratta il paper (14 su 15 sono di identificazione). L'informazione
si trova nell'abstract, nell'introduzione e nella sezione che presenta il metodo, ma il gold v0.3.1 accettava quasi
solo "Abstract". In v0.3.2 il gruppo di evidenza di ogni domanda MCQ (sempre uno) accetta in più, come
alternative: l'Introduzione e la sezione in cui il paper presenta il metodo nominato nella domanda, con tutte le
sue sottosezioni. I titoli sono quelli dell'indice del PDF (versione di `papers.csv`), non le citazioni dei modelli.
Per PHY-241021611-Q01 (ruolo del paper, nessun metodo) si aggiunge solo l'Introduzione.

## Answer key (`02_private_answer_key/answer_key_private.csv`, colonna `evidence`)

| Domanda | v0.3.1 | Aggiunte in v0.3.2 |
|---|---|---|
| PHY-170502355-Q01 | `II Method` | `Introduction`; `Method` |
| PHY-171210321-Q01 | `Abstract; III Generative Adversarial Networks; IV The CaloGAN` | `I Introduction`; `A Model Architecture`; `B Loss Formulation`; `C Training Strategy` |
| PHY-180500850-Q01 | `Abstract` | `1 Introduction`; `2 Inputs and problem formulation` |
| PHY-180701954-Q01 | `Abstract` | `1 Introduction`; `3 Fast simulation approach`; `3.1 Generative adversarial networks (GANs)`; `3.2 Wasserstein GANs`; `3.3 Label conditioning`; `3.4 Strategy and network training` |
| PHY-210605285-Q01 | `Abstract` | `I Introduction`; `II Density Estimation with Normalizing Flows`; `IV CaloFlow`; `A Flow I: learning the energy depositions per layer`; `B Flow II: learning the shower shapes`; `C Sampling from CaloFlow` |
| PHY-220611898-Q01 | `Abstract; Introduction` | `I Introduction`; `II Generative models using gradients of the data` |
| PHY-221006204-Q01 | `Abstract` | `1 Introduction`; `4 Models`; `4.1 Variational autoencoders`; `4.2 Generative adversarial networks`; `4.3 Comparison with prior work` |
| PHY-221007430-Q01 | `Abstract; 2 Methodology` | `1 Introduction`; `2.1 Dataset and Preprocessing`; `2.2 Deep Generative Models`; `2.3 CaloDVAE` |
| PHY-221115380-Q01 | `Abstract; 2 Method` | `1 Introduction`; `2.1 Learning the Manifold of Calorimeter Showers`; `2.2 Intrinsic Dimension Estimation` |
| PHY-230504847-Q01 | `3 Model` | `Introduction`; `Model`; `Diffusion Point Cloud Generator`; `Encoder & Latent Flow`; `Shower Flow & Post-Diffusion Calibration` |
| PHY-230511934-Q01 | `Abstract; III iCaloFlow` | `Introduction`; `iCaloFlow`; `Overview`; `Architectures`; `Teacher MAFs`; `Student IAFs`; `Training`; `Teachers`; `Students` |
| PHY-230803876-Q01 | `Abstract` | `Introduction`; `Diffusion Models`; `Methods`; `Preprocessing`; `Diffusion Specifics`; `Network Architecture` |
| PHY-230811700-Q01 | `Abstract; III SuperCalo` | `Introduction`; `SuperCalo`; `Coarse voxelization`; `Architecture and training`; `Comparison with other approaches`; `Comparing coarse voxelization choices` |
| PHY-240211575-Q01 | `Abstract` | `Introduction`; `Model description` |
| PHY-241021611-Q01 | `Abstract` | `Introduction` |

`gold_evidence_groups_json`, `gold_evidence_sections_json` e `evidence_annotations_template.csv` sono
rigenerati con `scripts/build_assets.py` (51 gruppi di evidenza, come in v0.3.1). Domande, risposte gold, key point,
lessico e `scripts/evaluate_v3.py` sono invariati; `tests/selftest_v3.py` passa.

## Limite noto

Accettare tutte le sottosezioni della sezione del metodo è generoso: alcune non sono pertinenti alla domanda
(es. `2.1 Dataset and Preprocessing` per PHY-221007430-Q01). Con questa scelta l'evidence F1 delle MCQ della
pipeline A sale da 29–52% a 84–99% e, come l'accuratezza MCQ, distingue poco i modelli.
