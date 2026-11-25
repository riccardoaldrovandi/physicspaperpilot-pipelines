# PhysicsPaperPilot v0.3.1 — changelog rispetto a v0.3

Data: 2026-11-04. Base: release v0.3 (cartella `PhysicsPaperPilotEval`, lasciata invariata).
Motivazione e verifiche: `docs/PhysicsPaperPilot_v0.3.1.pdf` nel repository paper_analyzer.

Le 45 risposte gold sono state verificate sul testo dei PDF nelle versioni di `papers.csv`
(scaricate da arXiv e controllate: versioni identiche). Il contenuto scientifico delle risposte
è corretto; le modifiche riguardano le sezioni di evidenza, due key point, una domanda ambigua,
una risposta incompleta e lo script di scoring.

## Answer key (`02_private_answer_key/answer_key_private.csv`, colonna `evidence`)

| Domanda | v0.3 | v0.3.1 | Motivo |
|---|---|---|---|
| PHY-171210321-Q02 | `V Performance / V.1–V.3` | `V Performance / II Dataset / Abstract` | `V.1–V.3` non è un'intestazione; le tre particelle sono elencate anche in Dataset e Abstract |
| PHY-180500850-Q02 | `Input data / dataset description` | `2 Inputs and problem formulation` | le due etichette non esistono nel paper |
| PHY-180701954-Q02 | `Abstract / Experimental setup` | `Abstract / 2 Experimental setup / 2.2 Reference dataset` | la simulazione GEANT4 è descritta in 2.2 |
| PHY-210605285-Q02 | `IV CaloFlow` | `IV CaloFlow / IV.A Flow I: … / IV.B Flow II: …` | le due sottosezioni citate dalla domanda come alternative |
| PHY-210605285-Q03 | 2 gruppi: `Introduction / motivation`, `CaloFlow advantages` | 1 gruppo: `I Introduction / II Density Estimation with Normalizing Flows / VI Conclusions` | `motivation` e `CaloFlow advantages` non esistono; l'elenco dei vantaggi è nell'Introduction |
| PHY-230511934-Q01 | `Title / III iCaloFlow` | `Abstract / III iCaloFlow` | `Title` non è una sezione |
| PHY-230511934-Q02 (gruppo 2) | `III.2 Architectures` | `III.2 Architectures / Teacher MAFs / Student IAFs` | sotto-sottosezioni come alternative |
| PHY-230511934-Q03 | `III.2 Architectures / flow definitions` | `III.1 Overview` | Flow-3 e i suoi condizionamenti sono definiti in III.A Overview (Tabella I), non in Architectures; `flow definitions` non esiste |
| PHY-230803876-Q02 | `V Geometric Innovations` | `V Geometric Innovations / V.1 Optimizations for Cylindrical Geometry / V.2 GLaM: Geometry Latent Mapping` | sottosezioni come alternative |
| PHY-241021611-Q03 | `Abstract / challenge description` | `Abstract / Introduction` | `challenge description` non esiste |

`gold_evidence_groups_json`, `gold_evidence_sections_json`, `keypoints_lexicon.csv` e
`evidence_annotations_template.csv` sono rigenerati con `scripts/build_assets.py` (51 gruppi di evidenza invece di 52).

## Key point (colonna `key_points` e lessico)

- Rimossi PHY-180500850-Q03-KP04 ("Do not invent one fixed speed-up factor") e PHY-241021611-Q03-KP04
  ("Recognizes the paper as a challenge report rather than one new model"): sono istruzioni o giudizi sulla
  risposta, non contenuti del paper; una risposta corretta non li contiene e la key-point coverage li conta come mancanti.
  Il vincolo di non inventare un fattore di speed-up resta nel testo della domanda 1805.00850-Q03.
- Aggiunto PHY-240211575-Q03-KP05 ("A logit transform is applied, followed by re-standardization."), con alias in `build_assets.py`.
- Totale key point: 87 (v0.3: 88).

## Risposte e domande

- PHY-240211575-Q03, `gold_answer`: completata con la trasformazione logit (α = 10⁻⁶) e la nuova standardizzazione
  descritte nella stessa frase del paper (Sec. II.B).
- PHY-221115380-Q03, domanda (`questions_public.csv`): "Name one quantitative comparison…" diventa
  "Which quantitative metric does the paper report in its results table (Table 1)…". La versione v0.3 ammetteva più
  risposte corrette (anche l'AUC del classificatore, 0.78), ma i key point riconoscono solo la χ² separation power.

## Scoring (`scripts/evaluate_v3.py`)

1. **Sezioni**: un'intestazione citata corrisponde a quella gold anche se coincide il solo titolo senza numerazione
   (`A. Graph creation`, `II.A Graph creation`, `Graph creation` ↔ `II.1 Graph creation`; `1 Introduction` ↔ `Introduction`).
   Le etichette gold seguono la numerazione della versione HTML di arXiv, i PDF stampano lettere REVTeX, numeri arabi o nessun numero.
2. **Esponenti in apice**: `10⁵` e `10⁻⁶` sono convertiti in `10^5`, `10^-6` prima della normalizzazione Unicode
   (NFKC li trasformava in `105` e `10-6`), sia nel controllo numerico sia nel matching dei key point.
3. **Potenze di dieci nel testo normalizzato**: `10^-6` diventa `10 to minus 6` (il segno andava perso) e la regola non
   unisce più il numero alla parola seguente (`10^5 on` diventava `10 to 5on`).

## Test (`tests/selftest_v3.py`)

Aggiunti controlli unitari per i punti 1–3; il test sulle 45 risposte gold cita ora i titoli senza numerazione.

## Effetto misurato sulle risposte gold

| Input | v0.3 evidence-section F1 | v0.3.1 | v0.3 numeric coverage | v0.3.1 |
|---|---|---|---|---|
| Risposte gold, etichette gold esatte | 100 | 100 | 66.7 | 100 |
| Risposte gold, titoli senza numero | 58.9 | 100 | 66.7 | 100 |
| Risposte gold, intestazioni come stampate nel PDF | 62.6 | 100 | 66.7 | 100 |

## Invariato

Domande (salvo PHY-221115380-Q03), risposte MCQ, target numerici e tolleranze, formule delle metriche, aggregazione
(media per paper, poi media tra paper), assenza di punteggio composito e di LLM giudice. Lo stato di validazione resta
`candidate`: la revisione indipendente da parte di esperti di dominio è ancora necessaria.
