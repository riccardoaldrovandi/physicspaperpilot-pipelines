# PhysicsPaperPilot v0.3.1 — benchmark pilota senza LLM giudice

> Release v0.3.1: correzioni a sezioni di evidenza, key point e scoring rispetto a v0.3. Dettagli in `CHANGELOG_v0.3.1.md`.

**15 paper · 45 domande · 15 multiple choice · 23 short answer · 7 open-ended**

Valutazione interamente automatica: nessun LLM generativo viene usato per assegnare punteggi.

## Struttura della cartella

- `01_questions_for_models/questions_public.csv`: 45 domande senza risposte.
- `02_private_answer_key/answer_key_private.csv`: risposte gold, key points originali, target numerici, evidenze e gruppi di sezioni accettabili. **Non consegnare ai modelli testati.**
- `02_private_answer_key/keypoints_lexicon.csv`: key points atomici, ID e frasi accettate dal matcher deterministico.
- `02_private_answer_key/evidence_annotations_template.csv`: template per aggiungere pagine, paragraph/span ID e citazioni testuali verificate.
- `02_private_answer_key/KEYPOINT_ANNOTATION_GUIDELINES.md`: istruzioni per revisionare il lessico dei concetti gold.
- `03_pilot_run/predictions_template.csv`: un record per domanda; duplicare per ogni modello.
- `03_pilot_run/prompt_for_tested_models.md`: prompt uniforme per ottenere risposta ed evidenze.
- `03_pilot_run/README_SCORING_IT.md`: formule, limiti e procedura.
- `scripts/evaluate_v3.py`: valutatore automatico senza LLM giudice.
- `tests/selftest_v3.py`: test di coerenza dello scoring su risposte gold sintetiche.
- `SCORING_CONFIG.json`: configurazione esplicita delle regole di scoring.
- `VALIDATION_STATUS.md`: verifiche tecniche eseguite e validazioni scientifiche ancora necessarie.
- `scripts/build_assets.py`: script che costruisce il lessico e i gruppi di evidenza della release a partire dalla answer key candidate.
- `REFERENCES.md`: riferimenti delle metriche.

## Procedura 

1. Selezionare 4 modelli e registrare versione/snapshot, data e parametri.
2. Fornire a tutti lo stesso paper/versione indicata in `papers.csv` e la stessa domanda.
3. Chiedere al modello di restituire `ANSWER` e intestazioni esatte delle sezioni in `EVIDENCE`.
4. Copiare la risposta in `model_answer` e le sezioni in `cited_evidence` (separate da `;`). Per le multiple choice compilare anche `mcq_choice`.
5. Eseguire:

```bash
python scripts/evaluate_v3.py --input 03_pilot_run/predictions_template.csv --outdir 04_results
```

6. Consultare `summary_by_model.csv`, `summary_by_paper.csv`, `summary_by_question_type.csv` e `scored_answers.csv`.

## Metriche attive

| Tipo | Metriche |
|---|---|
| Multiple choice | Accuracy |
| Short answer | Normalized Exact Match, Token Precision/Recall/F1, key-point coverage |
| Open-ended | Token F1, ROUGE-L, key-point coverage |
| Tutte | Evidence-section P/R/F1 sulle intestazioni citate |
| Domande con numeri gold | Numeric target coverage entro tolleranze |
| Short/open, opzionale | BERTScore con encoder locale |

**Non c'è un punteggio globale unico.** Accuracy, similarità testuale, key-point coverage, localizzazione delle sezioni e correttezza numerica sono riportate separatamente.

## Attenzione: che cosa è e che cosa non è misurato

- `keypoint_coverage` è una misura automatica di copertura/recall dei key points, basata su frasi e varianti lessicali predefinite. Non identifica tutte le affermazioni false aggiuntive e quindi non è key-point precision/F1 né factuality.
- `evidence_section_f1` misura se il modello cita le sezioni gold. **Non è ancora passage-level Evidence F1**: il gold attuale non contiene per tutte le domande paragrafi o frasi di evidenza verificati.
- EM, Token F1, ROUGE-L e BERTScore confrontano la risposta con la reference; non certificano la correttezza fisica.
- La answer key e il lessico sono ancora candidate: prima di pubblicare risultati occorre una revisione indipendente da parte di esperti.

Per dettagli e formule leggere `03_pilot_run/README_SCORING_IT.md`.
