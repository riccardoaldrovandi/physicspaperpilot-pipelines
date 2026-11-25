# Scoring — interamente automatico, senza LLM giudice

## 1. Principio

Tutti i punteggi dell'Esperimento 1 sono calcolati da `scripts/evaluate_v3.py`. Lo script non interroga API e non usa un LLM generativo come giudice. Confronta le risposte con la answer key, un lessico di key points e le sezioni di evidenza annotate.

L'unica componente opzionale basata su un modello pre-addestrato è BERTScore: usa un encoder per calcolare similarità contestuale, senza generare testo e senza assegnare giudizi in linguaggio naturale.

## 2. Metriche per tipologia di domanda

| Tipologia | Metrica | Interpretazione |
|---|---|---|
| Multiple choice (15) | Accuracy | Quota di opzioni corrette |
| Short answer (23) | Normalized Exact Match | Corrispondenza normalizzata con la risposta di riferimento |
| Short answer (23) | Token Precision / Recall / F1 | Sovrapposizione di token con la risposta gold, in stile SQuAD/QASPER |
| Short answer + open-ended | Key-point coverage (recall) | Quota dei concetti gold riconosciuti nella risposta tramite lessico deterministico |
| Open-ended (7) | ROUGE-L P/R/F1 | Somiglianza lessicale/ordinata rispetto al testo gold |
| Short + open-ended | BERTScore P/R/F1 (opzionale) | Similarità semantica tramite encoder locale, non un giudice LLM |
| Tutte le domande | Evidence-section P/R/F1 | Precisione e copertura delle intestazioni di sezione citate rispetto alle sezioni gold |
| Domande con target numerici | Numeric target coverage | Valori attesi presenti entro tolleranze predefinite |

Le metriche restano separate. Non viene calcolato un unico score composito.

## 3. Key-point coverage: come funziona

Ogni domanda short/open ha uno o più key points atomici in `02_private_answer_key/keypoints_lexicon.csv`. Per ogni key point il file contiene:

- un ID stabile (`keypoint_id`);
- la descrizione del concetto atteso;
- frasi accettate e varianti lessicali (`accepted_phrases_json`).

Lo script normalizza Unicode, maiuscole/minuscole, punteggiatura e alcune notazioni scientifiche. Un key point è marcato come presente se trova nella risposta almeno una frase accettata oppure se supera il controllo conservativo sui token-ancora.

Formula:

`Key-point coverage = key points matched / total gold key points`

Questa quantità è una **coverage/recall deterministica**, non una precision o un F1 di claim. Lo script non può sapere automaticamente quali affermazioni extra non previste dal gold siano false. Per questo non si calcola key-point precision e non si chiama questa metrica factual correctness.

Il lessico è modificabile e deve essere revisionato: parafrasi non previste possono essere falsi negativi; alias troppo generici possono creare falsi positivi. Le varianti sono visibili nel CSV, quindi la valutazione è ispezionabile e riproducibile. Il fallback token-based accetta un key point quando almeno l'85% dei suoi token-ancora informativi (minimo tre) compare nella risposta, senza considerare l'ordine delle parole.

## 4. Evidence-section P/R/F1: che cosa misura davvero

Nella answer key v0.3 le evidenze preesistenti sono localizzate principalmente a livello di sezione; non tutte hanno ancora pagina, paragrafo o citazione testuale. Per questo lo script misura **section citation matching**, non il supporto semantico del testo e non la passage-level Evidence F1.

La answer key contiene gruppi di evidenza (`gold_evidence_groups_json`). Ogni gruppo corrisponde a una voce evidence originale. Quando la voce propone più intestazioni separate da `/`, queste sono trattate come alternative accettabili per quel gruppo.

Il modello deve citare le intestazioni esatte nella colonna `cited_evidence`, separate da punto e virgola. Lo script abbina le citazioni ai gruppi gold: una sezione può essere una delle alternative ammesse per un gruppo. Un gruppo è coperto se almeno una delle sue intestazioni alternative viene citata. Citazioni aggiuntive che corrispondono a intestazioni alternative dello stesso gruppo non vengono contate due volte; citazioni che non corrispondono a nessun gruppo sono considerate non pertinenti.

- TP: numero di gruppi gold coperti;
- FP: numero di citazioni che non corrispondono ad alcun gruppo gold;
- FN: numero di gruppi gold non coperti.

`Evidence-section Precision = TP / (TP + FP)`

`Evidence-section Recall = TP / (TP + FN)`

`Evidence-section F1 = 2 × P × R / (P + R)`

**Limite:** una citazione corretta del titolo della sezione non dimostra che il modello abbia usato il paragrafo corretto o che ogni frase della risposta sia supportata. Per una vera Evidence F1 a livello di passaggio occorre completare `evidence_annotations_template.csv` con pagina, paragraph/span ID e citazione esatta, poi richiedere al modello di produrre citazioni allineate allo stesso sistema di identificatori.

## 5. Metriche testuali

### Normalized Exact Match

Normalizza la risposta e il riferimento (minuscole, Unicode, punteggiatura e tokenizzazione). Vale 1 se coincidono, 0 altrimenti. Parafrasi valide possono ottenere zero.

### Token Precision / Recall / F1

I token in comune vengono contati con molteplicità, come nella valutazione QA di SQuAD. La metrica è riproducibile ma lessicale: sinonimi e parafrasi possono essere penalizzati.

### ROUGE-L

Usa la longest common subsequence. È una misura di sovrapposizione testuale, non una verifica di correttezza scientifica.

### BERTScore (opzionale)

Calcolato localmente con un encoder (default `roberta-large`). Non genera testo e non è un LLM-as-a-Judge. Richiede PyTorch/Transformers e il download iniziale dei pesi. Riportarlo separatamente da EM/F1/ROUGE.

## 6. Controllo numerico

Per le domande con `numeric_targets_json`, lo script estrae i numeri e li confronta con i valori gold entro `rel_tolerance` e `abs_tolerance`. Le tolleranze vanno controllate sul paper e congelate prima del confronto definitivo. Il controllo numerico da solo non verifica che il valore sia associato alla particella, all'unità o alla condizione corretta: per questo va letto insieme a key points e token metrics.

## 7. Esecuzione

Dalla cartella principale:

```bash
python scripts/evaluate_v3.py --input 03_pilot_run/predictions_template.csv --outdir 04_results
```

BERTScore opzionale:

```bash
pip install -r requirements_bertscore.txt
python scripts/evaluate_v3.py --input 03_pilot_run/predictions_template.csv --outdir 04_results --with-bertscore --bert-model roberta-large
```

Test interno dello scoring:

```bash
python tests/selftest_v3.py
```

## 8. Output

- `scored_answers.csv`: metriche e diagnostica per ogni risposta;
- `summary_by_model.csv`: medie macro per paper e modello;
- `summary_by_paper.csv`: risultati per paper;
- `summary_by_question_type.csv`: risultati per tipo di domanda;
- `run_warnings.txt`: risposte mancanti, ID duplicati/sconosciuti, citazioni mancanti.

Le medie aggregate vengono calcolate prima all'interno di ciascun paper e poi tra paper, così ogni paper ha lo stesso peso. Le metriche restano separate.

## 9. Limiti da riportare in ogni uso scientifico

1. La answer key e il lessico dei key points sono ancora candidate e richiedono revisione di dominio indipendente.
2. Key-point coverage dipende dal lessico accettato; una parafrasi non prevista può non essere riconosciuta.
3. Evidence-section F1 verifica l'indicazione della sezione, non la correttezza del supporto a livello di frase/paragrafo.
4. EM, Token F1, ROUGE-L e BERTScore sono metriche di similarità con risposte di riferimento, non factuality metrics.
5. Non viene calcolata Groundedness completa senza annotazioni di claim support/refute o valutazione umana.
6. Il corpus è concentrato sulla simulazione degli sciami nei calorimetri; non rappresenta l'intera fisica.
