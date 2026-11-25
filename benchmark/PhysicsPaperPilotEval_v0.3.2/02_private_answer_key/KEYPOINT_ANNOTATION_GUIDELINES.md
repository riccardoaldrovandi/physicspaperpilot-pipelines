# Linee guida per revisionare il lessico dei key points

Il file `keypoints_lexicon.csv` traduce i concetti gold in regole di matching deterministiche. Lo script non usa un LLM per interpretare le risposte.

## Campi

- `question_id`: domanda di riferimento.
- `keypoint_id`: identificativo stabile del concetto.
- `key_point`: formulazione scientifica del concetto atteso.
- `accepted_phrases_json`: frasi o varianti che lo script considera sufficienti per riconoscere il concetto.
- `matching_rule`: descrizione del matching usato.
- `validation_status`: stato di revisione.

## Come avviene il matching

1. Lo script normalizza Unicode, simboli scientifici, maiuscole/minuscole e punteggiatura.
2. Se una frase accettata compare come sequenza di token nella risposta, il key point è considerato coperto.
3. In alternativa, lo script usa un controllo conservativo: almeno l'85% dei token-ancora informativi del key point (minimo tre) deve comparire nella risposta, senza richiedere lo stesso ordine.

## Come revisionare le varianti

- Aggiungere sinonimi tecnici realmente equivalenti nel contesto del paper.
- Aggiungere forme alternative della stessa notazione (per esempio `π+` e `pi plus`) quando hanno lo stesso significato.
- Non aggiungere parole troppo generiche (`model`, `energy`, `result`) come alias autonomi.
- Controllare che due key points distinti non siano riconosciuti dalla stessa espressione generica.
- Verificare ogni alias sulla risposta gold e su alcune parafrasi manuali valide e non valide.
- Dopo la revisione, cambiare `validation_status` in `reviewed_by_domain_annotator` e registrare annotatore/data in un changelog.

## Limite

La copertura è una misura di recall/coverage rispetto ai concetti gold. Non rileva automaticamente tutte le affermazioni extra scorrette del modello e non è precision o F1 di factual claims. Una parafrasi valida non inclusa nel lessico può essere un falso negativo; un alias troppo ampio può essere un falso positivo.
