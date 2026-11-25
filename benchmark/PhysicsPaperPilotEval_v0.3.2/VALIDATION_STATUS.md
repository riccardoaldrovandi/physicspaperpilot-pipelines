# Stato 

- 15 paper.
- 45 domande: 15 multiple choice, 23 short answer, 7 open-ended.
- 87 key points atomici associati alle domande short/open (v0.3: 88).
- 51 record di evidenza a livello di sezione (v0.3: 52).
- 51 righe nel template di annotazione puntuale delle evidenze.
- Nessun LLM generativo usato dallo script di scoring.

## Verifiche tecniche eseguite

`python tests/selftest_v3.py` viene eseguito con risposte gold sintetiche e citazioni di sezione previste. Controlla che:

- le multiple choice gold ottengano accuracy 1;
- tutte le risposte gold short/open coprano i key points associati secondo il lessico attuale;
- i riferimenti di sezione gold coprano i gruppi di evidenza;
- il matching delle alternative di sezione non le conti come false positive duplicate.

Il test è un controllo di coerenza del codice e del formato, **non una validazione scientifica indipendente** delle 45 domande, delle risposte, delle varianti lessicali o delle sezioni gold.

## Da completare 

1. Revisione indipendente di domande e risposte da parte di esperti del dominio.
2. Revisione del lessico dei key points e delle sue varianti.
3. ~~Verifica che ogni etichetta di sezione corrisponda a un'intestazione esistente~~: fatta in v0.3.1 sui PDF delle versioni di `papers.csv` (vedi `CHANGELOG_v0.3.1.md`).
4. Completamento di `evidence_annotations_template.csv` con pagina, paragraph/span ID e citazione esatta.
5. Congelamento di versioni dei paper, answer key, lessico, script e parametri.
