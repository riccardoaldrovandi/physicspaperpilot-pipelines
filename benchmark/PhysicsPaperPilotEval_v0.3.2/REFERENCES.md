# Riferimenti metodologici

1. **SQuAD / Exact Match e token F1** — Rajpurkar et al. (2016), *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, EMNLP 2016. https://aclanthology.org/D16-1264/ . Riferimento per normalized EM e token-overlap F1.
2. **QASPER** — Dasigi et al. (2021), *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers*, NAACL 2021. https://aclanthology.org/2021.naacl-main.365/ . Riferimento per QA su paper scientifici e annotazione delle evidenze di supporto.
3. **ROUGE** — Lin (2004), *ROUGE: A Package for Automatic Evaluation of Summaries*, Text Summarization Branches Out. https://aclanthology.org/W04-1013/ . ROUGE-L usa la longest common subsequence.
4. **BERTScore** — Zhang et al. (2020), *BERTScore: Evaluating Text Generation with BERT*, ICLR 2020. https://arxiv.org/abs/1904.09675 . Similarità contestuale basata su encoder; non è un generative LLM judge.
5. **SciFact** — Wadden et al. (2020), *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020. https://aclanthology.org/2020.emnlp-main.609/ . Riferimento per futura estensione a claim verification e supporting evidence annotate.
6. **SymPy** — documentazione per semplificazione e verifica simbolica, per una futura categoria di domande matematiche: https://docs.sympy.org/latest/tutorial/simplification.html .

## Scelte operative specifiche 

- Il key-point coverage è un matching deterministico tramite un lessico esplicito e ispezionabile; non è una metrica standard universale. La sua riproducibilità dipende dalla versione del lessico, che va versionata e congelata.
- Evidence-section P/R/F1 è un adattamento del principio di retrieval delle evidenze, ma usa intestazioni di sezione come unità di corrispondenza. Non va presentato come la stessa metrica di passage-level Evidence F1 di QASPER.
