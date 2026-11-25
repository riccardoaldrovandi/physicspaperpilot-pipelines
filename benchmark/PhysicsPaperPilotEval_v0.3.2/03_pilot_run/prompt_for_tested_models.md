# Prompt uniforme per i modelli valutati — PhysicsPaperPilot

Usare lo stesso prompt, la stessa versione del PDF e le stesse condizioni per tutti i modelli.

```text
Read the supplied scientific paper and answer the question using only information supported by the paper.

Rules:
- Answer in English.
- Be concise, but include all scientifically relevant points needed to answer the question.
- Preserve numerical values, signs, exponents and units as reported in the paper.
- Do not add facts from external knowledge.
- If the paper does not provide the requested information, write: Not specified in the paper.
- For multiple-choice questions, put the option letter (A, B, C or D) first, followed by a brief explanation.
- Cite the exact section heading(s) that support the answer. Use the heading as written in the paper, not a paraphrased description.
- Cite only sections that directly support the answer. If more than one section is needed, separate headings with semicolons.
- Do not invent page, figure, table or equation identifiers. If the evidence is only identifiable by section, cite the section heading.

Return exactly these two fields, with no additional commentary:
ANSWER: <your answer>
EVIDENCE: <exact section heading 1>; <exact section heading 2>

Question:
[INSERT QUESTION]
```

## Inserimento nel CSV

- Copiare il testo dopo `ANSWER:` nella colonna `model_answer`.
- Copiare i titoli dopo `EVIDENCE:` nella colonna `cited_evidence`, separati da `;`.
- Per una risposta senza evidenza individuabile, lasciare `cited_evidence` vuoto. Lo script assegnerà zero alla metrica di localizzazione delle sezioni e segnalerà la riga.
- Per le domande multiple choice, compilare anche `mcq_choice` con la lettera scelta.

**Non fornire ai modelli** i file della cartella `02_private_answer_key`.
