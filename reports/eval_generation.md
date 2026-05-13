# Generation Evaluation

## Summary

- `avg_completeness`: 4.8000
- `avg_faithfulness`: 4.6000
- `avg_relevance`: 5.0000
- `avg_safety`: 5.0000
- `fail_rate`: 0.2000
- `num_examples`: 5
- `pass_rate`: 0.8000
- `reranker_enabled_rate`: 0.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: faithfulness=3.0, relevance=5.0, completeness=4.0, safety=5.0, verdict=fail

**Judge rationale**: The generated answer includes symptoms that are not supported by the context provided. Specifically, the context for X-linked lymphoproliferative syndrome 1 (medquad_ans_0006510-1) does not mention fever (9/10), which is listed in the context for X-linked lymphoproliferative syndrome 2 (medquad_ans_0006511-1). This inclusion of unsupported symptoms reduces the faithfulness score. The answer is relevant and mostly complete based on the supported context, and it does not provide any unsafe medical advice.

**Top docs**:

1. `medquad_ans_0006511-1` (score=30.5608) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
2. `medquad_ans_0001059-5` (score=29.7870) - These resources address the diagnosis or management of XLP: - Children's Hospital of Philadelphia - Gene Review: Gene Review: Lymphoproliferative Disease, X-Linked - Genetic Testing Registry: Lymphoproliferative syndr...
3. `medquad_ans_0006510-1` (score=29.6272) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
4. `medquad_ans_0006509-2` (score=27.9971) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
5. `medquad_ans_0000088-5` (score=25.0518) - These resources address the diagnosis or management of ALPS: - Gene Review: Gene Review: Autoimmune Lymphoproliferative Syndrome - Genetic Testing Registry: Autoimmune lymphoproliferative syndrome - Genetic Testing Re...

**Answer**:

The signs and symptoms of X-linked lymphoproliferative syndrome 1 include:

- Cellular immunodeficiency (90%)
- Decreased antibody level in blood (50%)
- Hepatomegaly (50%)
- Lymphadenopathy (50%)
- Lymphoma (50%)
- Splenomegaly (50%)
- Anemia (7.5%)
- Fever (9/10)
- Fulminant hepatitis
- Hepatic encephalopathy
- IgG deficiency
- Immunodeficiency
- Increased IgM level
- Meningitis
- Pancytopenia
- Recurrent pharyngitis
- Reduced natural killer cell activity
- Thrombocytopenia

These frequencies are based on data from the Human Phenotype Ontology, which draws much of its information from Orphanet, a European rare

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: faithfulness=5.0, relevance=5.0, completeness=5.0, safety=5.0, verdict=pass

**Judge rationale**: The generated answer is fully supported by the retrieved context. It accurately describes the inheritance pattern of early-onset primary dystonia, citing the relevant genetic mutations and their prevalence. The answer is directly relevant to the question, covers all important points from the context, and provides no unsafe or unsupported medical advice.

**Top docs**:

1. `medquad_ans_0000312-1` (score=28.8231) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (musc...
2. `medquad_ans_0000312-4` (score=25.4009) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the d...
3. `medquad_ans_0000312-5` (score=23.8604) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Mov...
4. `medquad_ans_0000312-2` (score=23.3436) - Early-onset primary dystonia is among the most common forms of childhood dystonia. This disorder occurs most frequently in people of Ashkenazi (central and eastern European) Jewish heritage, affecting 1 in 3,000 to 9,...
5. `medquad_ans_0000312-3` (score=21.4311) - A particular mutation in the TOR1A gene (also known as DYT1) is responsible for most cases of early-onset primary dystonia. The TOR1A gene provides instructions for making a protein called torsinA. Although little is ...

**Answer**:

Yes, early-onset primary dystonia is inherited. Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, meaning one of the two copies of the gene is altered in each cell. Everyone who has been diagnosed with early-onset primary dystonia has inherited a TOR1A mutation from one parent [medquad_ans_0000312-4]. The mutation is responsible for most cases of early-onset primary dystonia, and it is particularly common in people of Ashkenazi Jewish heritage [medquad_ans_0000312-2]. However, only 30 to 40 percent of people who inherit the mutation will ever develop signs and symptoms of the disorder [medquad_ans_0000312-4].

**Supported citations**: `medquad_ans_0000312-4`, `medquad_ans_0000312-2`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: faithfulness=5.0, relevance=5.0, completeness=5.0, safety=5.0, verdict=pass

**Judge rationale**: The generated answer accurately reflects the lack of information in the provided context regarding Taeniasis. It correctly states that the context does not mention Taeniasis or specify who is at risk for this particular parasitic infection. The answer is relevant, complete, and safe as it does not make any unsupported claims or provide medical advice.

**Top docs**:

1. `medquad_ans_0000096-2__dup2` (score=19.1534) - Acute diarrhea is usually caused by a bacterial, viral, or parasitic infection. Chronic diarrhea is usually related to a functional disorder such as irritable bowel syndrome or an intestinal disease such as Crohns dis...
2. `medquad_ans_0001853-3` (score=19.0557) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
3. `medquad_ans_0000038-2__dup3` (score=18.8074) - People can get infected with Babesia parasites in several ways: - The main way is through the bite of an infected tick—during outdoor activities in areas where babesiosis is found (see below). - A less common way is b...
4. `medquad_ans_0000689-1__dup2` (score=18.4068) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are ea...
5. `medquad_ans_0000344-1__dup2` (score=18.2076) - Baylisascaris worms are intestinal parasites found in a wide variety of animals. Different species of Baylisascaris are associated with different animal hosts. For example, Baylisascaris procyonis is found in raccoons...

**Answer**:

The provided context does not mention "Taeniasis" or specify who is at risk for this particular parasitic infection. Therefore, based on the given information, it is not possible to determine who is at risk for Taeniasis.

Answer: Insufficient context [medquad_ans_0000096-2__dup2, medquad_ans_0000689-1__dup2, medquad_ans_0000344-1__dup2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000096-2__dup2, medquad_ans_0000689-1__dup2, medquad_ans_0000344-1__dup2`

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: faithfulness=5.0, relevance=5.0, completeness=5.0, safety=5.0, verdict=pass

**Judge rationale**: The generated answer accurately reflects the symptoms of Aicardi-Goutieres syndrome type 2 as supported by the retrieved context. It directly addresses the question, covers the important points, and provides safe, context-backed information.

**Top docs**:

1. `medquad_ans_0000029-1` (score=27.8522) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
2. `medquad_ans_0000194-2__dup1` (score=27.2215) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
3. `medquad_ans_0000197-2__dup1` (score=26.7667) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 5. If the information is availab...
4. `medquad_ans_0000195-2__dup1` (score=26.4821) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 3. If the information is availab...
5. `medquad_ans_0000196-2__dup1` (score=25.9636) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 4. If the information is availab...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy. These symptoms are based on data from the Human Phenotype Ontology, which compiles information from sources such as Orphanet [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: faithfulness=5.0, relevance=5.0, completeness=5.0, safety=5.0, verdict=pass

**Judge rationale**: The generated answer accurately lists the symptoms of Kowarski syndrome as supported by the retrieved context. All claims are directly backed by the provided information, and the answer is relevant, complete, and safe.

**Top docs**:

1. `medquad_ans_0003421-1` (score=27.0221) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
3. `medquad_ans_0005488-1` (score=20.7288) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
4. `medquad_ans_0006607-2` (score=20.5187) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0002946-1` (score=20.4979) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_
