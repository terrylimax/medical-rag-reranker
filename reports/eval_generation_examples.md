# Baseline Generation Examples (No Reranker)

- retriever: `bm25`
- llm_model: `qwen/qwen3-4b-instruct-2507`
- top_k: `5`
- reranker_enabled: `False`
- num_examples: `5`

## Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Top docs**:

1. `medquad_ans_0006511-1` (score=30.5608) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the information is available, the ...
2. `medquad_ans_0001059-5` (score=29.7870) - These resources address the diagnosis or management of XLP: - Children's Hospital of Philadelphia - Gene Review: Gene Review: Lymphoproliferative Disease, X-Linked - Genetic Testing Registry: Lymphoproliferative syndrome 1, X-linked - Ge...
3. `medquad_ans_0006510-1` (score=29.6272) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the information is available, the ...
4. `medquad_ans_0006509-2` (score=27.9971) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is available, the tabl...
5. `medquad_ans_0000088-5` (score=25.0518) - These resources address the diagnosis or management of ALPS: - Gene Review: Gene Review: Autoimmune Lymphoproliferative Syndrome - Genetic Testing Registry: Autoimmune lymphoproliferative syndrome - Genetic Testing Registry: Autoimmune l...

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

**Citations detected**: _none_
**Supported citations**: _none_
**Unsupported citations**: _none_

## Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Top docs**:

1. `medquad_ans_0000312-1` (score=28.8231) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (muscle contractions), tw...
2. `medquad_ans_0000312-4` (score=25.4009) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the disorder and may neve...
3. `medquad_ans_0000312-5` (score=23.8604) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Movement - uncontrolled...
4. `medquad_ans_0000312-2` (score=23.3436) - Early-onset primary dystonia is among the most common forms of childhood dystonia. This disorder occurs most frequently in people of Ashkenazi (central and eastern European) Jewish heritage, affecting 1 in 3,000 to 9,000 people in this p...
5. `medquad_ans_0000312-3` (score=21.4311) - A particular mutation in the TOR1A gene (also known as DYT1) is responsible for most cases of early-onset primary dystonia. The TOR1A gene provides instructions for making a protein called torsinA. Although little is known about its func...

**Answer**:

Yes, early-onset primary dystonia is inherited. Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, meaning one of the two copies of the gene is altered in each cell. Everyone who has been diagnosed with early-onset primary dystonia has inherited a TOR1A mutation from one parent [medquad_ans_0000312-4]. The mutation is responsible for most cases of early-onset primary dystonia, and it is particularly common in people of Ashkenazi Jewish heritage [medquad_ans_0000312-2]. However, only 30 to 40 percent of people who inherit the mutation will ever develop signs and symptoms of the disorder [medquad_ans_0000312-4].

**Citations detected**: `medquad_ans_0000312-4`, `medquad_ans_0000312-2`
**Supported citations**: `medquad_ans_0000312-4`, `medquad_ans_0000312-2`
**Unsupported citations**: _none_

## Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Top docs**:

1. `medquad_ans_0000096-2__dup2` (score=19.1534) - Acute diarrhea is usually caused by a bacterial, viral, or parasitic infection. Chronic diarrhea is usually related to a functional disorder such as irritable bowel syndrome or an intestinal disease such as Crohns disease. The most commo...
2. `medquad_ans_0001853-3` (score=19.0557) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increased risk of developing...
3. `medquad_ans_0000038-2__dup3` (score=18.8074) - People can get infected with Babesia parasites in several ways: - The main way is through the bite of an infected tick—during outdoor activities in areas where babesiosis is found (see below). - A less common way is by getting a transfus...
4. `medquad_ans_0000689-1__dup2` (score=18.4068) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are easily treated and som...
5. `medquad_ans_0000344-1__dup2` (score=18.2076) - Baylisascaris worms are intestinal parasites found in a wide variety of animals. Different species of Baylisascaris are associated with different animal hosts. For example, Baylisascaris procyonis is found in raccoons and Baylisascaris c...

**Answer**:

The provided context does not mention "Taeniasis" or specify who is at risk for this particular parasitic infection. Therefore, based on the given information, it is not possible to determine who is at risk for Taeniasis.

Answer: Insufficient context [medquad_ans_0000096-2__dup2, medquad_ans_0000689-1__dup2, medquad_ans_0000344-1__dup2].

**Citations detected**: `medquad_ans_0000096-2__dup2, medquad_ans_0000689-1__dup2, medquad_ans_0000344-1__dup2`
**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000096-2__dup2, medquad_ans_0000689-1__dup2, medquad_ans_0000344-1__dup2`

## Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Top docs**:

1. `medquad_ans_0000029-1` (score=27.8522) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. However, about 20 percent ...
2. `medquad_ans_0000194-2__dup1` (score=27.2215) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is available, the table below ...
3. `medquad_ans_0000197-2__dup1` (score=26.7667) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 5. If the information is available, the table below ...
4. `medquad_ans_0000195-2__dup1` (score=26.4821) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 3. If the information is available, the table below ...
5. `medquad_ans_0000196-2__dup1` (score=25.9636) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 4. If the information is available, the table below ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy. These symptoms are based on data from the Human Phenotype Ontology, which compiles information from sources such as Orphanet [medquad_ans_0000194-2__dup1].

**Citations detected**: `medquad_ans_0000194-2__dup1`
**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

## Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Top docs**:

1. `medquad_ans_0003421-1` (score=27.0221) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how often the symptom i...
2. `medquad_ans_0000108-2__dup1` (score=20.7436) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table below includes how o...
3. `medquad_ans_0005488-1` (score=20.7288) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how often the symptom is ...
4. `medquad_ans_0006607-2` (score=20.5187) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, the table below includ...
5. `medquad_ans_0002946-1` (score=20.4979) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below includes how often the...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Citations detected**: `medquad_ans_0003421-1`
**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_
