# Generation Evaluation

## Summary

- `avg_answer_chars`: 561.3200
- `avg_answer_relevance`: 0.7709
- `avg_citation_presence_rate`: 0.8333
- `avg_context_relevance`: 0.9233
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 5635.9483
- `avg_generation_latency_ms`: 2898.6475
- `avg_groundedness`: 0.7888
- `avg_insufficient_context`: 0.0367
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 2680.5486
- `avg_retrieval_latency_ms`: 56.4365
- `avg_supported_citation_rate`: 0.7606
- `avg_unsupported_citation_rate`: 0.0728
- `num_examples`: 300
- `reranker_enabled_rate`: 1.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006510-1` (score=0.9994) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
2. `medquad_ans_0006509-2` (score=0.0004) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
3. `medquad_ans_0006511-1` (score=0.0001) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
4. `medquad_ans_0004907-3` (score=0.0001) - What causes pityriasis lichenoides et varioliformis acuta? The exact underlying cause of pityriasis lichenoides et varioliformis acuta (PLEVA) is unknown. Some scientists suspect that it may occur due to an exaggerate...
5. `medquad_ans_0000088-5` (score=0.0001) - These resources address the diagnosis or management of ALPS: - Gene Review: Gene Review: Autoimmune Lymphoproliferative Syndrome - Genetic Testing Registry: Autoimmune lymphoproliferative syndrome - Genetic Testing Re...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), and other features such as fulminant hepatitis, encephalitis, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000312-4` (score=0.9879) - Mutations in the TOR1A gene are inherited in an autosomal dominant pattern, which means one of the two copies of the gene is altered in each cell. Many people who have a mutation in this gene are not affected by the d...
2. `medquad_ans_0000299-4` (score=0.0034) - When dopa-responsive dystonia is caused by mutations in the GCH1 gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In so...
3. `medquad_ans_0000311-4` (score=0.0001) - Early-onset glaucoma can have different inheritance patterns. Primary congenital glaucoma is usually inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. Most o...
4. `medquad_ans_0000112-1__dup1` (score=0.0001) - The dystonias are movement disorders in which sustained muscle contractions cause twisting and repetitive movements or abnormal postures. The movements, which are involuntary and sometimes painful, may affect a single...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Yes, early-onset primary dystonia is inherited. Everyone who has been diagnosed with early-onset primary dystonia has inherited a TOR1A mutation from one parent, and the mutation in the TOR1A gene is inherited in an autosomal dominant pattern [medquad_ans_0000312-4].

**Supported citations**: `medquad_ans_0000312-4`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.857, groundedness=0.075, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000344-1__dup2` (score=0.0019) - Baylisascaris worms are intestinal parasites found in a wide variety of animals. Different species of Baylisascaris are associated with different animal hosts. For example, Baylisascaris procyonis is found in raccoons...
2. `medquad_ans_0000689-1__dup2` (score=0.0009) - Parasites are living things that use other living things - like your body - for food and a place to live. You can get them from contaminated food or water, a bug bite, or sexual contact. Some parasitic diseases are ea...
3. `medquad_ans_0000376-1__dup1` (score=0.0007) - Each year, 48 million people in the U.S. get sick from contaminated food. Common culprits include bacteria, parasites and viruses. Symptoms range from mild to serious. They include - Upset stomach - Abdominal cramps -...
4. `medquad_ans_0000265-5__dup1` (score=0.0002) - Loa loa parasites are found in West and Central Africa. Ten countries have areas where there are high rates of infection (i.e., where more than 40% of the people who live in that area report that they have had eye wor...
5. `medquad_ans_0000038-5__dup3` (score=0.0001) - In symptomatic people, babesiosis usually is diagnosed by examining blood specimens under a microscope and seeing Babesia parasites inside red blood cells. To be sure the diagnosis is correct, your health care provide...

**Answer**:

The provided context does not mention Taeniasis or who is at risk for this specific parasitic infection. Therefore, based on the given information, it is not possible to determine who is at risk for Taeniasis. [no supporting context]

**Supported citations**: _none_
**Unsupported citations**: `no supporting context`

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.9994) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0000195-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000196-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000193-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.9993) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0001388-1` (score=0.0001) - What are the signs and symptoms of COASY Protein-Associated Neurodegeneration ? The Human Phenotype Ontology provides the following list of signs and symptoms for COASY Protein-Associated Neurodegeneration . If the in...
4. `medquad_ans_0004384-2` (score=0.0001) - What are the signs and symptoms of Neuronal ceroid lipofuscinosis 10 ? The Human Phenotype Ontology provides the following list of signs and symptoms for Neuronal ceroid lipofuscinosis 10 . If the information is avail...
5. `medquad_ans_0000141-2__dup1` (score=0.0001) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.902, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000450-4` (score=0.9842) - Hemophilia A and hemophilia B are inherited in an X-linked recessive pattern. The genes associated with these conditions are located on the X chromosome, which is one of the two sex chromosomes. In males (who have onl...
2. `medquad_ans_0000069-3__dup1` (score=0.0040) - A defect in one of the genes that determines how the body makes blood clotting factor VIII or IX causes hemophilia. These genes are located on the X chromosomes (KRO-muh-somz). Chromosomes come in pairs. Females have ...
3. `medquad_ans_0000102-1__dup3` (score=0.0001) - Normally, if you get hurt, your body forms a blood clot to stop the bleeding. For blood to clot, your body needs cells called platelets and proteins known as clotting factors. If you have a bleeding disorder, you eith...
4. `medquad_ans_0002788-1` (score=0.0001) - Hemophilia B is a bleeding disorder that slows the blood clotting process. People with this disorder experience prolonged bleeding or oozing following an injury or surgery. In severe cases of hemophilia, heavy bleedin...
5. `medquad_ans_0000069-5__dup1` (score=0.0001) - If you or your child appears to have a bleeding problem, your doctor will ask about your personal and family medical histories. This will reveal whether you or your family members, including women and girls, have blee...

**Answer**:

Yes, hemophilia is inherited. Hemophilia A and hemophilia B are inherited in an X-linked recessive pattern, meaning the genes associated with these conditions are located on the X chromosome [medquad_ans_0000450-4]. In males, who have only one X chromosome, one altered copy of the gene is sufficient to cause the condition. In females, who have two X chromosomes, a mutation would need to occur in both copies of the gene to cause the disorder, making it very rare for females to have hemophilia [medquad_ans_0000450-4]. Females with one altered copy are typically carriers and may have reduced levels of clotting factors, putting them at risk for bleeding in certain situations [medquad_ans_0000450-4]. Additionally, the inheritance pattern shows that each daughter has a 50% chance of inheriting the hemophilia gene

**Supported citations**: `medquad_ans_0000450-4`
**Unsupported citations**: _none_

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.970, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.9994) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0004087-1` (score=0.0001) - What are the signs and symptoms of Mitochondrial Membrane Protein-Associated Neurodegeneration ? The Human Phenotype Ontology provides the following list of signs and symptoms for Mitochondrial Membrane Protein-Associ...
3. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0002061-3` (score=0.0001) - What causes Ehlers-Danlos syndrome progeroid type? Ehlers-Danlos syndrome progeroid type is caused by changes (mutations) in both of an individual's copies of the B4GALT7 gene, which is located on chromosome 5. This g...
5. `medquad_ans_0000141-2__dup1` (score=0.0001) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=0.833, groundedness=0.537, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004747-1` (score=0.9758) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People ...
2. `medquad_ans_0000775-1` (score=0.6112) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
3. `medquad_ans_0000093-5` (score=0.0009) - These resources address the diagnosis or management of ADNFLE: - Gene Review: Gene Review: Autosomal Dominant Nocturnal Frontal Lobe Epilepsy - Genetic Testing Registry: Epilepsy, nocturnal frontal lobe, type 1 - Gene...
4. `medquad_ans_0000775-2` (score=0.0005) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
5. `medquad_ans_0000774-1` (score=0.0001) - Paroxysmal extreme pain disorder is a condition characterized by skin redness and warmth (flushing) and attacks of severe pain in various parts of the body. The area of flushing typically corresponds to the site of th...

**Answer**:

Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. The disorder is characterized by sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to a deficiency of red blood cells (hemolytic anemia) and resulting in signs and symptoms such as fatigue, weakness, abnormally pale skin (pallor), shortness of breath, and an increased heart rate. Affected individuals may pass dark-colored urine due to the presence of hemoglobin, a

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.941, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.6004) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
2. `medquad_ans_0000221-1__dup2` (score=0.4409) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...
3. `medquad_ans_0000037_2-1` (score=0.0423) - Key Points - Rectal cancer is a disease in which malignant (cancer) cells form in the tissues of the rectum. - Health history affects the risk of developing rectal cancer. - Signs of rectal cancer include a change in ...
4. `medquad_ans_0000010-17` (score=0.0222) - Parents, siblings, or children of a person who has had colorectal cancer are somewhat more likely to develop this type of cancer themselves. This is especially true if the relative had the cancer at a young age. If ma...
5. `medquad_ans_0000037_4-1` (score=0.0212) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...

**Answer**:

Colorectal cancer is a type of cancer that occurs when malignant tumors form in the lining of the large intestine, also known as the large bowel. The colon and rectum are part of the large intestine, which is part of the digestive system [medquad_ans_0000010-10]. It is common in both men and women and the risk of developing it rises after age 50 [medquad_ans_0000221-1__dup2]. Additional risk factors include having colorectal polyps, a family history of colorectal cancer, ulcerative colitis or Crohn's disease, eating a diet high in fat, or smoking [medquad_ans_0000221-1__dup2]. Symptoms may include diarrhea or constipation, a feeling that the bowel does not empty completely, blood in the stool, narrow stools, gas pains or cramps, feeling full

**Supported citations**: `medquad_ans_0000010-10`, `medquad_ans_0000221-1__dup2`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=0.833, groundedness=0.525, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003317-1` (score=0.9712) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...
2. `medquad_ans_0000544-1` (score=0.9670) - Junctional epidermolysis bullosa (JEB) is one of the major forms of epidermolysis bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. Blisters and skin erosions form in...
3. `medquad_ans_0002121-1` (score=0.0042) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
4. `medquad_ans_0000544-2` (score=0.0003) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
5. `medquad_ans_0005032-1` (score=0.0001) - Pretibial epidermolysis bullosa is a rare form of epidermolysis bullosa, a condition characterized by fragile skin that blisters easily in response to minor injury or friction. In the pretibial form, specifically, the...

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe, with blistering present from birth or early infancy, affecting large regions of the body and mucous membranes, leading to complications such as chronic malnutrition, scarring, granulation tissue formation, and increased susceptibility to infections. Many infants with Herlitz JEB do not survive beyond the first year of life. The Non-Herlitz type includes several subtypes that cause mild to severe blistering, often limited to the hands, feet, knees, and elbows, and may improve after the newborn period. JEB is inherited in an autosomal recessive pattern and is caused by

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.945, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000668-3` (score=0.9991) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
2. `medquad_ans_0000287-3` (score=0.0001) - Dihydropyrimidinase deficiency is caused by mutations in the DPYS gene, which provides instructions for making an enzyme called dihydropyrimidinase. This enzyme is involved in the breakdown of molecules called pyrimid...
3. `medquad_ans_0000668-2` (score=0.0001) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought tha...
4. `medquad_ans_0000477-3` (score=0.0001) - Hereditary xanthinuria type I is caused by mutations in the XDH gene. This gene provides instructions for making an enzyme called xanthine dehydrogenase. This enzyme is involved in the normal breakdown of purines, whi...
5. `medquad_ans_0000668-1` (score=0.0001) - Molybdenum cofactor deficiency is a rare condition characterized by brain dysfunction (encephalopathy) that worsens over time. Babies with this condition appear normal at birth, but within a week they have difficulty ...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. Mutations in the MOCS1 gene cause type A, mutations in the MOCS2 gene cause type B, and mutations in the GPHN gene cause type C of the disorder [medquad_ans_0000668-3]. These mutations reduce or eliminate the function of the associated protein, impairing molybdenum cofactor biosynthesis [medquad_ans_0000668-3].

**Supported citations**: `medquad_ans_0000668-3`
**Unsupported citations**: _none_

### Example 12 (`0000756-1`)

**Question**: What is (are) ornithine translocase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004581-1` (score=0.9212) - Ornithine transcarbamylase (OTC) deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become to...
2. `medquad_ans_0000756-1` (score=0.3175) - Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The...
3. `medquad_ans_0000755-1` (score=0.0462) - Ornithine transcarbamylase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high...
4. `medquad_ans_0000755-4` (score=0.0067) - Ornithine transcarbamylase deficiency is an X-linked disorder. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. A characte...
5. `medquad_ans_0000968-1__dup1` (score=0.0005) - Carnitine-acylcarnitine translocase deficiency is a condition that prevents the body from converting certain fats called long-chain fatty acids into energy, particularly during periods without food (fasting). Carnitin...

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is especially sensitive to the effects of excess ammonia. Ornithine translocase deficiency varies widely in its severity and age of onset. An infant with ornithine translocase deficiency may be lacking in energy (lethargic) or refuse to eat, or have poorly controlled breathing or body temperature. Some babies with this disorder may experience seizures or unusual body movements, or go into a coma. Episodes of illness may coincide with the introduction of high-protein formulas or solid foods into the diet. In most affected individuals, signs and symptoms of ornithine translocase deficiency do not appear until later in life. Later-onset forms of ornithine translocase deficiency are usually less severe than the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000362-1__dup2` (score=0.9954) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
2. `medquad_ans_0000286-1__dup2` (score=0.0065) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
3. `medquad_ans_0000045-28` (score=0.0010) - The general number for Medicare is 1-800-Medicare (1-800-633-4227). TTY users should call 1-877-486-2048. You can also visit http://www.medicare.gov. The "Medicare & You" handbook is mailed out to all Medicare enrolle...
4. `medquad_ans_0000060-35` (score=0.0003) - These quit smoking websites offer you free, accurate information and professional assistance to help support the immediate and long-term needs of people trying to quit smoking. The National Cancer Institute sponsors -...
5. `medquad_ans_0000042-18` (score=0.0001) - Many agencies and organizations in the community provide assistance and information to people who have low vision and to their families and caregivers. State agencies for the blind and visually impaired can make refer...

**Answer**:

Yes, there is information about financial assistance. If you don't have health insurance or need help with costs that aren't covered, financial assistance might be available through certain government programs and nonprofit organizations. You can also discuss concerns about paying your medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2].

**Supported citations**: `medquad_ans_0000362-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0004105-1` (score=0.8755) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
2. `medquad_ans_0000115-4__dup1` (score=0.0039) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
3. `medquad_ans_0000721-1__dup1` (score=0.0004) - Pneumococci are a type of streptococcus bacteria. The bacteria spread through contact with people who are ill or by healthy people who carry the bacteria in the back of their nose. Pneumococcal infections can be mild ...
4. `medquad_ans_0000598-1__dup2` (score=0.0001) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enter...
5. `medquad_ans_0000115-2__dup1` (score=0.0001) - Anyone experiencing symptoms of meningitis or encephalitis should see a doctor immediately. Antibiotics for most types of meningitis can greatly reduce the risk of dying from the disease. Antiviral medications may be ...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, although recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1].

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003319-3` (score=0.9871) - What causes juvenile amyotrophic lateral sclerosis? Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases the underlying gene abnormality...
2. `medquad_ans_0003331-3` (score=0.9829) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
3. `medquad_ans_0000309-3__dup1` (score=0.4638) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the...
4. `medquad_ans_0000551-5` (score=0.0001) - These resources address the diagnosis or management of juvenile primary lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Genetic Testing Registry: Juvenile primary lateral sclerosis These resour...
5. `medquad_ans_0003319-4` (score=0.0001) - How might juvenile amyotrophic lateral sclerosis be treated? Treatments and therapies are available to relieve symptoms and improve the quality of life of people with juvenile ALS. Medications, such as those that redu...

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases, the underlying gene abnormality cannot be determined. Juvenile ALS may be inherited in an autosomal dominant (as in ALS type 4) or autosomal recessive (as in ALS type 2) fashion [medquad_ans_0003319-3]. Additionally, mutations in genes such as C9orf72, SOD1, TARDBP, FUS, ANG, ALS2, SETX, and VAPB have been associated with familial ALS, which may also contribute to sporadic cases [medquad_ans_0000309-3__dup1]. However, the exact mechanism by which these mutations lead to disease is not fully understood.

**Supported citations**: `medquad_ans_0003319-3`, `medquad_ans_0000309-3__dup1`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=1.000, groundedness=0.793, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000060-10` (score=0.0046) - You can keep your nervous system healthy by taking these steps: - Eat healthy meals and follow the meal plan that you and your doctor or dietitian have worked out. - If you drink alcoholic beverages, limit your intake...
2. `medquad_ans_0000219-4__dup3` (score=0.0007) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men ag...
3. `medquad_ans_0000193-6` (score=0.0006) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
4. `medquad_ans_0000591-1__dup1` (score=0.0003) - Summary : Medicines cure infectious diseases, prevent problems from chronic diseases, and ease pain. But medicines can also cause harmful reactions if not used correctly. Errors can happen in the hospital, at the doct...
5. `medquad_ans_0000032-3__dup2` (score=0.0003) - Don't Ignore Hearing Problems Some people have a hearing problem without realizing it. Others might think they have a problem, but are too embarrassed to tell their doctor, friends, or family. You can help identify a ...

**Answer**:

The provided context does not include specific information or recommendations on what to do for "What I need to know about Erectile Dysfunction." While it states that erectile dysfunction (ED) affects men of all races and ages, with higher prevalence in older men, and that it becomes more common with age, there are no actionable steps or resources directed toward managing or understanding ED in detail [medquad_ans_0000219-4__dup3]. Therefore, based on the given context, it is not possible to provide a response to the question about "What I need to know about Erectile Dysfunction."

**Supported citations**: `medquad_ans_0000219-4__dup3`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.9702) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0000202-5` (score=0.0019) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Regist...
3. `medquad_ans_0000576-5` (score=0.0002) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
4. `medquad_ans_0000395-5` (score=0.0001) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
5. `medquad_ans_0000631-1__dup1` (score=0.0001) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often become necessary in the first or second decade of life. Prevention of chest infections may be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that people with Ullrich congenital muscular dystrophy may benefit from cyclosporin A, although more studies are needed to confirm its benefits [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=1.000, groundedness=0.792, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002834-4` (score=0.0002) - How is hereditary lymphedema type II inherited? Hereditary lymphedema type II appears to have an autosomal dominant pattern of inheritance, which means that one copy of an altered gene in each cell is sufficient to ca...
2. `medquad_ans_0004005-4` (score=0.0001) - How is microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) inherited? MOPD1 is thought to be inherited in an autosomal recessive manner. This means that affected individuals have abnormal gene changes (mu...
3. `medquad_ans_0000653-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcep...
4. `medquad_ans_0004006-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
5. `medquad_ans_0004005-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) is a genetic condition that is mainly characterized by intrauterine and post-natal growth retardation; an abnormally small head size (microcephaly); abn...

**Answer**:

The provided context does not specify the inheritance pattern of microcephalic osteodysplastic primordial dwarfism type II (MOPDII). While the context details the clinical features of MOPDII, such as short stature, microcephaly, skeletal abnormalities, and other signs, it does not mention how the condition is inherited. Therefore, based on the given information, the inheritance pattern of MOPDII cannot be determined.

[medquad_ans_0000653-1]

**Supported citations**: `medquad_ans_0000653-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.112, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000527-4` (score=0.0262) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
2. `medquad_ans_0001084-3__dup1` (score=0.0030) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
3. `medquad_ans_0003718-4` (score=0.0022) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
4. `medquad_ans_0000562-4` (score=0.0002) - When Klippel-Feil syndrome is caused by mutations in the GDF6 or GDF3 genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder....
5. `medquad_ans_0004408-3` (score=0.0001) - How is nevoid basal cell carcinoma syndrome inherited? Nevoid basal cell carcinoma syndrome (NBCCS) is caused by a change (mutation) in the PTCH1 gene and is inherited in an autosomal dominant way. This means that if ...

**Answer**:

The provided context does not contain any information about Ochoa syndrome or its inheritance pattern. Therefore, it is not possible to determine based on the given text whether Ochoa syndrome is inherited.

Answer: Insufficient context [no_source].

**Supported citations**: _none_
**Unsupported citations**: `no_source`

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.808, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.9960) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
2. `medquad_ans_0000531-2` (score=0.0001) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
3. `medquad_ans_0000531-3` (score=0.0001) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic m...
4. `medquad_ans_0006555-1` (score=0.0001) - 15q13.3 microdeletion syndrome is a type of contiguous gene deletion syndrome. Individuals with this microdeletion may have very different signs and symptoms from other affected individuals (even within the same famil...
5. `medquad_ans_0003266-1` (score=0.0001) - Isodicentric chromosome 15 syndrome is a chromosome abnormality that affects many different parts of the body. As the name suggests, people with this condition have an extra chromosome (called an isodicentric chromoso...

**Answer**:

The provided context does not specify treatments for isodicentric chromosome 15 syndrome. While it mentions resources such as genetic counseling, drug therapy, surgery and rehabilitation, palliative care, and diagnostic tests as available for conditions in general, it does not detail specific treatments for isodicentric chromosome 15 syndrome [medquad_ans_0000531-5]. Therefore, based on the given information, the treatments for isodicentric chromosome 15 syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=0.9931) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0000272-4__dup1` (score=0.0002) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cau...
3. `medquad_ans_0002890-3` (score=0.0001) - What causes Hirschsprung disease? There are a number of different causes of Hirschsprung disease (HSCR). For example, HSCR may occur as: A part of a syndrome In association with a chromosome anomaly (such as trisomy 2...
4. `medquad_ans_0000998-2` (score=0.0001) - About 20 cases of Troyer syndrome have been reported in the Old Order Amish population of Ohio. It has not been found outside this population.
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene, which give the body instructions to make a protein called spartin. This protein is present in many body tissues, including those of the nervous system, and is thought to play various roles needed for cell function. Troyer syndrome is assumed to be caused by a loss of function of the spartin protein. More research is needed to fully understand how these mutations lead to the features of the syndrome [medquad_ans_0006224-3].

**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.953, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000983-3` (score=0.9990) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
2. `medquad_ans_0000869-3` (score=0.0004) - Restless legs syndrome likely results from a combination of genetic and environmental factors, many of which are unknown. Studies suggest that restless legs syndrome is related to a shortage (deficiency) of iron in ce...
3. `medquad_ans_0000543-3` (score=0.0001) - Joubert syndrome and related disorders can be caused by mutations in at least 10 genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopi...
4. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000254-3` (score=0.0001) - Crohn disease is related to chromosomes 5 and 10. Variations of the ATG16L1, IRGM, and NOD2 genes increase the risk of developing Crohn disease. The IL23R gene is associated with Crohn disease. A variety of genetic an...

**Answer**:

Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. These mutations change the structure of CaV1.2 channels, causing them to stay open much longer than usual, which leads to an abnormal influx of calcium ions into cells. This results in altered heart function and can cause arrhythmia [medquad_ans_0000983-3].

**Supported citations**: `medquad_ans_0000983-3`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000061-3__dup3` (score=0.9930) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
2. `medquad_ans_0000061-10` (score=0.0139) - Rheumatoid arthritis is characterized by inflammation of the joint lining. This inflammation causes warmth, redness, swelling, and pain around the joints. The pain of rheumatoid arthritis varies greatly from person to...
3. `medquad_ans_0000061-14` (score=0.0093) - Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation. Still others, called disease-modifying anti-rheumatic drugs or DMARDs, can often slow th...
4. `medquad_ans_0000061-13` (score=0.0024) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation...
5. `medquad_ans_0000061-4__dup3` (score=0.0007) - Most Symptoms Are Treatable Doctors use a variety of approaches to treat rheumatoid arthritis. The goals of treatment are to help relieve pain, reduce swelling, slow down or help prevent joint damage, increase the abi...

**Answer**:

The symptoms of rheumatoid arthritis include pain and stiffness in the joints, warmth, redness, swelling, and pain around the joints. People also often feel sick, tired, and may experience fever. Rheumatoid arthritis generally occurs in a symmetrical pattern, meaning if one knee or hand is affected, the other is also likely to be affected [medquad_ans_0000061-3__dup3].

**Supported citations**: `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=0.800, groundedness=0.532, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.9718) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.9287) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0003148-1` (score=0.0017) - Idiopathic thrombocytopenic purpura (ITP) is a bleeding disorder characterized by too few platelets in the blood. This is because platelets are being destroyed by the immune system. Symptoms may include bruising, nose...
4. `medquad_ans_0000837-1` (score=0.0011) - Protein C deficiency is a disorder that increases the risk of developing abnormal blood clots; the condition can be mild or severe. Individuals with mild protein C deficiency are at risk of a type of blood clot known ...
5. `medquad_ans_0000838-1` (score=0.0007) - Protein S deficiency is a disorder of blood clotting. People with this condition have an increased risk of developing abnormal blood clots. Individuals with mild protein S deficiency are at risk of a type of clot call...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It occurs when small blood vessels in the body become inflamed and leak, leading to a rash that appears as many small raised bruises. HSP may develop as an immune response to an infection and is most common in children between 2 and 6 years of age, though it can affect people of any age. Most cases go away on their own without treatment, and the main goal of treatment, when needed, is to relieve symptoms such as joint pain, abdominal pain, or swelling. In some cases, over-the-counter medicines are used, and in severe arthritis, prednisone, a steroid medicine, may be prescribed. Most people recover completely, though kidney damage is the most likely long-term complication, and in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.9994) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0003162-2` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition ...
3. `medquad_ans_0003163-1` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
4. `medquad_ans_0002992-4` (score=0.0001) - Is hyper IgD syndrome inherited? Hyper IgD syndrome is inherited in an autosomal recessive manner, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recess...
5. `medquad_ans_0000579-1__dup1` (score=0.0001) - Autosomal recessive hyper IgE syndrome (AR-HIES) is a very rare primary immunodeficiency syndrome characterized by highly elevated blood levels of immunoglobulin E (IgE), recurrent staphylococcal skin abscesses, and r...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.9991) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000543-3` (score=0.0003) - Joubert syndrome and related disorders can be caused by mutations in at least 10 genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopi...
3. `medquad_ans_0000142-3` (score=0.0002) - C3 glomerulopathy is associated with changes in many genes. Most of these genes provide instructions for making proteins that help regulate a part of the body's immune response known as the complement system. This sys...
4. `medquad_ans_0001041-3` (score=0.0002) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000832-3` (score=0.0001) - In most cases, the genetic cause of progressive supranuclear palsy is unknown. Rarely, the disease results from mutations in the MAPT gene. Certain normal variations (polymorphisms) in the MAPT gene have also been ass...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia, which are involved in chemical signaling pathways. Mutations in these genes likely lead to problems with the structure and function of cilia, disrupting important chemical signaling within cells. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. In some individuals, no mutations in these five genes are identified, and the genetic cause of the disorder is unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.9988) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
2. `medquad_ans_0003700-1` (score=0.0001) - Lymphedema distichiasis syndrome is a condition that affects the normal function of the lymphatic system (part of the immune system that produces and transports fluids and immune cells throughout the body). People wit...
3. `medquad_ans_0000610-1` (score=0.0001) - Lymphedema-distichiasis syndrome is a condition that affects the normal function of the lymphatic system, which is a part of the circulatory and immune systems. The lymphatic system produces and transports fluids and ...
4. `medquad_ans_0000610-3` (score=0.0001) - Lymphedema-distichiasis syndrome is caused by mutations in the FOXC2 gene. The FOXC2 gene provides instructions for making a protein that plays a critical role in the formation of many organs and tissues before birth....
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The prevalence of lymphedema-distichiasis syndrome is unknown [medquad_ans_0000610-2].

**Supported citations**: `medquad_ans_0000610-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=1.000, groundedness=0.942, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0002853-4` (score=0.0003) - Is genetic testing available for hereditary sensory neuropathy type 1? At least four genes responsible for hereditary sensory neuropathy type 1 (HSN1) have been found: HSN1A (the most common form) is associated with m...
2. `medquad_ans_0001483-5` (score=0.0001) - Is genetic testing available for congenital bilateral absence of the vas deferens (CBAVD)? GeneTests lists the names of laboratories that are performing genetic testing for CBAVD. To view the contact information for t...
3. `medquad_ans_0000011-5__dup1` (score=0.0001) - Is genetic testing available for abetalipoproteinemia? Yes. The Genetic Testing Registry (GTR) provides information about the genetic tests available for abetalipoproteinemia. The intended audience for the GTR is heal...
4. `medquad_ans_0005415-4` (score=0.0001) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
5. `medquad_ans_0006498-3` (score=0.0001) - Is genetic testing available for X-linked congenital stationary night blindness? Yes. About 45% of individuals with XLCSNB have the complete form, which is caused by mutations in the NYX gene. The other 55% have the i...

**Answer**:

Yes, information about genetic testing is available for certain conditions. For hereditary sensory neuropathy type 1 (HSN1), at least four genes have been identified (HSN1A, HSN1B, HSN1C, HSN1D, HSN1E), and the Genetic Testing Registry (GTR) provides information about genetic testing for HSN1A. However, clinical laboratories offering testing for other types of HSN1 are not currently known, and patients should consult a health care provider or genetics professional to locate testing options [medquad_ans_0002853-4].

For congenital bilateral absence of the vas deferens (CBAVD), GeneTests lists laboratories performing genetic testing, though most do not accept direct contact from patients; patients should work with a health care provider or genetics professional to access testing [medquad_ans_0001483-5].

For

**Supported citations**: `medquad_ans_0002853-4`, `medquad_ans_0001483-5`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.550, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.8681) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0004739-1` (score=0.0244) - Parkes Weber syndrome (PWS) is a rare congenital condition causing an individual to have a large number of abnormal blood vessels. The main characteristics of PWS typically include a capillary malformation on the skin...
3. `medquad_ans_0005994-1` (score=0.0049) - Systemic capillary leak syndrome is a condition in which fluid and proteins leak out of tiny blood vessels and flow into surrounding tissues, resulting in dangerously low blood pressure. Attacks frequently last for se...
4. `medquad_ans_0000641-1` (score=0.0032) - Megalencephaly-capillary malformation syndrome (MCAP) is a disorder characterized by overgrowth of several tissues in the body. Its primary features are a large brain (megalencephaly) and abnormalities of small blood ...
5. `medquad_ans_0000654-1` (score=0.0004) - Microcephaly-capillary malformation syndrome is an inherited disorder characterized by an abnormally small head size (microcephaly) and abnormalities of small blood vessels in the skin called capillaries (capillary ma...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or develop during childhood. In most cases, capillary malformations do not cause health problems and may be the only sign of the disorder. However, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries. These abnormalities can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure, with some complications

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.9994) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
2. `medquad_ans_0001994-1` (score=0.0010) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the s...
3. `medquad_ans_0001995-1` (score=0.0003) - What are the signs and symptoms of Dystonia 19? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 19. If the information is available, the table below includes how often the s...
4. `medquad_ans_0003444-2` (score=0.0001) - What are the symptoms of La Crosse (LAC) encephalitis? Most people infected with LAC encephalitis do not have symptoms. Those that do become ill may initially have fever, headache, vomiting and lethargy (tiredness). S...
5. `medquad_ans_0006128-1` (score=0.0001) - What are the signs and symptoms of Torsion dystonia with onset in infancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Torsion dystonia with onset in infancy. If the information ...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (5%), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive features, retrocollis [medquad_ans_0001992-1].

**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.714, groundedness=0.958, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000899-5` (score=0.9950) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
2. `medquad_ans_0005548-5` (score=0.9652) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
3. `medquad_ans_0000899-3` (score=0.0001) - In most cases of septo-optic dysplasia, the cause of the disorder is unknown. Researchers suspect that a combination of genetic and environmental factors may play a role in causing this disorder. Proposed environmenta...
4. `medquad_ans_0000001-1__dup1` (score=0.0001) - The septum pellucidum (SP) is a thin membrane located at the midline of the brain between the two cerebral hemispheres, or halves of the brain.. It is connected to the corpus callosum -- a collection of nerve fibers t...
5. `medquad_ans_0000001-2__dup1` (score=0.0001) - Absence of the SP alone is not a disorder but is instead a characteristic noted in children with septo-optic dysplasia or other developmental anomalies.

**Answer**:

Treatment for septo-optic dysplasia is symptomatic and includes hormone replacement therapy for hormone deficiencies. Optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5].

**Supported citations**: `medquad_ans_0005548-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=0.900, groundedness=0.994, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000208-7` (score=0.9931) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney da...
2. `medquad_ans_0000208-11` (score=0.9898) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men ...
3. `medquad_ans_0000213-10` (score=0.0054) - People can prevent urinary retention before it occurs by treating some of the potential causes. For example, men with benign prostatic hyperplasia should take prostate medications as prescribed by their health care pr...
4. `medquad_ans_0000213-9` (score=0.0052) - Complications of urinary retention and its treatments may include - UTIs - bladder damage - kidney damage - urinary incontinence after prostate, tumor, or cancer surgery UTIs. Urine is normally sterile, and the normal...
5. `medquad_ans_0000211-4__dup2` (score=0.0039) - Urinary incontinence in men results when the brain does not properly signal the bladder, the sphincters do not squeeze strongly enough, or both. The bladder muscle may contract too much or not enough because of a prob...

**Answer**:

The complications of benign prostatic hyperplasia (prostate enlargement) may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney damage - bladder stones. Most men with benign prostatic hyperplasia do not develop these complications. However, kidney damage in particular can be a serious health threat when it occurs [medquad_ans_0000208-7].

Complications related to treatment include side effects from medications such as hives, rash, itching, shortness of breath, rapid or irregular heartbeat, painful erection lasting hours, swelling of the face, eyes, lips, tongue, throat, arms, hands, feet, ankles, or lower legs, difficulty breathing or swallowing, chest pain, dizziness or fainting when standing up suddenly, sudden decrease or loss of vision, blurred vision, sudden decrease or loss of hearing,

**Supported citations**: `medquad_ans_0000208-7`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.889, groundedness=0.797, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000001-3__dup1` (score=0.0001) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-t...
2. `medquad_ans_0003726-1` (score=0.0001) - What are the signs and symptoms of Macrocephaly-capillary malformation? The Human Phenotype Ontology provides the following list of signs and symptoms for Macrocephaly-capillary malformation. If the information is ava...
3. `medquad_ans_0000001-1__dup1` (score=0.0001) - The septum pellucidum (SP) is a thin membrane located at the midline of the brain between the two cerebral hemispheres, or halves of the brain.. It is connected to the corpus callosum -- a collection of nerve fibers t...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The provided context does not mention any treatments for absence of the septum pellucidum. [medquad_ans_0000001-3__dup1] states that absence of the septum pellucidum is not life-threatening and is part of septo-optic dysplasia, with prognosis depending on associated symptoms, but it does not address treatments. Therefore, based on the given information, no treatments are specified.

**Supported citations**: `medquad_ans_0000001-3__dup1`
**Unsupported citations**: _none_

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.517, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.9992) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0000128-2__dup1` (score=0.0065) - A child who has a febrile seizure usually doesn't need to be hospitalized. If the seizure is prolonged or is accompanied by a serious infection, or if the source of the infection cannot be determined, a doctor may rec...
3. `medquad_ans_0000128-3__dup1` (score=0.0031) - The vast majority of febrile seizures are short and harmless. There is no evidence that short febrile seizures cause brain damage. Multiple or prolonged seizures are a risk factor for epilepsy but most children who ex...
4. `medquad_ans_0002322-3` (score=0.0001) - What causes febrile ulceronecrotic Mucha-Habermann disease? The cause of FUMHD is not known (idiopathic). A hypersensitivity to an infectious agent is suggested to be the main cause. Single cases of people with FUMHD ...
5. `medquad_ans_0002322-1` (score=0.0001) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sor...

**Answer**:

The symptoms of Febrile Ulceronecrotic Mucha-Habermann disease (FUMHD) include red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). The lesions can progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can become life

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002833-4` (score=0.9639) - How is hereditary leiomyomatosis and renal cell cancer inherited? Hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the ...
2. `medquad_ans_0000137-4` (score=0.0005) - Most cases of breast cancer are not caused by inherited genetic factors. These cancers are associated with somatic mutations in breast cells that are acquired during a person's lifetime, and they do not cluster in fam...
3. `medquad_ans_0000019_1-3` (score=0.0001) - Some ovarian, fallopian tube, and primary peritoneal cancers are caused by inherited gene mutations (changes). The genes in cells carry the hereditary information that is received from a persons parents. Hereditary ov...
4. `medquad_ans_0000026_3-2` (score=0.0001) - Having certain genetic syndromes or other conditions can increase the risk of Wilms tumor. Anything that increases the risk of getting a disease is called a risk factor. Having a risk factor does not mean that you wil...
5. `medquad_ans_0002833-1` (score=0.0001) - Hereditary leiomyomatosis and renal cell cancer (HLRCC) is a condition that causes benign tumors of smooth muscle tissue in the skin (cutaneous leiomyomas) and in the uterus in females (uterine leiomyomas, or fibroids...

**Answer**:

Yes, hereditary leiomyomatosis and renal cell cancer (HLRCC) is inherited in an autosomal dominant pattern, which means that having one mutated copy of the gene in each cell is enough to cause symptoms of the condition [medquad_ans_0002833-4].

**Supported citations**: `medquad_ans_0002833-4`
**Unsupported citations**: _none_

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.9994) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
2. `medquad_ans_0000389-1` (score=0.0001) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
3. `medquad_ans_0002374-2` (score=0.0001) - What are the symptoms of fibrous dysplasia? Fibrous dysplasia may cause no symptoms, mild symptoms, or severe symptoms. The most common symptoms are bone pain, bone deformities, fractures, and skin pigmentation differ...
4. `medquad_ans_0000389-3` (score=0.0001) - Mutations in the FLNA gene cause frontometaphyseal dysplasia. The FLNA gene provides instructions for producing the protein filamin A, which helps build the network of protein filaments (cytoskeleton) that gives struc...
5. `medquad_ans_0003571-1` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of the frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of the finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of the finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, and ureteral st

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.909, groundedness=0.780, answer_relevance=0.909

**Top docs**:

1. `medquad_ans_0000193-6` (score=0.9410) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
2. `medquad_ans_0000128-10` (score=0.8487) - Eat a high-fiber diet to help prevent problems. Talk to your doctor about using fiber products like Benefiber, Citrucel, or Metamucil. Daily use can help you get the fiber you need if you do not get it through your di...
3. `medquad_ans_0000591-1__dup1` (score=0.0480) - Summary : Medicines cure infectious diseases, prevent problems from chronic diseases, and ease pain. But medicines can also cause harmful reactions if not used correctly. Errors can happen in the hospital, at the doct...
4. `medquad_ans_0000035-9` (score=0.0041) - People can manage their diabetes with meal planning, physical activity, and if needed, medications. More information about taking care of type 1 or type 2 diabetes is provided in the NIDDK health topics: - What I need...
5. `medquad_ans_0000057-22` (score=0.0026) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions....

**Answer**:

The provided context does not include specific information about treatments for diverticular disease. While it mentions that eating a high-fiber diet can help reduce diverticular disease symptoms and problems, and lists examples of high-fiber foods, it does not outline comprehensive treatments for diverticular disease. Therefore, based on the given context, the treatments for "What I need to know about Diverticular Disease" cannot be fully answered.

[medquad_ans_0000128-10]

**Supported citations**: `medquad_ans_0000128-10`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.534, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.9995) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
4. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomeg

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003911-2` (score=0.9540) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a men...
2. `medquad_ans_0000084-2__dup1` (score=0.0318) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
3. `medquad_ans_0005174-2` (score=0.0010) - What treatments are available for pudendal neuralgia? There are no established guidelines for the treatment of pudendal neuralgia. Treatment may include medications, the use of a specialized cushion while sitting, ner...
4. `medquad_ans_0001354-2` (score=0.0003) - What treatments for metastatic clear cell renal cell carcinoma are available in North America? There are several treatments for metastatic clear cell renal cell carcinoma available in North America. IL-2 and sunitinib...
5. `medquad_ans_0002796-4` (score=0.0002) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...

**Answer**:

The treatment for meningiomas varies depending on the location of the meningioma and the symptoms it causes. Careful observation may be the best course of action for some individuals. When treatment is necessary, surgery and radiation are the most common forms of treatment. Radiation may be used if the meningioma cannot be operated on or if it is only partially removed by surgery. Radiation is also used in cases of malignant, atypical, or recurrent tumors. Other treatments that have been tried or are being explored include hydroxyurea, epidermal growth factor receptor inhibitors, platelet-derived growth factor receptor inhibitors, vascular endothelial growth factor inhibitors, immunotherapy to stimulate the immune system, and somatostatin analogs which prevent the release of growth hormones [medquad_ans_0003911-2].

**Supported citations**: `medquad_ans_0003911-2`
**Unsupported citations**: _none_

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-8` (score=0.9738) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
2. `medquad_ans_0000015_1-8` (score=0.9636) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers...
3. `medquad_ans_0000006_6-8` (score=0.0341) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health...
4. `medquad_ans_0000006_4-8` (score=0.0284) - Key Points - There are different types of treatment for patients with central nervous system atypical teratoid/rhabdoid tumor. - Children with atypical teratoid/rhabdoid tumor should have their treatment planned by a ...
5. `medquad_ans_0000006_7-7` (score=0.0256) - Check the list of NCI-supported cancer clinical trials that are now accepting patients with childhood central nervous system germ cell tumor. For more specific results, refine the search by using other search features...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and participation in clinical trials for new treatments [medquad_ans_0000006_7-8]. Treatment is typically planned by a team of healthcare providers who are experts in treating cancer in children, including a pediatric oncologist and/or a radiation oncologist, along with other specialists such as a pediatric neurosurgeon, neurologist, endocrinologist, ophthalmologist, pediatric nurse specialist, rehabilitation specialist, psychologist, and social worker [medquad_ans_0000006_7-8]. Some treatments are standard, while others are being tested in clinical trials [medquad_ans_0000006_7-8]. Patients may consider participating in clinical trials before, during, or after starting treatment [med

**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=0.889, groundedness=0.986, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0004401-1` (score=0.9127) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
2. `medquad_ans_0000729-1` (score=0.8874) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0000123-4__dup1` (score=0.0019) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
4. `medquad_ans_0000002-4__dup1` (score=0.0017) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge abut the brain and nervous system, and to use that knowledge to reduce the burden of neurological dis...
5. `medquad_ans_0000065-4__dup1` (score=0.0015) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...

**Answer**:

Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene and is inherited in an autosomal recessive pattern. Other features may include a fatty liver, a weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. Signs and symptoms vary greatly among affected individuals. There is currently no treatment to correct the underlying metabolic problem [medquad_ans_0004401-1][medquad_ans_0000729-1].

**Supported citations**: `medquad_ans_0004401-1`, `medquad_ans_0000729-1`
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.479, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000981-1__dup1` (score=0.4161) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
2. `medquad_ans_0006546-1` (score=0.0613) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
3. `medquad_ans_0000092-1__dup1` (score=0.0066) - Cytomegalovirus (CMV) is a virus found throughout the world that infects between 50 to 80 percent of all adults in the United States by the age of 40. CMV is in the same family of viruses that causes cold sores (herpe...
4. `medquad_ans_0000193-1__dup2` (score=0.0005) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may dev...
5. `medquad_ans_0000228-2__dup2` (score=0.0001) - After an incubation of 9-16 days, infection with Hendra virus can lead to respiratory illness with severe flu-like signs and symptoms. In some cases, illness may progress to encephalitis. Although infection with Hendr...

**Answer**:

Zika virus is a virus spread primarily through the bite of an infected mosquito. It can also be transmitted from a pregnant mother to her child, through sexual contact with an affected male partner, and through blood transfusions. Most people infected with Zika virus do not develop symptoms, but those who do may experience mild illness with symptoms such as fever, rash, joint pain, and conjunctivitis, which typically begin 2 to 7 days after being bitten by an infected mosquito. The virus has been linked to serious birth defects, including microcephaly and other brain anomalies in babies born to mothers infected during pregnancy. There is no vaccine or specific medicine to treat Zika virus, and management focuses on rest, hydration, and use of acetaminophen. The Centers for Disease Control and Prevention recommends that pregnant women avoid travel to areas with active Zika virus transmission and take precautions to prevent mosquito bites [medquad_ans_0000981-

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000019-3__dup2` (score=0.9850) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
2. `medquad_ans_0000060-3__dup2` (score=0.3281) - Research suggests thatcoronary heart disease(CHD) begins with damage to the lining and inner layers of the coronary (heart) arteries. Several factors contribute to this damage. They include: Smoking, including secondh...
3. `medquad_ans_0002911-3` (score=0.0069) - What causes Holt-Oram syndrome? Holt-Oram syndrome is caused by changes (mutations) in the TBX5 gene. This gene gives the body instructions for making a protein involved in the development of the heart and upper limbs...
4. `medquad_ans_0000126-3__dup2` (score=0.0012) - Doctors often don't know what causes tetralogy of Fallot and other congenital heart defects. Some conditions or factors that occur during pregnancy may raise your risk of having a child who has tetralogy of Fallot. Th...
5. `medquad_ans_0000019-8` (score=0.0004) - Researchers are still learning about broken heart syndrome, and no treatments have been shown to prevent it. For people who have experienced the condition, the risk of recurrence is low. An emotionally upsetting or se...

**Answer**:

Broken heart syndrome is believed to be caused by extreme emotional or physical stress, which triggers a surge of fight or flight hormones such as adrenaline and noradrenaline. This hormone surge overwhelms and "stuns" the heart muscle, leading to temporary changes in heart muscle cells or coronary blood vessels. Potential causes include hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=1.000, groundedness=0.500, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000892-1__dup1` (score=0.8647) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...
2. `medquad_ans_0000005-1__dup5` (score=0.4499) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
3. `medquad_ans_0000448-1` (score=0.2328) - Hashimoto thyroiditis is a condition that affects the function of the thyroid, which is a butterfly-shaped gland in the lower neck. The thyroid makes hormones that help regulate a wide variety of critical body functio...
4. `medquad_ans_0000008-3__dup4` (score=0.0036) - Hyperthyroidism has several causes, including - Graves disease - thyroid nodules - thyroiditis, or inflammation of the thyroid - consuming too much iodine - overmedicating with synthetic thyroid hormone, which is used...
5. `medquad_ans_0000448-2` (score=0.0024) - Hashimoto thyroiditis affects 1 to 2 percent of people in the United States. It occurs more often in women than in men, which may be related to hormonal factors. The condition is the most common cause of thyroid under...

**Answer**:

Hashimoto thyroiditis is a condition that affects the function of the thyroid, a butterfly-shaped gland in the lower neck. It is a form of chronic inflammation that can damage the thyroid, reducing its ability to produce hormones. In Hashimoto thyroiditis, the immune system attacks the thyroid gland, causing inflammation and interfering with its ability to produce thyroid hormones. This immune attack is due to an autoimmune response where large numbers of white blood cells called lymphocytes accumulate in the thyroid and produce antibodies that initiate the autoimmune process. Hashimoto thyroiditis often leads to reduced thyroid function, or hypothyroidism, which occurs when the thyroid does not make enough thyroid hormone for the body's needs. Hypothyroidism slows down many bodily functions, including metabolism, body temperature regulation, heart rate, menstrual cycles, and weight. One of the first signs of Hashimoto thyroiditis is an enlargement of the thyroid called a goiter [medquad_ans_00004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=1.000, groundedness=0.504, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.7181) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
2. `medquad_ans_0006463-1` (score=0.6708) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
3. `medquad_ans_0001043-2` (score=0.0003) - Wolman disease is estimated to occur in 1 in 350,000 newborns.
4. `medquad_ans_0002259-1` (score=0.0003) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in body organs such as the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are typically healthy at birth but soon develop symptoms including enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor nutrient absorption. The disease is severe and life-threatening, with most affected children not surviving past early childhood. The estimated incidence is 1 in 350,000 newborns. New therapies such as bone marrow transplantation and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.9955) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0004053-3` (score=0.0003) - What causes microscopic polyangiitis (MPA)? The cause of MPA is unknown. It is not contagious, does not usually run in families, and is not a form of cancer. The immune system is thought to play a critical role in the...
3. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
4. `medquad_ans_0002753-1` (score=0.0001) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...
5. `medquad_ans_0002946-1` (score=0.0001) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.875, groundedness=0.948, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000655-3` (score=0.9980) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
2. `medquad_ans_0000593-3` (score=0.0024) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a ...
3. `medquad_ans_0000142-3` (score=0.0002) - C3 glomerulopathy is associated with changes in many genes. Most of these genes provide instructions for making proteins that help regulate a part of the body's immune response known as the complement system. This sys...
4. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000290-3` (score=0.0001) - Distal arthrogryposis type 1 can be caused by mutations in at least two genes: TPM2 and MYBPC1. These genes are active (expressed) in muscle cells, where they interact with other muscle proteins to help regulate the t...

**Answer**:

Genetic changes related to microphthalmia include mutations in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from chromosomal abnormalities affecting one or more genes. In the case of Lenz microphthalmia syndrome, mutations in at least two genes on the X chromosome are thought to be responsible, with only the BCOR gene currently identified. The BCOR gene provides instructions for making a protein called the BCL6 corepressor, which helps regulate the activity of other genes and appears to play an important role in early embryonic development. A mutation in the BCOR gene changes the structure of the BCL6 corepressor protein, disrupting normal eye and organ development before birth [medquad_ans_0000593-3]. Most genetic changes associated with isolated microphthalmia have been identified in only a small number of affected individuals [medquad_ans_00006

**Supported citations**: `medquad_ans_0000593-3`
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.738, answer_relevance=0.767

**Top docs**:

1. `medquad_ans_0000028-15` (score=0.0029) - Physicians often prescribe high doses of non-steroidal anti-inflammatory drugs (NSAIDs) or steroids for a sudden attack of gout. NSAIDs are taken by mouth and corticosteroids are either taken by mouth or injected into...
2. `medquad_ans_0000028-9` (score=0.0026) - Most people with gout have too much uric acid in their blood, a condition called hyperuricemia. Uric acid is a substance that results from the breakdown of purines, which are part of all human tissue and are found in ...
3. `medquad_ans_0000028-4__dup3` (score=0.0005) - Symptoms Can Be Controlled With proper treatment, most people with gout are able to control their symptoms and live productive lives. The goals for treatment are to ease the pain that comes from sudden attacks, preven...
4. `medquad_ans_0000028-3__dup3` (score=0.0005) - Attacks Usually Start at Night Gout is a form of arthritis that causes sudden onset of intense pain and swelling in the joints, which also may be warm and red. Gout typically attacks one joint at a time, and the attac...
5. `medquad_ans_0000028-13` (score=0.0001) - Gout is a form of arthritis that frequently affects joints in the lower part of the body such as the knees, ankles, or toes. The affected joint may become swollen, red, or warm. Attacks usually occur at night. Sometim...

**Answer**:

The provided context does not explicitly describe the symptoms of gout. It discusses treatments and the underlying condition of hyperuricemia but does not list specific symptoms such as pain, swelling, redness, or joint inflammation associated with gout. Therefore, based on the given information, the symptoms of gout cannot be determined.

[medquad_ans_0000028-9] and [medquad_ans_0000028-4__dup3] mention that symptoms can be controlled with treatment and that the goals include easing pain from sudden attacks and preventing future attacks, but they do not specify what the symptoms are. Thus, the context is insufficient to answer the question directly.

**Supported citations**: `medquad_ans_0000028-9`, `medquad_ans_0000028-4__dup3`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.667, groundedness=0.087, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000309-5` (score=0.0001) - These resources address the diagnosis or management of dystonia 6: - Gene Review: Gene Review: Dystonia Overview - Genetic Testing Registry: Dystonia 6, torsion These resources from MedlinePlus offer information about...
2. `medquad_ans_0000307-1__dup1` (score=0.0001) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
3. `medquad_ans_0000860-3` (score=0.0001) - Recombinant 8 syndrome is caused by a rearrangement of chromosome 8 that results in a deletion of a piece of the short (p) arm and a duplication of a piece of the long (q) arm. The deletion and duplication result in t...
4. `medquad_ans_0000309-3` (score=0.0001) - Dystonia 6 is caused by mutations in the THAP1 gene. This gene provides instructions for making a protein that is a transcription factor, which means that it attaches (binds) to specific regions of DNA and regulates t...
5. `medquad_ans_0000312-1` (score=0.0001) - Early-onset primary dystonia is a condition characterized by progressive problems with movement, typically beginning in childhood. Dystonia is a movement disorder that involves involuntary tensing of the muscles (musc...

**Answer**:

The provided context does not contain information about the cause of Dystonia 8. Therefore, it is not possible to answer the question based on the given text. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.889, groundedness=0.978, answer_relevance=0.444

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.9993) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
2. `medquad_ans_0000573-2` (score=0.0033) - Lactose intolerance in infancy resulting from congenital lactase deficiency is a rare disorder. Its incidence is unknown. This condition is most common in Finland, where it affects an estimated 1 in 60,000 newborns. A...
3. `medquad_ans_0000114-10` (score=0.0001) - People may find it helpful to talk with a health care provider or a registered dietitian about a dietary plan. A dietary plan can help people manage the symptoms of lactose intolerance and make sure they get enough nu...
4. `medquad_ans_0000131-2__dup2` (score=0.0001) - Lactose intolerance means you have symptoms such as bloating, diarrhea, and gas after you have milk or milk products. If your small intestine does not produce much lactase, you cannot break down much lactose. Lactose ...
5. `medquad_ans_0002825-2` (score=0.0001) - What are the signs and symptoms of Hereditary fructose intolerance? The symptoms of HFI include: Poor feeding as a baby Irritability Increased or prolonged neonatal jaundice Vomiting Convulsions Excessive sleepiness I...

**Answer**:

Lysinuric protein intolerance affects an estimated 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.800, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004373-2` (score=0.0001) - What are the signs and symptoms of Neurofibromatosis type 1? People affected by neurofibromatosis type 1 (NF1) have an increased risk of developing many different types of tumors (both cancerous and noncancerous). Alm...
2. `medquad_ans_0000170-8` (score=0.0001) - Treatment for RAS includes lifestyle changes, medications, and surgery and aims to - prevent RAS from getting worse - treat RVH - relieve the blockage of the renal arteries RAS that has not led to RVH or caused a sign...
3. `medquad_ans_0000170-10` (score=0.0001) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart ...
4. `medquad_ans_0000840-2__dup1` (score=0.0001) - What are the signs and symptoms of Branchiootorenal syndrome? Signs and symptoms of branchiootorenal syndrome can vary greatly from person to person and even between people within the same family. Hearing loss is the ...
5. `medquad_ans_0005294-1` (score=0.0001) - What are the signs and symptoms of Renal dysplasia-limb defects syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Renal dysplasia-limb defects syndrome. If the information is...

**Answer**:

The provided context does not contain information about the symptoms of Renal Artery Stenosis. Therefore, a response cannot be generated based on the given context. [medquad_ans_0004373-2]

**Supported citations**: `medquad_ans_0004373-2`
**Unsupported citations**: _none_

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=1.000, groundedness=0.788, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.9908) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
2. `medquad_ans_0004947-4` (score=0.0002) - What treatments are available for itching related to polycythemia vera? There are several treatments for the itching (pruritus) related to polycythemia vera (PV). No single treatment has been found to be effective for...
3. `medquad_ans_0005174-2` (score=0.0001) - What treatments are available for pudendal neuralgia? There are no established guidelines for the treatment of pudendal neuralgia. Treatment may include medications, the use of a specialized cushion while sitting, ner...
4. `medquad_ans_0002796-4` (score=0.0001) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
5. `medquad_ans_0003103-4` (score=0.0001) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...

**Answer**:

The provided context does not contain specific information about treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While the context lists these conditions in the context of genetic testing resources [medquad_ans_0004947-4], it does not detail any available treatments for these syndromes. Therefore, based on the given information, the treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis cannot be determined.

**Supported citations**: `medquad_ans_0004947-4`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.9994) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000353-1` (score=0.0001) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control ...
3. `medquad_ans_0002593-1` (score=0.0001) - Glucocorticoid-remediable aldosteronism is one of three types of familial hyperaldosteronism and was first described in 1966. Aldosterone is a hormone manufactured by the adrenal glands which helps the body retain wat...
4. `medquad_ans_0000340-1` (score=0.0001) - Familial adenomatous polyposis (FAP) is an inherited disorder characterized by cancer of the large intestine (colon) and rectum. People with the classic type of familial adenomatous polyposis may begin to develop mult...
5. `medquad_ans_0000680-1` (score=0.0001) - Mucopolysaccharidosis type III (MPS III), also known as Sanfilippo syndrome, is a progressive disorder that mainly affects the brain and spinal cord (central nervous system). People with MPS III generally do not displ...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=1.000, groundedness=0.952, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005761-3` (score=0.9580) - How is spinocerebellar ataxia type 11 inherited? SCA11 is inherited in an autosomal dominant manner. The rate of de novo mutations is not known. Each child of an individual with SCA11 has a 50% chance of inheriting th...
2. `medquad_ans_0004530-2` (score=0.0114) - Is olivopontocerebellar atrophy inherited? Olivopontocerebellar atrophy (OPCA) may be associated with conditions that are inherited (genetic), or it may occur sporadically. Genetic forms of OPCA may be inherited in an...
3. `medquad_ans_0000032-1__dup1` (score=0.0001) - Ataxia often occurs when parts of the nervous system that control movement are damaged. People with ataxia experience a failure of muscle control in their arms and legs, resulting in a lack of balance and coordination...
4. `medquad_ans_0005768-2` (score=0.0001) - What are the signs and symptoms of Spinocerebellar ataxia 2? Early symptoms of spinocerebellar ataxia may include uncoordinated movement (ataxia) and leg cramps. Other symptoms may include tremor; decreased muscle ton...
5. `medquad_ans_0005761-1` (score=0.0001) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...

**Answer**:

Yes, Spinocerebellar ataxia type 11 (SCA11) is inherited in an autosomal dominant manner. Each child of an individual with SCA11 has a 50% chance of inheriting the mutation [medquad_ans_0005761-3].

**Supported citations**: `medquad_ans_0005761-3`
**Unsupported citations**: _none_

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0003040-1` (score=0.9993) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
2. `medquad_ans_0000503-1` (score=0.0001) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carr...
3. `medquad_ans_0000016-2__dup3` (score=0.0001) - Anemia of inflammation and chronic disease is a type of anemia that commonly occurs with chronic, or long term, illnesses or infections. Cancer and inflammatory disorders, in which abnormal activation of the immune sy...
4. `medquad_ans_0000026-3` (score=0.0001) - African iron overload was first noted in rural central and southern African populations among people who drink a traditional beer brewed in uncoated steel drums that allow iron (a component of steel) to leach into the...
5. `medquad_ans_0000503-2` (score=0.0001) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.

**Answer**:

The symptoms of Hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth. Additionally, the condition leads to abnormally small and pale red blood cells (microcytic and hypochromic), and iron accumulates in the liver, which can impair liver function over time, typically becoming apparent in adolescence or early adulthood [medquad_ans_0000503-1].

**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=1.000, groundedness=0.718, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000117-4` (score=0.9978) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
2. `medquad_ans_0002290-1` (score=0.0001) - Familial pemphigus vulgaris refers to a cluster of pemphigus vulgaris within a family. Pemphigus vulgaris is a rare autoimmune condition that is characterized by blisters and sores on the skin and mucus membranes. Alt...
3. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0000115-1` (score=0.0001) - Benign chronic pemphigus, often called Hailey-Hailey disease, is a rare skin condition that usually appears in early adulthood. The disorder is characterized by red, raw, and blistered areas of skin that occur most of...
5. `medquad_ans_0004796-1` (score=0.0001) - Pemphigus vulgaris is an autoimmune disorder that involves blistering of the skin and mucous membranes. It occurs almost exclusively in middle-aged or older people. Many cases begin with blisters in the mouth, followe...

**Answer**:

The provided context does not specify whether benign chronic pemphigus (Hailey-Hailey disease) is inherited. While it describes the symptoms, triggers, and severity of the condition, there is no information about its inheritance pattern. Therefore, based on the given context, it cannot be determined if benign chronic pemphigus is inherited.

[medquad_ans_0000115-1]

**Supported citations**: `medquad_ans_0000115-1`
**Unsupported citations**: _none_

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000916-4__dup1` (score=0.9533) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0000915-1__dup1` (score=0.0001) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
5. `medquad_ans_0000916-1__dup1` (score=0.0001) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is enough to cause the condition. [medquad_ans_0000916-4__dup1]

**Supported citations**: `medquad_ans_0000916-4__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.900, groundedness=0.954, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000483-3` (score=0.9990) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family o...
2. `medquad_ans_0000193-3` (score=0.0035) - Coats plus syndrome results from mutations in the CTC1 gene. This gene provides instructions for making a protein that plays an important role in structures known as telomeres, which are found at the ends of chromosom...
3. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
4. `medquad_ans_0000483-2` (score=0.0001) - Histiocytosis-lymphadenopathy plus syndrome is a rare disorder, affecting approximately 100 individuals worldwide.
5. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene [medquad_ans_0000483-3]. These mutations reduce or eliminate the activity of the equilibrative nucleoside transporter 3 (ENT3) protein, which is responsible for transporting nucleosides out of lysosomes and into mitochondria [medquad_ans_0000483-3].

**Supported citations**: `medquad_ans_0000483-3`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004450-3` (score=0.9363) - How is Noonan syndrome inherited? Noonan syndrome is inherited in an autosomal dominant manner. This means that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. ...
2. `medquad_ans_0004457-1` (score=0.0001) - Noonan-like syndrome with loose anagen hair is characterized by facial features suggestive of Noonan syndrome (macrocephaly, high forehead, wide-set eyes or hypertelorism, palpebral ptosis, and low-set and posteriorly...
3. `medquad_ans_0004451-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004453-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004456-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Yes, Noonan syndrome is typically inherited in an autosomal dominant manner, meaning that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. Each child of a person with Noonan syndrome has a 50% (1 in 2) chance to inherit the condition. However, many cases are due to a new mutation that occurs for the first time in the affected person and are not inherited from an affected parent [medquad_ans_0004450-3]. [medquad_ans_0004451-1]

**Supported citations**: `medquad_ans_0004450-3`, `medquad_ans_0004451-1`
**Unsupported citations**: _none_

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.875, groundedness=0.799, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0001093-5` (score=0.9952) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
2. `medquad_ans_0000445-3__dup1` (score=0.0260) - How might arginase deficiency be treated? The treatment and management of arginase deficiency is generally focused on lowering arginine levels and preventing hyperammonemia (an accumulation of ammonia in the blood). T...
3. `medquad_ans_0003264-5` (score=0.0018) - How is isobutyryl-CoA dehydrogenase deficiency (IBD deficiency) treated? There is no standard treatment protocol for IBD deficiency. Infants diagnosed through newborn screening are encouraged to work with a metabolic ...
4. `medquad_ans_0004456-3` (score=0.0016) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004450-4` (score=0.0016) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain information about the treatments for 3-methylglutaconyl-CoA hydratase deficiency. [medquad_ans_0001093-5] lists resources that address the diagnosis or management of this condition, such as genetic testing and information on metabolic acidosis, but does not specify any treatments. Therefore, based on the given context, the treatments for 3-methylglutaconyl-CoA hydratase deficiency cannot be determined.

**Supported citations**: `medquad_ans_0001093-5`
**Unsupported citations**: _none_

### Example 61 (`0003000-1`)

**Question**: What is (are) Hyperkalemic periodic paralysis ?

**Scores**: context_relevance=0.833, groundedness=0.994, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003000-1` (score=0.9120) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move...
2. `medquad_ans_0000505-1` (score=0.7986) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...
3. `medquad_ans_0000494-1` (score=0.7255) - Hyperkalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness or paralysis, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to...
4. `medquad_ans_0003059-1` (score=0.6926) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...
5. `medquad_ans_0000126-1__dup1` (score=0.0111) - Familial periodic paralyses are a group of inherited neurological disorders caused by mutations in genes that regulate sodium and calcium channels in nerve cells. They are characterized by episodes in which the affect...

**Answer**:

Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until about age 25, after which they may occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods, stress, fatigue, and long periods without food. Muscle strength improves between attacks, although many affected people continue to experience mild stiffness, particularly in muscles of the face and hands. This condition is caused by mutations in the SCN4A gene and is inherited in an autosomal dominant fashion [medquad_ans_0003000-1].

**Supported citations**: `medquad_ans_0003000-1`
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.900, groundedness=0.849, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.9944) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
2. `medquad_ans_0000084-2__dup1` (score=0.0073) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
3. `medquad_ans_0000225-3` (score=0.0001) - CFEOM1 and rare cases of CFEOM3 result from mutations in the KIF21A gene. This gene provides instructions for making a protein called a kinesin, which is essential for the transport of materials within cells. Research...
4. `medquad_ans_0000225-2` (score=0.0001) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few fami...
5. `medquad_ans_0000017_1-7` (score=0.0001) - Key Points - There are different types of treatments for patients with intraocular melanoma. - Five types of standard treatment are used: - Surgery - Watchful Waiting - Radiation therapy - Photocoagulation - Thermothe...

**Answer**:

The provided context does not contain information about treatments for congenital fibrosis of the extraocular muscles (CFEOM). While the context mentions resources related to the diagnosis and management of CFEOM, including genetic testing and information on extraocular muscle function, it does not specify any treatments. Therefore, based on the available information, the treatments for congenital fibrosis of the extraocular muscles cannot be determined. [medquad_ans_0000225-5]

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.521, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.6731) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0000152-1__dup1` (score=0.4376) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressiv...
3. `medquad_ans_0000075-1__dup2` (score=0.0710) - Chorea is an abnormal involuntary movement disorder, one of a group of neurological disorders called dyskinesias, which are caused by overactivity of the neurotransmitter dopamine in the areas of the brain that contro...
4. `medquad_ans_0000013-9__dup1` (score=0.0022) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...
5. `medquad_ans_0000075-3__dup1` (score=0.0002) - The prognosis for individuals with chorea varies depending on the type of chorea and the associated disease. Huntington's disease is a progressive, and ultimately, fatal disease. Syndenham's chorea is treatable and cu...

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. It is caused by a mutation in the gene for a protein called huntingtin, which results in the cytosine, adenine, and guanine (CAG) building blocks of DNA repeating many more times than is normal. Symptoms, which progressively worsen, include uncontrolled movements (called chorea), abnormal body postures, changes in behavior, emotion, judgment, and cognition, impaired coordination, slurred speech, and difficulty feeding and swallowing. HD typically begins between ages 30 and 50, though an earlier onset form called juvenile HD occurs under age 20, with symptoms such as unsteadiness, rigidity, difficulty at school, and seizures. Individuals born with the defective gene have a 50 percent chance of inheriting it and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000416-2` (score=0.9992) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
2. `medquad_ans_0002601-2` (score=0.0001) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the inform...
3. `medquad_ans_0002601-1` (score=0.0001) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0000416-1` (score=0.0001) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...
5. `medquad_ans_0001074-1` (score=0.0001) - 17 alpha()-hydroxylase/17,20-lyase deficiency is a condition that affects the function of certain hormone-producing glands called the gonads (ovaries in females and testes in males) and the adrenal glands. The gonads ...

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.678, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0004453-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
2. `medquad_ans_0004450-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004451-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004454-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004452-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not specify the number of people affected by Noonan syndrome. [medquad_ans_0004453-1] [medquad_ans_0004450-1] [medquad_ans_0004451-1] [medquad_ans_0004454-1] describe the features, genetic causes, and inheritance pattern of Noonan syndrome but do not include epidemiological data on prevalence or affected population size. Therefore, the number of people affected by Noonan syndrome cannot be determined from the given information.

**Supported citations**: `medquad_ans_0004453-1`, `medquad_ans_0004450-1`, `medquad_ans_0004451-1`, `medquad_ans_0004454-1`
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.571, groundedness=0.989, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000053-5__dup2` (score=0.9966) - Body lice infestation is diagnosed by finding eggs and crawling lice in the seams of clothing. Sometimes a body louse can be seen crawling or feeding on the skin. Although body lice and nits can be large enough to be ...
2. `medquad_ans_0000339-5__dup1` (score=0.9943) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
3. `medquad_ans_0000214-5__dup1` (score=0.0870) - Misdiagnosis of head lice infestation is common. The diagnosis of head lice infestation is best made by finding a live nymph or adult louse on the scalp or hair of a person. Because adult and nymph lice are very small...
4. `medquad_ans_0000339-1__dup3` (score=0.0001) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels o...
5. `medquad_ans_0000214-1__dup3` (score=0.0001) - The head louse, or Pediculus humanus capitis, is a parasitic insect that can be found on the head, eyebrows, and eyelashes of people. Head lice feed on human blood several times a day and live close to the human scalp...

**Answer**:

Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.896, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.9688) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0004450-4` (score=0.0015) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004454-3` (score=0.0015) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004456-3` (score=0.0015) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004453-3` (score=0.0015) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. However, due to the limited number of diagnosed cases and the periodic nature of the disease, evaluating effective treatment options is difficult, and responses to these treatments vary. It is recommended that treatment decisions be made in consultation with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.982, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.6838) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-1` (score=0.6256) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0000373-2` (score=0.0002) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...
4. `medquad_ans_0000373-3` (score=0.0002) - Researchers have identified changes in five regions of the X chromosome that are linked to FG syndrome in affected families. Mutations in a gene called MED12, which is located in one of these regions, appear to be the...
5. `medquad_ans_0000373-4` (score=0.0001) - FG syndrome is inherited in an X-linked recessive pattern. The genes likely associated with this disorder, including MED12, are located on the X chromosome, which is one of the two sex chromosomes. In males (who have ...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in the MED12 gene are the most common cause (FG syndrome 1). Other genes involved include FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) [medquad_ans_0002359-1].

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.9994) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
3. `medquad_ans_0000141-2__dup1` (score=0.0001) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.833, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000084-2__dup1` (score=0.0022) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
2. `medquad_ans_0002796-4` (score=0.0001) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
3. `medquad_ans_0000738-3__dup1` (score=0.0001) - What treatments are available for birdshot chorioretinopathy? Unfortunately, there is currently no cure for birdshot chorioretinopathy. Because this condition is rare, there are no established guidelines for treatment...
4. `medquad_ans_0005174-2` (score=0.0001) - What treatments are available for pudendal neuralgia? There are no established guidelines for the treatment of pudendal neuralgia. Treatment may include medications, the use of a specialized cushion while sitting, ner...
5. `medquad_ans_0003103-4` (score=0.0001) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...

**Answer**:

The provided context does not contain information about the treatments for Mucolipidoses. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.9994) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
2. `medquad_ans_0003240-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile ...
3. `medquad_ans_0005084-1` (score=0.0001) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
4. `medquad_ans_0000525-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy is a liver disorder that occurs in pregnant women. Cholestasis is a condition that impairs the release of a digestive fluid called bile from liver cells. As a result, bile builds ...
5. `medquad_ans_0000525-3` (score=0.0001) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, intrahepatic cholestasis, pruritus, short stature, and splenomegaly [medquad_ans_0005084-2].

**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.800, groundedness=0.878, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.9968) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
2. `medquad_ans_0000213-4__dup2` (score=0.0061) - Urinary retention in men becomes more common with age. - In men 40 to 83 years old, the overall incidence of urinary retention is 4.5 to 6.8 per 1,000 men.2 - For men in their 70s, the overall incidence increases to 1...
3. `medquad_ans_0000070-9` (score=0.0002) - Leaking urine can happen for many reasons. Sometimes urinary incontinence happens for a little while. Short periods of leaking urine can happen because of - urinary tract infections - constipation - some medicines. ur...
4. `medquad_ans_0000070-7` (score=0.0001) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is ...
5. `medquad_ans_0000211-5__dup1` (score=0.0001) - The types of UI in men include - urgency incontinence - stress incontinence - functional incontinence - overflow incontinence - transient incontinence Urgency Incontinence Urgency incontinence happens when a man urina...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6]. Therefore, the prevalence of urinary incontinence in men ranges from 11% to 34%.

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005415-1` (score=0.3207) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
2. `medquad_ans_0000882-1` (score=0.2402) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
3. `medquad_ans_0005415-4` (score=0.0002) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
4. `medquad_ans_0000882-2` (score=0.0002) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the c...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Two major types have been identified: autosomal recessive and autosomal dominant Robinow syndrome. These types are distinguished by the severity of their signs and symptoms and by their pattern of inheritance. Autosomal recessive Robinow syndrome is characterized by skeletal abnormalities such as shortening of the long bones in the arms and legs, abnormally short fingers and toes (brachydactyly), wedge-shaped spinal bones (hemivertebrae) leading to kyphoscoliosis, fused or missing ribs, and short stature. Affected individuals also have distinctive facial features, including a broad forehead, prominent and widely spaced eyes, a short nose with an upturned tip, a wide nasal bridge, and a broad, triangle-shaped mouth, which are sometimes described as "fetal facies." Other features may include underdeveloped genitalia, dental

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 74 (`0000551-1`)

**Question**: What is (are) juvenile primary lateral sclerosis ?

**Scores**: context_relevance=0.857, groundedness=0.993, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000551-1` (score=0.9132) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...
2. `medquad_ans_0003331-1` (score=0.8675) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and stiffness of muscles in the arms, legs, and face. This disorder damages motor neurons, which are specialized nerve cells ...
3. `medquad_ans_0000238-1__dup1` (score=0.1857) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper ...
4. `medquad_ans_0003319-1` (score=0.0260) - Juvenile amyotrophic lateral sclerosis (ALS) is a type of motor neuron disease which leads to problems with muscle control and movement. Signs and symptoms of juvenile ALS tend to present by age 25 years or younger. U...
5. `medquad_ans_0005053-1` (score=0.0215) - Primary lateral sclerosis is a type of motor neuron disease, where nerve cells that control voluntary muscle movement breakdown and die. In primary lateral sclerosis only the upper motor neurons in the brain are affec...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. It is caused by damage to motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress slowly over many years. Early symptoms include clumsiness, muscle weakness, and spasticity in the legs, and difficulty with balance. As the condition progresses, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern [medquad_ans_0003331-1].

**Supported citations**: `medquad_ans_0003331-1`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.925, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-1__dup3` (score=0.4616) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...
2. `medquad_ans_0000188-1__dup2` (score=0.0856) - Leukemia is cancer of the white blood cells. It is the most common type of childhood cancer. Your blood cells form in your bone marrow. White blood cells help your body fight infection. In leukemia, the bone marrow pr...
3. `medquad_ans_0000040-6` (score=0.0127) - Researchers are conducting clinical trials in many parts of the country. Clinical trials test an intervention such as a drug, therapy, medical device, or behavior in many people to see if it is safe and effective. Cli...
4. `medquad_ans_0000001_4-6` (score=0.0086) - Key Points - After chronic myelogenous leukemia has been diagnosed, tests are done to find out if the cancer has spread. - Chronic myelogenous leukemia has 3 phases. - Chronic phase - Accelerated phase - Blastic phase...
5. `medquad_ans_0000001_4-1` (score=0.0015) - Key Points - Chronic myelogenous leukemia is a disease in which the bone marrow makes too many white blood cells. - Leukemia may affect red blood cells, white blood cells, and platelets. - Signs and symptoms of chroni...

**Answer**:

Leukemia is a cancer of the blood cells, specifically originating in the bone marrow, where it produces abnormal white blood cells called leukemia cells. These abnormal cells initially function almost normally but, over time, crowd out healthy blood cells, impairing the blood's ability to carry out its normal functions such as fighting infection, carrying oxygen, and controlling bleeding. There are four common types of adult leukemia: chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic forms develop slowly and are more common in older adults, while acute forms progress rapidly; acute lymphocytic leukemia is more common in children. Symptoms may include fevers, frequent infections, fatigue, swollen lymph nodes, weight loss, and easy bleeding or bruising, though these symptoms can also result from infections or other conditions [medquad_ans_0000040-1__dup3].

**Supported citations**: `medquad_ans_0000040-1__dup3`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=1.000, groundedness=0.903, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002485-1` (score=0.7954) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
2. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
3. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
4. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor composed of mature neurons. It most commonly occurs in individuals between the ages of 10 and 30, although it can affect all age groups. The most frequent site is the temporal lobe of the brain, but gangliocytomas can arise in other parts of the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. These tumors are often associated with epilepsy and may present with signs and symptoms depending on their location, such as seizures (most commonly), increased brain pressure, endocrine disorders, and focal symptoms. Gangliocytomas are generally slow-growing and non-malignant. Treatment typically involves surgical removal [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.978, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.7352) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0004068-1` (score=0.0810) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. Characteristic features include underdeveloped cheek bones, a very small lower jaw, cleft lip and/or palate, abnormalities...
3. `medquad_ans_0000658-1` (score=0.0291) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. The severity of this disorder varies among affected individuals. Children with Miller syndrome are born with underdevelope...
4. `medquad_ans_0000130-4__dup1` (score=0.0011) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.
5. `medquad_ans_0000658-2` (score=0.0004) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barré syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Like Guillain-Barré syndrome, symptoms may be preceded by a viral illness. Additional symptoms include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1__dup1].

**Supported citations**: `medquad_ans_0000130-1__dup1`
**Unsupported citations**: _none_

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006570-3` (score=0.9947) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
2. `medquad_ans_0006579-3` (score=0.0223) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at ...
3. `medquad_ans_0006562-3` (score=0.0002) - What causes 17q23.2q23.2 microdeletion syndrome? The syndrome is caused by an interstitial deletion (a deletion that does not involve the ends of a chromosome) encompassing bands 23.1 to 23.2 on the long (q) arm of ch...
4. `medquad_ans_0001084-3` (score=0.0001) - Most people with 22q11.2 deletion syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22 in each cell. This region contains 30 to 40 genes, many of which have ...
5. `medquad_ans_0001084-2` (score=0.0001) - 22q11.2 deletion syndrome affects an estimated 1 in 4,000 people. However, the condition may actually be more common than this estimate because doctors and researchers suspect it is underdiagnosed due to its variable ...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location designated q11.2. Most people with 22q11.2 deletion syndrome are missing a piece of the chromosome that contains about 30 to 40 genes, many of which have not been well characterized. Some affected people have smaller deletions. The deletion of a particular gene, TBX1, is probably responsible for many of the syndrome's characteristic signs (such as heart defects, a cleft palate, distinctive facial features, hearing loss, and low calcium levels). Loss of this gene may also contribute to behavioral problems. The loss of another gene, COMT, may also cause increased risk of behavioral problems and mental illness in affected people. The other genes that are deleted likely contribute to the various features of

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-2__dup3` (score=0.9619) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...
2. `medquad_ans_0000148-3__dup2` (score=0.0060) - For most healthy people who receive treatment soon after the outbreak of blisters, the lesions heal, the pain subsides within 3 to 5 weeks, and the blisters often leave no scars. However, shingles is a serious threat ...
3. `medquad_ans_0000062-4__dup3` (score=0.0025) - Burning, Itching, Tingling, Then a Rash An outbreak of shingles usually begins with a burning, itching, or tingling sensation on the back, chest, or around the rib cage or waist. It is also common for the face or eye ...
4. `medquad_ans_0000062-1__dup5` (score=0.0012) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single s...
5. `medquad_ans_0000149-4__dup1` (score=0.0006) - The NINDS supports research on shingles and shingles-related conditions. Current studies focus on the relationship between the persistence of neurotropic viruses and development of neurological diseases including herp...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. When a person recovers from chickenpox, the virus remains in the body, hiding in nerve cells, usually in the spine. It can become active again when the immune system weakens, triggering a reactivation of the virus [medquad_ans_0000062-2__dup3].

**Supported citations**: `medquad_ans_0000062-2__dup3`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.795, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.9994) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0000333-2` (score=0.0001) - Fabry disease affects an estimated 1 in 40,000 to 60,000 males. This disorder also occurs in females, although the prevalence is unknown. Milder, late-onset forms of the disorder are probably more common than the clas...
3. `medquad_ans_0000392-1` (score=0.0001) - Fryns syndrome is a condition that affects the development of many parts of the body. The features of this disorder vary widely among affected individuals and overlap with the signs and symptoms of several other disor...
4. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0000123-4__dup1` (score=0.0001) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...

**Answer**:

The provided context does not describe "Fryns Hofkens Fabry syndrome" as a valid or recognized medical condition. The context separately references Fryns syndrome, Fabry disease, and mentions a list of signs and symptoms under "Fryns Hofkens Fabry syndrome" in a misleading or erroneous format.

Fryns syndrome is a condition affecting development of many body parts, with common features including congenital diaphragmatic hernia, abnormalities of fingers and toes, and distinctive facial features [medquad_ans_0000392-1]. Fabry disease is a different disorder affecting males at a prevalence of 1 in 40,000 to 60,000, with symptoms related to enzyme deficiency and organ damage, though not linked to the symptoms listed under Fryns syndrome [medquad_ans_0000333-2].

There is no valid medical evidence or context

**Supported citations**: `medquad_ans_0000392-1`, `medquad_ans_0000333-2`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.800, groundedness=0.774, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000024_2-8` (score=0.0064) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000006_3-8` (score=0.0053) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000004_6-7` (score=0.0025) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000040_1-6` (score=0.0021) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000025_1-8` (score=0.0014) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for osteoporosis. [medquad_ans_0000024_2-8] and [medquad_ans_0000006_3-8] discuss clinical trials for cancer treatments, such as chemoprevention and radiosensitizers, and do not reference osteoporosis. Therefore, no information is available in the context about research for osteoporosis.

**Supported citations**: `medquad_ans_0000024_2-8`, `medquad_ans_0000006_3-8`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=1.000, groundedness=0.910, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.9966) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
2. `medquad_ans_0000286-1__dup2` (score=0.0033) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
3. `medquad_ans_0000925-1__dup1` (score=0.0015) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
4. `medquad_ans_0000463-1__dup1` (score=0.0012) - Summary : An herb is a plant or plant part used for its scent, flavor, or therapeutic properties. Herbal medicines are one type of dietary supplement. They are sold as tablets, capsules, powders, teas, extracts, and f...
5. `medquad_ans_0000875-1__dup1` (score=0.0001) - Summary : Being a teenager is hard. You're under stress to be liked, do well in school, get along with your family, and make big decisions. You can't avoid most of these pressures, and worrying about them is normal. B...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise quick fixes, miracle cures, ancient remedies, secret ingredients, or scientific breakthroughs for common health issues like weight loss, memory loss, sexual performance, and joint pain. They target individuals with serious conditions such as cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Health fraud can be dangerous, waste money, and prevent people from accessing effective treatments. To protect yourself, recognize red flags and consult a doctor or healthcare professional before taking unproven treatments [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.917, groundedness=0.745, answer_relevance=0.517

**Top docs**:

1. `medquad_ans_0000015-24` (score=0.0006) - People with type 1 diabetes control their blood sugar with insulin -- either with shots or an insulin pen. Many people with type 2 diabetes can control blood glucose levels with diet and exercise alone. Others require...
2. `medquad_ans_0000009-3__dup3` (score=0.0006) - There are several things you can do to lower your risk for cataract. They include - having regular eye exams - quitting smoking - wearing sunglasses - taking care of other health problems - maintaining a healthy weigh...
3. `medquad_ans_0000015-33` (score=0.0004) - Your weight affects your health in many ways. Being overweight can keep your body from making and using insulin properly. It can also cause high blood pressure. If you are overweight or obese, choose sensible ways to ...
4. `medquad_ans_0000080-1__dup3` (score=0.0002) - If you have diabetes,\* the best time to control your blood glucose, also called blood sugar, is before you get pregnant. High blood glucose levels can be harmful to your baby during the first weeks of pregnancyeven be...
5. `medquad_ans_0000199-1__dup2` (score=0.0002) - You have two kidneys, each about the size of your fist. Their main job is to filter wastes and excess water out of your blood to make urine. They also keep the body's chemical balance, help control blood pressure, and...

**Answer**:

The provided context does not mention symptoms of preventing diabetes problems or specifically address eye health. Therefore, it is not possible to answer the question based on the given information. [medquad_ans_0000015-24]

**Supported citations**: `medquad_ans_0000015-24`
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000527-4` (score=0.9916) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
2. `medquad_ans_0000001-4` (score=0.9843) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chro...
3. `medquad_ans_0001084-3__dup1` (score=0.0010) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
4. `medquad_ans_0003718-4` (score=0.0002) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
5. `medquad_ans_0000562-4` (score=0.0001) - When Klippel-Feil syndrome is caused by mutations in the GDF6 or GDF3 genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder....

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern [medquad_ans_0000001-4].

**Supported citations**: `medquad_ans_0000001-4`
**Unsupported citations**: _none_

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.900, groundedness=0.926, answer_relevance=0.380

**Top docs**:

1. `medquad_ans_0001063-3` (score=0.9982) - Mutations in the ABCB7 gene cause X-linked sideroblastic anemia and ataxia. The ABCB7 gene provides instructions for making a protein that is critical for heme production. Heme is a component of the hemoglobin protein...
2. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
3. `medquad_ans_0001063-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
4. `medquad_ans_0001062-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0006525-1` (score=0.0001) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...

**Answer**:

Mutations in the ABCB7 gene cause X-linked sideroblastic anemia. These mutations slightly alter the structure of the ABCB7 protein, disrupting its usual role in heme production and iron homeostasis [medquad_ans_0001063-3]. The ABCB7 protein is critical for heme production and the formation of proteins containing iron-sulfur clusters, and its dysfunction leads to anemia due to insufficient hemoglobin production [medquad_ans_0001063-3]. The condition is inherited in an X-linked recessive pattern, meaning that in males, one altered copy of the gene is sufficient to cause the condition [medquad_ans_0001063-4].

**Supported citations**: `medquad_ans_0001063-3`, `medquad_ans_0001063-4`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.952, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.3290) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
2. `medquad_ans_0003601-1` (score=0.1475) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0003610-1` (score=0.0108) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
4. `medquad_ans_0000576-1` (score=0.0054) - LAMA2-related muscular dystrophy is a disorder that causes weakness and wasting (atrophy) of muscles used for movement (skeletal muscles). This condition generally appears in one of two ways: as a severe, early-onset ...
5. `medquad_ans_0003604-1` (score=0.0029) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...

**Answer**:

Limb-girdle muscular dystrophy is a group of diseases that cause weakness and wasting of the muscles in the arms and legs, particularly affecting the proximal muscles closest to the body, such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0000602-1]. It is a progressive condition leading to a loss of muscle strength and bulk over time, with onset possibly occurring in childhood, adolescence, young adulthood, or later in life [medquad_ans_0003601-1]. The severity, age of onset, and features vary among the subtypes and may differ even within the same family [medquad_ans_0000602-1]. Most forms are inherited in an autosomal recessive manner, while several rare forms follow an autosomal dominant pattern [medquad_ans_0003601-1]. Intelligence is

**Supported citations**: `medquad_ans_0000602-1`, `medquad_ans_0003601-1`
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=0.833, groundedness=0.856, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000016-26` (score=0.9879) - National Eye Institute National Institutes of Health 2020 Vision Place Bethesda, MD 20892-3655 301-496-5248 E-mail: 2020@nei.nih.gov www.nei.nih.gov Find eye health organizations that address diabetic eye disease.
2. `medquad_ans_0000016-24` (score=0.9810) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...
3. `medquad_ans_0000778-1__dup1` (score=0.9644) - The retina is a layer of tissue in the back of your eye that senses light and sends images to your brain. In the center of this nerve tissue is the macula. It provides the sharp, central vision needed for reading, dri...
4. `medquad_ans_0000042-2__dup3` (score=0.8272) - Eye Diseases and Health Conditions Most people develop low vision because of eye diseases and health conditions like macular degeneration, cataracts, glaucoma, and diabetes. Your eye care professional can tell the dif...
5. `medquad_ans_0000275-1__dup1` (score=0.4466) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...

**Answer**:

Diabetic eye problems include proliferative retinopathy and macular edema. Proliferative retinopathy can develop without symptoms and, if advanced, puts individuals at high risk for vision loss or blindness. Macular edema can also develop without symptoms at any of the four stages of diabetic retinopathy. Individuals can develop both conditions and still see fine, but they remain at high risk for vision loss. Early detection and timely treatment can prevent vision loss [medquad_ans_0000016-24]. These conditions are part of a broader category known as diabetic eye disease, which affects the retina and can lead to serious vision issues [medquad_ans_0000778-1__dup1].

**Supported citations**: `medquad_ans_0000016-24`, `medquad_ans_0000778-1__dup1`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=0.800, groundedness=0.484, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000973-1__dup1` (score=0.2974) - Wilson disease is a rare inherited disorder that prevents your body from getting rid of extra copper. You need a small amount of copper from food to stay healthy. Too much copper is poisonous. Normally, your liver rel...
2. `medquad_ans_0001037-1` (score=0.2708) - Wilson disease is an inherited disorder in which excessive amounts of copper accumulate in the body, particularly in the liver, brain, and eyes. The signs and symptoms of Wilson disease usually first appear between th...
3. `medquad_ans_0000133-1__dup3` (score=0.2184) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...
4. `medquad_ans_0006449-1` (score=0.0779) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...
5. `medquad_ans_0000133-11` (score=0.0049) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the i...

**Answer**:

Wilson disease is a rare inherited disorder that prevents the body from getting rid of extra copper. You need a small amount of copper from food to stay healthy, but too much copper is poisonous. Normally, the liver releases extra copper into bile, a digestive fluid; however, in Wilson disease, copper builds up in the liver and is released directly into the bloodstream, causing damage to the liver, brain, kidneys, and eyes. The condition is present at birth, with symptoms typically starting between ages 5 and 35. It initially attacks the liver, the central nervous system, or both. A hallmark sign is a rusty brown ring around the cornea of the eye, known as the Kayser-Fleischer ring. Diagnosis is made through physical exam and laboratory tests, and treatment involves medications to remove excess copper, a low-copper diet, and avoidance of high-copper foods such as shellfish, liver, chocolate, mushrooms, and nuts. Drinking

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001503-2` (score=0.9990) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
2. `medquad_ans_0003748-2` (score=0.0001) - What are the signs and symptoms of Majeed syndrome? Majeed syndrome is characterized by recurrent episodes of fever and inflammation in the bones and skin. There are two main features of Majeed syndrome: Chronic recur...
3. `medquad_ans_0001503-1` (score=0.0001) - Congenital dyserythropoietic anemia type 2 (CDA II) is an inherited blood disorder characterized by mild to severe anemia. It is usually diagnosed in adolescence or early adulthood. Many affected individuals have yell...
4. `medquad_ans_0000307-3` (score=0.0001) - Mutations in the GATA1 gene cause dyserythropoietic anemia and thrombocytopenia. The GATA1 gene provides instructions for making a protein that attaches (binds) to specific regions of DNA and helps control the activit...
5. `medquad_ans_0000223-1` (score=0.0001) - Congenital dyserythropoietic anemia (CDA) is an inherited blood disorder that affects the development of red blood cells. This disorder is one of many types of anemia, which is a condition characterized by a shortage ...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen (splenomegaly), anemia of inadequate production, endopolyploidy on chromosome studies of bone marrow, reduced activity of N-acetylglucosaminyltransferase II, reticulocytosis, and iron overload. Iron overload can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.897, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000038-3__dup2` (score=0.0255) - Kidney Disease Kidney disease is often called a "silent" disease, because most people have no symptoms with early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NO...
2. `medquad_ans_0000136-12` (score=0.0136) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - What I need ...
3. `medquad_ans_0000179-14` (score=0.0136) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - What I need ...
4. `medquad_ans_0000138-11` (score=0.0123) - You and your doctor will work together to choose a treatment that's best for you. The publications of the NIDDK Kidney Failure Series can help you learn about the specific issues you will face. Booklets - Treatment Me...
5. `medquad_ans_0000170-9` (score=0.0021) - Limiting intake of fats, cholesterol, sodium, and sugar can help prevent atherosclerosis, which can lead to RAS. Most sodium in the diet comes from salt. A healthy diet that prevents people from becoming overweight or...

**Answer**:

Kidney disease is often called a "silent" disease because most people have no symptoms in the early stages, and individuals might feel fine until their kidneys have almost stopped working. It is characterized by damaged kidneys that filter less than 15% of the blood filtered by healthy kidneys, leading to a condition known as uremia, where waste products build up in the blood. Kidney disease can also result in hormonal imbalances and disruptions in mineral balance in the blood, causing complications such as itching, sleep problems, restless legs, weak bones, joint problems, and depression. Diagnosis is typically made through blood and urine tests, which assess kidney function using the glomerular filtration rate (GFR). Key risk factors include diabetes, high blood pressure, heart disease, and a family history of kidney failure [medquad_ans_0000038-3__dup2].

**Supported citations**: `medquad_ans_0000038-3__dup2`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.943, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000058-30` (score=0.8510) - Through research, doctors are trying to find new, more effective ways to treat prostate cancer. Cryosurgery -- destroying cancer by freezing it -- is under study as an alternative to surgery and radiation therapy. To ...
2. `medquad_ans_0000036_1-7` (score=0.7404) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000036_1-6` (score=0.7046) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopha...
4. `medquad_ans_0000058-23` (score=0.1778) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But ...
5. `medquad_ans_0000058-20` (score=0.0191) - Doctors use tests to detect prostate abnormalities, but tests cannot show whether abnormalities are cancer or another, less serious condition. The results from these tests will help the doctor decide whether to check ...

**Answer**:

Treatments for prostate cancer include cryosurgery, which uses a cryoprobe to freeze and destroy cancer cells, often guided by ultrasound [medquad_ans_0000058-30][medquad_ans_0000036_1-7]. High-intensity focused ultrasound therapy uses high-energy sound waves to destroy cancer cells, with an endorectal probe used to deliver the sound waves [medquad_ans_0000036_1-7]. Proton beam radiation therapy, a type of external radiation therapy using protons, is being studied for treating prostate cancer [medquad_ans_0000036_1-7]. Hormonal therapy is used, either after radiation therapy to help when cancer has spread to nearby tissues or to shrink tumors before surgery or radiation, and is also being tested in combination with vaccines to prevent recurrence [medquad_ans_00

**Supported citations**: `medquad_ans_0000058-30`, `medquad_ans_0000036_1-7`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.861, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.9973) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000431-1__dup1` (score=0.9972) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
3. `medquad_ans_0000362-1__dup2` (score=0.9971) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
4. `medquad_ans_0000589-1__dup2` (score=0.9958) - Summary : Medicare is the U.S. government's health insurance program for people age 65 or older. Some people under age 65 can qualify for Medicare, too. They include those with disabilities, permanent kidney failure, ...
5. `medquad_ans_0000429-1__dup1` (score=0.0017) - Summary : Health facilities are places that provide health care. They include hospitals, clinics, outpatient care centers, and specialized care centers, such as birthing centers and psychiatric care centers. When you ...

**Answer**:

Yes, there is information about health insurance. Health insurance is a contract between you and your insurance company, where you buy a plan and the company agrees to pay part of your medical expenses when you need care [medquad_ans_0000431-1__dup1]. Many people obtain health insurance through their employers, and employer-sponsored plans are often managed care plans that contract with healthcare providers to offer care at reduced costs [medquad_ans_0000431-1__dup1]. Additionally, government health insurance programs such as Medicare and Medicaid are available to eligible individuals. Medicare serves people age 65 or older and some under 65 with disabilities or chronic conditions [medquad_ans_0000589-1__dup2]. Medicaid is a government health insurance program that assists low-income individuals, with eligibility based on factors such as age, pregnancy, disability, income, and immigration status

**Supported citations**: `medquad_ans_0000431-1__dup1`, `medquad_ans_0000589-1__dup2`
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.979, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.4641) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0000635-1` (score=0.4579) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
3. `medquad_ans_0000635-4` (score=0.0009) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...
4. `medquad_ans_0000635-3` (score=0.0006) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
5. `medquad_ans_0000636-1` (score=0.0001) - MECP2-related severe neonatal encephalopathy is a neurological disorder that primarily affects males and causes brain dysfunction (encephalopathy). Affected males have a small head size (microcephaly), poor muscle ton...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia, delayed motor milestones (e.g., sitting up, crawling), recurrent infections, poor or absent speech, seizures, and/or spasticity. The condition results from an extra copy (duplication) of the MECP2 gene in each cell, which is generally caused by a duplication of genetic material on the long (q) arm of the X chromosome. It is inherited in an X-linked manner. In males, a duplication of the only copy of the MECP2 gene in each cell is sufficient to cause the condition, whereas in females, a duplication of one of the two copies typically does not cause the disorder due to X-inactivation, which inactivates one X chromosome in each somatic cell [medquad_ans_0003860-1]

**Supported citations**: `medquad_ans_0003860-1`
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=1.000, groundedness=0.102, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000028_2-3` (score=0.0017) - Signs of melanoma include a change in the way a mole or pigmented area looks. These and other signs and symptoms may be caused by melanoma or by other conditions. Check with your doctor if you have any of the followin...
2. `medquad_ans_0000594-1__dup2` (score=0.0002) - Melanoma is the most serious type of skin cancer. Often the first sign of melanoma is a change in the size, shape, color, or feel of a mole. Most melanomas have a black or black-blue area. Melanoma may also appear as ...
3. `medquad_ans_0000063-8` (score=0.0002) - Many Areas of Research Scientists are constantly searching for new ways to detect skin cancer, assess risk, and predict patient outcomes. They are interested in finding new treatments and new ways to deliver drugs and...
4. `medquad_ans_0005046-1` (score=0.0001) - Primary melanoma of the gastrointestinal (GI) tract refers to a melanoma starting in the stomach, intestines, salivary glands, mouth, esophagus, liver, pancreas, gallbladder, or rectum. Melanoma is a disease in which ...
5. `medquad_ans_0000080-3__dup1` (score=0.0001) - The prognosis for individuals with Coffin-Lowry syndrome varies depending on the severity of symptoms. Early intervention may improve the outlook for patients. Life span is reduced in some individuals with Coffin-Lowr...

**Answer**:

The provided context does not contain information about the outlook for melanoma, such as survival rates, prognosis, or long-term outcomes. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000268-4` (score=0.0128) - Czech dysplasia is inherited in an autosomal dominant pattern, which means one copy of the altered COL2A1 gene in each cell is sufficient to cause the disorder. All known individuals with Czech dysplasia inherited the...
2. `medquad_ans_0000648-4` (score=0.0005) - Metatropic dysplasia is considered an autosomal dominant disorder because one mutated copy of the TRPV4 gene in each cell is sufficient to cause the condition. Most cases of metatropic dysplasia are caused by new muta...
3. `medquad_ans_0000745-4` (score=0.0002) - Most cases of oculodentodigital dysplasia are inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person in...
4. `medquad_ans_0000250-4` (score=0.0001) - Craniometaphyseal dysplasia can have different inheritance patterns. In most cases this condition is inherited in an autosomal dominant pattern, which means one altered copy of the ANKH gene in each cell is sufficient...
5. `medquad_ans_0001374-4` (score=0.0001) - How is cleidocranial dysplasia inherited? Cleidocranial dysplasia is inherited in an autosomal dominant manner, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases,...

**Answer**:

The provided context does not mention boomerang dysplasia or its inheritance pattern. Therefore, it is not possible to determine based on the given information whether boomerang dysplasia is inherited.

Answer: Insufficient context [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.900, groundedness=0.421, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000466-3` (score=0.9991) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
2. `medquad_ans_0000836-3` (score=0.0001) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...
3. `medquad_ans_0000690-3` (score=0.0001) - Multiple familial trichoepithelioma can be caused by mutations in the CYLD gene. This gene provides instructions for making a protein that helps regulate nuclear factor-kappa-B. Nuclear factor-kappa-B is a group of re...
4. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
5. `medquad_ans_0002838-1` (score=0.0001) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...

**Answer**:

The genetic changes related to hereditary multiple osteochondromas are mutations in the EXT1 and EXT2 genes. These mutations cause the production of defective exostosin-1 or exostosin-2 proteins, which form a complex in the Golgi apparatus responsible for modifying heparan sulfate. When this modification is impaired due to mutations in EXT1 or EXT2, heparan sulfate becomes nonfunctional, leading to the development of osteochondromas. Mutations in the EXT1 gene result in hereditary multiple osteochondromas type 1, while mutations in the EXT2 gene cause type 2. EXT1 mutations are estimated to account for 55 to 75 percent of all cases, and symptoms tend to be more severe in type 1. Approximately 15 percent of individuals with the condition have no identifiable mutation in either EXT1 or EXT2 [medquad_ans_0000466

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.740, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000746-1` (score=0.0001) - Oculofaciocardiodental (OFCD) syndrome is a condition that affects the development of the eyes (oculo-), facial features (facio-), heart (cardio-) and teeth (dental). This condition occurs only in females. The eye abn...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
4. `medquad_ans_0004510-1` (score=0.0001) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip tha...
5. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...

**Answer**:

The provided context does not specify the number of people affected by oculofaciocardiodental (OFCD) syndrome. [medquad_ans_0000746-1] describes the signs and symptoms of the condition but does not include population frequency or prevalence data. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000746-1`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.0631) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0001350-1` (score=0.0046) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
3. `medquad_ans_0000189-2` (score=0.0005) - Type I citrullinemia is the most common form of the disorder, affecting about 1 in 57,000 people worldwide. Type II citrullinemia is found primarily in the Japanese population, where it occurs in an estimated 1 in 100...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described: Type I (also known as classic citrullinemia), which is caused by mutations in the ASS1 gene and inherited in an autosomal recessive pattern, and typically becomes evident in the first few days of life with symptoms including lethargy, poor feeding, vomiting, seizures, and loss of consciousness; and Type II, which chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma, and may appear during adulthood, often triggered by medications, infections, surgery, or alcohol intake. Type II can also develop in individuals who had neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) as infants [medquad_ans_0000189-1]medquad*ans*

**Supported citations**: `medquad_ans_0000189-1`
**Unsupported citations**: _none_

### Example 99 (`0000718-1__dup1`)

**Question**: What is (are) Bilateral frontal polymicrogyria ?

**Scores**: context_relevance=0.833, groundedness=0.992, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000719-1__dup1` (score=0.9717) - Bilateral frontoparietal polymicrogyria (BFPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). BFPP specifically affects the frontal and parietal lobes on both sides ...
2. `medquad_ans_0000718-1__dup1` (score=0.9290) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
3. `medquad_ans_0000722-1__dup1` (score=0.0592) - Bilateral perisylvian polymicrogyria (BPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). Signs and symptoms include partial paralysis of muscles on both sides of th...
4. `medquad_ans_0004956-1` (score=0.0117) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. Specifically, the surface of the brain develops too many folds which are unusually small. The signs and symptoms associate...
5. `medquad_ans_0000807-1` (score=0.0061) - Polymicrogyria is a condition characterized by abnormal development of the brain before birth. The surface of the brain normally has many ridges or folds, called gyri. In people with polymicrogyria, the brain develops...

**Answer**:

Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area located behind the Sylvius fissure or the area located behind the Rolando sulcus. Signs and symptoms include delayed motor and language milestones; spastic (stiffness) hemiparesis (weakness in one side of the body) or quadriparesis (weakness in all four limbs of the body); and mild to moderate intellectual disability. Seizures may also be present. The cause for the frontal form of polymicrogyria is still not known. Treatment is based on the signs and symptoms present in each person [medquad_ans_0000718-1__dup1].

**Supported citations**: `medquad_ans_0000718-1__dup1`
**Unsupported citations**: _none_

### Example 100 (`0000944-5`)

**Question**: What are the treatments for spondylocostal dysostosis ?

**Scores**: context_relevance=0.857, groundedness=0.886, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000944-5` (score=0.9918) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic ...
2. `medquad_ans_0005860-4` (score=0.0968) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respi...
3. `medquad_ans_0003783-5` (score=0.0001) - Is genetic testing available for mandibulofacial dysostosis with microcephaly? Yes. Genetic testing is available for mandibulofacial dysostosis with microcephaly (MFDM) and confirms the diagnosis in virtually all peop...
4. `medquad_ans_0000944-2` (score=0.0001) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.
5. `medquad_ans_0000944-3` (score=0.0001) - Mutations in at least four genes are known to cause spondylocostal dysostosis: Mutations in the DLL3 gene cause spondylocostal dysostosis type 1; mutations in the MESP2 gene cause spondylocostal dysostosis type 2; mut...

**Answer**:

Treatment for spondylocostal dysostosis includes intensive medical care, such as treatment of respiratory infections, bone surgery, and orthopedic treatment, particularly due to the risk of breathing difficulties from a small, malformed chest [medquad_ans_0005860-4].

**Supported citations**: `medquad_ans_0005860-4`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.934, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.9994) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0000786-3` (score=0.0001) - Perry syndrome results from mutations in the DCTN1 gene. This gene provides instructions for making a protein called dynactin-1, which is involved in the transport of materials within cells. To move materials, dynacti...
3. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0000786-1` (score=0.0001) - Perry syndrome is a progressive brain disease that is characterized by four major features: a pattern of movement abnormalities known as parkinsonism, psychiatric changes, weight loss, and abnormally slow breathing (h...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function, respiratory insufficiency, sleep disturbance, tremor, weight loss, developmental regression, hallucinations, hypotension, apathy, bradykinesia, central hypoventilation, dysarthria, hypoventilation, inappropriate behavior, insomnia, mask-like facies, parkinsonism, rapidly progressive course, rigidity, short-stepped shuffling gait, vertical supranuclear gaze palsy, and weak voice. These symptoms occur in approximately 90% of patients for most of the listed features, with some symptoms such as developmental regression, hallucinations, and hypotension occurring in about 7.5% of patients [medquad_ans_0004825-1]. The underlying cause is mutations in the DCTN1 gene, which disrupt the function of dynactin-1 and impair intracellular transport in neurons,

**Supported citations**: `medquad_ans_0004825-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.904, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0006398-2` (score=0.9992) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
2. `medquad_ans_0006399-1` (score=0.9876) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...
3. `medquad_ans_0001027-5` (score=0.0001) - These resources address the diagnosis or management of warfarin resistance: - American Society of Hematology: Antithrombotic Therapy - MedlinePlus Drugs & Supplements: Warfarin - PharmGKB These resources from MedlineP...
4. `medquad_ans_0001028-2` (score=0.0001) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
5. `medquad_ans_0000005-7` (score=0.0001) - Antiphospholipid antibody syndrome (APS) has no cure. However, medicines can help prevent complications. The goals of treatment are to prevent blood clots from forming and keep existing clots from getting larger. You ...

**Answer**:

The signs and symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet. However, specific symptom frequencies are not detailed in the provided context [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.964, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0004262-1` (score=0.7196) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
2. `medquad_ans_0000701-1` (score=0.6933) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
3. `medquad_ans_0000701-2` (score=0.0005) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.
4. `medquad_ans_0002259-1` (score=0.0002) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0001388-1` (score=0.0001) - What are the signs and symptoms of COASY Protein-Associated Neurodegeneration ? The Human Phenotype Ontology provides the following list of signs and symptoms for COASY Protein-Associated Neurodegeneration . If the in...

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. It is caused by mutations in the SMAD4 gene and inherited in an autosomal dominant pattern, with most cases resulting from a new mutation [medquad_ans_0004262-1]. The condition affects multiple systems and functions of the body, with features such as delayed development of language and motor skills, hearing loss (sensorineural, conductive, or mixed), growth restriction beginning before birth and continuing through adolescence, skin stiffness, and a muscular appearance [medquad_ans_0000701-1]. Typical facial features include short palpebral fissures, a short philtrum, midface hypoplasia, a small mouth with

**Supported citations**: `medquad_ans_0004262-1`, `medquad_ans_0000701-1`
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006057-2` (score=0.9994) - What are the signs and symptoms of Thanatophoric dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Thanatophoric dysplasia. If the information is available, the table below ...
2. `medquad_ans_0000976-4` (score=0.0001) - Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell is sufficient to cause the condition. Virtually all cases of thanatophoric dysplasia are cau...
3. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
4. `medquad_ans_0002946-1` (score=0.0001) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...
5. `medquad_ans_0006057-4` (score=0.0001) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of tha...

**Answer**:

The symptoms of Thanatophoric dysplasia include abnormality of the femur, abnormality of the metaphyses, abnormality of the sacroiliac joint, aplasia/hypoplasia of the lungs, bowing of the long bones, brachydactyly syndrome, cognitive impairment, cutis laxa, depressed nasal bridge, increased nuchal translucency, macrocephaly, malar flattening, micromelia, muscular hypotonia, narrow chest, platyspondyly, respiratory insufficiency, short stature, short thorax, skeletal dysplasia, small face, split hand, abnormality of neuronal migration, frontal bossing, hearing impairment, intrauterine growth retardation, kyphosis, polyhydramnios, proptosis, ventriculomegaly, abnormality of the kidney, acanthosis nigricans, atria septal defect,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.9990) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
2. `medquad_ans_0000381-3` (score=0.0001) - Mutations in the FMR1 gene cause fragile X syndrome. The FMR1 gene provides instructions for making a protein called FMRP. This protein helps regulate the production of other proteins and plays a role in the developme...
3. `medquad_ans_0002417-4` (score=0.0001) - How is fragile X syndrome inherited? Fragile X syndrome (FXS) is inherited in an X-linked dominant manner. A condition is X-linked if the responsible gene is located on the X chromosome. The inheritance is dominant if...
4. `medquad_ans_0000383-5` (score=0.0001) - These resources address the diagnosis or management of FXTAS: - Fragile X Research Foundation of Canada: FXTAS - Gene Review: Gene Review: FMR1-Related Disorders - Genetic Testing Registry: Fragile X tremor/ataxia syn...
5. `medquad_ans_0000384-1` (score=0.0001) - Fragile XE syndrome is a genetic disorder that impairs thinking ability and cognitive functioning. Most affected individuals have mild intellectual disability. In some people with this condition, cognitive function is...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=0.750, groundedness=0.954, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000056-1__dup1` (score=0.2599) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease that occurs when the thickening of blood vessel walls blocks the fl...
2. `medquad_ans_0000889-1__dup1` (score=0.2255) - CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited disease of the blood vessels that occurs when the thickening of blood vessel walls blocks the flow ...
3. `medquad_ans_0000166-1` (score=0.1349) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow...
4. `medquad_ans_0000056-3__dup1` (score=0.0014) - Symptoms usually progress slowly. By age 65, the majority of persons with CADASIL have cognitive problems and dementia. Some will become dependent due to multiple strokes.
5. `medquad_ans_0000166-2` (score=0.0007) - CADASIL is likely a rare condition; however, its prevalence is unknown.

**Answer**:

CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease caused by a mutation in the NOTCH3 gene, which leads to thickening of the blood vessel walls and blocks blood flow to the brain, particularly affecting small blood vessels in the white matter of the brain [medquad_ans_0000056-1__dup1]. It is characterized by migraine headaches, multiple strokes, cognitive deterioration, seizures, vision problems, psychiatric symptoms such as severe depression and behavioral changes, and an increased risk of heart attack [medquad_ans_0000056-1__dup1]. The disease progresses to dementia and typically presents with symptoms in the mid-30s, though onset can vary widely and some individuals may not show signs until later in life. CADASIL is inherited

**Supported citations**: `medquad_ans_0000056-1__dup1`
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004187-1` (score=0.9426) - Multiple endocrine neoplasia type 2A (MEN 2A) is is an inherited disorder caused by mutations in the RET gene. Individuals with MEN 2A are at high risk of developing medullary carcinoma of the thyroid. About 50% will ...
2. `medquad_ans_0000688-1` (score=0.6657) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
3. `medquad_ans_0000014-5__dup1` (score=0.0088) - In about 80 percent of people with primary hyperparathyroidism, a benign, or noncancerous, tumor called an adenoma has formed in one of the parathyroid glands.2 The tumor causes the gland to become overactive. In most...
4. `medquad_ans_0005097-1` (score=0.0039) - A prolactinoma is a tumor of the pituitary gland, which controls production of many hormones. A prolactinoma causes increased levels of the hormone prolactin. The symptoms of prolactinoma may include unusual milk prod...
5. `medquad_ans_0000025_2-2` (score=0.0021) - Having certain syndromes can increase the risk of pancreatic NETs. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not havin...

**Answer**:

Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. Many different types of tumors are associated with multiple endocrine neoplasia. Type 1 frequently involves tumors of the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004155-3` (score=0.9959) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
2. `medquad_ans_0004155-5` (score=0.0001) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...
3. `medquad_ans_0000035-9` (score=0.0001) - People can manage their diabetes with meal planning, physical activity, and if needed, medications. More information about taking care of type 1 or type 2 diabetes is provided in the NIDDK health topics: - What I need...
4. `medquad_ans_0004155-4` (score=0.0001) - How is mucopolysaccharidosis I (MPS I) inherited? MPS I is inherited from both parents in an autosomal recessive pattern.
5. `medquad_ans_0000200-3__dup1` (score=0.0001) - The mucopolysaccharidoses syndromes share many clinical features but have varying degrees of severity. Most individuals with a mucopolysaccharidosis syndrome generally experience a period of normal development followe...

**Answer**:

Mutations in the IDUA gene cause mucopolysaccharidosis type I (MPS I). The IDUA gene provides instructions for producing an enzyme involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme, leading to the accumulation of GAGs within cells, specifically inside the lysosomes. This accumulation results in lysosomal storage disorders [medquad_ans_0004155-3].

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.920

**Top docs**:

1. `medquad_ans_0000825-3` (score=0.9992) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
2. `medquad_ans_0000832-3` (score=0.0001) - In most cases, the genetic cause of progressive supranuclear palsy is unknown. Rarely, the disease results from mutations in the MAPT gene. Certain normal variations (polymorphisms) in the MAPT gene have also been ass...
3. `medquad_ans_0000825-4` (score=0.0001) - When this condition is caused by mutations in the FLCN gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, ...
4. `medquad_ans_0000825-2` (score=0.0001) - Primary spontaneous pneumothorax is more common in men than in women. This condition occurs in 7.4 to 18 per 100,000 men each year and 1.2 to 6 per 100,000 women each year.
5. `medquad_ans_0000825-1` (score=0.0001) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the space between the lungs and the chest cavity (called the pleural space) that can result in the partial or complete collapse of a lung. This ty...

**Answer**:

The genetic changes related to primary spontaneous pneumothorax are mutations in the FLCN gene. These mutations can cause primary spontaneous pneumothorax, although they appear to be a very rare cause of this condition [medquad_ans_0000825-3]. The FLCN gene provides instructions for making a protein called folliculin, which is found in lung connective tissue cells and in cells lining the alveoli. Researchers believe folliculin may help control cell growth and division and may play a role in repairing and re-forming lung tissue after damage. However, the exact mechanism by which FLCN gene mutations lead to bleb formation and increased risk of pneumothorax is not fully understood [medquad_ans_0000825-3].

**Supported citations**: `medquad_ans_0000825-3`
**Unsupported citations**: _none_

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.926, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000527-1__dup2` (score=0.9744) - A stroke is a medical emergency. There are two types - ischemic and hemorrhagic. Ischemic stroke is the most common type. It is usually caused by a blood clot that blocks or plugs a blood vessel in the brain. This kee...
2. `medquad_ans_0000857-1__dup2` (score=0.9664) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by ...
3. `medquad_ans_0000066-13` (score=0.9599) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...
4. `medquad_ans_0000124-1__dup2` (score=0.9582) - A stroke occurs if the flow of oxygen-rich blood to a portion of the brain is blocked. Without oxygen, brain cells start to die after a few minutes. Sudden bleeding in the brain also can cause a stroke if it damages b...
5. `medquad_ans_0000066-25` (score=0.9534) - Another imaging technique used for stroke patients is the magnetic resonance imaging or MRI scan. MRI uses magnetic fields to detect a variety of changes in the brain and blood vessels caused by a stroke. One effect o...

**Answer**:

A stroke is a medical emergency that occurs when blood flow to the brain stops. There are two main types: ischemic stroke, which is the most common and accounts for approximately 80 percent of all strokes, caused by a blood clot that blocks or plugs a blood vessel in the brain, and hemorrhagic stroke, caused by a blood vessel breaking and bleeding into the brain [medquad_ans_0000527-1\_\_dup2][medquad_ans_0000857-1__dup2][medquad_ans_0000066-13]. Within minutes, brain cells begin to die due to the interruption of blood flow. Transient ischemic attacks (TIAs), or "mini-strokes," occur when the blood supply to the brain is briefly interrupted and can indicate a higher risk for a more serious stroke [medquad_ans_0000527-1\_\_

**Supported citations**: `medquad_ans_0000527-1__dup2`, `medquad_ans_0000857-1__dup2`, `medquad_ans_0000066-13`
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-3` (score=0.9953) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been s...
2. `medquad_ans_0000145-2__dup3` (score=0.0019) - The causes of Goodpasture syndrome are not fully understood. People who smoke or use hair dyes appear to be at increased risk for this condition. Exposure to hydrocarbon fumes, metallic dust, and certain drugs, such a...
3. `medquad_ans_0000247-3__dup1` (score=0.0011) - What causes alopecia universalis? The exact underlying cause of alopecia universalis (AU) is not currently known. AU is an advanced form of alopecia areata (AA), a condition that leads to round patches of hair loss. A...
4. `medquad_ans_0002442-3` (score=0.0001) - What causes frontal fibrosing alopecia? The exact underlying cause of frontal fibrosing alopecia (FFA) is unknown. FFA is thought to be an autoimmune condition in which an affected person's immune system mistakenly at...
5. `medquad_ans_0006276-6` (score=0.0001) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...

**Answer**:

The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been suggested that the condition may result from premature keratinization (development of keratin) of the inner root sheath, which forms the channel for the growing hair. The inner root sheath conforms in configuration to the abnormal outline of the hair shaft. It thus forms an irregular, rigid tube that then alters the shape of the emerging hair. While it is assumed that the condition is autosomal dominant and thus due to changes (mutations) in a gene, no responsible gene has been identified [medquad_ans_0006276-3].

**Supported citations**: `medquad_ans_0006276-3`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000887-4` (score=0.9860) - Most cases of Russell-Silver syndrome are sporadic, which means they occur in people with no history of the disorder in their family. Less commonly, Russell-Silver syndrome can run in families. In some affected famili...
2. `medquad_ans_0005440-4` (score=0.9638) - Is Russell-Silver syndrome inherited? Most cases of Russell-Silver syndrome (RSS) are sporadic (not inherited), which means they occur in people with no family history of RSS. Less commonly, Russell-Silver syndrome is...
3. `medquad_ans_0000527-4` (score=0.1646) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
4. `medquad_ans_0001084-3__dup1` (score=0.0020) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
5. `medquad_ans_0003718-4` (score=0.0005) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...

**Answer**:

Most cases of Russell-Silver syndrome are sporadic and not inherited, meaning they occur in people with no family history of the disorder. Less commonly, Russell-Silver syndrome is inherited, either in an autosomal dominant or autosomal recessive pattern [medquad_ans_0000887-4]. In autosomal dominant inheritance, one copy of a genetic change in each cell is sufficient to cause the disorder, and each child of an affected individual has a 50% chance of inheriting the change [medquad_ans_0005440-4]. In autosomal recessive inheritance, both copies of the gene must be altered, and each parent is a carrier without symptoms [medquad_ans_0005440-4]. Therefore, Russell-Silver syndrome can be inherited in some families, but most cases are not inherited [medquad_ans_0005440-4

**Supported citations**: `medquad_ans_0000887-4`, `medquad_ans_0005440-4`
**Unsupported citations**: _none_

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=1.000, groundedness=0.949, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.9971) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000286-1__dup2` (score=0.0008) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
3. `medquad_ans_0000043-6__dup1` (score=0.0001) - - Reach and maintain a reasonable body weight. - Make wise food choices most of the time. - Be physically active every day. - Take your prescribed medicines. Doing these things can reduce your risk of developing type ...
4. `medquad_ans_0000022-9` (score=0.0001) - You can do a lot to reduce your risk of getting type 2 diabetes. Being more physically active, reducing fat and calorie intake, and losing a little weight can help you lower your chances of developing type 2 diabetes....
5. `medquad_ans_0000157-5__dup1` (score=0.0001) - You already know you need to watch how much you drink. Any food that is liquid at room temperature also contains water. These foods include soup, Jell-O, and ice cream. Many fruits and vegetables contain lots of water...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It often limits the amount of food you can take in, and some types affect how you digest and absorb nutrients. All types of surgery carry risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off, and lifelong medical follow-up is required. [medquad_ans_0000969-1__dup2]

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.812, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000889-5` (score=0.9947) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
2. `medquad_ans_0004454-3` (score=0.0007) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004452-2` (score=0.0007) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004453-3` (score=0.0007) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004451-3` (score=0.0007) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain information about the treatments for Saethre-Chotzen syndrome. While resources for diagnosis and management of Saethre-Chotzen syndrome are listed, such as the Gene Review, Genetic Testing Registry, and MedlinePlus Encyclopedia, no specific treatments are described in the context. Therefore, based on the available information, the treatments for Saethre-Chotzen syndrome cannot be determined. [medquad_ans_0000889-5]

**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.491, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.9214) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
2. `medquad_ans_0000512-1` (score=0.2314) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
3. `medquad_ans_0000758-1__dup2` (score=0.1210) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000078-4__dup2` (score=0.0025) - Sometimes doctors can find out what is causing pulmonary fibrosis (lung scarring). For example, exposure to environmental pollutants and certain medicines can cause the disease. Environmental pollutants include inorga...
5. `medquad_ans_0000078-7` (score=0.0019) - Doctors may prescribe medicines, oxygen therapy, pulmonary rehabilitation (PR), and lung transplant to treat idiopathic pulmonary fibrosis (IPF). Medicines Currently, no medicines are proven to slow the progression of...

**Answer**:

Idiopathic pulmonary fibrosis (IPF) is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, making them unable to transport oxygen into the bloodstream effectively. It usually affects people between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, gradual weight loss, and in some cases, widened and rounded tips of the fingers and toes (clubbing) due to oxygen shortage. IPF has no known cause in most cases and is referred to as "idiopathic." When more than one family member has the disease, it is called familial IPF. The disease progresses over time, leading to reduced oxygen supply to the body's organs and tissues, and it is associated with serious complications such as respiratory failure, pulmonary hypertension, heart failure, pulmonary embolism, pneumonia, and lung cancer. Currently,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.917, groundedness=0.796, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000213-4__dup1` (score=0.9974) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
2. `medquad_ans_0000006_3-8` (score=0.0861) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000025_1-8` (score=0.0148) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000004_2-2` (score=0.0109) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000213-3__dup1` (score=0.0006) - Early identification of and treatment for individuals with neuroleptic malignant syndrome improves outcome. If clinically indicated, a low potency neuroleptic can be reintroduced very slowly when the individual recove...

**Answer**:

The provided context does not contain specific information about research or clinical trials being conducted for Neuroleptic Malignant Syndrome. While the NINDS supports research on neurological disorders such as neuroleptic malignant syndrome, including efforts to find ways to prevent and treat the disorder, no details about specific clinical trials or ongoing research are given [medquad_ans_0000213-4__dup1]. Therefore, based on the available context, the exact nature of current research or clinical trials for this condition cannot be determined.

**Supported citations**: `medquad_ans_0000213-4__dup1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.857, groundedness=0.882, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000221-8` (score=0.9934) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
2. `medquad_ans_0000221-10` (score=0.0012) - To prevent kidney stones, you need to know what caused your kidney stone. Your doctor may ask you to try to catch the kidney stone as it passes in your urine. The kidney stone can then be sent to a lab to find out wha...
3. `medquad_ans_0000221-4__dup2` (score=0.0001) - Doctors have found four main types of kidney stones: - The most common types of stones contain calcium. Calcium is a normal part of a healthy diet. Calcium that is not used by the bones and muscles goes to the kidneys...
4. `medquad_ans_0000203-4__dup2` (score=0.0001) - Kidney stones can form when substances in the urinesuch as calcium, oxalate, and phosphorusbecome highly concentrated. Certain foods may promote stone formation in people who are susceptible, but scientists do not bel...
5. `medquad_ans_0000221-12` (score=0.0001) - - A kidney stone is a solid piece of material that forms in a kidney when there are high levels of certain substances in the urine. These substances are normally found in the urine and do not cause problems at lower l...

**Answer**:

To diagnose kidney stones in adults, your doctor will perform a physical exam and ask about your medical history, including family history of kidney stones, diet, digestive problems, and other health issues. The doctor may conduct urine, blood, and imaging tests to complete the diagnosis. Urine tests can detect infections or substances in the urine that form stones. Blood tests can identify underlying problems that contribute to stone formation. Imaging tests are used to locate kidney stones in the body and may also reveal conditions that caused the stones to form [medquad_ans_0000221-8].

**Supported citations**: `medquad_ans_0000221-8`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000205-6` (score=0.4465) - The job of the sphincter muscles is to hold urine in the bladder by squeezing the urethra shut. If the urethral sphincter fails to stay closed, urine may leak out of the bladder. When nerve signals are coordinated pro...
2. `medquad_ans_0000041_1-1` (score=0.0700) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
3. `medquad_ans_0000929-1__dup1` (score=0.0185) - The urethra is the tube that allows urine to pass out of the body. In men, it's a long tube that runs through the penis. It also carries semen in men. In women, it's short and is just above the vagina. Urethral proble...
4. `medquad_ans_0000063-27` (score=0.0102) - Once cancer has been found, the doctor will need to determine the extent, or stage, of the cancer. Through staging, the doctor can tell if the cancer has spread and, if so, to what parts of the body. More tests may be...
5. `medquad_ans_0000041_1-5` (score=0.0036) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. -...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. It begins in cells that line the urethra. A history of bladder cancer can affect the risk of urethral cancer. Signs of urethral cancer include bleeding or trouble with urination. Tests that examine the urethra and bladder are used to detect and diagnose urethral cancer. Certain factors affect prognosis and treatment options [medquad_ans_0000041_1-1].

**Supported citations**: `medquad_ans_0000041_1-1`
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=1.000, groundedness=0.002, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004671-4` (score=0.0251) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible...
2. `medquad_ans_0000308-4` (score=0.0001) - Dyskeratosis congenita can have different inheritance patterns. When dyskeratosis congenita is caused by DKC1 gene mutations, it is inherited in an X-linked recessive pattern. The DKC1 gene is located on the X chromos...
3. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0004287-1` (score=0.0001) - Myotonia congenita is a genetic condition characterized by the inability of the skeletal muscles to quickly relax after a voluntary movement. The symptoms associated with the condition typically appear in childhood an...
5. `medquad_ans_0004725-1` (score=0.0001) - Paramyotonia congenita is an inherited condition that affects muscles used for movement (skeletal muscles), mainly in the face, neck, arms, and hands. Symptoms begin in infancy or early childhood and include episodes ...

**Answer**:

The provided context does not contain information about the inheritance of paramyotonia congenita. Therefore, it is not possible to determine based on the given text whether paramyotonia congenita is inherited.

Answer: Insufficient context [medquad_ans_0004671-4, medquad_ans_0000308-4, medquad_ans_0001853-3, medquad_ans_0004287-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0004671-4, medquad_ans_0000308-4, medquad_ans_0001853-3, medquad_ans_0004287-1`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.540, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.9987) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000125-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the BTD gene in each cell have mutations. The parents of an individual with biotinidase deficiency each carry one copy of the m...
3. `medquad_ans_0000735-1__dup1` (score=0.0001) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form ...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0000125-1` (score=0.0001) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few ...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.857, groundedness=0.350, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000846-1__dup1` (score=0.0003) - Summary : Playing sports can be fun, but it can also be dangerous if you are not careful. You can help prevent injuries by - Getting a physical to make sure you are healthy before you start playing your sport - Wearin...
2. `medquad_ans_0000505-1__dup1` (score=0.0002) - Summary : We usually think of air pollution as being outdoors, but the air in your house or office could also be polluted. Sources of indoor pollution include - Mold and pollen - Tobacco smoke - Household products and...
3. `medquad_ans_0000019-1__dup3` (score=0.0001) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air p...
4. `medquad_ans_0000035-19` (score=0.0001) - The recommended blood test for checking your cholesterol levels is called a fasting lipoprotein profile. It will show your - total cholesterol - low-density lipoprotein (LDL), or bad cholesterol -- the main source of ...
5. `medquad_ans_0000066-1__dup3` (score=0.0001) - Asthma is a chronic disease that affects your airways. Your airways are tubes that carry air in and out of your lungs. If you have asthma, the inside walls of your airways become sore and swollen. In the United States...

**Answer**:

No, there is no information about water pollution in the provided context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.857, groundedness=0.973, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.9937) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000132-2__dup1` (score=0.0002) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the mus...
3. `medquad_ans_0000132-1__dup1` (score=0.0001) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend t...
4. `medquad_ans_0000132-4__dup1` (score=0.0001) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to the neurological conditions that cause foot drop in its laboratories at the National Institutes of Health (NIH), and als...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000145-1` (score=0.7351) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
2. `medquad_ans_0000916-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000915-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
4. `medquad_ans_0000145-2` (score=0.0001) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. The age at which affected individuals first experience symptoms varies greatly; however, most people with this condition develop pain or weakness by adolescence. Camurati-Engelmann disease is caused by a mutation in the TGFB1 gene which is inherited in an autosomal dominant fashion. In some instances, people have the gene mutation that causes Camurati-Engelmann disease but never develop the characteristic features of this condition. In others, features are present, but a mutation cannot be identified. These cases are referred to as Camurati-Engelmann disease type

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.9994) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
2. `medquad_ans_0000316-4` (score=0.0001) - Emery-Dreifuss muscular dystrophy can have several different patterns of inheritance. When this condition is caused by mutations in the EMD gene, it is inherited in an X-linked recessive pattern. A condition is consid...
3. `medquad_ans_0003612-1` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy, type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy, type 2G. If the informatio...
4. `medquad_ans_0000316-1` (score=0.0001) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...
5. `medquad_ans_0003610-2` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy, type 2C? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy, type 2C. If the informatio...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include abnormality of the neck, Achilles tendon contracture, atrioventricular block, childhood onset, decreased cervical spine flexion due to contractures of posterior cervical muscles, elbow flexion contracture, elevated serum creatine phosphokinase, juvenile onset, pectus excavatum, primary atrial arrhythmia, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, and X-linked recessive inheritance [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.764, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000072-5` (score=0.9945) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
2. `medquad_ans_0000054-17` (score=0.0006) - Here are links to more information about P.A.D. from the National Heart, Lung, and Blood Institute. - What Is Peripheral Arterial Disease? - Atherosclerosis - The DASH Eating Plan - Facts about P.A.D. - Facts about P....
3. `medquad_ans_0000054-16` (score=0.0002) - The National Heart, Lung, and Blood Institute (NHLBI) supports research aimed at learning more about peripheral arterial disease (P.A.D.). For example, NHLBI-supported research on P.A.D. includes studies that - explor...
4. `medquad_ans_0000277-2__dup2` (score=0.0001) - There is no cure for Zellweger syndrome, nor is there a standard course of treatment. Since the metabolic and neurological abnormalities that cause the symptoms of Zellweger syndrome are caused during fetal developmen...
5. `medquad_ans_0000072-2` (score=0.0001) - Arterial tortuosity syndrome is a rare disorder; its prevalence is unknown. About 100 cases have been reported in the medical literature.

**Answer**:

The provided context does not specify the treatments for arterial tortuosity syndrome. While it mentions resources for the diagnosis or management of arterial tortuosity syndrome, including genetic testing and information from the National Heart, Lung, and Blood Institute, no specific treatments are detailed [medquad_ans_0000072-5]. Therefore, based on the given information, the treatments for arterial tortuosity syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
3. `medquad_ans_0001112-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...
4. `medquad_ans_0001100-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
5. `medquad_ans_0001104-5` (score=0.0001) - Is genetic testing available for Charcot-Marie-Tooth disease type 2F? Yes. GeneTests lists the names of laboratories that are performing clincial genetic testing for Charcot-Marie-Tooth disease type 2F. To view the co...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, and segmental peripheral demyelination/remyelination [medquad_ans_0001092-1__dup1].

**Supported citations**: `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.9994) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
2. `medquad_ans_0006446-2` (score=0.0001) - What are the signs and symptoms of Williams syndrome? The signs and symptoms of Williams syndrome can be variable, but the disorder is generally characterized by mild to moderate intellectual disability a distinctive ...
3. `medquad_ans_0000870-1` (score=0.0001) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
4. `medquad_ans_0001625-1` (score=0.0001) - What are the signs and symptoms of Cranioacrofacial syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cranioacrofacial syndrome. If the information is available, the table be...
5. `medquad_ans_0000962-1` (score=0.0001) - Supravalvular aortic stenosis (SVAS) is a heart defect that develops before birth. This defect is a narrowing (stenosis) of the large blood vessel that carries blood from the heart to the rest of the body (the aorta)....

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis include exudative retinal detachment and pulmonic stenosis [medquad_ans_0005321-1].

**Supported citations**: `medquad_ans_0005321-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0001838-1` (score=0.3892) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is...
2. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
3. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0000549-1__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and is inherited in an autosomal recessive fashion [medquad_ans_0001838-1].

**Supported citations**: `medquad_ans_0001838-1`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005860-3` (score=0.9863) - Is spondylothoracic dysostosis genetic? Yes. Spondylothoracic dysostosis is caused by mutations in the MESP2 gene. It is inherited in an autosomal recessive fashion.
2. `medquad_ans_0000944-4` (score=0.9726) - Spondylocostal dysostosis can have different inheritance patterns. Types 1, 2, 3, and 4 are inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of ...
3. `medquad_ans_0001034-4` (score=0.6560) - Weyers acrofacial dysostosis is inherited in an autosomal dominant pattern, which means one copy of the altered EVC or EVC2 gene in each cell is sufficient to cause the disorder. In most cases, an affected person inhe...
4. `medquad_ans_0000949-3` (score=0.0001) - The MESP2 gene provides instructions for a protein that plays a critical role in the development of vertebrae. Specifically, it is involved in separating vertebrae from one another during early development, a process ...
5. `medquad_ans_0005859-1` (score=0.0001) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevel...

**Answer**:

Yes, spondylothoracic dysostosis is inherited. It is caused by mutations in the MESP2 gene and is inherited in an autosomal recessive fashion [medquad_ans_0005860-3]. The MESP2 gene provides instructions for a protein that plays a critical role in the development of vertebrae, specifically in separating vertebrae during early development (somite segmentation); mutations in this gene lead to improper segmentation and fusion of spinal and rib bones [medquad_ans_0000949-3].

**Supported citations**: `medquad_ans_0005860-3`, `medquad_ans_0000949-3`
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.299, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000542-1` (score=0.9244) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
2. `medquad_ans_0003296-1` (score=0.8756) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
3. `medquad_ans_0000542-2` (score=0.0004) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...

**Answer**:

Jervell and Lange-Nielsen syndrome is a rare condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is a form of long QT syndrome, which results in the heart muscle taking longer than usual to recharge between beats. This disorder increases the risk of fainting (syncope) and sudden death beginning in early childhood. There are two types—type 1 and type 2—and it is inherited in an autosomal recessive fashion. The condition is uncommon, affecting an estimated 1.6 to 6 per 1 million people worldwide, with a higher prevalence in Denmark, where it affects at least 1 in 200,000 people [medquad_ans_0000542-1, medquad_ans_0003296-1, medquad_ans_0000542-2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000542-1, medquad_ans_0003296-1, medquad_ans_0000542-2`

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=0.917, groundedness=0.745, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000218-1` (score=0.0001) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
2. `medquad_ans_0004453-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004454-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004451-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004456-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not specify the number of people affected by congenital cataracts, facial dysmorphism, and neuropathy (CCFDN). [medquad_ans_0000218-1] describes the disorder's features and effects but does not include epidemiological data on prevalence or affected population size. Therefore, the exact number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000218-1`
**Unsupported citations**: _none_

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.769, groundedness=0.949, answer_relevance=0.892

**Top docs**:

1. `medquad_ans_0000040-3` (score=0.9988) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although th...
2. `medquad_ans_0000039-3` (score=0.0003) - Alpha thalassemia typically results from deletions involving the HBA1 and HBA2 genes. Both of these genes provide instructions for making a protein called alpha-globin, which is a component (subunit) of hemoglobin. Pe...
3. `medquad_ans_0000251-3` (score=0.0002) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that...
4. `medquad_ans_0000127-3__dup2` (score=0.0001) - Your body makes three types of blood cells: red blood cells, white blood cells, and platelets (PLATE-lets). Red blood cells contain hemoglobin, an iron-rich protein that carries oxygen from your lungs to all parts of ...
5. `medquad_ans_0000040-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The ATRX gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the g...

**Answer**:

The genetic changes related to alpha thalassemia X-linked intellectual disability syndrome are mutations in the ATRX gene. These mutations change the structure of the ATRX protein, which likely prevents it from effectively regulating gene expression. Reduced activity of the HBA1 and HBA2 genes, which are necessary for normal hemoglobin production, causes alpha thalassemia. Abnormal expression of other unidentified genes probably leads to developmental delay, distinctive facial features, and other signs and symptoms of the syndrome [medquad_ans_0000040-3].

**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.696, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002622-2` (score=0.9994) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
2. `medquad_ans_0000424-5` (score=0.0001) - These resources address the diagnosis or management of glycogen storage disease type IV: - Gene Review: Gene Review: Glycogen Storage Disease Type IV - Genetic Testing Registry: Glycogen storage disease, type IV - Med...
3. `medquad_ans_0002622-1` (score=0.0001) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
4. `medquad_ans_0002620-4` (score=0.0001) - How is glycogen storage disease type 4 inherited? Glycogen storage disease type 4 is inherited in an autosomal recessive manner. This means that an individual must have 2 abnormal copies of the GBE1 gene to be affecte...
5. `medquad_ans_0000422-1` (score=0.0001) - Glycogen storage disease type I (also known as GSDI or von Gierke disease) is an inherited disorder caused by the buildup of a complex sugar called glycogen in the body's cells. The accumulation of glycogen in certain...

**Answer**:

The symptoms of Glycogen storage disease type 6 include hypoglycemia (seen in 90% of patients), short stature (seen in 90% of patients), hepatomegaly (enlarged liver), increased hepatic glycogen content, and postnatal growth retardation [medquad_ans_0002622-2]. Hypoglycemia and short stature are the most commonly reported symptoms, with both occurring in approximately 90% of individuals [medquad_ans_0002622-2]. Symptoms typically begin in infancy or childhood and may improve significantly with age [medquad_ans_0002621]. The condition is caused by mutations in the PYGL gene and is inherited in an autosomal recessive manner [medquad_ans_0002622-1].

**Supported citations**: `medquad_ans_0002622-2`, `medquad_ans_0002622-1`
**Unsupported citations**: `medquad_ans_0002621`

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.762, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000042-6__dup1` (score=0.9951) - Diabetes Medications Hypoglycemia can occur as a side effect of some diabetes medications, including insulin and oral diabetes medicationspillsthat increase insulin production, such as - chlorpropamide (Diabinese) - g...
2. `medquad_ans_0000042-5__dup2` (score=0.9951) - Diabetes Medications Hypoglycemia can occur as a side effect of some diabetes medications, including insulin and oral diabetes medicationspillsthat increase insulin production, such as - chlorpropamide (Diabinese) - g...
3. `medquad_ans_0000015-16` (score=0.9892) - Type 1 diabetes is an autoimmune disease. In an autoimmune reaction, antibodies, or immune cells, attach to the body's own healthy tissues by mistake, signaling the body to attack them. At present, scientists do not k...
4. `medquad_ans_0000057-1__dup5` (score=0.0895) - Too much glucose in the blood for a long time can cause diabetes problems. This high blood glucose, also called blood sugar, can damage many parts of the body, such as the heart, blood vessels, eyes, and kidneys. Hear...
5. `medquad_ans_0001100-3` (score=0.0193) - 6q24-related transient neonatal diabetes mellitus is caused by the overactivity (overexpression) of certain genes in a region of the long (q) arm of chromosome 6 called 6q24. People inherit two copies of their genes, ...

**Answer**:

The provided context does not address the causes of diabetes. It discusses hypoglycemia as a side effect of certain diabetes medications but does not identify the underlying causes of diabetes. Therefore, based on the given information, the causes of diabetes cannot be determined. [medquad_ans_0000042-6__dup1] [medquad_ans_0000042-5__dup2]

**Supported citations**: `medquad_ans_0000042-6__dup1`, `medquad_ans_0000042-5__dup2`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=0.727, groundedness=0.769, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000006_3-8` (score=0.6321) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000040_1-6` (score=0.3080) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000024_2-8` (score=0.0591) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000025_1-8` (score=0.0199) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000004_2-2` (score=0.0138) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Diabetic Retinopathy. [medquad_ans_0000006_3-8] and [medquad_ans_0000040_1-6] discuss clinical trials for cancer treatments, not Diabetic Retinopathy. Therefore, no information is available in the context to answer the question.

**Supported citations**: `medquad_ans_0000006_3-8`, `medquad_ans_0000040_1-6`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.9100) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000977-1__dup1` (score=0.0014) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from m...
3. `medquad_ans_0000845-1__dup1` (score=0.0002) - Exercising is good for you, but sometimes you can injure yourself when you play sports or exercise. Accidents, poor training practices, or improper gear can cause them. Some people get hurt because they are not in sha...
4. `medquad_ans_0000419-1__dup1` (score=0.0001) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...
5. `medquad_ans_0000424-1__dup2` (score=0.0001) - Chances are you've bumped your head before. Usually, the injury is minor because your skull is hard and it protects your brain. But other head injuries can be more severe, such as a skull fracture, concussion, or trau...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen. The best way to prevent inhalation injuries is to limit exposure, especially when smoke is detected or fires are nearby [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.945, answer_relevance=0.475

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.9982) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...
2. `medquad_ans_0000134-9` (score=0.0005) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
3. `medquad_ans_0000134-3__dup2` (score=0.0002) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes horm...
4. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0000134-5__dup1` (score=0.0001) - Zollinger-Ellison syndrome signs and symptoms are similar to those of peptic ulcers. A dull or burning pain felt anywhere between the navel and midchest is the most common symptom of a peptic ulcer. This discomfort us...

**Answer**:

Zollinger-Ellison syndrome occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2].

**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.777, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.9949) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
2. `medquad_ans_0004453-3` (score=0.0001) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004452-2` (score=0.0001) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004451-3` (score=0.0001) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004454-3` (score=0.0001) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain information about the treatments for tetrahydrobiopterin deficiency. While it lists diagnostic resources such as genetic testing and serum phenylalanine screening, no specific treatments are mentioned [medquad_ans_0000974-5]. Therefore, based on the available context, the treatments for tetrahydrobiopterin deficiency cannot be determined.

**Supported citations**: `medquad_ans_0000974-5`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.869, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000736-2` (score=0.9991) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
2. `medquad_ans_0000736-1` (score=0.0001) - Nonsyndromic holoprosencephaly is an abnormality of brain development that also affects the head and face. Normally, the brain divides into two halves (hemispheres) during early development. Holoprosencephaly occurs w...
3. `medquad_ans_0000736-4` (score=0.0001) - Nonsyndromic holoprosencephaly is inherited in an autosomal dominant pattern, which means an alteration in one copy of a gene in each cell is usually sufficient to cause the disorder. However, not all people with a ge...
4. `medquad_ans_0000150-1__dup1` (score=0.0001) - Holoprosencephaly is a disorder caused by the failure of the prosencephalon (the embryonic forebrain) to sufficiently divide into the double lobes of the cerebral hemispheres. The result is a single-lobed brain struct...
5. `medquad_ans_0000584-1__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal recessive nonsyndromic congenital nuclear cataract? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive nonsyndromic cong...

**Answer**:

Nonsyndromic holoprosencephaly affects an estimated 1 in 10,000 newborns [medquad_ans_0000736-2]. This represents approximately 25 to 50 percent of all cases of holoprosencephaly [medquad_ans_0000736-2]. Therefore, the prevalence of nonsyndromic holoprosencephaly is about 1 in 10,000 newborns.

**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=1.000, groundedness=0.112, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000525-3` (score=0.0013) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
2. `medquad_ans_0003240-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile ...
3. `medquad_ans_0000525-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy is a liver disorder that occurs in pregnant women. Cholestasis is a condition that impairs the release of a digestive fluid called bile from liver cells. As a result, bile builds ...
4. `medquad_ans_0000829-1` (score=0.0001) - Progressive familial intrahepatic cholestasis (PFIC) is a disorder that causes progressive liver disease, which typically leads to liver failure. In people with PFIC, liver cells are less able to secrete a digestive f...
5. `medquad_ans_0003240-2` (score=0.0001) - What are the signs and symptoms of Intrahepatic cholestasis of pregnancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Intrahepatic cholestasis of pregnancy. If the information is...

**Answer**:

The provided context does not mention progressive familial intrahepatic cholestasis or describe genetic changes related to this condition. Therefore, based on the given information, it is not possible to answer the question. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.9994) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0003571-1` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...
5. `medquad_ans_0001388-1` (score=0.0001) - What are the signs and symptoms of COASY Protein-Associated Neurodegeneration ? The Human Phenotype Ontology provides the following list of signs and symptoms for COASY Protein-Associated Neurodegeneration . If the in...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=0.857, groundedness=0.075, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000228-4` (score=0.0001) - The various syndromes of which congenital hepatic fibrosis is often a feature can have different inheritance patterns. Most of these disorders are inherited in an autosomal recessive pattern, which means both copies o...
2. `medquad_ans_0000480-1__dup1` (score=0.0001) - Summary : HIV, the human immunodeficiency virus, kills or damages cells of the body's immune system. The most advanced stage of infection with HIV is AIDS, which stands for acquired immunodeficiency syndrome. HIV ofte...
3. `medquad_ans_0003166-1` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
4. `medquad_ans_0006510-1` (score=0.0001) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

The provided context does not contain information about hepatic veno-occlusive disease with immunodeficiency or its inheritance pattern. Therefore, it is not possible to determine based on the given text whether this condition is inherited.

Answer: Insufficient context [medquad_ans_0000228-4, medquad_ans_0000480-1__dup1, medquad_ans_0003166-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000228-4, medquad_ans_0000480-1__dup1, medquad_ans_0003166-1`

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.9993) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0000057-4__dup3` (score=0.0117) - Symptoms Vary With Disorders There are several types of taste disorders depending on how the sense of taste is affected. People who have taste disorders usually lose their ability to taste or can no longer perceive ta...
3. `medquad_ans_0000006-3__dup4` (score=0.0001) - Some people may have a balance problem without realizing it. Others might think they have a problem, but are too embarrassed to tell their doctor, friends, or family. Here are common symtoms experienced by people with...
4. `medquad_ans_0000057-22` (score=0.0001) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions....
5. `medquad_ans_0001754-1` (score=0.0001) - What are the signs and symptoms of Deafness, dystonia, and cerebral hypomyelination ? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness, dystonia, and cerebral hypomyelination...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. [medquad_ans_0005902-1]

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.938, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000159-7` (score=0.9732) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
2. `medquad_ans_0000159-8` (score=0.9253) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
3. `medquad_ans_0000159-4__dup2` (score=0.0088) - Lupus nephritis is diagnosed through urine and blood tests and a kidney biopsy: - Urinalysis. Urinalysis is testing of a urine sample. The urine sample is collected in a special container in a health care providers of...
4. `medquad_ans_0000165-3__dup2` (score=0.0015) - Kidney disease in children can be caused by - birth defects - hereditary diseases - infection - nephrotic syndrome - systemic diseases - trauma - urine blockage or reflux From birth to age 4, birth defects and heredit...
5. `medquad_ans_0000159-6` (score=0.0006) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent further damage to the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication such as cyclophosphamide or mycophenolate mofetil, typically used in combination with prednisone [medquad_ans_0000159-8]. If lupus nephritis is causing high blood pressure, medications to lower blood pressure may be needed, as this can significantly slow the progression of kidney disease [medquad_ans_0000159-8]. Additionally, people with kidney disease may need to consult a health care provider about dietary changes, such as reducing sodium intake, especially if they have high blood pressure [medquad_ans_0000159-7]. Diagnosis is confirmed through urine and blood tests, including

**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0000159-7`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.879, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000869-5` (score=0.9811) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndro...
2. `medquad_ans_0000277-2__dup2` (score=0.0023) - There is no cure for Zellweger syndrome, nor is there a standard course of treatment. Since the metabolic and neurological abnormalities that cause the symptoms of Zellweger syndrome are caused during fetal developmen...
3. `medquad_ans_0000776-1__dup2` (score=0.0006) - Restless legs syndrome (RLS) causes a powerful urge to move your legs. Your legs become uncomfortable when you are lying down or sitting. Some people describe it as a creeping, crawling, tingling, or burning sensation...
4. `medquad_ans_0000824-1__dup2` (score=0.0002) - Is it hard for you to fall asleep or stay asleep through the night? Do you wake up feeling tired or feel very sleepy during the day, even if you have had enough sleep? You might have a sleep disorder. The most common ...
5. `medquad_ans_0001104-6` (score=0.0002) - How might Charcot-Marie-Tooth disease type 2F be treated? Treatment for Charcot-Marie-Tooth disease type 2 mainly focuses on the specific symptoms present. Affected individuals are often managed by a team of various s...

**Answer**:

Treatments for Restless Legs Syndrome (RLS) include lifestyle changes such as maintaining regular sleep habits, practicing relaxation techniques, and engaging in moderate exercise during the day. If these measures do not alleviate symptoms, medications may be used to reduce the symptoms of RLS [medquad_ans_0000776-1__dup2]. Additionally, the National Heart, Lung, and Blood Institute provides information on how RLS is treated, which may include specific pharmacological options [medquad_ans_0000869-5]. Most people with RLS also have periodic limb movement disorder (PLMD), which may be managed as part of the overall treatment plan [medquad_ans_0000776-1__dup2].

**Supported citations**: `medquad_ans_0000776-1__dup2`, `medquad_ans_0000869-5`
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002573-1` (score=0.6410) - Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue th...
2. `medquad_ans_0002573-5` (score=0.0010) - Is genetic testing available for glioblastoma? Genetic testing is not available for many people with glioblastoma since most of these tumors occur sporadically (by chance) and are not caused by a genetic mutation. How...
3. `medquad_ans_0002575-1` (score=0.0004) - Gliomatosis cerebri is a type of brain cancer. It is a variant form of glioblastoma multiforme. It is characterized by scattered and widespread tumor cells that can cause the cerebrum, cerebellum, or brain stem to enl...
4. `medquad_ans_0002573-4` (score=0.0003) - Is glioblastoma inherited? Most glioblastomas are not inherited. They usually occur sporadically in people with no family history of tumors. However, they can rarely occur in people with certain genetic syndromes such...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue that helps repair brain damage in response to injury. Glioblastomas are often very aggressive and grow into surrounding brain tissue. Signs and symptoms, such as headache, nausea, vomiting, and/or drowsiness, may develop when the tumor begins to put excess pressure on the brain. Affected people may also experience other features depending on the size and location of the tumor. In most cases, the exact underlying cause is unknown; however, they can rarely occur in people with certain genetic syndromes such as neurofibromatosis type 1, Turcot syndrome, and Li Fraumeni syndrome. There is currently no cure for glioblastoma. Treatment is palliative and may include surgery

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0005496-3` (score=0.9944) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, s...
2. `medquad_ans_0000251-3__dup1` (score=0.0001) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.
3. `medquad_ans_0005496-4` (score=0.0001) - Is schizencephaly inherited? Schizencephaly is not thought to be inherited in most cases and it rarely affects more than one person in a family. A few cases of familial schizencephaly have been linked to changes (muta...
4. `medquad_ans_0005496-5` (score=0.0001) - Is genetic testing available for schizencephaly? In rare cases, people affected by schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Genetic testing is available f...
5. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, which supports a genetic cause in some cases. Schizencephaly has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.801, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000694-3` (score=0.0002) - Although the cause of multiple sclerosis is unknown, variations in dozens of genes are thought to be involved in multiple sclerosis risk. Changes in the HLA-DRB1 gene are the strongest genetic risk factors for develop...
2. `medquad_ans_0000316-1__dup1` (score=0.0001) - What are the signs and symptoms of Amyotrophic lateral sclerosis type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis type 6. If the information is a...
3. `medquad_ans_0000309-1__dup1` (score=0.0001) - Amyotrophic lateral sclerosis (ALS), also referred to as "Lou Gehrig's disease," is a progressive motor neuron disease which leads to problems with muscle control and movement. There are various types of ALS, which ar...
4. `medquad_ans_0000551-1` (score=0.0001) - Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. The features of this disorder are caused by damage to moto...
5. `medquad_ans_0000320-1__dup1` (score=0.0001) - What are the signs and symptoms of Amyotrophic lateral sclerosis-parkinsonism/dementia complex 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis-parkin...

**Answer**:

The provided context does not contain information about genetic changes related to amyotrophic lateral sclerosis. [medquad_ans_0000694-3] discusses genetic factors in multiple sclerosis, not amyotrophic lateral sclerosis type 6. [medquad_ans_0000316-1__dup1] lists signs and symptoms of amyotrophic lateral sclerosis type 6 but does not mention genetic changes. Therefore, based on the given context, the genetic changes related to amyotrophic lateral sclerosis cannot be determined.

**Supported citations**: `medquad_ans_0000694-3`, `medquad_ans_0000316-1__dup1`
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.981, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0003473-6` (score=0.9531) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
2. `medquad_ans_0004453-3` (score=0.0024) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0000277-2__dup2` (score=0.0012) - There is no cure for Zellweger syndrome, nor is there a standard course of treatment. Since the metabolic and neurological abnormalities that cause the symptoms of Zellweger syndrome are caused during fetal developmen...
4. `medquad_ans_0005174-2` (score=0.0004) - What treatments are available for pudendal neuralgia? There are no established guidelines for the treatment of pudendal neuralgia. Treatment may include medications, the use of a specialized cushion while sitting, ner...
5. `medquad_ans_0002796-4` (score=0.0002) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth, improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It has also been shown to raise blood glucose levels, reduce cholesterol, and increase muscle growth. IGF-1 and GH levels should be closely monitored during treatment because overdosage can cause health problems [medquad_ans_0003473-6].

**Supported citations**: `medquad_ans_0003473-6`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000016-19` (score=0.9650) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
2. `medquad_ans_0000016-15` (score=0.9647) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...
3. `medquad_ans_0000016-4__dup2` (score=0.8793) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
4. `medquad_ans_0000016-3__dup2` (score=0.1108) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year to detect the disease before it causes damage to your vision. Visio...
5. `medquad_ans_0000016-10` (score=0.0423) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year. Learn more about a comprehensive dilated eye exam.

**Answer**:

The treatments for diabetic retinopathy include:

- For the first three stages, no treatment is needed unless macular edema is present. To prevent progression, people with diabetes should control blood sugar, blood pressure, and blood cholesterol levels [medquad_ans_0000016-15].

- Proliferative retinopathy is treated with scatter laser treatment, which involves placing 1,000 to 2,000 laser burns in the retina away from the macula to shrink abnormal blood vessels. Two or more sessions are usually required, and this treatment may slightly reduce color and night vision. Side vision may be lost, but the rest of the sight can be preserved [medquad_ans_0000016-15].

- For macular edema, prompt treatment with anti-VEGF drugs (injected into the eye) results in better vision than laser treatment alone or steroid

**Supported citations**: `medquad_ans_0000016-15`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000116-8` (score=0.9757) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during ...
2. `medquad_ans_0001421-3` (score=0.9645) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
3. `medquad_ans_0000122-9` (score=0.0003) - Complications of ulcerative colitis can include - rectal bleedingwhen ulcers in the intestinal lining open and bleed. Rectal bleeding can cause anemia, which health care providers can treat with diet changes and iron ...
4. `medquad_ans_0000221-1__dup2` (score=0.0001) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...
5. `medquad_ans_0005174-2` (score=0.0001) - What treatments are available for pudendal neuralgia? There are no established guidelines for the treatment of pudendal neuralgia. Treatment may include medications, the use of a specialized cushion while sitting, ner...

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to have ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, which may include a reduced-fat diet, eliminating foods that contain caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If these changes alone are not enough, medications can be used to help control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids including budesonide and prednisone can be used to reduce inflammation, but they are usually only used to control sudden attacks of diarrhea. Long-term use of steroids is typically avoided because of unwanted side effects. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxyl

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.667, groundedness=0.968, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000418-7` (score=0.8627) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
2. `medquad_ans_0000094-11` (score=0.0397) - A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - a...
3. `medquad_ans_0000109-1__dup3` (score=0.0285) - Summary : If you have some kinds of heart or blood vessel disease, or if you have poor blood flow to your brain, your doctor may recommend that you take a blood thinner. Blood thinners reduce the risk of heart attack ...
4. `medquad_ans_0000904-1__dup2` (score=0.0200) - Your teeth are made of a hard, bonelike material. Inside the tooth are nerves and blood vessels. You need your teeth for many activities you may take for granted. These include eating, speaking and even smiling. But t...
5. `medquad_ans_0000396-3` (score=0.0161) - The FH gene provides instructions for making an enzyme called fumarase (also known as fumarate hydratase). Fumarase participates in an important series of reactions known as the citric acid cycle or Krebs cycle, which...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis), wash your hands with warm water and soap after handling raw meat. Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; therefore, these methods are not reliable. Freeze pork less than 6 inches thick for 20 days at 5°F (-15°C) to kill any worms. Freezing wild game meats may not effectively kill all worms because some worm species that infect wild game animals are freeze-resistant. Clean meat grinders thoroughly after each use. To help prevent Trichinella infection in animal populations, do not allow pigs or wild animals to eat uncooked meat, scraps, or carcasses of any animals, including rats, which may be infected with Trichinella [medquad_ans_0000418-7].

**Supported citations**: `medquad_ans_0000418-7`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000536-3` (score=0.9939) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic at...
2. `medquad_ans_0000535-3` (score=0.0029) - What causes autoimmune hemolytic anemia? In about half of cases, the cause of autoimmune hemolytic anemia cannot be determined (idiopathic or primary). This condition can also be caused by or occur with another disord...
3. `medquad_ans_0001415-3` (score=0.0013) - What causes cold agglutinin disease? Cold agglutinin disease is typically classified as primary (unknown cause) or secondary (caused by an underlying condition). Secondary cold agglutinin disease may be associated wit...
4. `medquad_ans_0000193-2__dup3` (score=0.0005) - Causes of cirrhosis include - heavy alcohol use - some drugs, medicines, and harmful chemicals - infections - chronic hepatitis B, C, or Dviral infections that attack the liver - autoimmune hepatitis, which causes the...
5. `medquad_ans_0000007_4-2` (score=0.0004) - Being infected with certain types of the hepatitis virus can cause hepatitis and increase the risk of liver cancer. Hepatitis is most commonly caused by the hepatitis virus. Hepatitis is a disease that causes inflamma...

**Answer**:

Autoimmune hepatitis is caused by a cell-mediated immunologic attack on the liver, which may be triggered by genetic factors, viral infections, or chemical agents. The condition sometimes occurs in relatives of people with autoimmune diseases, suggesting a genetic component [medquad_ans_0000536-3].

**Supported citations**: `medquad_ans_0000536-3`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.958, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.9580) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0000006_3-7` (score=0.0017) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
3. `medquad_ans_0000006_3-4` (score=0.0003) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor gr...
4. `medquad_ans_0000273-2__dup2` (score=0.0001) - There is no cure for TSC, although treatment is available for a number of the symptoms. Rapamycin and related drugs are not yet approved by the U.S. Food and Drug Administration (FDA) for any purpose in individuals wi...
5. `medquad_ans_0000006_3-9` (score=0.0001) - Key Points - There are different types of treatment for patients with childhood astrocytoma. - Children with astrocytomas should have their treatment planned by a team of health care providers who are experts in treat...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.980, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-1__dup3` (score=0.8728) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
2. `medquad_ans_0000013_2-2__dup3` (score=0.0018) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.
3. `medquad_ans_0000013_2-1__dup3` (score=0.0004) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treat...
4. `medquad_ans_0000013_3-4__dup2` (score=0.0003) - Treatment of atypical chronic myelogenous leukemia (CML) may include chemotherapy. Check the list of NCI-supported cancer clinical trials that are now accepting patients with atypical chronic myeloid leukemia, BCR-ABL...
5. `medquad_ans_0000013_3-2__dup3` (score=0.0001) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check ...

**Answer**:

Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In atypical chronic myelogenous leukemia (CML), the body tells too many blood stem cells to become a type of white blood cell called granulocytes. Some of these blood stem cells never become mature white blood cells. These immature white blood cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. The leukemia cells in atypical CML and CML look alike under a microscope. However, in atypical CML a certain chromosome change, called the "Philadelphia chromosome" is not present [medquad_ans_0000013_3-1__dup3].

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=1.000, groundedness=0.920, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.9960) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
2. `medquad_ans_0000996-4` (score=0.0003) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...
3. `medquad_ans_0000996-2` (score=0.0001) - Trisomy 13 occurs in about 1 in 16,000 newborns. Although women of any age can have a child with trisomy 13, the chance of having a child with this condition increases as a woman gets older.
4. `medquad_ans_0000997-3` (score=0.0001) - Most cases of trisomy 18 result from having three copies of chromosome 18 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
5. `medquad_ans_0003520-3` (score=0.0001) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development [medquad_ans_0000996-3]. This can occur due to nondisjunction during the formation of reproductive cells (eggs and sperm), resulting in a cell with an extra copy of chromosome 13 [medquad_ans_0000996-4]. In rare cases, trisomy 13 can result from a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or early fetal development [medquad_ans_0000996-3]. In these translocation cases, individuals have two normal copies of chromosome 13 plus an extra copy attached to another chromosome. Additionally, in rare instances, only part of chromosome 13 is

**Supported citations**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=1.000, groundedness=0.910, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.9866) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - G...
2. `medquad_ans_0004415-3` (score=0.8279) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
3. `medquad_ans_0003103-4` (score=0.0001) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
4. `medquad_ans_0004947-4` (score=0.0001) - What treatments are available for itching related to polycythemia vera? There are several treatments for the itching (pruritus) related to polycythemia vera (PV). No single treatment has been found to be effective for...
5. `medquad_ans_0001354-2` (score=0.0001) - What treatments for metastatic clear cell renal cell carcinoma are available in North America? There are several treatments for metastatic clear cell renal cell carcinoma available in North America. IL-2 and sunitinib...

**Answer**:

There is no specific treatment for Niemann-Pick disease type A. Supportive care from specialists such as a pulmonologist for respiratory problems, a cardiologist for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists may be helpful for managing symptoms [medquad_ans_0004415-3]. The provided context does not specify treatments for other types of Niemann-Pick disease beyond type A. For additional information, ongoing research efforts are available through the Research section of the web page [medquad_ans_0004415-3].

**Supported citations**: `medquad_ans_0004415-3`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.9994) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0000270-4` (score=0.0001) - Most cases of Dandy-Walker malformation are sporadic, which means they occur in people with no history of the disorder in their family. A small percentage of cases seem to run in families; however, Dandy-Walker malfor...
3. `medquad_ans_0002072-2` (score=0.0001) - What are the signs and symptoms of Ellis-Van Creveld syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Ellis-Van Creveld syndrome. If the information is available, the table ...
4. `medquad_ans_0000270-3` (score=0.0001) - Researchers have found mutations in a few genes that are thought to cause Dandy-Walker malformation, but these mutations account for only a small number of cases. Dandy-Walker malformation has also been associated wit...
5. `medquad_ans_0006103-1` (score=0.0001) - What are the signs and symptoms of Tibia absent polydactyly arachnoid cyst? The Human Phenotype Ontology provides the following list of signs and symptoms for Tibia absent polydactyly arachnoid cyst. If the informatio...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.893, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003789-3` (score=0.9941) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Man...
2. `medquad_ans_0000004_5-7` (score=0.0001) - Key Points - There are different types of treatment for patients with primary CNS lymphoma. - Three standard treatments are used: - Radiation therapy - Chemotherapy - Steroid therapy - New types of treatment are being...
3. `medquad_ans_0000004_7-1` (score=0.0001) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major ...
4. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0003789-1` (score=0.0001) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgk...

**Answer**:

Mantle cell lymphoma is caused by DNA damage within cells, which can result from a combination of factors. Many cases are associated with a chromosome translocation. While the exact cause is not fully known, it is considered an acquired disease rather than an inherited one [medquad_ans_0003789-3].

**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
3. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
5. `medquad_ans_0000549-1__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

The provided context does not contain information about Legg-Calv-Perthes disease or its inheritance. Therefore, it is not possible to determine whether Legg-Calv-Perthes disease is inherited based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.979, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000192-1__dup1` (score=0.9699) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage ...
2. `medquad_ans_0003965-1` (score=0.7330) - Metachromatic leukodystrophy is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially cells of the nervous system. This accumulation results in progressive destruction...
3. `medquad_ans_0000647-1` (score=0.2649) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
4. `medquad_ans_0003566-1` (score=0.1022) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
5. `medquad_ans_0000181-1__dup2` (score=0.0983) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...

**Answer**:

Metachromatic leukodystrophy (MLD) is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially in cells of the nervous system. This accumulation results in progressive destruction of the white matter of the brain, which consists of nerve fibers covered by myelin. Affected individuals experience progressive deterioration of intellectual functions and motor skills, such as the ability to walk, as well as loss of sensation in the extremities, incontinence, seizures, paralysis, and inability to speak [medquad_ans_0003965-1]. MLD is one of a group of genetic disorders called leukodystrophies, characterized by the toxic buildup of lipids and other storage materials in cells of the central nervous system and peripheral nerves, impairing the growth or development of the myelin sheath [medquad_ans_0000192-1__dup1]. It is caused by

**Supported citations**: `medquad_ans_0003965-1`, `medquad_ans_0000192-1__dup1`
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.9994) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0004109-2` (score=0.0001) - What are the signs and symptoms of Mondini dysplasia? Mondini dysplasia is a congenital malformation (present at birth). It may occur either unilaterally (in one ear) or bilaterally (in both ears). Most affected indiv...
4. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
5. `medquad_ans_0000390-1` (score=0.0001) - Frontonasal dysplasia is a condition that results from abnormal development of the head and face before birth. People with frontonasal dysplasia have at least two of the following features: widely spaced eyes (ocular ...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, and tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.527, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000251-1` (score=0.8085) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
2. `medquad_ans_0000251-2` (score=0.0005) - Cri-du-chat syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns. This condition is found in people of all ethnic backgrounds.
3. `medquad_ans_0000251-4` (score=0.0002) - Most cases of cri-du-chat syndrome are not inherited. The deletion occurs most often as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Affected people typicall...
4. `medquad_ans_0000251-3` (score=0.0001) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

Cri-du-chat syndrome is a chromosomal condition caused by a deletion of the end of the short (p) arm of chromosome 5, written as 5p-. It is characterized by a high-pitched cry that sounds like a cat's cry, intellectual disability, delayed development, small head size (microcephaly), low birth weight, weak muscle tone (hypotonia) in infancy, and distinctive facial features such as widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children are born with heart defects. The condition occurs in approximately 1 in 20,000 to 50,000 newborns and is found in people of all ethnic backgrounds. Most cases result from a random event during the formation of reproductive cells or early fetal development, with about 10% of cases inherited from an unaffected parent who carries a balanced translocation. Larger deletions

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.9992) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0000722-2__dup1` (score=0.0003) - What are the signs and symptoms of Bilateral perisylvian polymicrogyria? The signs and symptoms of bilateral perisylvian polymicrogyria (BPP) vary but may include: Partial paralysis of muscles on both sides of the fac...
3. `medquad_ans_0000718-1__dup1` (score=0.0001) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
4. `medquad_ans_0001483-2` (score=0.0001) - What are the signs and symptoms of Congenital bilateral absence of the vas deferens? The Human Phenotype Ontology provides the following list of signs and symptoms for Congenital bilateral absence of the vas deferens....
5. `medquad_ans_0000722-4__dup1` (score=0.0001) - Is bilateral perisylvian polymicrogyria inherited? In most cases, bilateral perisylvian polymicrogyria (BPP) occurs sporadically in people with no family history of the condition. Rarely, more than one family member m...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include: moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.973, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003414-3` (score=0.9965) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
2. `medquad_ans_0000674-3__dup1` (score=0.0003) - What causes Behr syndrome? The exact cause of Behr syndrome is not known; however, a genetic cause is suspected based on the families identified, thus far.
3. `medquad_ans_0003306-3` (score=0.0001) - What causes Jones syndrome? The exact, underlying genetic cause of Jones syndrome is not yet known.
4. `medquad_ans_0000765-3__dup1` (score=0.0001) - What causes blue rubber bleb nevus syndrome? Currently the cause of blue rubber bleb syndrome is not known.
5. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene, specifically a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein, written as Arg445His or R445H [medquad_ans_0003414-3]. It is unclear why the R445H mutation causes both hearing and vision loss in affected individuals.

**Supported citations**: `medquad_ans_0003414-3`
**Unsupported citations**: _none_

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004584-2` (score=0.9992) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0004589-2` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...
4. `medquad_ans_0004584-1` (score=0.0001) - Orofaciodigital syndrome 1 (OFD1), also called orofaciodigital syndrome type 1, is a condition that affects the development of the oral cavity (the mouth and teeth), facial features, and digits (fingers and toes). Thi...
5. `medquad_ans_0004592-1` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table ...

**Answer**:

The symptoms of Orofaciodigital syndrome 1 include oral features such as a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other facial abnormalities. Finger and toe abnormalities may include shortness (brachydactyly), webbing or joining (syndactyly), abnormally curved fingers (clinodactyly), or other abnormalities. Brain abnormalities such as cysts and kidney problems such as polycystic kidney disease may also occur. About half of individuals have some degree of learning disability, usually mild. Almost all individuals with OFD1 are female. Specific signs and symptoms with approximate frequencies include: bifid tongue (90%), broad alveolar ridges (90%), cleft

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=1.000, groundedness=0.898, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000130-1__dup3` (score=0.0026) - Summary : If you need a mastectomy, you have a choice about whether or not to have surgery to rebuild the shape of the breast. Instead of breast reconstruction, you could choose to wear a breast form that replaces the...
2. `medquad_ans_0000925-1__dup1` (score=0.0003) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
3. `medquad_ans_0000286-1__dup2` (score=0.0001) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
4. `medquad_ans_0002838-5` (score=0.0001) - How might hereditary multiple osteochondromas (HMO) be treated? Currently, there is no known medical treatment for HMO. Osteochondromas are not usually removed because they stop growing around age 12. Another consider...
5. `medquad_ans_0000219-7` (score=0.0001) - To find the cause of your ED, your doctor may - take your medical and sexual history - ask you questions about your mental health - give you a physical exam - test your blood - give you a nighttime erection test - per...

**Answer**:

Yes, information about surgery is available in the context. Specifically, the context discusses mastectomy and breast reconstruction as surgical options. It states that if a woman needs a mastectomy, she has a choice about whether or not to have surgery to rebuild the shape of the breast, which may be done at the same time as the mastectomy or later. Breast reconstruction can involve breast implants or using tissue taken from another part of the body, such as the lower abdomen, back, or buttocks. The type of reconstruction depends on factors like age, body type, and the type of cancer surgery. A plastic surgeon can help determine the best option [medquad_ans_0000130-1__dup3].

Additionally, the context notes that if radiation therapy is part of the treatment plan, a doctor may suggest waiting until after radiation therapy before proceeding with reconstruction [medquad_ans_0000130-1\_\_dup3

**Supported citations**: `medquad_ans_0000130-1__dup3`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.985, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0002935-1` (score=0.9831) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
2. `medquad_ans_0002934-1` (score=0.0011) - Human T-cell leukemia virus, type 1 (HTLV-1) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, some affected people may later de...
3. `medquad_ans_0000467-1__dup1` (score=0.0001) - Herpes is an infection that is caused by a herpes simplex virus (HSV). Oral herpes causes cold sores around the mouth or face. Genital herpes affects the genitals, buttocks or anal area. Genital herpes is a sexually t...
4. `medquad_ans_0000777-1` (score=0.0001) - PDGFRA-associated chronic eosinophilic leukemia is a form of blood cell cancer characterized by an elevated number of cells called eosinophils in the blood. These cells help fight infections by certain parasites and a...
5. `medquad_ans_0002935-4` (score=0.0001) - How is human T-cell leukemia virus, type 2 diagnosed? Human T-cell leukemia virus, type 2 (HTLV-2) is usually diagnosed based on blood tests that detect antibodies to the virus. However, HTLV-2 is often never suspecte...

**Answer**:

Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles. It can also be spread from mother to child during birth or breastfeeding. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic (show no symptoms) throughout life [medquad_ans_0002935-1].

**Supported citations**: `medquad_ans_0002935-1`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.975, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.5592) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
2. `medquad_ans_0001152-1` (score=0.0115) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD d...
3. `medquad_ans_0000758-1__dup2` (score=0.0098) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000038-1__dup2` (score=0.0032) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...
5. `medquad_ans_0000038-5__dup1` (score=0.0005) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...

**Answer**:

Interstitial lung diseases (ILDs) are a large group of diseases that inflame or scar the lungs. The inflammation and scarring, known as pulmonary fibrosis, make it difficult to get enough oxygen. These diseases can be caused by breathing in dust or other particles in the air, such as coal dust (black lung disease), farm dust (farmer's lung), asbestos fibers (asbestosis), iron from mines or welding fumes (siderosis), or silica dust (silicosis). Other causes include autoimmune diseases, occupational exposures to molds, gases, or fumes, and some types have no known cause. Treatment depends on the type of exposure and the stage of the disease and may involve medicines, oxygen therapy, or a lung transplant in severe cases [medquad_ans_0000522-1__dup1]. Additionally, children's interstitial and diffuse lung disease (chILD) is a group of rare lung

**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.117, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000527-4` (score=0.3595) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
2. `medquad_ans_0003718-4` (score=0.0005) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
3. `medquad_ans_0001084-3__dup1` (score=0.0003) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
4. `medquad_ans_0000562-4` (score=0.0002) - When Klippel-Feil syndrome is caused by mutations in the GDF6 or GDF3 genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder....
5. `medquad_ans_0000549-1__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

The provided context does not contain information about the inheritance of Melnick-Needles syndrome. Therefore, it is not possible to determine based on the given text whether Melnick-Needles syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006138-1` (score=0.9703) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
2. `medquad_ans_0006138-2` (score=0.0001) - What are the signs and symptoms of tracheobronchomalacia? Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway ...
3. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
4. `medquad_ans_0004384-2` (score=0.0001) - What are the signs and symptoms of Neuronal ceroid lipofuscinosis 10 ? The Human Phenotype Ontology provides the following list of signs and symptoms for Neuronal ceroid lipofuscinosis 10 . If the information is avail...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse [medquad_ans_0006138-1]. There are two forms of TBM: a congenital form (called primary TBM) that typically develops during infancy or early childhood and an acquired form (called secondary TBM) that is usually seen in adults [medquad_ans_0006138-1]. Some affected people may initially have no signs or symptoms, but the condition is typically progressive and most people will eventually develop characteristic features such as shortness of breath, cough, sputum retention (inability to clear mucus from the respiratory tract), and wheezing or stridor with breathing [medquad_ans_0006138-1

**Supported citations**: `medquad_ans_0006138-1`
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000531-1__dup2` (score=0.8110) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The t...
2. `medquad_ans_0006030-1` (score=0.4411) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
3. `medquad_ans_0000079-1__dup1` (score=0.0036) - While acute pain is a normal sensation triggered in the nervous system to alert you to possible injury and the need to take care of yourself, chronic pain is different. Chronic pain persists. Pain signals keep firing ...
4. `medquad_ans_0004770-1` (score=0.0017) - Patulous eustachian tube is a benign condition in which the eustachian tube stays open most of the time. The eustachian tube is the tube that runs between the middle ear and throat and regulates the ear pressure aroun...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.929, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000554-4` (score=0.9615) - Kallmann syndrome 1 (caused by ANOS1 gene mutations) has an X-linked recessive pattern of inheritance. The ANOS1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only on...
2. `medquad_ans_0003339-3` (score=0.9550) - How is Kallmann syndrome inherited? Kallmann syndrome (KS) may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner depending on the gene(s) responsible. For example: KS due to muta...
3. `medquad_ans_0001478-4` (score=0.6491) - Is congenital anosmia inherited? Most cases of isolated congenital anosmia (not associated with additional symptoms) occur sporadically in people with no family history of the condition. Rarely, more than one family m...
4. `medquad_ans_0003718-4` (score=0.0017) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
5. `medquad_ans_0003339-1` (score=0.0001) - Kallmann syndrome (KS) is a condition characterized primarily by hypogonadotropic hypogonadism (HH) and absent or diminished sense of smell (anosmia or hyposmia, respectively). HH is present from birth and is due to d...

**Answer**:

Yes, Kallmann syndrome can be inherited in different patterns depending on the genetic cause. It may be inherited in an X-linked recessive, autosomal dominant, or autosomal recessive manner [medquad_ans_0003339-3]. For example, Kallmann syndrome 1, caused by mutations in the ANOS1 (KAL1) gene, is inherited in an X-linked recessive pattern [medquad_ans_0000554-4]. Other forms, such as those caused by mutations in FGFR1, PROKR2, PROK2, CHD7, or FGF8 genes, are predominantly inherited in an autosomal dominant manner [medquad_ans_0003339-3]. In some cases, mutations in PROKR2 and PROK2 can also follow an autosomal recessive pattern [medquad_ans_000333

**Supported citations**: `medquad_ans_0003339-3`, `medquad_ans_0000554-4`
**Unsupported citations**: _none_

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.9994) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005970-1` (score=0.0001) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
3. `medquad_ans_0005967-1` (score=0.0001) - Syndactyly type 3 (SD3) is a limb abnormality present at birth that is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases only the soft tissue is fused, but in some cases the bone...
4. `medquad_ans_0005974-1` (score=0.0001) - Syndromic microphthalmia, type 3 is a rare condition that affects the eyes and other parts of the body. Babies with this condition are generally born without eyeballs (anophthalmia) or with eyes that are unusually sma...
5. `medquad_ans_0003952-1` (score=0.0001) - What are the signs and symptoms of Merlob Grunebaum Reisner syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Merlob Grunebaum Reisner syndrome. If the information is availab...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.959, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000232-1__dup2` (score=0.8652) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...
2. `medquad_ans_0000197-1__dup3` (score=0.1864) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
3. `medquad_ans_0000021-1__dup3` (score=0.0154) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Con...
4. `medquad_ans_0000008-1__dup3` (score=0.0084) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
5. `medquad_ans_0000242-1__dup3` (score=0.0079) - Coughing is a reflex that keeps your throat and airways clear. Although it can be annoying, coughing helps your body heal or protect itself. Coughs can be either acute or chronic. Acute coughs begin suddenly and usual...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to the lungs. It causes a cough that often brings up mucus and may also cause shortness of breath, wheezing, chest tightness, and a low fever. It is one type of COPD (chronic obstructive pulmonary disease). The inflamed bronchial tubes produce a lot of mucus, leading to persistent coughing and difficulty breathing. Cigarette smoking is the most common cause, and long-term exposure to air pollution, fumes, or dust can also contribute. Chronic bronchitis is a long-term condition that keeps coming back or never goes away completely. Treatment includes medicines to open the airways and help clear mucus, oxygen therapy if needed, and quitting smoking. [medquad_ans_0000197-1__dup3] [medquad_ans_00002

**Supported citations**: `medquad_ans_0000197-1__dup3`
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.750, groundedness=0.783, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000936-5` (score=0.9910) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscula...
2. `medquad_ans_0000935-5` (score=0.9884) - These resources address the diagnosis or management of spinal muscular atrophy: - Gene Review: Gene Review: Spinal Muscular Atrophy - Gene Review: Gene Review: Spinal Muscular Atrophy, X-Linked Infantile - Genetic Tes...
3. `medquad_ans_0000934-5` (score=0.0224) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - Med...
4. `medquad_ans_0000838-1__dup2` (score=0.0001) - Your spinal cord is a bundle of nerves that runs down the middle of your back. It carries signals back and forth between your body and your brain. It is protected by your vertebrae, which are the bone disks that make ...
5. `medquad_ans_0000840-1__dup2` (score=0.0001) - Spinal muscular atrophy (SMA) is a genetic disease that attacks nerve cells, called motor neurons, in the spinal cord. These cells communicate with your voluntary muscles - the ones you can control, like in your arms ...

**Answer**:

The provided context does not specify treatments for Spinal Muscular Atrophy. While it mentions resources related to diagnosis and management, such as diagnostic tests, drug therapy, genetic counseling, surgery and rehabilitation, and palliative care, it does not detail specific treatments for Spinal Muscular Atrophy [medquad_ans_0000935-5]. Therefore, based on the given information, the exact treatments are not available.

**Supported citations**: `medquad_ans_0000935-5`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.899, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000001_1-7` (score=0.9738) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
2. `medquad_ans_0000001_7-8` (score=0.0315) - Key Points - There are different types of treatment for children with acute myeloid leukemia (AML), chronic myelogenous leukemia (CML), juvenile myelomonocytic leukemia (JMML), or myelodysplastic syndromes (MDS). - Tr...
3. `medquad_ans_0000001_2-7` (score=0.0111) - Key Points - There are different types of treatment for patients with adult acute myeloid leukemia. - The treatment of adult AML usually has 2 phases. - Four types of standard treatment are used: - Chemotherapy - Radi...
4. `medquad_ans_0000001_6-7` (score=0.0069) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood ...
5. `medquad_ans_0000001_2-2` (score=0.0002) - Smoking, previous chemotherapy treatment, and exposure to radiation may affect the risk of adult AML. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean ...

**Answer**:

The treatments for Adult Acute Lymphoblastic Leukemia (ALL) include chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and biologic therapy. The treatment is typically divided into two phases: remission induction therapy, which aims to kill leukemia cells in the blood and bone marrow to achieve remission, and post-remission therapy, which targets any remaining leukemia cells to prevent relapse. Central nervous system (CNS) sanctuary therapy is also commonly given during each phase to address leukemia cells that may hide in the brain and spinal cord. Additionally, patients may consider participating in clinical trials, which are research studies testing new or improved treatments [medquad_ans_0000001_1-7].

**Supported citations**: `medquad_ans_0000001_1-7`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002439-3` (score=0.9899) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
2. `medquad_ans_0001042-3__dup1` (score=0.2452) - What causes cerebellar degeneration? Cerebellar degeneration can be caused by a variety of different conditions. Neurological diseases that can lead to cerebellar degeneration include: Acute and hemorrhagic stroke can...
3. `medquad_ans_0000388-3` (score=0.0001) - Mutations in the FXN gene cause Friedreich ataxia. This gene provides instructions for making a protein called frataxin. Although its role is not fully understood, frataxin appears to be important for the normal funct...
4. `medquad_ans_0000388-2` (score=0.0001) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
5. `medquad_ans_0000078-1` (score=0.0001) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. In people with Friedreich ataxia, this GAA segment is repeated 66 to more than 1,000 times, whereas normally it is repeated 5 to 33 times. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, severely reducing the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [medquad_ans_0002439-3].

**Supported citations**: `medquad_ans_0002439-3`
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.969, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000034-1__dup2` (score=0.9724) - Atrial fibrillation (AF) describes the rapid, irregular beating of the left atrium (upper chamber) of the heart. These rapid contractions of the heart are weaker than normal contractions, resulting in slow flow of blo...
2. `medquad_ans_0000013-4__dup2` (score=0.9705) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
3. `medquad_ans_0000341-1` (score=0.8466) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which cau...
4. `medquad_ans_0002230-1` (score=0.8236) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular respons...
5. `medquad_ans_0000341-2` (score=0.7779) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with...

**Answer**:

Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm, characterized by uncoordinated or erratic electrical activity in the heart's upper chambers (the atria). This leads to a fast and irregular heartbeat (arrhythmia) and can cause symptoms such as dizziness, chest pain, palpitations, shortness of breath, or fainting. If untreated, it increases the risk of stroke and sudden death. Complications can occur at any age, although some individuals may never experience health problems associated with the disorder [medquad_ans_0000341-1][medquad_ans_0002230-1].

**Supported citations**: `medquad_ans_0000341-1`, `medquad_ans_0002230-1`
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.8245) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0001637-1` (score=0.0001) - Autosomal dominant craniometaphyseal dysplasia is a genetic skeletal condition characterized by progressive thickening of bones in the skull (cranium) and abnormalities at the ends of long bones in the limbs (metaphys...
3. `medquad_ans_0002238-1` (score=0.0001) - What are the signs and symptoms of Familial congenital fourth cranial nerve palsy? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial congenital fourth cranial nerve palsy. If ...
4. `medquad_ans_0003378-1` (score=0.0001) - What are the signs and symptoms of Keratosis palmoplantaris striata 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Keratosis palmoplantaris striata 1. If the information is avail...
5. `medquad_ans_0000316-1__dup1` (score=0.0001) - What are the signs and symptoms of Amyotrophic lateral sclerosis type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis type 6. If the information is a...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia characterized by genetic conditions that affect bone development and growth. It is caused by mutations in the WTX gene and is inherited in an X-linked dominant manner. Features typically present at birth include skeletal abnormalities, particularly at the ends of long bones, sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected individuals may also experience developmental delay, hearing loss, heart defects, and ophthalmoplegia (paralysis of the muscles surrounding the eyes). Treatment is based on the specific signs and symptoms present in each individual [medquad_ans_0004627-1].

**Supported citations**: `medquad_ans_0004627-1`
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004097-2` (score=0.9992) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
2. `medquad_ans_0000230-1__dup2` (score=0.0001) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their...
3. `medquad_ans_0000141-1` (score=0.0001) - Buschke-Ollendorff syndrome is a hereditary disorder of connective tissues, which are tissues that provide strength and flexibility to structures throughout the body. Specifically, the condition is characterized by sk...
4. `medquad_ans_0000220-3` (score=0.0001) - Mutations in the FBN2 gene cause congenital contractural arachnodactyly. The FBN2 gene provides instructions for producing the fibrillin-2 protein. Fibrillin-2 binds to other proteins and molecules to form threadlike ...
5. `medquad_ans_0000034_1-1` (score=0.0001) - Key Points - Penile cancer is a disease in which malignant (cancer) cells form in the tissues of the penis. - Human papillomavirus infection may increase the risk of developing penile cancer. - Signs of penile cancer ...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints, inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), neurologic abnormalities (e

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=1.000, groundedness=0.207, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000004-23` (score=0.6805) - Currently, no treatment can stop Alzheimer's disease. However, four medications are used to treat its symptoms. These medicines may help maintain thinking, memory, and speaking skills for a limited time. They work by ...
2. `medquad_ans_0000004-32` (score=0.4979) - People with Alzheimer's disease, those with mild cognitive impairment, those with a family history of Alzheimers, and healthy people with no memory problems who want to help scientists test new treatments may be able ...
3. `medquad_ans_0000004-24` (score=0.1477) - Currently, no medicines or treatments are known to prevent Alzheimer's disease, but scientists are studying many possibilities. These possibilities include lifestyle factors such as exercise and physical activity, a h...
4. `medquad_ans_0000004-5__dup2` (score=0.0969) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate sta...
5. `medquad_ans_0000003-6__dup1` (score=0.0019) - Most people with Alzheimers disease are cared for at home by family members. Within families, caregiving is provided most often by wives and husbands, followed by daughters. As Alzheimers disease gets worse, the perso...

**Answer**:

The provided context does not contain information about treatments for Alzheimer's caregiving. Therefore, based on the given context, it is not possible to answer the question. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.900, groundedness=0.929, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000511-3` (score=0.9972) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
2. `medquad_ans_0000717-3` (score=0.0004) - Mutations in one of many genes can cause nemaline myopathy. These genes provide instructions for producing proteins that play important roles in skeletal muscles. Within skeletal muscle cells, these proteins are found...
3. `medquad_ans_0000547-3` (score=0.0003) - Juvenile idiopathic arthritis is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Its signs and symptoms res...
4. `medquad_ans_0000023-3` (score=0.0001) - The term "idiopathic" means that the cause of this condition is unknown. Adolescent idiopathic scoliosis probably results from a combination of genetic and environmental factors. Studies suggest that the abnormal spin...
5. `medquad_ans_0000512-3` (score=0.0001) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...

**Answer**:

The genetic changes related to idiopathic inflammatory myopathy include variations in several genes, most commonly within the human leukocyte antigen (HLA) complex. These HLA gene variations may influence the risk of developing the disorder, as the HLA complex helps the immune system distinguish the body's own proteins from foreign proteins. Specific variations of HLA genes have been associated with an increased risk of idiopathic inflammatory myopathy. Researchers are also studying variations in other immune-related genes to understand their contribution to the condition. However, the exact genetic mechanisms are not fully understood, and the term "idiopathic" indicates that the specific cause remains unknown [medquad_ans_0000511-3].

**Supported citations**: `medquad_ans_0000511-3`
**Unsupported citations**: _none_

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0005223-5` (score=0.9899) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
2. `medquad_ans_0000855-3` (score=0.0001) - The genes involved in pyruvate dehydrogenase deficiency each provide instructions for making a protein that is a component of a group of proteins called the pyruvate dehydrogenase complex. This complex plays an import...
3. `medquad_ans_0000855-2` (score=0.0001) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
4. `medquad_ans_0003451-1` (score=0.0001) - Lactate dehydrogenase deficiency is a condition that affects how the body breaks down sugar to use as energy in cells, primarily muscle cells. There are two types of lactate dehydrogenase deficiency: lactate dehydroge...
5. `medquad_ans_0000572-3` (score=0.0001) - Mutations in the LDHA gene cause lactate dehydrogenase-A deficiency, and mutations in the LDHB gene cause lactate dehydrogenase-B deficiency. These genes provide instructions for making the lactate dehydrogenase-A and...

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.974, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.9991) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
3. `medquad_ans_0006554-2` (score=0.0001) - What are the signs and symptoms of 15q11.2 microdeletion? The signs and symptoms in people with a 15q11.2 microdeletion can vary widely. Some people with the microdeletion don't have any apparent features, while other...
4. `medquad_ans_0006562-1` (score=0.0001) - 17q23.1q23.2 microdeletion syndrome is a condition caused by a small deletion of genetic material from chromosome 17. The deletion occurs at a location encompassing bands 23.1 to 23.2 on the long (q) arm of the chromo...
5. `medquad_ans_0000566-3` (score=0.0001) - Koolen-de Vries syndrome is caused by genetic changes that eliminate the function of one copy of the KANSL1 gene in each cell. Most affected individuals are missing a small amount of genetic material, including the KA...

**Answer**:

The symptoms of 17q23.1q23.2 microdeletion syndrome include developmental delay, microcephaly, short stature, heart defects, and hand, foot, and limb abnormalities. Specific signs and symptoms reported in the literature include:

- Developmental delay, particularly delays in speech [medquad_ans_0006562-2]
- Microcephaly (seen in 50% of patients) [medquad_ans_0006562-2]
- Short stature (seen in 50% of patients) [medquad_ans_0006562-2]
- Heart defects, including patent ductus arteriosus or atrial septal defects (each seen in 50% of patients) [medquad_ans_0006562-2]
- Arachnodactyly (long, thin fingers and toes)

**Supported citations**: `medquad_ans_0006562-2`
**Unsupported citations**: _none_

### Example 186 (`0003860-3`)

**Question**: What causes MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.941, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.9952) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0005347-3` (score=0.0079) - What causes Rett syndrome? Rett syndrome is typically caused by changes (mutations) in the MECP2 gene. This gene provides instructions for making a protein (MeCP2) needed for the development of the nervous system and ...
3. `medquad_ans_0000635-3` (score=0.0001) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
4. `medquad_ans_0000635-2` (score=0.0001) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...
5. `medquad_ans_0000635-4` (score=0.0001) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...

**Answer**:

MECP2 duplication syndrome is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome [medquad_ans_0003860-3]. The size of the duplication can vary, ranging from 100,000 to 900,000 base pairs (100 to 900 kilobases), but this variation does not affect the severity of the condition [medquad_ans_0000635-3]. The MECP2 gene encodes a protein essential for normal brain function, and its duplication leads to excess protein that cannot properly regulate the expression of other genes, resulting in irregular brain activity and the signs and symptoms of the syndrome [medquad_ans_0003860-3].

**Supported citations**: `medquad_ans_0003860-3`, `medquad_ans_0000635-3`
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.9994) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
2. `medquad_ans_0004635-2` (score=0.0002) - What are the signs and symptoms of Osteopetrosis autosomal recessive 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 2. If the information is ava...
3. `medquad_ans_0004632-2` (score=0.0001) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
4. `medquad_ans_0004632-1` (score=0.0001) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...
5. `medquad_ans_0004635-1` (score=0.0001) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...

**Answer**:

The symptoms of Osteopetrosis autosomal dominant type 2 include:

- Abnormality of epiphysis morphology (90%)
- Abnormality of the metacarpal bones (90%)
- Abnormality of the metaphyses (90%)
- Aseptic necrosis (90%)
- Bone pain (90%)
- Facial palsy (90%)
- Frontal bossing (90%)
- Joint dislocation (90%)
- Macrocephaly (90%)
- Osteoarthritis (90%)
- Osteomyelitis (90%)
- Recurrent fractures (90%)
- Short distal phalanx of finger (90%)
- Anemia (50%)
- Genu valgum (50%)
- Optic atrophy (50%)
- Short stature

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 188 (`0003459-4`)

**Question**: What are the treatments for Lambert Eaton myasthenic syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.992, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0003459-4` (score=0.9687) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...
2. `medquad_ans_0000277-2__dup2` (score=0.0692) - There is no cure for Zellweger syndrome, nor is there a standard course of treatment. Since the metabolic and neurological abnormalities that cause the symptoms of Zellweger syndrome are caused during fetal developmen...
3. `medquad_ans_0000235-5` (score=0.0063) - These resources address the diagnosis or management of congenital myasthenic syndrome: - Gene Review: Gene Review: Congenital Myasthenic Syndromes - Genetic Testing Registry: CHRNA1-Related Congenital Myasthenic Syndr...
4. `medquad_ans_0003103-4` (score=0.0001) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...
5. `medquad_ans_0004947-4` (score=0.0001) - What treatments are available for itching related to polycythemia vera? There are several treatments for the itching (pruritus) related to polycythemia vera (PV). No single treatment has been found to be effective for...

**Answer**:

Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis (where blood plasma is removed and replaced with fluid, protein, or donated plasma), IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. While there has been some evidence that either 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit (i.e., how much symptoms are improved) still needs to be determined [medquad_ans_0003459-4].

**Supported citations**: `medquad_ans_0003459-4`
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.9993) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
2. `medquad_ans_0002595-2` (score=0.0001) - What are the signs and symptoms of glucose-6-phosphate dehydrogenase (G6PD) deficiency? People with G6PD deficiency do not have signs of the disease unless their red blood cells are exposed to certain chemicals in foo...
3. `medquad_ans_0000065-3` (score=0.0001) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This ...
4. `medquad_ans_0000065-1` (score=0.0001) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
5. `medquad_ans_0000445-1__dup1` (score=0.0001) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...

**Answer**:

The symptoms of Arginase deficiency include aminoaciduria (90%), behavioral abnormality (90%), cognitive impairment (90%), neurological speech impairment (90%), EEG abnormality (50%), hemiplegia/hemiparesis (50%), hyperammonemia (50%), seizures (50%), anorexia, autosomal recessive inheritance, diaminoaciduria, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1].

**Supported citations**: `medquad_ans_0000445-2__dup1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.889, groundedness=0.981, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006376-6` (score=0.9676) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
2. `medquad_ans_0000139-5__dup1` (score=0.0074) - Treatment for von Willebrand disease (VWD) is based on the type of VWD you have and how severe it is. Most cases of VWD are mild, and you may need treatment only if you have surgery, tooth extraction, or an accident. ...
3. `medquad_ans_0001020-5` (score=0.0060) - These resources address the diagnosis or management of von Hippel-Lindau syndrome: - Brigham and Women's Hospital - Gene Review: Gene Review: Von Hippel-Lindau Syndrome - Genetic Testing Registry: Von Hippel-Lindau sy...
4. `medquad_ans_0000961-1__dup1` (score=0.0001) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancrea...
5. `medquad_ans_0001021-3` (score=0.0001) - Mutations in the VWF gene cause von Willebrand disease. The VWF gene provides instructions for making a blood clotting protein called von Willebrand factor, which is essential for the formation of blood clots. After a...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms but are still small to avoid permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6].

**Supported citations**: `medquad_ans_0006376-6`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.701, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004450-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
2. `medquad_ans_0004456-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004454-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004453-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004452-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not mention "Noonan syndrome 3" or any specific subtype of Noonan syndrome beyond general references to subtypes based on the responsible gene. Therefore, based on the available information, Noonan syndrome 3 cannot be identified or described.

Answer: Insufficient context to determine what Noonan syndrome 3 is. [medquad_ans_0004450-1]

**Supported citations**: `medquad_ans_0004450-1`
**Unsupported citations**: _none_

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.9991) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0002966-1` (score=0.0004) - What are the signs and symptoms of Hydrocephalus, costovertebral dysplasia, and Sprengel anomaly? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrocephalus, costovertebral dyspla...
3. `medquad_ans_0000419-1__dup1` (score=0.0001) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, hemivertebrae, neck muscle hypoplasia, rib segmentation abnormalities, cervical segmentation defects, spina bifida occulta, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2]. The deformity may also be associated with limb length discrepancy or other skeletal abnormalities [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.644

**Top docs**:

1. `medquad_ans_0000882-3` (score=0.9988) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...
2. `medquad_ans_0000543-3` (score=0.0004) - Joubert syndrome and related disorders can be caused by mutations in at least 10 genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopi...
3. `medquad_ans_0000869-3` (score=0.0003) - Restless legs syndrome likely results from a combination of genetic and environmental factors, many of which are unknown. Studies suggest that restless legs syndrome is related to a shortage (deficiency) of iron in ce...
4. `medquad_ans_0001041-3` (score=0.0002) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000667-3` (score=0.0001) - The causes of Moebius syndrome are unknown, although the condition probably results from a combination of environmental and genetic factors. Researchers are working to identify and describe specific genes related to t...

**Answer**:

Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene, which prevents cells from making any functional ROR2 protein, disrupting development before birth and leading to characteristic features of the syndrome. Autosomal dominant Robinow syndrome can be caused by mutations in the WNT5A or DVL1 gene, with the osteosclerotic form resulting from DVL1 gene mutations. The proteins from these genes are part of the same chemical signaling pathways as the ROR2 protein, and mutations alter the production or function of these proteins, impairing chemical signaling important for early development. In some cases, no mutation is identified in the ROR2, WNT5A, or DVL1 genes, and the cause remains unknown [medquad_ans_0000882-3].

**Supported citations**: `medquad_ans_0000882-3`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=1.000, groundedness=0.787, answer_relevance=0.727

**Top docs**:

1. `medquad_ans_0000058-9__dup1` (score=0.9918) - Your dietitian or doctor may suggest a special eating plan for you. You may have to avoid a diet high in protein, fat, sodium, and potassium. - Cut back on protein, especially animal products such as meat. Damaged kid...
2. `medquad_ans_0000179-12` (score=0.8733) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should tal...
3. `medquad_ans_0000142-4__dup2` (score=0.5006) - For children with CKD, learning about nutrition is vital because their diet can affect how well their kidneys work. Parents or guardians should always consult with their childs health care team before making any dieta...
4. `medquad_ans_0000165-6` (score=0.5006) - For children with CKD, learning about nutrition is vital because their diet can affect how well their kidneys work. Parents or guardians should always consult with their childs health care team before making any dieta...
5. `medquad_ans_0000158-13` (score=0.1524) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A person...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis" as a specific topic or resource. While the context mentions a health topic titled "Eat Right to Feel Right on Hemodialysis" in relation to healthy eating and kidney disease [medquad_ans_0000058-9__dup1], it does not define or describe what this specific resource is. Therefore, based on the available information, the content of "Kidney Failure: Eat Right to Feel Right on Hemodialysis" cannot be determined.

Answer: Insufficient context to determine what "Kidney Failure: Eat Right to Feel Right on Hemodialysis" is. [medquad_ans_0000058-9__dup1]

**Supported citations**: `medquad_ans_0000058-9__dup1`
**Unsupported citations**: _none_

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.889, groundedness=0.945, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.9994) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000455-3` (score=0.0001) - Mutations in the COL4A1 gene cause HANAC syndrome. The COL4A1 gene provides instructions for making one component of a protein called type IV collagen. Type IV collagen molecules attach to each other to form complex p...
3. `medquad_ans_0000366-3` (score=0.0001) - Mutations in the COL4A1 gene cause familial porencephaly. The COL4A1 gene provides instructions for making one component of a protein called type IV collagen. Type IV collagen molecules attach to each other to form co...
4. `medquad_ans_0000677-1` (score=0.0001) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild f...
5. `medquad_ans_0000199-3` (score=0.0001) - As the name suggests, mutations in the COL4A1 gene cause COL4A1-related brain small vessel disease. The COL4A1 gene provides instructions for making one component of a protein called type IV collagen. Type IV collagen...

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=0.750, groundedness=0.992, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.9368) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000159-6` (score=0.0019) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
3. `medquad_ans_0000017-10` (score=0.0002) - Most people with dry eye will not have serious problems, but severe dry eye may lead to inflammation, ulcers, or scars on the cornea, and some loss of vision. Permanent loss of vision from dry eye is uncommon.
4. `medquad_ans_0000063-19` (score=0.0001) - Skin cancer is caused by DNA damage, which can result from excess exposure to ultraviolet (UV) light. Having a previous occurrence of skin cancer or a close family member with the disease also puts you at risk for ski...
5. `medquad_ans_0001913-1` (score=0.0001) - Dowling-Degos disease is a skin condition characterized by a lacy or net-like (reticulate) pattern of abnormally dark skin coloring (hyperpigmentation), particularly in the body's folds and creases. Other features may...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - how big and deep your wound is, where it is, how long it takes to heal, your age, and your inherited tendency to scar. Scars usually fade over time but never go away completely [medquad_ans_0000795-1__dup1].

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.929, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0001045-3` (score=0.9992) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules call...
2. `medquad_ans_0001047-3` (score=0.0003) - X-linked chondrodysplasia punctata 1 is caused by genetic changes involving the ARSE gene. This gene provides instructions for making an enzyme called arylsulfatase E. The function of this enzyme is unknown, although ...
3. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
4. `medquad_ans_0001063-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0001062-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...

**Answer**:

Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. These mutations result in a deficiency of the adrenoleukodystrophy protein (ALDP), which is responsible for transporting very long-chain fatty acids (VLCFAs) into peroxisomes. The lack of this protein disrupts the transport and breakdown of VLCFAs, leading to their accumulation in the body. This accumulation is believed to be toxic to the adrenal cortex and myelin, and may trigger an inflammatory response in the brain, contributing to the breakdown of myelin and the signs and symptoms of the disorder [medquad_ans_0001045-3].

**Supported citations**: `medquad_ans_0001045-3`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.9981) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
2. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
3. `medquad_ans_0000988-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000988-1` (score=0.0001) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most fre...
5. `medquad_ans_0000027-1__dup3` (score=0.0001) - Amyloidosis occurs when abnormal proteins called amyloids build up and form deposits. The deposits can collect in organs such as the kidney and heart. This can cause the organs to become stiff and unable to work the w...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form is more common among people with African ancestry, affecting between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.825, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.9994) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0001795-2` (score=0.0001) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
3. `medquad_ans_0001754-1` (score=0.0001) - What are the signs and symptoms of Deafness, dystonia, and cerebral hypomyelination ? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness, dystonia, and cerebral hypomyelination...
4. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
5. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include those listed by the Human Phenotype Ontology, which are derived from data collected through Orphanet, a European rare disease database. The frequency of these symptoms is typically reported as a rough estimate of the percentage of patients who have the feature, often expressed as a fraction (e.g., 25/25 meaning all patients in a study had the symptom). However, the specific list of symptoms for Dens in dente and palatal invaginations is not provided in the given context [medquad_ans_0001780-1]. Therefore, the exact symptoms cannot be identified from the available information.

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.9993) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0004818-2` (score=0.0001) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discove...
3. `medquad_ans_0000782-1` (score=0.0001) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia mea...
4. `medquad_ans_0004818-3` (score=0.0001) - What are the recommended evaluations for patients diagnosed with periventricular nodular heterotopia? The following evaluations are recommended:[1823] Imaging exams of the brain to establish the diagnosis Evaluation b...
5. `medquad_ans_0002523-1` (score=0.0001) - What are the signs and symptoms of Genitopatellar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Genitopatellar syndrome. If the information is available, the table below ...

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2].

**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.727, groundedness=0.725, answer_relevance=0.727

**Top docs**:

1. `medquad_ans_0000040_1-6` (score=0.1000) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000006_3-8` (score=0.0382) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000019_2-7` (score=0.0372) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000004_6-7` (score=0.0279) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000024_2-8` (score=0.0114) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Landau-Kleffner Syndrome. [medquad_ans_0000040_1-6][medquad_ans_0000006_3-8]

**Supported citations**: `medquad_ans_0000040_1-6`, `medquad_ans_0000006_3-8`
**Unsupported citations**: _none_

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004290-1` (score=0.1678) - Myotonic dystrophy is an inherited condition that affects the muscles and other body systems. It is the most common form of muscular dystrophy that begins in adulthood, usually in a person's 20s or 30s. This condition...
2. `medquad_ans_0000306-1` (score=0.1412) - Muscular dystrophies are a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy). The Duchenne and Becker types of muscular dystrophy are two related conditions that primarily ...
3. `medquad_ans_0001935-1` (score=0.0195) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
4. `medquad_ans_0002213-1` (score=0.0009) - Facioscapulohumeral muscular dystrophy is a disorder characterized by muscle weakness and wasting (atrophy). This condition gets its name from the areas of the body that are affected most often: muscles in the face (f...
5. `medquad_ans_0003605-1` (score=0.0003) - What are the signs and symptoms of Limb-girdle muscular dystrophy type 2E? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy type 2E. If the information ...

**Answer**:

Muscular dystrophy is a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy) [medquad_ans_0000306-1]. These conditions affect skeletal muscles used for movement and, in some cases, cardiac muscle [medquad_ans_0000306-1]. Examples include Duchenne and Becker muscular dystrophy, which are caused by different mutations in the same gene and primarily affect males [medquad_ans_0000306-1]. Both types are associated with cardiomyopathy, a heart condition that weakens the cardiac muscle and can progress to dilated cardiomyopathy, leading to symptoms such as shortness of breath, fatigue, arrhythmia, and leg swelling [medquad_ans_0000306-1].

**Supported citations**: `medquad_ans_0000306-1`
**Unsupported citations**: _none_

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.8062) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0005020-1` (score=0.0002) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
3. `medquad_ans_0005022-1` (score=0.0001) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...
4. `medquad_ans_0002679-2` (score=0.0001) - What are the signs and symptoms of Greig cephalopolysyndactyly syndrome? The symptoms of Greig cephalopolysyndactyly syndrome (GCPS) are highly variable, ranging from mild to severe. People with this condition typical...
5. `medquad_ans_0000549-1__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.707, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002519-3` (score=0.0001) - How is hereditary geniospasm inherited? Hereditary geniospasm is inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene in each body cell is sufficient to cause s...
2. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
3. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
4. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0003571-1` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...

**Answer**:

The provided context does not include information about the symptoms of geniospasm. Therefore, based on the given text, it is not possible to answer the question. [medquad_ans_0002519-3] discusses the inheritance pattern of hereditary geniospasm but does not list its symptoms.

**Supported citations**: `medquad_ans_0002519-3`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.988, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000512-3` (score=0.9928) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
2. `medquad_ans_0000023-3` (score=0.0002) - The term "idiopathic" means that the cause of this condition is unknown. Adolescent idiopathic scoliosis probably results from a combination of genetic and environmental factors. Studies suggest that the abnormal spin...
3. `medquad_ans_0000078-4__dup2` (score=0.0001) - Sometimes doctors can find out what is causing pulmonary fibrosis (lung scarring). For example, exposure to environmental pollutants and certain medicines can cause the disease. Environmental pollutants include inorga...
4. `medquad_ans_0000511-3` (score=0.0001) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
5. `medquad_ans_0000512-2` (score=0.0001) - Idiopathic pulmonary fibrosis has an estimated prevalence of 13 to 20 per 100,000 people worldwide. About 100,000 people are affected in the United States, and 30,000 to 40,000 new cases are diagnosed each year. Famil...

**Answer**:

Genetic changes related to idiopathic pulmonary fibrosis include mutations in the TERC and TERT genes, which have been found in about 15 percent of all cases of familial pulmonary fibrosis and a smaller percentage of cases of sporadic idiopathic pulmonary fibrosis. These genes provide instructions for making components of an enzyme called telomerase, which maintains telomeres at the ends of chromosomes. [medquad_ans_0000512-3]

**Supported citations**: `medquad_ans_0000512-3`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000813-4` (score=0.9934) - Potocki-Shaffer syndrome follows an autosomal dominant inheritance pattern, which means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. In some cases, an affected per...
2. `medquad_ans_0000527-4` (score=0.7178) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
3. `medquad_ans_0001084-3__dup1` (score=0.0072) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
4. `medquad_ans_0000562-4` (score=0.0001) - When Klippel-Feil syndrome is caused by mutations in the GDF6 or GDF3 genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder....
5. `medquad_ans_0003718-4` (score=0.0001) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...

**Answer**:

Yes, Potocki-Shaffer syndrome is inherited in an autosomal dominant pattern, which means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. [medquad_ans_0000813-4]

**Supported citations**: `medquad_ans_0000813-4`
**Unsupported citations**: _none_

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=1.000, groundedness=0.137, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000925-1__dup1` (score=0.9907) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
2. `medquad_ans_0000286-1__dup2` (score=0.0002) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
3. `medquad_ans_0000822-5` (score=0.0001) - These resources address the diagnosis or management of PMAH: - Eunice Kennedy Shriver National Institute of Child Health and Human Development: How Do Health Care Providers Diagnose Adrenal Gland Disorders? - Eunice K...
4. `medquad_ans_0000613-1__dup2` (score=0.0001) - A miscarriage is the loss of pregnancy from natural causes before the 20th week of pregnancy. Most miscarriages occur very early in the pregnancy, often before a woman even knows she is pregnant. There are many differ...
5. `medquad_ans_0000223-12` (score=0.0001) - - A urinary tract infection (UTI) is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. - Most UTIs are caused by bacteria that live in the bowel, ...

**Answer**:

No, there is no information about child care in the provided context. [medquad_ans_0000925-1__dup1, medquad_ans_0000286-1__dup2, medquad_ans_0000822-5, medquad_ans_0000613-1__dup2] discuss topics such as infant health, dietary supplements, adrenal gland disorders, and miscarriage, but none address child care.

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000925-1__dup1, medquad_ans_0000286-1__dup2, medquad_ans_0000822-5, medquad_ans_0000613-1__dup2`

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.800, groundedness=0.954, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.9995) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000016-2__dup1` (score=0.0001) - Drug therapy including verapamil may help to reduce the severity and duration of attacks of paralysis associated with the more serious form of alternating hemiplegia
3. `medquad_ans_0000016-3__dup1` (score=0.0001) - Children with the benign form of alternating hemiplegia have a good prognosis. Those who experience the more severe form have a poor prognosis because intellectual and mental capacities do not respond to drug therapy,...
4. `medquad_ans_0000046-4` (score=0.0001) - Alternating hemiplegia of childhood is considered an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of alternating hemiplegia of chi...
5. `medquad_ans_0000016-1__dup1` (score=0.0001) - Alternating hemiplegia is a rare neurological disorder that develops in childhood, most often before the child is 18 months old. The disorder is characterized by recurrent episodes of paralysis that involve one or bot...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=0.833, groundedness=0.955, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000159-3__dup2` (score=0.9977) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
2. `medquad_ans_0000159-6` (score=0.0062) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
3. `medquad_ans_0000569-1__dup2` (score=0.0002) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus eryt...
4. `medquad_ans_0000186-3__dup1` (score=0.0001) - The prognosis for lupus varies widely depending on the organs involved and the intensity of the inflammatory reaction. The course of lupus is commonly chronic and relapsing, often with long periods of remission. Most ...
5. `medquad_ans_0000186-1__dup1` (score=0.0001) - Lupus (also called systemic lupus erythematosus) is a disorder of the immune system. Normally, the immune system protects the body against invading infections and cancers. In lupus, the immune system is over-active an...

**Answer**:

The symptoms of lupus include joint pain or swelling, muscle pain, fever with no known cause, fatigue, red rashes often on the face (also called the "butterfly rash"), and swelling in the legs, feet, or ankles [medquad_ans_0000159-3__dup2]. Additionally, lupus can cause skin issues such as discoid lupus (a rash that doesn't go away) and subacute cutaneous lupus (sores after sun exposure) [medquad_ans_0000569-1__dup2]. The immune system in lupus attacks healthy cells and tissues, which can damage joints, skin, blood vessels, and organs [medquad_ans_0000569-1__dup2].

**Supported citations**: `medquad_ans_0000159-3__dup2`, `medquad_ans_0000569-1__dup2`
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.531, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-1` (score=0.2765) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
2. `medquad_ans_0001397-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
3. `medquad_ans_0001395-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001398-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
5. `medquad_ans_0000073-1__dup2` (score=0.0065) - Cerebro-oculo-facio-skeletal syndrome (COFS) is a pediatric, genetic, degenerative disorder that involves the brain and the spinal cord. It is characterized by craniofacial and skeletal abnormalities, severely reduced...

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging (progeria). It is associated with severe photosensitivity, failure to thrive, microcephaly, and impaired development of the nervous system. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain visible on brain scans. The syndrome is divided into three subtypes: classical (type I or type A), which has an onset in early childhood (usually after age 1 year) and is sometimes called "moderate" or "classic" Cockayne syndrome; type II (type B), which is more severe and presents at birth with growth and developmental abnormalities, also known as cerebro-oculo-facio-skeletal (COFS) syndrome or Pena-Shokeir syndrome type II; and type III (type C), which is the mildest form and appears later in childhood. Cockayne

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.833, groundedness=0.929, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000413-1__dup1` (score=0.9751) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the t...
2. `medquad_ans_0000029-6__dup1` (score=0.9748) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...
3. `medquad_ans_0000029-8__dup1` (score=0.9484) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regu...
4. `medquad_ans_0000029-7__dup1` (score=0.2683) - Gum disease is caused by dental plaque -- a sticky film of bacteria that builds up on teeth. Regular brushing and flossing help get rid of plaque. But plaque that is not removed can harden and form tartar that brushin...
5. `medquad_ans_0000059-3__dup3` (score=0.0161) - The following chart shows the most common mouth problems from diabetes. Problem What It Is Symptoms Treatment gingivitis - unhealthy or inflamed gums - red, swollen, and bleeding gums - daily brushing and flossing - r...

**Answer**:

Gum (periodontal) disease is an infection of the tissues that hold your teeth in place. It ranges from a mild form called gingivitis, characterized by red, swollen, and bleeding gums, to a more severe form called periodontitis, where the gums pull away from the teeth and form infected pockets. If left untreated, periodontitis can lead to the destruction of the bones, gums, and connective tissue that support the teeth, potentially resulting in tooth loss [medquad_ans_0000029-6__dup1]. Gingivitis is a mild form of gum disease that does not involve loss of bone or tissue and can usually be reversed with daily brushing and flossing and regular cleanings by a dentist or dental hygienist [medquad_ans_0000029-8__dup1]. The disease is caused by dental plaque, a sticky film of bacteria that builds up

**Supported citations**: `medquad_ans_0000029-6__dup1`, `medquad_ans_0000029-8__dup1`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000494-2` (score=0.9992) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
2. `medquad_ans_0003059-2` (score=0.0001) - What are the signs and symptoms of Hypokalemic periodic paralysis? Hypokalemic periodic paralysis involves attacks of muscle weakness or loss of muscle movement (paralysis) that come and go. The weakness or paralysis ...
3. `medquad_ans_0003059-4` (score=0.0001) - How is hypokalemic periodic paralysis inherited? This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, ...
4. `medquad_ans_0003059-1` (score=0.0001) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...
5. `medquad_ans_0003000-1` (score=0.0001) - Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move...

**Answer**:

An estimated 1 in 200,000 people are affected by hyperkalemic periodic paralysis [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.8399) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
3. `medquad_ans_0002946-1` (score=0.0001) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0001388-1` (score=0.0001) - What are the signs and symptoms of COASY Protein-Associated Neurodegeneration ? The Human Phenotype Ontology provides the following list of signs and symptoms for COASY Protein-Associated Neurodegeneration . If the in...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles, platyspondyly (flattening of the spine bones), and cortical thinning. Cranial involvement is minimal, with some individuals showing mild hyperostosis of the skull base and thickening of the frontal and occipital bones. Pyle disease is inherited in an autosomal recessive manner [medquad_ans_0005212-1].

**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.800, groundedness=0.935, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000108-3__dup2` (score=0.0331) - Major Causes Pulmonary embolism (PE) usually begins as a blood clot in a deep vein of the leg. This condition is called deep vein thrombosis. The clot can break free, travel through the bloodstream to the lungs, and b...
2. `medquad_ans_0000108-4__dup2` (score=0.0002) - Pulmonary embolism (PE) occurs equally in men and women. The risk increases with age. For every 10 years after age 60, the risk of having PE doubles. Certain inherited conditions, such as factor V Leiden, increase the...
3. `medquad_ans_0000104-1__dup4` (score=0.0001) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
4. `medquad_ans_0000108-8` (score=0.0001) - Preventing pulmonary embolism (PE) begins with preventing deep vein thrombosis (DVT). Knowing whether you're at risk for DVT and taking steps to lower your risk are important. Exercise your lower leg muscles if you're...
5. `medquad_ans_0000456-1` (score=0.0001) - Hereditary antithrombin deficiency is a disorder of blood clotting. People with this condition are at higher than average risk for developing abnormal blood clots, particularly a type of clot that occurs in the deep v...

**Answer**:

Deep vein thrombosis (DVT) is caused by blood flow being restricted and slowing down in the deep veins of the legs. This can happen if a person does not move around for long periods, such as after some types of surgery, during a long trip in a car or airplane, or if they must stay in bed for an extended time. Blood clots are more likely to develop in veins that are damaged from surgery or injured in other ways [medquad_ans_0000108-3__dup2]. Additionally, other risk factors include having certain diseases or conditions such as stroke, paralysis, chronic heart disease, or high blood pressure, as well as being overweight or obese, sitting for long periods, pregnancy, and the 6-week period after pregnancy [medquad_ans_0000108-4__dup2].

**Supported citations**: `medquad_ans_0000108-3__dup2`, `medquad_ans_0000108-4__dup2`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=0.714, groundedness=0.000, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000665-1` (score=0.0088) - Mitochondrial trifunctional protein deficiency is a rare condition that prevents the body from converting certain fats to energy, particularly during periods without food (fasting). Signs and symptoms of mitochondrial...
2. `medquad_ans_0004079-1` (score=0.0011) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...
3. `medquad_ans_0000661-1` (score=0.0010) - Mitochondrial complex III deficiency is a genetic condition that can affect several parts of the body, including the brain, kidneys, liver, heart, and the muscles used for movement (skeletal muscles). Signs and sympto...
4. `medquad_ans_0000665-2` (score=0.0004) - Mitochondrial trifunctional protein deficiency is a rare disorder; its incidence is unknown.
5. `medquad_ans_0000661-2` (score=0.0001) - The prevalence of mitochondrial complex III deficiency is unknown, although the condition is thought to be rare.

**Answer**:

The provided context does not contain any information about Mitochondrial complex IV deficiency. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=0.857, groundedness=0.522, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.4373) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0000026-1__dup1` (score=0.0015) - Apraxia (called "dyspraxia" if mild) is a neurological disorder characterized by loss of the ability to execute or carry out skilled movements and gestures, despite having the desire and the physical ability to perfor...
3. `medquad_ans_0000495-1__dup1` (score=0.0007) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
4. `medquad_ans_0000026-2__dup1` (score=0.0003) - Generally, treatment for individuals with apraxia includes physical, speech,or occupational therapy. If apraxia is a symptom of another disorder, the underlying disorder should be treated.
5. `medquad_ans_0000078-2` (score=0.0002) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, making it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4; in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Type 1 individuals often develop

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=0.909, groundedness=0.543, answer_relevance=0.909

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.9399) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0001496-1` (score=0.0002) - Congenital deafness with vitiligo and achalasia is a syndrome characterized by deafness present from birth (congenital), associated with short stature, vitiligo, muscle wasting and achalasia (swallowing difficulties)....
3. `medquad_ans_0001728-1` (score=0.0001) - Deafness and myopia syndrome is rare condition that affects both hearing and vision. Beginning at birth or in early infancy, people with this condition have moderate to profound hearing loss in both ears that generall...
4. `medquad_ans_0000897-4` (score=0.0001) - Sensorineural deafness and male infertility is inherited in an autosomal recessive pattern, which means both copies of chromosome 15 in each cell have a deletion. The parents of an individual with sensorineural deafne...
5. `medquad_ans_0001549-1` (score=0.0001) - Congenital primary aphakia (CPA) is a rare eye condition that is present at birth in which the lens is missing. In some cases, CPA can be associated with other eye abnormalities including microphthalmia, absence of th...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). Rarely, affected individuals have some underdeveloped inner ear structures in one or both ears. The abnormalities of the inner ear cause a form of hearing loss called sensorineural deafness that is present from birth (congenital). Because the inner ear is important for balance as well as hearing, development of motor skills, such as sitting and crawling, may be delayed in affected infants. In addition, people with LAMM syndrome often have abnormally small outer ears (microtia) with narrow ear canals. They can also have unusually small, widely spaced teeth (microdontia) [medquad_ans_000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000064-4` (score=0.9898) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in ...
2. `medquad_ans_0000527-4` (score=0.9783) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
3. `medquad_ans_0001553-4` (score=0.0773) - How is congenital radio-ulnar synostosis inherited? Congenital radio-ulnar synostosis appears to be inherited in an autosomal dominant manner in some cases. This means that one mutated copy of the disease-causing gene...
4. `medquad_ans_0001084-3__dup1` (score=0.0015) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
5. `medquad_ans_0003718-4` (score=0.0006) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...

**Answer**:

Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases result from new mutations and occur in individuals with no family history of the disorder, although affected individuals can pass the condition to their offspring [medquad_ans_0000064-4].

**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: _none_

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.1319) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-3__dup2` (score=0.0073) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
3. `medquad_ans_0000341-1__dup3` (score=0.0047) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
4. `medquad_ans_0000359-1__dup1` (score=0.0039) - A fever is a body temperature that is higher than normal. It is not an illness. It is part of your body's defense against infection. Most bacteria and viruses that cause infections do well at the body's normal tempera...
5. `medquad_ans_0001304-1` (score=0.0037) - Chromosomes are the structures found in every cell of the body that contain our DNA, the instructions that tell our body what to do. Humans have 23 pairs of chromosomes, which means that each human cell contains 46 ch...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air that contains airborne barnyard dust contaminated by dried placental material, birth fluids, and excreta of infected animals. Other modes of transmission, including tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission, are rare. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, people with acute Q fever develop a chronic form of the disease. Treatment of the acute form is with antibiotics,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.936, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000038-5__dup1` (score=0.9936) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
2. `medquad_ans_0000038-9` (score=0.0145) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...
3. `medquad_ans_0000043-17` (score=0.0127) - If you smoke cigarettes, you are at much higher risk for lung cancer than a person who has never smoked. The risk of dying from lung cancer is 23 times higher for men who smoke and 13 times higher for women who smoke ...
4. `medquad_ans_0004072-4` (score=0.0024) - Is genetic testing available for Milroy disease? Yes. The Genetic Testing Registry (GTR) provides information about the labs that offer genetic testing for Milroy disease. The intended audience for the GTR is health c...
5. `medquad_ans_0000032_2-2` (score=0.0011) - Smoking is the major risk factor for small cell lung cancer. Anything that increases your chance of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having ri...

**Answer**:

Children who have a family history of interstitial lung disease or childhood interstitial lung disease (chILD), those with an inherited surfactant disorder or a family history of such disorders, children with problems involving aspiration (inhaling food, liquid, or vomit into the lungs), children with immune system disorders, children exposed to environmental lung irritants such as molds and chemicals, children with systemic or autoimmune diseases (e.g., collagen vascular disease or inflammatory bowel disease), children undergoing cancer treatments like radiation or chemotherapy, and children who have had a bone marrow or lung transplant are at increased risk for childhood interstitial lung disease [medquad_ans_0000038-5__dup1]. Additionally, certain types of chILD are more common in infants and young children, while others can occur in children of any age [medquad_ans_0000038-5__dup1].

**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=1.000, groundedness=0.949, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.9011) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0003405-1` (score=0.0039) - Kluver Bucy syndrome is a rare behavioral impairment characterized by inappropriate sexual behaviors and mouthing of objects. Other signs and symptoms, include a diminished ability to visually recognize objects, loss ...
3. `medquad_ans_0000441-3` (score=0.0010) - GRN-related frontotemporal dementia results from mutations in the GRN gene. This gene provides instructions for making a protein called granulin (also known as progranulin). Granulin is active in many different tissue...
4. `medquad_ans_0000391-3` (score=0.0009) - FTDP-17 is caused by mutations in the MAPT gene. This gene is located on chromosome 17, which is how the disease got its name. The MAPT gene provides instructions for making a protein called tau. This protein is found...
5. `medquad_ans_0006001-1` (score=0.0006) - Takayasu arteritis is a condition that causes inflammation of the main blood vessel that carries blood from the heart to the rest of the body (aorta) and its associated branched blood vessels. As a result of the infla...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, located at the lower sides of the skull beneath the temple. Affected individuals often have no signs or symptoms other than a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically without a family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.506, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.9565) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
2. `medquad_ans_0001082-1` (score=0.9285) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0001082-2` (score=0.0004) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
4. `medquad_ans_0000855-2` (score=0.0003) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
5. `medquad_ans_0000793-1` (score=0.0001) - Phosphoglycerate dehydrogenase deficiency is a condition characterized by an unusually small head size (microcephaly); impaired development of physical reactions, movements, and speech (psychomotor retardation); and r...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This deficiency leads to the buildup of isoleucine and abnormal levels of organic acids in the blood, urine, and tissues, which can be toxic and cause a range of health problems. Symptoms may include poor feeding, lack of energy (lethargy), vomiting, irritability, and in severe cases, difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills such as standing and walking. Symptoms can be triggered by fasting, infections, or high-protein diets. The disorder is more common among Hmong populations in southeast Asia and Hmong Americans, with an estimated incidence of 1 in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005840-2` (score=0.9994) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information ...
2. `medquad_ans_0005840-3` (score=0.0001) - How is spondyloepiphyseal dysplasia congenita inherited? Spondyloepiphyseal dysplasia (SEDC) is typically inherited in an autosomal dominant manner. This means that one altered (mutated) gene in each cell is sufficien...
3. `medquad_ans_0000947-1` (score=0.0001) - Spondyloepiphyseal dysplasia congenita is an inherited bone growth disorder that results in short stature (dwarfism), skeletal abnormalities, and problems with vision and hearing. This condition affects the bones of t...
4. `medquad_ans_0005840-1` (score=0.0001) - Spondyloepiphyseal dysplasia congenita is an inherited disorder of bone growth that affects the bones of the spine and ends of the long bones in the arms and legs. Features of this condition include short stature (dwa...
5. `medquad_ans_0000947-3` (score=0.0001) - Spondyloepiphyseal dysplasia congenita is one of a spectrum of skeletal disorders caused by mutations in the COL2A1 gene. This gene provides instructions for making a protein that forms type II collagen. This type of ...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies (90%), abnormality of epiphysis morphology (90%), micromelia (90%), narrow chest (90%), short neck (90%), short stature (90%), short thorax (90%), skeletal dysplasia (90%), broad forehead (50%), cleft palate (50%), hyperlordosis (50%), hypertelorism (50%), malar flattening (50%), osteoarthritis (50%), talipes (50%), cataract (7.5%), glaucoma (7.5%), hearing impairment (7.5%), kyphosis (7.5%), myopia (7.5%), nystagmus (7.5%), retinal detachment (7.5%), and scoliosis (7.5%)

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005219-3` (score=0.9912) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...
2. `medquad_ans_0000179-4__dup1` (score=0.0011) - The NINDS conducts and supports a broad program of basic and clinical research on epilepsy including Lennox-Gastaut syndrome. These studies are aimed at finding the causes of these disorders, improving the diagnosis, ...
3. `medquad_ans_0000853-2` (score=0.0001) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
4. `medquad_ans_0000004-4__dup1` (score=0.0001) - The NINDS supports broad and varied programs of research on epilepsy and developmental disorders. This research is aimed at discovering new ways to prevent, diagnose, and treat epilepsy and developmental disorders and...
5. `medquad_ans_0000818-5` (score=0.0001) - These resources address the diagnosis or management of PRICKLE1-related progressive myoclonus epilepsy with ataxia: - Gene Review: Gene Review: PRICKLE1-Related Progressive Myoclonus Epilepsy with Ataxia - Genetic Tes...

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The ALDH7A1 gene provides instructions for making an enzyme called α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the protein building block (amino acid) lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. Pyridoxine plays a role in many processes in the body, such as the breakdown of amino acids and the production of chemicals that transmit signals in the brain (neurotransmitters). It is unclear how a lack of pyridoxine causes the seizures that are characteristic of this condition. Some individuals with pyridox

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=1.000, groundedness=0.940, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.9949) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
2. `medquad_ans_0000004_6-5` (score=0.0027) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...
3. `medquad_ans_0000001_6-6` (score=0.0023) - Key Points Past treatment for cancer and certain genetic conditions affect the risk of having childhood ALL. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does no...
4. `medquad_ans_0000003_5-2` (score=0.0006) - Having certain diseases and inherited disorders can increase the risk of childhood soft tissue sarcoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mea...
5. `medquad_ans_0000004_1-5` (score=0.0003) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The patient's signs and symptoms. - The stage of the can...

**Answer**:

Individuals at risk for childhood Hodgkin lymphoma include those who are infected with the Epstein-Barr virus, infected with the human immunodeficiency virus (HIV), have certain diseases of the immune system, have a personal history of mononucleosis ("mono"), or have a parent or sibling with a personal history of Hodgkin lymphoma [medquad_ans_0000004_6-2]. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma due to its effect on the immune system [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.8953) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0001795-1` (score=0.7929) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0000088-11` (score=0.0098) - Most people with celiac disease have a significant improvement in symptoms when they follow a gluten-free diet. Health care providers typically refer people to a dietitian who specializes in treating people with the d...
4. `medquad_ans_0000768-1__dup1` (score=0.0079) - A rash is an area of irritated or swollen skin. Many rashes are itchy, red, painful, and irritated. Some rashes can also lead to blisters or patches of raw skin. Rashes are a symptom of many different medical problems...
5. `medquad_ans_0000084-2` (score=0.0050) - Atopic dermatitis is a common disorder that affects 10 to 20 percent of children and 5 to 10 percent of adults.

**Answer**:

Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These lesions are more common on the knees, elbows, buttocks, and shoulder blades. The condition is often associated with the inability to digest gluten and may present with symptoms such as fluid-filled sores, red lesions resembling hives, and itchiness, redness, and burning. A burning sensation may precede lesion formation, and the lesions are usually scratched off by the time a patient presents for a physical exam, appearing as erosions and excoriations. It is also associated with celiac disease, and patients may experience dental enamel defects in permanent teeth. Less than 20 percent of people with DH have symptoms of celiac disease [medquad_ans_0001795-1]. The disease is typically treated with the drug dapsone [medquad_ans_000

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.9926) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
2. `medquad_ans_0000040-2__dup1` (score=0.9899) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...
3. `medquad_ans_0000040-3__dup1` (score=0.9090) - Most individuals with myotonia congenita lead long, productive lives. Although muscle stiffness may interfere with walking, grasping, chewing, and swallowing, it is usually relieved with exercise.
4. `medquad_ans_0000040-4__dup1` (score=0.0001) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to myotonia congenita and also supports additional research through grants to major research institutions across the countr...
5. `medquad_ans_0003103-4` (score=0.0001) - What treatment is available for Tay syndrome? Treatments for Tay syndrome are symptomatic. There is no cure for ichthyosis, only treatments to help manage symptoms. The main treatment for ichthyosis is to hydrate (moi...

**Answer**:

Most people with myotonia congenita do not require special treatments; stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some success with medications such as quinine or anticonvulsant drugs such as phenytoin. Physical therapy and other rehabilitative therapies are also sometimes used to improve muscle function [medquad_ans_0000040-2__dup1].

**Supported citations**: `medquad_ans_0000040-2__dup1`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=1.000, groundedness=0.485, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000133-13` (score=0.8860) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
2. `medquad_ans_0000133-11` (score=0.8679) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the i...
3. `medquad_ans_0000133-12` (score=0.0097) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
4. `medquad_ans_0000276-4__dup1` (score=0.0004) - The National Institute of Neurological Disorders and Stroke, the Eunice Kennedy Shriver National Institute of Child Health and Human Development, and other institutes of the National Institutes of Health (NIH) conduct...
5. `medquad_ans_0000133-9` (score=0.0002) - A health care provider may use several tests and exams to diagnose Wilson disease, including the following: - medical and family history - physical exam - blood tests - urine tests - liver biopsy - imaging tests Healt...

**Answer**:

For Wilson disease, treatment involves a lifelong effort to reduce and control copper levels in the body. This may include medications, changes in eating, diet, and nutrition, and in some cases, a liver transplant. People should reduce dietary copper intake by avoiding high-copper foods such as shellfish, liver, mushrooms, nuts, and chocolate, especially during initial treatment; however, they should consult a health care provider to determine if these foods can be consumed in moderation during maintenance treatment. Additionally, individuals should check the copper levels in their tap water, as water from copper pipes or wells may contain copper residue; running water helps reduce copper levels. People should not use copper containers or cookware to store or prepare food or drinks. It is also important to discuss the use of complementary and alternative medical practices, including vitamins and dietary supplements, with a health care provider, as some supplements contain copper. If a supplement is recommended, a pharmacist can suggest copper-free options.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.930, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.9980) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000543-3` (score=0.0001) - Joubert syndrome and related disorders can be caused by mutations in at least 10 genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopi...
3. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
4. `medquad_ans_0000592-4` (score=0.0001) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of som...
5. `medquad_ans_0000592-1` (score=0.0001) - Lennox-Gastaut syndrome is a form of severe epilepsy that begins in childhood. It is characterized by multiple types of seizures and intellectual disability. People with Lennox-Gastaut syndrome begin having frequent s...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3]. In about one-third of cases, the cause is unknown and described as cryptogenic, meaning there is no history of epilepsy, neurological problems, or delayed development prior to onset [medquad_ans_0000592-3]. In the remaining two-thirds of cases, the syndrome is symptomatic and associated with an existing neurological problem, such as brain injury, blood flow problems, infections, or brain malformations like cortical dysplasia, or it may occur as part of a genetic disorder such as tuberous sclerosis complex [medquad_ans_0000592-3]. However, no specific genes have been identified as directly causing Lennox-Gastaut syndrome.

**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.969, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000066-2` (score=0.9994) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
2. `medquad_ans_0000041-2` (score=0.0004) - Alpha-1 antitrypsin deficiency occurs worldwide, but its prevalence varies by population. This disorder affects about 1 in 1,500 to 3,500 individuals with European ancestry. It is uncommon in people of Asian descent. ...
3. `medquad_ans_0000066-1` (score=0.0001) - Arginine:glycine amidinotransferase deficiency is an inherited disorder that primarily affects the brain. People with this disorder have mild to moderate intellectual disability and delayed speech development. Some af...
4. `medquad_ans_0000066-3` (score=0.0001) - Mutations in the GATM gene cause arginine:glycine amidinotransferase deficiency. The GATM gene provides instructions for making the enzyme arginine:glycine amidinotransferase. This enzyme participates in the two-step ...
5. `medquad_ans_0001074-1` (score=0.0001) - 17 alpha()-hydroxylase/17,20-lyase deficiency is a condition that affects the function of certain hormone-producing glands called the gonads (ovaries in females and testes in males) and the adrenal glands. The gonads ...

**Answer**:

The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families. [medquad_ans_0000066-2]

**Supported citations**: `medquad_ans_0000066-2`
**Unsupported citations**: _none_

### Example 231 (`0002399-2`)

**Question**: What are the symptoms of Focal dermal hypoplasia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0002399-2` (score=0.9990) - What are the signs and symptoms of Focal dermal hypoplasia? Focal dermal hypoplasia is usually evident from birth and primarily affects the skin, skeleton, eyes, and face. The signs and symptoms of vary widely, althou...
2. `medquad_ans_0002401-1` (score=0.0004) - What are the signs and symptoms of Focal facial dermal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Focal facial dermal dysplasia. If the information is available, the ...
3. `medquad_ans_0000379-4` (score=0.0001) - Focal dermal hypoplasia is inherited in an X-linked dominant pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromos...
4. `medquad_ans_0002399-3` (score=0.0001) - How is this condition inherited? Focal dermal hypoplasia is caused by mutations in the PORCN gene and is inherited in an X-linked dominant manner. Many cases of focal dermal hypoplasia result from a new mutation and o...
5. `medquad_ans_0002400-1` (score=0.0001) - Focal dystonia is a movement disorder that is localized to a specific part of the body. The dystonias are a group of movement problems characterized by involuntary, sustained muscle contractions, tremors, and other un...

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths (papillomas) around the nostrils, lips, anus, and female genitalia, and may also be present in the throat, causing problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly), and a deep split in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.909, groundedness=0.944, answer_relevance=0.545

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.9752) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000269-2__dup2` (score=0.0017) - There are three different filarial species that can cause lymphatic filariasis in humans. Most of the infections worldwide are caused by Wuchereria bancrofti. In Asia, the disease can also be caused by Brugia malayi a...
3. `medquad_ans_0000031-3__dup3` (score=0.0008) - The major risk factors for carotid artery disease, listed below, also are the major risk factors for coronary heart disease (also called coronary artery disease) and peripheral artery disease. Diabetes. With this dise...
4. `medquad_ans_0000050-5__dup1` (score=0.0002) - Who Should Be Tested? The United States Preventive Service Task Force recommends that women aged 65 and older be screened (tested) for osteoporosis, as well as women aged 60 and older who are at increased risk for an ...
5. `medquad_ans_0000038-2__dup3` (score=0.0002) - People can get infected with Babesia parasites in several ways: - The main way is through the bite of an infected tick—during outdoor activities in areas where babesiosis is found (see below). - A less common way is b...

**Answer**:

People at risk for trichinellosis include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating even small amounts of undercooked meat during preparation or cooking puts individuals at risk. Additionally, individuals who consume undercooked wild game, especially bear meat, are at risk for acquiring the disease [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.928, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.9994) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0000108-3__dup1` (score=0.0019) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of co...
3. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
4. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
5. `medquad_ans_0001388-1` (score=0.0001) - What are the signs and symptoms of COASY Protein-Associated Neurodegeneration ? The Human Phenotype Ontology provides the following list of signs and symptoms for COASY Protein-Associated Neurodegeneration . If the in...

**Answer**:

The symptoms of Dravet syndrome include absence seizures, ataxia, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2]. Additionally, as children with Dravet syndrome get older, their decline in cognitive function stabilizes and may improve slightly, though most teenagers remain dependent on caregivers, and the degree of cognitive impairment correlates with the frequency of seizures [medquad_ans_0000108-3__dup1].

**Supported citations**: `medquad_ans_0001924-2`, `medquad_ans_0000108-3__dup1`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000826-3` (score=0.9974) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, whi...
2. `medquad_ans_0000011-13` (score=0.0010) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
3. `medquad_ans_0000832-3` (score=0.0001) - In most cases, the genetic cause of progressive supranuclear palsy is unknown. Rarely, the disease results from mutations in the MAPT gene. Certain normal variations (polymorphisms) in the MAPT gene have also been ass...
4. `medquad_ans_0000694-3` (score=0.0001) - Although the cause of multiple sclerosis is unknown, variations in dozens of genes are thought to be involved in multiple sclerosis risk. Changes in the HLA-DRB1 gene are the strongest genetic risk factors for develop...
5. `medquad_ans_0000089-4__dup1` (score=0.0001) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...

**Answer**:

Genetic changes related to prion disease include mutations in the PRNP gene, which occur in between 10 and 15 percent of all prion disease cases. These mutations result in the production of an abnormally shaped protein called PrPSc from one copy of the gene. This abnormal protein can bind to the normal prion protein (PrPC) and promote its transformation into PrPSc, leading to the accumulation of abnormal protein in the brain and neuronal damage [medquad_ans_0000826-3].

**Supported citations**: `medquad_ans_0000826-3`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.984, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.9994) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0006470-1` (score=0.0001) - What are the signs and symptoms of WT limb blood syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for WT limb blood syndrome. If the information is available, the table below in...
4. `medquad_ans_0005127-2` (score=0.0001) - What are the signs and symptoms of Pseudoachondroplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudoachondroplasia. If the information is available, the table below includ...
5. `medquad_ans_0000817-2__dup1` (score=0.0001) - What are the signs and symptoms of Brachydactyly type C? The Human Phenotype Ontology provides the following list of signs and symptoms for Brachydactyly type C. If the information is available, the table below includ...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.919, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.9994) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0004524-1` (score=0.0001) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0000771-1__dup1` (score=0.0001) - The cornea and lens of your eye helps you focus. Refractive errors are vision problems that happen when the shape of the eye keeps you from focusing well. The cause could be the length of the eyeball (longer or shorte...
4. `medquad_ans_0000764-1__dup1` (score=0.0001) - Summary : Radiation is energy that travels in the form of waves or high-speed particles. It occurs naturally in sunlight. Man-made radiation is used in X-rays, nuclear weapons, nuclear power plants and cancer treatmen...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which collects information from sources such as Orphanet, with frequencies often listed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.739, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000979-1` (score=0.0001) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (pl...
2. `medquad_ans_0006007-1` (score=0.0001) - TAR syndrome is characterized by the absence of a bone called the radius in each forearm, short stature, and thrombocytopenia. The thrombocytopenia often appears first in infancy but becomes less severe or returns to ...
3. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
4. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0000549-1__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...

**Answer**:

The provided context does not specify the number of people affected by thrombocytopenia-absent radius (TAR) syndrome. [medquad_ans_0000979-1], [medquad_ans_0006007-1], and [medquad_ans_0002259-1] describe the signs, symptoms, and genetic basis of TAR syndrome but do not provide population prevalence or affected individual counts. Therefore, the exact number of people affected cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000979-1`, `medquad_ans_0006007-1`, `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.818, groundedness=0.134, answer_relevance=0.745

**Top docs**:

1. `medquad_ans_0000019_3-7` (score=0.9961) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
2. `medquad_ans_0000040_1-6` (score=0.9929) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000019_2-7` (score=0.9904) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000024_2-8` (score=0.9893) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000006_3-8` (score=0.9880) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials specifically for Tay-Sachs Disease. Therefore, it is not possible to answer the question based on the given information. [medquad_ans_0000019_3-7, medquad_ans_0000040_1-6]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000019_3-7, medquad_ans_0000040_1-6`

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.4357) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000008-2` (score=0.0004) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
3. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
4. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
5. `medquad_ans_0000141-2__dup1` (score=0.0001) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. The condition also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green [medquad_ans_00000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=1.000, groundedness=0.840, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000039-9` (score=0.9869) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
2. `medquad_ans_0000039-11` (score=0.9823) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
3. `medquad_ans_0000039-26` (score=0.9592) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circl...
4. `medquad_ans_0000061-13` (score=0.8930) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation...
5. `medquad_ans_0000049-4__dup3` (score=0.0870) - Treatment Goals: Manage Pain, Improve Function Osteoarthritis treatment plans often include ways to manage pain and improve function. Such plans can include exercise, rest and joint care, pain relief, weight control, ...

**Answer**:

The provided context does not list specific treatments for knee replacement surgery. It only describes the reasons for knee replacement, such as easing pain and disability caused by arthritis or joint problems, and mentions that treatments before knee replacement may include exercises, weight loss, walking aids, shoe inserts, and medicines [medquad_ans_0000039-11]. However, no details about treatments _for_ knee replacement itself are given in the context. Therefore, based on the available information, the treatments for knee replacement cannot be determined.

Answer: Insufficient context to determine treatments for knee replacement. [medquad_ans_0000039-11]

**Supported citations**: `medquad_ans_0000039-11`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.784, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000607-1__dup1` (score=0.0022) - Summary : A metabolic panel is a group of tests that measures different chemicals in the blood. These tests are usually done on the fluid (plasma) part of blood. The tests provide information about your body's chemica...
2. `medquad_ans_0000090-1__dup2` (score=0.0001) - Metabolicsyndrome is the name for a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. The term "metabolic" refers to the biochemical processes involv...
3. `medquad_ans_0000058-4__dup2` (score=0.0001) - Certain risk factors make it more likely that you'll develop coronary heart disease (CHD) and have a heart attack. You can control many of these risk factors. Risk Factors You Can Control The major risk factors for a ...
4. `medquad_ans_0001795-2` (score=0.0001) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
5. `medquad_ans_0000608-1__dup1` (score=0.0001) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...

**Answer**:

The provided context does not mention any symptoms of metabolic syndrome. It describes the risk factors associated with metabolic syndrome but does not list symptoms [medquad_ans_0000090-1__dup2]. Therefore, based on the given information, the symptoms of metabolic syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000090-1__dup2`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=0.833, groundedness=0.015, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
2. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
3. `medquad_ans_0002062-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...
4. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The provided context does not contain information about the symptoms of Pyknoachondrogenesis. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003965-2` (score=0.9993) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
2. `medquad_ans_0003966-1` (score=0.0163) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
3. `medquad_ans_0003965-4` (score=0.0002) - Who might consider genetic carrier testing for a family history of metachromatic leukodystrophy? If someone has a family history of metachromatic leukodystrophy (MLD) or someone is known to be a carrier for MLD, indiv...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0000141-2__dup1` (score=0.0001) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, limitation of joint mobility, muscular hypotonia, nystagmus, optic atrophy, aganglionic megacolon, ataxia, Babinski sign, bulbar palsy, cholecystitis, chorea, delusions, dysarthria, dystonia, EMG: neuropathic changes, emotional lability, gallbladder dysfunction, hallucinations, hyporeflexia, increased CSF protein, intellectual disability, loss of speech, mental deterioration, peripheral demyelination, progressive peripheral neuropathy, spastic tetraplegia, tetraplegia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000186-4` (score=0.9678) - When chronic granulomatous disease is caused by mutations in the CYBB gene, the condition is inherited in an X-linked recessive pattern. The CYBB gene is located on the X chromosome, which is one of the two sex chromo...
2. `medquad_ans_0001323-4` (score=0.0029) - Is chronic inflammatory demyelinating polyneuropathy (CIDP) inherited? CIDP is not known to be inherited and is considered an acquired disorder. No clear genetic predisposition or other predisposing factors for CIDP h...
3. `medquad_ans_0004341-3` (score=0.0004) - Is nephrocalcinosis inherited? Nephrocalcinosis may be caused by a large variety of things, including underlying disorders, certain medications and supplements, and infections. Nephrocalcinosis itself is not inherited...
4. `medquad_ans_0000186-1` (score=0.0001) - Chronic granulomatous disease is a disorder that causes the immune system to malfunction, resulting in a form of immunodeficiency. Immunodeficiencies are conditions in which the immune system is not able to protect th...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Yes, chronic granulomatous disease can be inherited. When caused by mutations in the CYBB gene, it is inherited in an X-linked recessive pattern. In this pattern, males are more frequently affected because they have only one X chromosome, and one altered copy of the gene is sufficient to cause the condition. Females require mutations in both copies of the gene to be affected, which is rare. Additionally, when caused by mutations in the CYBA, NCF1, NCF2, or NCF4 genes, the condition is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell must be mutated [medquad_ans_0000186-4].

**Supported citations**: `medquad_ans_0000186-4`
**Unsupported citations**: _none_

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.948, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000061-3__dup1` (score=0.9886) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
2. `medquad_ans_0000061-2__dup1` (score=0.0046) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...
3. `medquad_ans_0000061-1__dup1` (score=0.0001) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
4. `medquad_ans_0000062-1__dup2` (score=0.0001) - Central pain syndrome is a neurological condition caused by damage to or dysfunction of the central nervous system (CNS), which includes the brain, brainstem, and spinal cord. This syndrome can be caused by stroke, mu...
5. `medquad_ans_0000001-6__dup2` (score=0.0001) - Early diagnosis is essential for effective treatment of Acanthamoeba keratitis. Several prescription eye medications are available for treatment. However, the infection can be difficult to treat. The best treatment re...

**Answer**:

The outlook for central cord syndrome varies, but most people who have the condition due to trauma experience some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery. Those who receive medical intervention soon after injury often have good outcomes, with many recovering substantial function, including the ability to walk, although some impairment may remain [medquad_ans_0000061-3__dup1]. Recovery typically begins in the legs, then progresses to the bladder, and hand function recovers last, if at all. Improvement is generally better in younger patients compared to those over the age of 50 [medquad_ans_0000061-3__dup1]. While there is no cure, some individuals recover near-normal function, and early surgical intervention may improve recovery chances, especially in cases of persistent spinal cord compression and neurological deterioration [medquad_ans_00000

**Supported citations**: `medquad_ans_0000061-3__dup1`
**Unsupported citations**: _none_

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.631, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000094-3__dup3` (score=0.9953) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way ...
2. `medquad_ans_0000094-12` (score=0.0025) - During the prodrome and vomiting phases of cyclic vomiting syndrome, a person will generally take in little or no nutrition by mouth. During the recovery phase, the person may be quite hungry as soon as the vomiting s...
3. `medquad_ans_0000261-3` (score=0.0017) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnor...
4. `medquad_ans_0000094-11` (score=0.0004) - A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - a...
5. `medquad_ans_0000094-7__dup1` (score=0.0002) - The severe vomiting and retching that define cyclic vomiting syndrome increase the chance of developing several complications, including dehydration, esophagitis, a Mallory-Weiss tear, and tooth decay. - Dehydration m...

**Answer**:

The cause of cyclic vomiting syndrome is unknown. However, some experts believe that possible problems with bodily functions may contribute to the cause, such as gastrointestinal motility, central nervous system function, autonomic nervous system function, hormone imbalances, and in children, an abnormal inherited gene [medquad_ans_0000094-3__dup3]. Researchers have also proposed that the condition may be related to changes in brain function, hormonal abnormalities, and gastrointestinal problems, and many believe it is a migraine-like condition involving changes in signaling between nerve cells in certain areas of the brain [medquad_ans_00000261-3].

**Supported citations**: `medquad_ans_0000094-3__dup3`
**Unsupported citations**: `medquad_ans_00000261-3`

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=1.000, groundedness=0.933, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000180-1__dup2` (score=0.6764) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
2. `medquad_ans_0000875-1__dup1` (score=0.0002) - Summary : Being a teenager is hard. You're under stress to be liked, do well in school, get along with your family, and make big decisions. You can't avoid most of these pressures, and worrying about them is normal. B...
3. `medquad_ans_0000046-1__dup4` (score=0.0001) - Summary : Antidepressants are medicines that treat depression. Your doctor can prescribe them for you. They work to balance some of the natural chemicals in our brains. It may take several weeks for them to help. Ther...
4. `medquad_ans_0000006-4__dup1` (score=0.0001) - Several components of the NIH support research on developmental disorders such as ADHD. Research programs of the NINDS, the National Institute of Mental Health (NIMH), and the National Institute of Child Health and Hu...
5. `medquad_ans_0000432-1__dup2` (score=0.0001) - Summary : Health literacy refers to how well a person can get the health information and services that they need, and how well they understand them. It is also about using them to make good health decisions. It involv...

**Answer**:

Yes, there is information about child mental health. Early recognition and treatment of mental illnesses in children are important because once mental illness develops, it becomes a regular part of the child's behavior and is more difficult to treat. Warning signs that may indicate a more serious problem include problems in more than one setting (at school, at home, with peers), changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to younger behaviors like bedwetting, signs of being upset such as sadness or tearfulness, self-destructive behaviors like head-banging or sudden injuries, and repeated thoughts of death. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history. Treatments may include medicines and talk therapy [medquad_ans_0000180-1__dup2].

**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001618-2` (score=0.9976) - What are the signs and symptoms of Cowden syndrome? Cowden syndrome is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. Approximately 99% of people affected b...
2. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
3. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
4. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
5. `medquad_ans_0002946-1` (score=0.0001) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous growths (hamartomas) on the skin and/or in the mouth, with approximately 99% of affected individuals having these by the third decade of life. A majority develop hamartomatous polyps in the gastrointestinal tract. Other common symptoms include benign diseases of the breast, thyroid, and endometrium; a rare noncancerous brain tumor called Lhermitte-Duclos disease; enlarged head (macrocephaly); autism spectrum disorder; intellectual disability; and vascular abnormalities. Specific signs and symptoms reported in the Human Phenotype Ontology include abnormality of the pupil, abnormality of the tongue, aplasia/hypoplasia of the cerebellum, arteriovenous malformation, cognitive impairment, conjunctival hamartoma, dental malocclusion, epibulbar dermoid, exostoses, foot polydactyly, genu

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.699, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.9929) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000576-5` (score=0.0004) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
3. `medquad_ans_0006267-3` (score=0.0003) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
4. `medquad_ans_0000395-2` (score=0.0001) - Fukuyama congenital muscular dystrophy is seen almost exclusively in Japan, where it is the second most common form of childhood muscular dystrophy (after Duchenne muscular dystrophy). Fukuyama congenital muscular dys...
5. `medquad_ans_0000395-3` (score=0.0001) - Fukuyama congenital muscular dystrophy is caused by mutations in the FKTN gene. This gene provides instructions for making a protein called fukutin. Although the exact function of fukutin is unclear, researchers predi...

**Answer**:

The provided context does not specify treatments for Fukuyama congenital muscular dystrophy. While it details the genetic cause (mutations in the FKTN gene) and incidence, no information on specific therapeutic interventions or management strategies is included [medquad_ans_0000395-3][medquad_ans_0000395-5]. Therefore, based on the available information, treatments for Fukuyama congenital muscular dystrophy cannot be determined.

**Supported citations**: `medquad_ans_0000395-3`, `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.860, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000004_1-4` (score=0.0148) - Tests that examine the lymph nodes are used to detect (find) and diagnose adult Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general sign...
2. `medquad_ans_0000004_6-3` (score=0.0001) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with ...
3. `medquad_ans_0000004_6-5` (score=0.0001) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...
4. `medquad_ans_0000004_6-2` (score=0.0001) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
5. `medquad_ans_0000573-1__dup1` (score=0.0001) - Lymphoma is a cancer of a part of the immune system called the lymph system. There are many types of lymphoma. One type is Hodgkin disease. The rest are called non-Hodgkin lymphomas. Non-Hodgkin lymphomas begin when a...

**Answer**:

The provided context does not contain specific information on how to diagnose childhood Hodgkin lymphoma. While it details tests used to detect and diagnose adult Hodgkin lymphoma, such as physical exam and history, complete blood count, blood chemistry studies, sedimentation rate, lymph node biopsy, and immunophenotyping, there is no mention of diagnostic procedures specific to childhood Hodgkin lymphoma. Therefore, based on the given context, the diagnosis of childhood Hodgkin lymphoma cannot be fully addressed.

[medquad_ans_0000004_1-4]

**Supported citations**: `medquad_ans_0000004_1-4`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.112, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0003718-4` (score=0.2713) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
2. `medquad_ans_0000562-4` (score=0.0204) - When Klippel-Feil syndrome is caused by mutations in the GDF6 or GDF3 genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder....
3. `medquad_ans_0000527-4` (score=0.0131) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
4. `medquad_ans_0004408-3` (score=0.0045) - How is nevoid basal cell carcinoma syndrome inherited? Nevoid basal cell carcinoma syndrome (NBCCS) is caused by a change (mutation) in the PTCH1 gene and is inherited in an autosomal dominant way. This means that if ...
5. `medquad_ans_0001084-3__dup1` (score=0.0028) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).

**Answer**:

The provided context does not contain any information about KBG syndrome or its inheritance pattern. Therefore, it is not possible to determine based on the given text whether KBG syndrome is inherited.

Answer: Insufficient context [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.667, groundedness=0.533, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000012-1__dup2` (score=0.9683) - Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is...
2. `medquad_ans_0000197-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000196-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000195-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000193-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is severe and a late-onset form that has less impact upon neurological function. The early-onset form affects about 20 percent of all babies who have AGS and is characterized by neurological and liver abnormalities, such as enlargement of the liver and spleen and elevated liver enzymes. Jittery behavior and poor feeding ability mimic congenital viral infection. The later-onset form begins after the first weeks or months of normal development, with symptoms including progressive decline in head growth, weak or stiffened muscles (spasticity), cognitive and developmental delays, irritability, inconsolable crying, intermittent fever, seizures, loss of developmental skills, puffy swelling on the fingers, toes, and ears resembling chilblains,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.833, groundedness=0.805, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000006_2-7` (score=0.9924) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000016_1-8` (score=0.9921) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000030_1-7` (score=0.9907) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000004_1-6` (score=0.9872) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000015_1-9` (score=0.9843) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any specific research or clinical trials being conducted for Renal Cell Cancer. Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000006_2-7][medquad_ans_0000016_1-8]

**Supported citations**: `medquad_ans_0000006_2-7`, `medquad_ans_0000016_1-8`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000190-9` (score=0.9146) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
2. `medquad_ans_0000121-13` (score=0.0249) - - Smoking has been found to increase the risk of cancers of the mouth, esophagus, stomach, and pancreas. Research suggests that smoking may also increase the risk of cancers of the liver, colon, and rectum. - Smoking ...
3. `medquad_ans_0000184-2__dup2` (score=0.0003) - Cirrhosis has various causes. Many people with cirrhosis have more than one cause of liver damage. The list below shows common causes of cirrhosis in the United States.2 While chronic hepatitis C and alcohol-related l...
4. `medquad_ans_0000190-4__dup2` (score=0.0003) - Most complications of primary biliary cirrhosis are related to cirrhosis and start after primary biliary cirrhosis progresses to cirrhosis. In some cases, portal hypertension and esophageal varices may develop before ...
5. `medquad_ans_0000257-2` (score=0.0003) - Cirrhosis affects more than 600,000 people in the United States; cryptogenic cirrhosis likely accounts for 5 to 30 percent of these cases.

**Answer**:

For primary biliary cirrhosis, a healthcare provider may prescribe ursodiol (Actigall, Urso) to treat the condition. Early treatment with this medication reduces the likelihood of needing a liver transplant and improves survival. A health care provider may consider a liver transplant when cirrhosis leads to liver failure or treatment for complications is ineffective [medquad_ans_0000190-9].

**Supported citations**: `medquad_ans_0000190-9`
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=1.000, groundedness=0.988, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.9778) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0002946-1` (score=0.0001) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...
3. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People who have seizure disorders are also at risk in the water. Drowning can happen quickly and silently [medquad_ans_0000299-1__dup2].

**Supported citations**: `medquad_ans_0000299-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=0.857, groundedness=0.990, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.9604) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
2. `medquad_ans_0000313-1` (score=0.2940) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
3. `medquad_ans_0000313-2` (score=0.0003) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...
4. `medquad_ans_0002061-1` (score=0.0002) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
5. `medquad_ans_0002049-5` (score=0.0001) - How is Ehlers-Danlos syndrome diagnosed? A diagnosis of Ehlers-Danlos syndrome is typically based on the presence of characteristic signs and symptoms. Depending on the subtype suspected, some of the following tests m...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is an inherited connective tissue disorder caused by defects in a protein called collagen. It is characterized by hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera. This type is caused by mutations in the PLOD1 gene and is inherited in an autosomal recessive manner. Treatment is focused on preventing serious complications and relieving associated signs and symptoms [medquad_ans_0002058-1].

**Supported citations**: `medquad_ans_0002058-1`
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.9994) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0000438-3` (score=0.0001) - Mutations in the LBR gene cause Greenberg dysplasia. This gene provides instructions for making a protein called the lamin B receptor. One region of this protein, called the sterol reductase domain, plays an important...
4. `medquad_ans_0001860-1` (score=0.0001) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
5. `medquad_ans_0001612-1` (score=0.0001) - What are the signs and symptoms of Cortisone reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Cortisone reductase deficiency. If the information is available, th...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity. These symptoms occur in approximately 90% of patients for cognitive impairment and microcephaly [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.714, groundedness=0.921, answer_relevance=0.257

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.9987) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
3. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
4. `medquad_ans_0002062-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, spondylocheirodysplastic type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, spondylocheirodyspl...
5. `medquad_ans_0004384-2` (score=0.0001) - What are the signs and symptoms of Neuronal ceroid lipofuscinosis 10 ? The Human Phenotype Ontology provides the following list of signs and symptoms for Neuronal ceroid lipofuscinosis 10 . If the information is avail...

**Answer**:

The prevalence of monilethrix is unknown. [medquad_ans_0000669-2]

**Supported citations**: `medquad_ans_0000669-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.3190) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0002259-1` (score=0.0002) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...
4. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0000141-2__dup1` (score=0.0001) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. Most people with Hartnup disease are asymptomatic, but those who do show symptoms typically experience onset between the ages of 3 and 9, occasionally in adulthood. Signs and symptoms may include skin photosensitivity, neurologic findings, psychiatric symptoms, and ocular findings. Mental development is usually normal, though a few cases with intellectual impairment have been reported. Treatment may include a high-protein diet, protection from sunlight, vitamin supplementation, avoidance of certain drugs/medications, and in some cases, nicotinamide supplements and tryptophan ethyl ester [medquad_ans_0002733-1].

**Supported citations**: `medquad_ans_0002733-1`
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=1.000, groundedness=0.792, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000428-1` (score=0.0001) - Glycogen storage disease type VII (GSDVII) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in muscle cells. A lack of glycogen breakdown interferes with the function of mu...
2. `medquad_ans_0000427-1` (score=0.0001) - Glycogen storage disease type VI (also known as GSDVI or Hers disease) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in liver cells. A lack of glycogen breakdown interfe...
3. `medquad_ans_0002615-3` (score=0.0001) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an impor...
4. `medquad_ans_0002620-1` (score=0.0001) - Glycogen storage disease type 4 (GSD 4) is part of a group of disorders which lead to abnormal accumulation of glycogen (a storage form of glucose) in various parts of the body. Symptoms of GSD 4 usually begin in infa...
5. `medquad_ans_0000424-5` (score=0.0001) - These resources address the diagnosis or management of glycogen storage disease type IV: - Gene Review: Gene Review: Glycogen Storage Disease Type IV - Genetic Testing Registry: Glycogen storage disease, type IV - Med...

**Answer**:

The provided context does not mention the specific genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-1] describes the clinical features and types of GSDVII but does not specify the underlying genetic mutations or changes. Therefore, based on the given information, the genetic changes related to GSDVII cannot be determined.

**Supported citations**: `medquad_ans_0000428-1`
**Unsupported citations**: _none_

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.821, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000977-5` (score=0.9933) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
2. `medquad_ans_0000138-4__dup2` (score=0.0001) - When kidneys are diseased or damaged, they do not make enough EPO. As a result, the bone marrow makes fewer red blood cells, causing anemia. When blood has fewer red blood cells, it deprives the body of the oxygen it ...
3. `medquad_ans_0000513-3` (score=0.0001) - Mutations in the AMN or CUBN gene can cause Imerslund-Grsbeck syndrome. The AMN gene provides instructions for making a protein called amnionless, and the CUBN gene provides instructions for making a protein called cu...
4. `medquad_ans_0000977-2` (score=0.0001) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
5. `medquad_ans_0005625-4` (score=0.0001) - How might sideroblastic anemia pyridoxine-refractory autosomal recessive be treated? Currently there is not a cure for sideroblastic anemia pyridoxine-refractory autosomal recessive, however with proper treatment the ...

**Answer**:

The provided context does not specify the treatments for thiamine-responsive megaloblastic anemia syndrome. While it mentions resources that address the diagnosis or management of the condition, such as Gene Review, Genetic Testing Registry, and MedlinePlus Encyclopedia, no explicit information on treatments is given [medquad_ans_0000977-5]. Therefore, based on the available context, the treatments cannot be determined.

**Supported citations**: `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.782, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000173-5__dup1` (score=0.0240) - People with a solitary kidney do not need to eat a special diet. However, people with reduced kidney function may need to make changes to their diet to slow the progression of kidney disease. More information about re...
2. `medquad_ans_0000201-9` (score=0.0019) - - Hematuria is blood in the urine. - Most people with microscopic hematuria do not have symptoms. People with gross hematuria have urine that is pink, red, or cola-colored due to the presence of red blood cells (RBCs)...
3. `medquad_ans_0000038-22` (score=0.0012) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and m...
4. `medquad_ans_0000135-2__dup2` (score=0.0006) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
5. `medquad_ans_0000804-5` (score=0.0003) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autoso...

**Answer**:

The provided context does not contain information about what to do for Polycystic Kidney Disease. [medquad_ans_0000173-5__dup1] discusses general kidney health, diet, blood pressure control, and injury prevention, but does not address Polycystic Kidney Disease specifically. [medquad_ans_0000201-9] mentions hematuria and its causes, including polycystic kidney disease as a more serious cause, but does not provide guidance or recommendations for managing it. Therefore, based on the given context, no actionable information about Polycystic Kidney Disease is available.

**Supported citations**: `medquad_ans_0000173-5__dup1`, `medquad_ans_0000201-9`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.909, groundedness=0.173, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000025_1-8` (score=0.9866) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000006_3-8` (score=0.9800) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000061-5__dup2` (score=0.9730) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research ...
4. `medquad_ans_0000061-9` (score=0.0115) - Scientists believe that rheumatoid arthritis may result from the interaction of many factors such as genetics, hormones, and the environment. Although rheumatoid arthritis sometimes runs in families, the actual cause ...
5. `medquad_ans_0000031_1-8` (score=0.0004) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Rheumatoid Arthritis. Therefore, based on the given information, it is not possible to answer the question. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.958, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.9770) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
2. `medquad_ans_0004450-4` (score=0.0133) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004454-3` (score=0.0133) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004453-3` (score=0.0133) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004452-2` (score=0.0133) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

Surgery is typically the treatment for Glass-Chapman-Hockley syndrome, with the goal of increasing the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.737, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000868-5` (score=0.9961) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
2. `medquad_ans_0004451-3` (score=0.0911) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004452-2` (score=0.0911) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004456-3` (score=0.0911) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004454-3` (score=0.0911) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain information about the treatments for Renpenning syndrome. Therefore, a specific answer cannot be given based on the available resources [medquad_ans_0000868-5].

**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.694, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
2. `medquad_ans_0001092-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
3. `medquad_ans_0001094-2__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1E? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1E. If the information is ava...
4. `medquad_ans_0001100-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
5. `medquad_ans_0001093-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 2G. [medquad_ans_0001110-1] and [medquad_ans_0001092-1__dup1] discuss symptoms of Charcot-Marie-Tooth disease type 2L and type 1C, respectively, but there is no mention of type 2G. Therefore, the requested information is not available in the context.

**Supported citations**: `medquad_ans_0001110-1`, `medquad_ans_0001092-1__dup1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=0.833, groundedness=0.681, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0004384-2` (score=0.0001) - What are the signs and symptoms of Neuronal ceroid lipofuscinosis 10 ? The Human Phenotype Ontology provides the following list of signs and symptoms for Neuronal ceroid lipofuscinosis 10 . If the information is avail...
2. `medquad_ans_0002946-1` (score=0.0001) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...
3. `medquad_ans_0005488-1` (score=0.0001) - What are the signs and symptoms of Scheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for Scheie syndrome . If the information is available, the table below includes how o...
4. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...
5. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...

**Answer**:

The provided context does not contain information about the symptoms of Coccygodynia. [medquad_ans_0004384-2] and [medquad_ans_0002946-1] discuss symptoms of Neuronal ceroid lipofuscinosis 10 and Hurler-Scheie syndrome, respectively, but neither mentions Coccygodynia. Therefore, no symptoms can be identified from the given context.

**Supported citations**: `medquad_ans_0004384-2`, `medquad_ans_0002946-1`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.9994) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0004026-1` (score=0.0001) - What are the signs and symptoms of Microcephaly, holoprosencephaly, and intrauterine growth retardation? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, holoprosencepha...
3. `medquad_ans_0004549-2` (score=0.0001) - What are the signs and symptoms of Opitz G/BBB syndrome? Opitz G/BBB syndrome mainly affects structures along the midline of the body. The most common features of the condition are wide-spaced eyes (hypertelorism); de...
4. `medquad_ans_0003270-1` (score=0.0001) - Agenesis of the corpus callosum (ACC) is a birth defect in which the structure that connects the two sides of the brain (the corpus callosum) is partially or completely absent. This birth defect can occur as an isolat...
5. `medquad_ans_0000009-1__dup1` (score=0.0001) - Agenesis of the corpus callosum (ACC) is one of several disorders of the corpus callosum, the structure that connects the two hemispheres (left and right) of the brain. In ACC the corpus callosum is partially or compl...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. Preaxial hand polydactyly, proptosis, and seizures are also listed as possible symptoms. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which draws from Orphanet and other sources, with frequencies often expressed as percentages or fractions [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003402-5` (score=0.9964) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
2. `medquad_ans_0002431-3` (score=0.0011) - How is Freeman Sheldon syndrome diagnosed? Freeman Sheldon syndrome may be suspected based on medical history and physical examination which reveal characteristic features such as a small mouth, flat mask-like face, c...
3. `medquad_ans_0001276-5` (score=0.0009) - How is chromosome 3p- syndrome diagnosed? There are several different specialized tests that can be used to diagnose a chromosome 3p- syndrome. These include: Karyotype - a karyotype is a laboratory test that produces...
4. `medquad_ans_0000089-5__dup1` (score=0.0001) - Your doctor will diagnose Marfan syndrome based on your medical and family histories, a physical exam, and test results. He or she also will consult a set of guidelines called Ghent criteria, which are used to diagnos...
5. `medquad_ans_0001404-4` (score=0.0001) - How is Coffin-Siris syndrome diagnosed? Diagnosis of Coffin-Siris syndrome is largely based upon the presence or absence of common signs and symptoms in the individual. While formal diagnostic criteria have not been e...

**Answer**:

Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. KFS can be associated with a wide range of other abnormalities involving many parts of the body. Therefore, other initial exams are needed to detect additional physical abnormalities or underlying conditions. These include: examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present in each person with KFS [medquad_ans_0003402

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.529, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001497-1` (score=0.9537) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
2. `medquad_ans_0000222-1` (score=0.0661) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
3. `medquad_ans_0002458-1` (score=0.0068) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...
4. `medquad_ans_0000222-2` (score=0.0004) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
5. `medquad_ans_0000222-3` (score=0.0002) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...

**Answer**:

Congenital diaphragmatic hernia (CDH) is a defect in the diaphragm resulting from abnormal development before birth. The diaphragm, composed of muscle and other fibrous tissue, normally separates the organs in the abdomen from those in the chest. CDH ranges in severity from a thinned area in the diaphragm to its complete absence, leading to an abnormal opening (hernia) that allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This crowding can cause underdevelopment of the lungs (pulmonary hypoplasia), potentially resulting in life-threatening breathing difficulties apparent at birth. CDH may occur with other malformations and is classified by position, such as Bochdalek hernia (80–90% of cases, involving the side or back of the diaphragm) and Morgagni hernia (about 2% of cases

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=0.800, groundedness=0.988, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001088-1__dup1` (score=0.0761) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
2. `medquad_ans_0000173-1` (score=0.0431) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
3. `medquad_ans_0001116-1` (score=0.0044) - Charcot-Marie-Tooth type 4 (CMT4) is a congenital neurologic hereditary disease, part of a group of peripheral neuropathies known as Charcot-Marie-Tooth disease (CMT). It is classified in CMT4A, CMT4B1, CMT4B2, CMT4C,...
4. `medquad_ans_0001094-1__dup1` (score=0.0020) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
5. `medquad_ans_0001104-5` (score=0.0003) - Is genetic testing available for Charcot-Marie-Tooth disease type 2F? Yes. GeneTests lists the names of laboratories that are performing clincial genetic testing for Charcot-Marie-Tooth disease type 2F. To view the co...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Defects in many different genes cause different forms of this disease. Common symptoms may include foot drop, foot deformity, loss of lower leg muscle, numbness in the foot or leg, slapping gait, and weakness of the hips, legs, or feet. The disease usually becomes apparent in adolescence or early adulthood, but onset may occur anytime from early childhood through late adulthood. Symptoms vary in severity, even among family members, and may include balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities such as high arches or flat feet, and decreased sensitivity to touch [medquad_ans_0000173-1]medquad_ans_0001088-1

**Supported citations**: `medquad_ans_0000173-1`
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=1.000, groundedness=0.964, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000669-4` (score=0.9908) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
4. `medquad_ans_0000549-1__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal dominant caf au lait spots ? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal dominant caf au lait spots . If the information is...
5. `medquad_ans_0003571-1` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy - dystonia - motor neuropathy ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoencephalopathy - dystonia - motor neuropat...

**Answer**:

Yes, monilethrix can be inherited. When caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder. In rare cases, the condition results from a new mutation and is not inherited. When caused by mutations in the DSG4 gene, it is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell have mutations [medquad_ans_0000669-4].

**Supported citations**: `medquad_ans_0000669-4`
**Unsupported citations**: _none_

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.904, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000215-3__dup2` (score=0.0924) - Most UTIs are caused by bacteria that live in the bowel. The bacterium Escherichia coli (E. coli) causes the vast majority of UTIs. The urinary tract has several systems to prevent infection. The points where the uret...
2. `medquad_ans_0000214-1__dup4` (score=0.0212) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
3. `medquad_ans_0000215-1__dup3` (score=0.0212) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
4. `medquad_ans_0000223-2__dup2` (score=0.0116) - Most UTIs are caused by bacteria that live in the bowel, the part of the digestive tract where stool is changed from liquid to solid. The bacterium Escherichia coli (E. coli) causes most UTIs. The urinary tract has se...
5. `medquad_ans_0000210-1__dup4` (score=0.0076) - Urinary incontinence is the loss of bladder control, which results in the accidental loss of urine. A child with UI may not stay dry during the day or night. Some UI is caused by a health problem such as - a urinary t...

**Answer**:

Urinary Tract Infections (UTIs) in children are infections in the urinary tract caused primarily by bacteria, most commonly Escherichia coli (E. coli), which live in the bowel. These infections occur when bacteria enter the urinary tract and overcome the body's natural defenses, such as one-way valves at the ureters that prevent urine reflux and immune defenses that flush microbes out. Regular urination helps maintain urinary tract sterility by flushing out bacteria, while holding in urine or inadequate fluid intake allows bacteria to grow. Chronic constipation, defined as fewer than two bowel movements per week, can increase UTI risk by pressing on the bladder and blocking urine flow, allowing bacterial growth. Children who are prone to such infections are more likely to develop UTIs, similar to how some children are prone to other infections like coughs or ear infections [medquad_ans_0000215-3__dup2]. UTIs can affect

**Supported citations**: `medquad_ans_0000215-3__dup2`
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=0.875, groundedness=0.942, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000135-2__dup2` (score=0.0063) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
2. `medquad_ans_0000135-1__dup4` (score=0.0007) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...
3. `medquad_ans_0000804-5` (score=0.0005) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autoso...
4. `medquad_ans_0000804-2` (score=0.0003) - Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States. The autosomal dominant form of the disease is much more common than the autosomal recessive form. Au...
5. `medquad_ans_0000167-4__dup2` (score=0.0001) - Estimates of PKDs prevalence range from one in 400 to one in 1,000 people.1 According to the United States Renal Data System, PKD accounts for 2.2 percent of new cases of kidney failure each year in the United States....

**Answer**:

Complications of Polycystic Kidney Disease (PKD) include high blood pressure and problems with blood vessels in the brain and heart [medquad_ans_0000135-2__dup2].

**Supported citations**: `medquad_ans_0000135-2__dup2`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=1.000, groundedness=0.507, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.8358) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000123-4__dup2` (score=0.0034) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
3. `medquad_ans_0000123-6` (score=0.0006) - Viral gastroenteritis is usually diagnosed based on symptoms alone. People who have symptoms that are severe or last for more than a few days may want to see a health care provider for additional tests. A health care ...
4. `medquad_ans_0000391-1__dup1` (score=0.0002) - Have you ever had the "stomach flu?" What you probably had was gastroenteritis - not a type of flu at all. Gastroenteritis is an inflammation of the lining of the intestines caused by a virus, bacteria or parasites. V...
5. `medquad_ans_0000638-1__dup1` (score=0.0001) - What are the signs and symptoms of Bare lymphocyte syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Bare lymphocyte syndrome 2. If the information is available, the table ...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, and body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in adults, often responsible for epidemics, especially from October to April. Symptoms typically appear 1 to 2 days after exposure and last for 1 to 3 days. There is no specific medicine to treat norovirus infections; treatment includes bed rest and plenty of fluids to prevent dehydration. Prevention can be achieved through proper hand washing and safe food preparation [medquad_ans_0000649-

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.900, groundedness=0.923, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000392-3` (score=0.0234) - The cause of Fryns syndrome is unknown. The disorder is thought to be genetic because it tends to run in families and has features similar to those of other genetic disorders. Duplications and deletions in several chr...
2. `medquad_ans_0000222-3` (score=0.0110) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...
3. `medquad_ans_0000222-2` (score=0.0001) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
4. `medquad_ans_0004319-1` (score=0.0001) - What are the signs and symptoms of Natal teeth, intestinal pseudoobstruction and patent ductus? The Human Phenotype Ontology provides the following list of signs and symptoms for Natal teeth, intestinal pseudoobstruct...
5. `medquad_ans_0000222-1` (score=0.0001) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia include duplications and deletions in several chromosome regions, which have been associated with the condition and some of its features, particularly in the context of Fryns syndrome [medquad_ans_0000392-3]. Some syndromes in which congenital diaphragmatic hernia occurs, such as Donnai-Barrow, Fryns, and Pallister-Killian mosaic syndrome, are caused by changes in single genes or chromosomal abnormalities affecting multiple genes [medquad_ans_0000222-3]. Researchers are also investigating changes in specific genes involved in diaphragm development, including transcription factors that regulate gene expression and genes that encode proteins involved in cell structure or cell migration during embryonic development [medquad_ans_0000222-3]. However, no specific genetic change has been identified that causes all of the signs and symptoms

**Supported citations**: `medquad_ans_0000392-3`, `medquad_ans_0000222-3`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004534-3` (score=0.9950) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
2. `medquad_ans_0003881-3` (score=0.0021) - What causes megalencephaly, polymicrogyria, and hydrocephalus (MPPH) syndrome? The cause of MPPH syndrome is unknown. Infection during pregnancy or fetal accident is thought to be unlikely.
3. `medquad_ans_0003540-3` (score=0.0005) - What causes Lemierre syndrome? In about 90% of cases, Lemierre syndrome is caused by Fusobacterium necrophorum; however, the syndrome has also been reported with other bacteria, including Stapylococcus aureus, Bactero...
4. `medquad_ans_0000765-3__dup1` (score=0.0003) - What causes blue rubber bleb nevus syndrome? Currently the cause of blue rubber bleb syndrome is not known.
5. `medquad_ans_0002391-3` (score=0.0002) - What causes Floating-Harbor syndrome? The exact cause of Floating-Harbor syndrome is not known. Autosomal dominant inheritance has been suggested.

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by mutations in several genes, including RAG1 and RAG2 (mapped to chromosome band 11p13), ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4. Some cases have also been associated with 22q11 microdeletion syndrome [medquad_ans_0004534-3].

**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: _none_

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=1.000, groundedness=0.976, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.9864) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0000032-1__dup1` (score=0.0264) - Ataxia often occurs when parts of the nervous system that control movement are damaged. People with ataxia experience a failure of muscle control in their arms and legs, resulting in a lack of balance and coordination...
3. `medquad_ans_0000495-1__dup1` (score=0.0014) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
4. `medquad_ans_0000078-2` (score=0.0005) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
5. `medquad_ans_0000496-1__dup1` (score=0.0001) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive movement problems due to a loss (atrophy) of nerve cells in the cerebellum, the part of the brain that coordinates movement. Signs and symptoms first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. Individuals may also experience difficulty with movements that involve judging distance or scale (dysmetria), abnormal eye movements (nystagmus), and problems following the movements of objects with the eyes. The movement problems are slowly progressive and may eventually require the use of a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.875, groundedness=0.963, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000050-2__dup1` (score=0.9721) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
2. `medquad_ans_0000049-2__dup1` (score=0.9721) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
3. `medquad_ans_0000468-3` (score=0.0005) - Mutations in the SEPT9 gene cause hereditary neuralgic amyotrophy. The SEPT9 gene provides instructions for making a protein called septin-9, which is part of a group of proteins called septins. Septins are involved i...
4. `medquad_ans_0000845-1__dup1` (score=0.0004) - Exercising is good for you, but sometimes you can injure yourself when you play sports or exercise. Accidents, poor training practices, or improper gear can cause them. Some people get hurt because they are not in sha...
5. `medquad_ans_0000050-3__dup1` (score=0.0001) - The site and type of brachial plexus injury determines the prognosis. For avulsion and rupture injuries, there is no potential for recovery unless surgical reconnection is made in a timely manner. The potential for re...

**Answer**:

The treatments for brachial plexus injuries include physical therapy and, in some cases, surgery [medquad_ans_0000050-2__dup1].

**Supported citations**: `medquad_ans_0000050-2__dup1`
**Unsupported citations**: _none_

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.943, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000216-8` (score=0.9936) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
2. `medquad_ans_0000216-7` (score=0.0299) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
3. `medquad_ans_0000216-2__dup1` (score=0.0293) - Many types of defects in the urinary tract can cause urine blockage: - Vesicoureteral reflux (VUR). Most children with VUR are born with a ureter that did not grow long enough during development in the womb. The valve...
4. `medquad_ans_0000216-4__dup1` (score=0.0033) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...
5. `medquad_ans_0000217-6` (score=0.0020) - The most common tests used to diagnose VUR include - Voiding cystourethrogram (VCUG). VCUG is an x-ray image of the bladder and urethra taken during urination, also called voiding. The bladder and urethra are filled w...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), ureterocele, and genetic conditions like prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects. Additionally, spina bifida and other birth defects affecting the spinal cord can lead to urine blockage [medquad_ans_0000216-8]. These defects can result in urine backing up, causing conditions such as hydroureter and hydronephrosis [medquad_ans_0000216-8]. Some defects may be detected before or after birth through prenatal tests like ultrasound, amniocentesis, or chorionic villus sampling, or through imaging

**Supported citations**: `medquad_ans_0000216-8`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.778, groundedness=0.857, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0000526-2` (score=0.9993) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
2. `medquad_ans_0000526-4` (score=0.0001) - Intranuclear rod myopathy is an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases are not inherited; they result from new mutations in ...
3. `medquad_ans_0000526-3` (score=0.0001) - Intranuclear rod myopathy is caused by a mutation in the ACTA1 gene. This gene provides instructions for making a protein called skeletal alpha ()-actin, which is part of the actin protein family. Actin proteins are i...
4. `medquad_ans_0001461-1` (score=0.0001) - Cone-rod dystrophies (CRDs) are a group of inherited eye disorders that affect both the cone and rod cells of the retina (photosenstitive receptor cells). In contrast to rod-cone dystrophies, individuals experience de...
5. `medquad_ans_0000526-1` (score=0.0001) - Intranuclear rod myopathy is a disorder that primarily affects skeletal muscles, which are muscles that the body uses for movement. People with intranuclear rod myopathy have severe muscle weakness (myopathy) and poor...

**Answer**:

The exact prevalence of intranuclear rod myopathy is unknown [medquad_ans_0000526-2]. Therefore, the number of people affected by this disorder cannot be determined from the provided context.

**Supported citations**: `medquad_ans_0000526-2`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000058-3__dup4` (score=0.9849) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
2. `medquad_ans_0000036-12` (score=0.9806) - Abnormal blood pressure is higher than 120/80 mmHg. If either your systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hy...
3. `medquad_ans_0000036-21` (score=0.9779) - In most cases, the goal is probably to keep your blood pressure below 140/90 mmHg (130/80 if you have diabetes or chronic kidney disease). Normal blood pressure is less than 120/80. Ask your doctor what your blood pre...
4. `medquad_ans_0000565-1__dup1` (score=0.9669) - You've probably heard that high blood pressure is a problem. Sometimes blood pressure that is too low can also cause problems. Blood pressure is the force of your blood pushing against the walls of your arteries. Each...
5. `medquad_ans_0000036-9` (score=0.9619) - Blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If your blood pressure rises and stays high over time, its called high blood pressure. High blood pressure ...

**Answer**:

High blood pressure is defined as blood pressure that is higher than 140/90 mmHg. If either the systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hypertension. However, high blood pressure specifically refers to readings at or above 140/90 mmHg. The goal in most cases is to keep blood pressure below 140/90 mmHg, and for individuals with diabetes or chronic kidney disease, the goal is below 130/80 mmHg [medquad_ans_0000036-12][medquad_ans_0000036-21]. High blood pressure forces the heart to work harder, can strain the heart, damage blood vessels, and

**Supported citations**: `medquad_ans_0000036-12`, `medquad_ans_0000036-21`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=0.750, groundedness=0.014, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0001058-4` (score=0.9962) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males...
2. `medquad_ans_0001061-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
3. `medquad_ans_0000607-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
4. `medquad_ans_0001053-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0001052-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...

**Answer**:

The provided context does not specify whether X-linked dystonia-parkinsonism is inherited. While the context discusses X-linked and X-linked recessive inheritance patterns in general, it does not mention X-linked dystonia-parkinsonism or its mode of inheritance. Therefore, based on the given information, it is not possible to determine if X-linked dystonia-parkinsonism is inherited. [no supporting context]

**Supported citations**: _none_
**Unsupported citations**: `no supporting context`

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.531, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000702-1__dup1` (score=0.9065) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
2. `medquad_ans_0000120-1` (score=0.8601) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
3. `medquad_ans_0000120-2` (score=0.0002) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, sometimes lead to coma. Ketoacidotic attacks are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene. It is estimated to affect fewer than 1 in 1 million newborns [medquad_ans_0000702-1\_\_dup

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.889, groundedness=0.848, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000006_5-7` (score=0.9969) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
2. `medquad_ans_0000006_9-4` (score=0.0041) - Tests that examine the brain and spinal cord are used to detect (find) childhood ependymoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs of ...
3. `medquad_ans_0000006_2-8` (score=0.0031) - Key Points - There are different types of treatment for children with brain and spinal cord tumors. - Children with brain or spinal cord tumors should have their treatment planned by a team of health care providers wh...
4. `medquad_ans_0000006_5-5` (score=0.0004) - Tests that examine the brain are used to detect (find) childhood brain stem glioma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs of health, i...
5. `medquad_ans_0000006_5-9` (score=0.0003) - Key Points - There are different types of treatment for children with brain stem glioma. - Children with brain stem glioma should have their treatment planned by a team of health care providers who are experts in trea...

**Answer**:

The context does not provide information on the stages of Childhood Brain Stem Glioma. [medquad_ans_0000006_5-7] states that there is no standard staging system for childhood brain stem glioma. Therefore, the stages of Childhood Brain Stem Glioma cannot be determined from the provided context.

**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.533, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.3112) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000266-2__dup1` (score=0.0036) - MRI imaging is often used to evaluate individuals with these symptoms, and can be used to diagnose the location of the tethering, lower than normal position of the conus medullaris, or presence of a tumor or fatty mas...
3. `medquad_ans_0000061-1__dup1` (score=0.0023) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
4. `medquad_ans_0000839-1__dup1` (score=0.0017) - Your spinal cord is a bundle of nerves that runs down the middle of your back. It carries signals back and forth between your body and your brain. A spinal cord injury disrupts the signals. Spinal cord injuries usuall...
5. `medquad_ans_0000258-1__dup2` (score=0.0011) - A spinal cord injury usually begins with a sudden, traumatic blow to the spine that fractures or dislocates vertebrae. The damage begins at the moment of injury when displaced bone fragments, disc material, or ligamen...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is closely linked to spina bifida and results from improper growth of the neural tube during fetal development. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to the degree of strain on the spinal cord over time and may

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.712, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000034-9` (score=0.0009) - Focal neuropathy appears suddenly and affects specific nerves, most often in the head, torso, or leg. Focal neuropathy may cause - inability to focus the eye - double vision - aching behind one eye - paralysis on one ...
2. `medquad_ans_0000136-5__dup2` (score=0.0001) - The symptoms of dialysis-related amyloidosis may include - pain, stiffness, and fluid in the joints. - abnormal, fluid-containing sacs, called cysts, in some bones. - carpal tunnel syndrome, caused by unusual buildup ...
3. `medquad_ans_0000058-1__dup1` (score=0.0001) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
4. `medquad_ans_0000034-2__dup4` (score=0.0001) - The causes are probably different for different types of diabetic neuropathy. Researchers are studying how prolonged exposure to high blood glucose causes nerve damage. Nerve damage is likely due to a combination of f...
5. `medquad_ans_0000419-1__dup1` (score=0.0001) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...

**Answer**:

The provided context does not contain information about the outlook or prognosis for carpal tunnel syndrome. While it describes symptoms and contributing factors, there is no mention of recovery timeline, long-term outcomes, or treatment effectiveness. Therefore, based on the given context, the outlook for carpal tunnel syndrome cannot be determined. [medquad_ans_0000136-5\_\_dup2][medquad_ans_0000058-1__dup1]

**Supported citations**: `medquad_ans_0000136-5__dup2`, `medquad_ans_0000058-1__dup1`
**Unsupported citations**: _none_

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.809, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000296-5` (score=0.9939) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphrag...
2. `medquad_ans_0004451-3` (score=0.0057) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004454-3` (score=0.0057) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004456-3` (score=0.0057) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004450-4` (score=0.0057) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain specific information about the treatments for Donnai-Barrow syndrome. While resources for diagnosis and management of Donnai-Barrow syndrome are listed, such as Gene Review, Genetic Testing Registry, and MedlinePlus Encyclopedia entries on related conditions (e.g., diaphragmatic hernia, hearing loss, omphalocele), no explicit treatments are described. Therefore, based on the available context, the treatments for Donnai-Barrow syndrome cannot be determined.

[medquad_ans_0000296-5]

**Supported citations**: `medquad_ans_0000296-5`
**Unsupported citations**: _none_

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=0.750, groundedness=0.517, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000189-3__dup2` (score=0.6440) - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Experts often classify porphyrias as acute or cutaneous based on the symptoms a person experiences:...
2. `medquad_ans_0000729-1__dup1` (score=0.4636) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. Th...
3. `medquad_ans_0000811-1` (score=0.3049) - Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marr...
4. `medquad_ans_0004987-1` (score=0.0456) - Porphyria cutanea tarda (PCT) is a form of porphyria that primarily affects the skin. People affected by this condition generally experience "photosensitivity," which causes painful, blistering lesions to develop on s...
5. `medquad_ans_0000117-1__dup1` (score=0.0233) - Acute intermittent porphyria (AIP) is one of the liver (hepatic) porphyrias. AIP is caused by low levels of porphobilinogen deaminase (PBGD), an enzyme also often called hydroxymethylbilane synthase. The low levels of...

**Answer**:

Porphyria is a group of genetic disorders caused by problems with how the body makes a substance called heme. Heme is found throughout the body, especially in the blood and bone marrow, where it carries oxygen. There are two main types of porphyrias: one that affects the skin and another that affects the nervous system. Skin-type porphyrias cause blisters, itching, and swelling of the skin when exposed to sunlight. The nervous system type, known as acute porphyria, presents with symptoms such as pain in the chest, abdomen, limbs, or back; muscle numbness, tingling, paralysis, or cramping; vomiting; constipation; and personality changes or mental disorders. These symptoms come and go and can be triggered by factors such as certain medicines, smoking, alcohol, infections, stress, and sun exposure. Attacks develop over hours or days and can last for days or weeks. Diagnosis requires blood, urine, and stool

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=0.833, groundedness=0.986, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0001900-1` (score=0.9606) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual...
2. `medquad_ans_0000585-1__dup1` (score=0.9521) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
3. `medquad_ans_0000674-1__dup1` (score=0.0385) - Behr syndrome is a disorder mainly characterized by early-onset optic atrophy, ataxia, and spasticity. Other signs and symptoms may be present and vary from person to person. Although the exact cause is unknown, the s...
4. `medquad_ans_0003414-1` (score=0.0250) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, whic...
5. `medquad_ans_0002854-1` (score=0.0006) - Hereditary sensory neuropathy type IE (HSNIE) is a progressive disorder of the central and peripheral nervous systems. Symptoms typically begin by age 20 to 35 and include sensory impairment of the lower legs and feet...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is suggested to be inherited in an autosomal recessive manner [medquad_ans_0000585-1__dup1].

**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.921, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000794-3` (score=0.9991) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process...
2. `medquad_ans_0000795-3` (score=0.9983) - Phosphoglycerate mutase deficiency is caused by mutations in the PGAM2 gene. This gene provides instructions for making an enzyme called phosphoglycerate mutase, which is involved in a critical energy-producing proces...
3. `medquad_ans_0000856-3` (score=0.0254) - Pyruvate kinase deficiency is caused by mutations in the PKLR gene. The PKLR gene is active in the liver and in red blood cells, where it provides instructions for making an enzyme called pyruvate kinase. The pyruvate...
4. `medquad_ans_0000770-3` (score=0.0010) - Mutations in the PANK2 gene cause pantothenate kinase-associated neurodegeneration. The PANK2 gene provides instructions for making an enzyme called pantothenate kinase 2. This enzyme is active in mitochondria, the en...
5. `medquad_ans_0000652-3` (score=0.0007) - Mutations in the MVK gene cause mevalonate kinase deficiency. The MVK gene provides instructions for making the mevalonate kinase enzyme. This enzyme is involved in the production of cholesterol, which is later conver...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene [medquad_ans_0000794-3].

**Supported citations**: `medquad_ans_0000794-3`
**Unsupported citations**: _none_

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.900, groundedness=0.821, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000608-3` (score=0.0125) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...
2. `medquad_ans_0000849-3` (score=0.0008) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in thi...
3. `medquad_ans_0000142-3` (score=0.0002) - C3 glomerulopathy is associated with changes in many genes. Most of these genes provide instructions for making proteins that help regulate a part of the body's immune response known as the complement system. This sys...
4. `medquad_ans_0001041-3` (score=0.0002) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000832-3` (score=0.0001) - In most cases, the genetic cause of progressive supranuclear palsy is unknown. Rarely, the disease results from mutations in the MAPT gene. Certain normal variations (polymorphisms) in the MAPT gene have also been ass...

**Answer**:

The provided context does not mention genetic changes related to pulmonary veno-occlusive disease. [medquad_ans_0000608-3] discusses genetic mutations in lung cancer, such as in the EGFR and KRAS genes, and mutations in the BMPR2 gene in pulmonary arterial hypertension, but it does not address pulmonary veno-occlusive disease. Therefore, based on the given information, the genetic changes related to pulmonary veno-occlusive disease cannot be determined.

**Supported citations**: `medquad_ans_0000608-3`
**Unsupported citations**: _none_

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=1.000, groundedness=0.732, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
2. `medquad_ans_0001006-2` (score=0.0001) - Worldwide, tyrosinemia type I affects about 1 in 100,000 individuals. This type is more common in Norway where 1 in 60,000 to 74,000 individuals are affected. Tyrosinemia type I is even more common in Quebec, Canada w...
3. `medquad_ans_0001006-1` (score=0.0001) - Tyrosinemia is a genetic disorder characterized by disruptions in the multistep process that breaks down the amino acid tyrosine, a building block of most proteins. If untreated, tyrosine and its byproducts build up i...
4. `medquad_ans_0006260-1` (score=0.0001) - Tyrosinemia type 1 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a shortage of the enzyme fumarylacetoacetate h...
5. `medquad_ans_0003520-3` (score=0.0001) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...

**Answer**:

The provided context does not specify the exact genetic changes related to tyrosinemia. It describes the prevalence and symptoms of the three types of tyrosinemia but does not detail the specific genetic mutations or alterations responsible for the disorder. Therefore, based on the given information, the genetic changes related to tyrosinemia cannot be determined. [medquad_ans_0001006-1] [medquad_ans_0001006-2]

**Supported citations**: `medquad_ans_0001006-1`, `medquad_ans_0001006-2`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=1.000, groundedness=0.822, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000084-2__dup1` (score=0.8171) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
2. `medquad_ans_0005174-2` (score=0.0001) - What treatments are available for pudendal neuralgia? There are no established guidelines for the treatment of pudendal neuralgia. Treatment may include medications, the use of a specialized cushion while sitting, ner...
3. `medquad_ans_0002796-4` (score=0.0001) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
4. `medquad_ans_0002675-2` (score=0.0001) - What are the signs and symptoms of Graves' disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Graves' disease. If the information is available, the table below includes how oft...
5. `medquad_ans_0001354-2` (score=0.0001) - What treatments for metastatic clear cell renal cell carcinoma are available in North America? There are several treatments for metastatic clear cell renal cell carcinoma available in North America. IL-2 and sunitinib...

**Answer**:

The provided context does not specify treatments for Graves' disease. While it lists signs and symptoms of Graves' disease, including hyperactivity, weight loss, goiter, and proptosis, there is no information about available treatments [medquad_ans_0002796-4]. Therefore, based on the given context, the treatments for Graves' disease cannot be determined.

**Supported citations**: `medquad_ans_0002796-4`
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.478, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.2434) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000039-3__dup1` (score=0.0037) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
3. `medquad_ans_0000039-4__dup1` (score=0.0015) - The biochemical defects that underlie several NCLs have recently been discovered. An enzyme called palmitoyl-protein thioesterase has been shown to be insufficiently active in the infantile form of Batten disease (thi...
4. `medquad_ans_0000545-2` (score=0.0009) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, wher...
5. `medquad_ans_0000039-2__dup1` (score=0.0007) - As yet, no specific treatment is known that can halt or reverse the symptoms of Batten disease. However, seizures can sometimes be reduced or controlled with anticonvulsant drugs, and other medical problems can be tre...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling, and vision loss is often an early symptom that may lead to suspicion during an eye exam. The disease progresses over time, leading to cognitive impairment, worsening seizures, progressive loss of sight and motor skills, and eventually blindness, bedridden state, and dementia. It is often fatal by the late teens or twenties. The biochemical defects in Batten disease include insufficient activity of the enzyme palmitoyl-protein thioesterase in the infantile form (CLN1), a deficiency of an acid protease in the late infantile form (CLN2), and a mutated gene in juvenile Batten disease (CLN3), although

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=1.000, groundedness=0.993, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002061-1` (score=0.5247) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
2. `medquad_ans_0000313-1` (score=0.0070) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
3. `medquad_ans_0000313-2` (score=0.0002) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...
4. `medquad_ans_0002061-5` (score=0.0001) - How might Ehlers-Danlos syndrome progeroid type be treated? Individuals with Ehlers-Danlos Syndrome progeroid type can benefit from a variety of treatments depending on their symptoms. Affected children with weak musc...
5. `medquad_ans_0002061-3` (score=0.0001) - What causes Ehlers-Danlos syndrome progeroid type? Ehlers-Danlos syndrome progeroid type is caused by changes (mutations) in both of an individual's copies of the B4GALT7 gene, which is located on chromosome 5. This g...

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a genetic disorder of the connective tissue that primarily affects the skin, hair, and skeletal system. Symptoms usually appear by childhood or adolescence and include unusually flexible joints, loose elastic skin, and easy scarring. Unique features of this type include sparse scalp hair and eyebrows, and loose elastic skin on the face, which cause affected individuals to look older than their age. Additional symptoms may include bone weakness, weak muscle tone, mild intellectual disability, and delayed growth in affected children. This type is caused by mutations in the B4GALT7 gene and is inherited in an autosomal recessive pattern [medquad_ans_0002061-1].

**Supported citations**: `medquad_ans_0002061-1`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.942, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.9991) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000416-2` (score=0.0001) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
3. `medquad_ans_0002601-1` (score=0.0001) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0000416-1` (score=0.0001) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...
5. `medquad_ans_0004664-2` (score=0.0001) - What causes an ovarian sex cord tumor with annular tubules? Approximately one third of ovarian sex cord tumors with annual tubules (SCTATs) develop because of an underlying genetic condition called Peutz Jeghers syndr...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the metabolism of the amino acid histidine and in the production of folate [medquad_ans_0000416-3]. The specific relationship between these genetic changes and the severity of the condition (mild or severe) is unclear [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001169-2` (score=0.9994) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
2. `medquad_ans_0001169-1` (score=0.0001) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
3. `medquad_ans_0004472-1` (score=0.0001) - What are the signs and symptoms of Nystagmus 1, congenital, X- linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Nystagmus 1, congenital, X- linked. If the information is avail...
4. `medquad_ans_0001170-1` (score=0.0001) - X-linked dominant chondrodysplasia punctata (CDPX2), also known as Conradi-Hnermann-Happle syndrome, is a rare form of skeletal dysplasia characterized by skeletal malformations, skin abnormalities, cataracts and shor...
5. `medquad_ans_0001047-1` (score=0.0001) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include abnormality of the vertebral column, anosmia, cataract, depressed nasal bridge, epiphyseal stippling, hearing impairment, hypogonadism, ichthyosis, microcephaly, short distal phalanx of finger, short nasal septum, short nose, short stature, and X-linked recessive inheritance [medquad_ans_0001169-2]. Additional common features include shortened fingers, a flat nose, breathing abnormalities, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development [medquad_ans_0001169-1]. The condition is caused by changes in the ARSE gene on the X chromosome and is inherited in an X-linked recessive manner, occurring almost exclusively in males [medquad_ans_000

**Supported citations**: `medquad_ans_0001169-2`, `medquad_ans_0001169-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=1.000, groundedness=0.761, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000132-7` (score=0.0006) - A health care provider may use several tests and exams to diagnose Whipple disease, including the following: - medical and family history - physical exam - blood tests - upper GI endoscopy and enteroscopy A patient ma...
2. `medquad_ans_0002796-4` (score=0.0002) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...
3. `medquad_ans_0005174-2` (score=0.0002) - What treatments are available for pudendal neuralgia? There are no established guidelines for the treatment of pudendal neuralgia. Treatment may include medications, the use of a specialized cushion while sitting, ner...
4. `medquad_ans_0004947-4` (score=0.0002) - What treatments are available for itching related to polycythemia vera? There are several treatments for the itching (pruritus) related to polycythemia vera (PV). No single treatment has been found to be effective for...
5. `medquad_ans_0000132-5__dup1` (score=0.0002) - Signs and symptoms of Whipple disease can vary widely from person to person. The most common symptoms of Whipple disease are - diarrhea - weight loss caused by malabsorption A person may not have diarrhea. Instead, ot...

**Answer**:

The provided context does not mention any treatments for Whipple disease. Therefore, it is not possible to answer the question based on the given information. [medquad_ans_0000132-7]

**Supported citations**: `medquad_ans_0000132-7`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=1.000, groundedness=0.707, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000676-4__dup1` (score=0.9747) - How might Bell's palsy be treated?
2. `medquad_ans_0000676-2__dup1` (score=0.0061) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0005077-3` (score=0.0034) - How is progressive bulbar palsy treated? Treatments aim to help people cope with the symptoms of progressive bulbar palsy, such as feeding tubes, devices to help with talking, and medicines to treat muscle spasms, wea...
4. `medquad_ans_0005695-6` (score=0.0003) - How might spastic diplegia cerebral palsy be treated? Treatment of spastic diplegia cerebral palsy varies based on the signs and symptoms present in each person and the severity of the condition. Affected people are o...
5. `medquad_ans_0000042-4__dup2` (score=0.0002) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge of the brain and nervous system and to use that knowledge to reduce the burden of neurological diseas...

**Answer**:

The provided context does not contain information on the treatments for Bell's palsy. [medquad_ans_0000676-4__dup1] mentions how Bell's palsy might be treated, but the specific treatments are not detailed in the given text. Therefore, based on the available context, the treatments for Bell's palsy cannot be determined.

**Supported citations**: `medquad_ans_0000676-4__dup1`
**Unsupported citations**: _none_
