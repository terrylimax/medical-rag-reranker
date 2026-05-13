# Generation Evaluation

## Summary

- `avg_answer_chars`: 576.2500
- `avg_answer_relevance`: 0.7698
- `avg_citation_presence_rate`: 0.8133
- `avg_context_relevance`: 0.9060
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 9197.1970
- `avg_generation_latency_ms`: 5745.4948
- `avg_groundedness`: 0.7776
- `avg_insufficient_context`: 0.0367
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 3293.7230
- `avg_retrieval_latency_ms`: 157.6921
- `avg_supported_citation_rate`: 0.7317
- `avg_unsupported_citation_rate`: 0.0817
- `generation_remote_concurrency`: 4.0000
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
3. `medquad_ans_0000129-4` (score=0.0007) - Blau syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most affected individuals have one parent with the condition. In ...
4. `medquad_ans_0000311-4` (score=0.0001) - Early-onset glaucoma can have different inheritance patterns. Primary congenital glaucoma is usually inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. Most o...
5. `medquad_ans_0000112-1__dup1` (score=0.0001) - The dystonias are movement disorders in which sustained muscle contractions cause twisting and repetitive movements or abnormal postures. The movements, which are involuntary and sometimes painful, may affect a single...

**Answer**:

Yes, early-onset primary dystonia is inherited. Everyone who has been diagnosed with early-onset primary dystonia has inherited a TOR1A mutation from one parent, and the mutation in the TOR1A gene is inherited in an autosomal dominant pattern [medquad_ans_0000312-4].

**Supported citations**: `medquad_ans_0000312-4`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=1.000, groundedness=0.893, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000397-2__dup1` (score=0.9876) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...
2. `medquad_ans_0000397-1__dup2` (score=0.0824) - Taeniasis in humans is a parasitic infection caused by the tapeworm species Taenia saginata (beef tapeworm), Taenia solium (pork tapeworm), and Taenia asiatica (Asian tapeworm). Humans can become infected with these t...
3. `medquad_ans_0000397-7` (score=0.0109) - One way to prevent taeniasis is to cook meat to safe temperatures. A food thermometer should be used to measure the internal temperature of cooked meat. Do not sample meat until it is cooked. USDA recommends the follo...
4. `medquad_ans_0000121-3__dup2` (score=0.0032) - Cysticercosis is an infection caused by the larvae of the tapeworm, Taenia solium. A person with an adult tapeworm, which lives in the person’s gut, sheds eggs in the stool. The infection with the adult tapeworm is ca...
5. `medquad_ans_0000344-1__dup2` (score=0.0019) - Baylisascaris worms are intestinal parasites found in a wide variety of animals. Different species of Baylisascaris are associated with different animal hosts. For example, Baylisascaris procyonis is found in raccoons...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef or pork. Specifically, individuals in under-developed communities with poor sanitation, as well as those in regions where contaminated raw beef or pork is consumed, such as Eastern Europe, Russia, eastern Africa, Latin America, sub-Saharan Africa, India, and Asia, are at higher risk. In the United States, taeniasis is more common among Latin American immigrants, particularly those who consume raw or undercooked pork or beef [medquad_ans_0000397-2\_\_dup1][medquad_ans_0000397-1__dup2]. Additionally, people with poor hygiene who have taeniasis may shed tapeworm eggs in their feces, potentially contaminating their environment and increasing the risk of transmission to themselves or others [medquad_ans_0000397-2\_\_dup1

**Supported citations**: `medquad_ans_0000397-2__dup1`, `medquad_ans_0000397-1__dup2`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.9994) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0000029-4` (score=0.0001) - Aicardi-Goutieres syndrome can have different inheritance patterns. In most cases it is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an i...
3. `medquad_ans_0000193-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000194-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000195-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.9993) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0000005-6__dup1` (score=0.0001) - Clinical trials are part of clinical research and at the heart of all treatment advances. Clinical trials look at new ways to prevent, detect, or treat disease. The National Institute of Mental Health at NIH supports ...
3. `medquad_ans_0001404-4` (score=0.0001) - How is Coffin-Siris syndrome diagnosed? Diagnosis of Coffin-Siris syndrome is largely based upon the presence or absence of common signs and symptoms in the individual. While formal diagnostic criteria have not been e...
4. `medquad_ans_0000593-1__dup1` (score=0.0001) - Axenfeld-Rieger syndrome is a group of eye disorders that affects the development of the eye. Common eye symptoms include cornea defects, which is the clear covering on the front of the eye, and iris defects, which is...
5. `medquad_ans_0000594-1__dup1` (score=0.0001) - Axenfeld-Rieger syndrome is a group of eye disorders that affects the development of the eye. Common eye symptoms include cornea defects, which is the clear covering on the front of the eye, and iris defects, which is...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=1.000, groundedness=0.937, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000450-4` (score=0.9842) - Hemophilia A and hemophilia B are inherited in an X-linked recessive pattern. The genes associated with these conditions are located on the X chromosome, which is one of the two sex chromosomes. In males (who have onl...
2. `medquad_ans_0000069-3__dup1` (score=0.0040) - A defect in one of the genes that determines how the body makes blood clotting factor VIII or IX causes hemophilia. These genes are located on the X chromosomes (KRO-muh-somz). Chromosomes come in pairs. Females have ...
3. `medquad_ans_0000102-1__dup3` (score=0.0001) - Normally, if you get hurt, your body forms a blood clot to stop the bleeding. For blood to clot, your body needs cells called platelets and proteins known as clotting factors. If you have a bleeding disorder, you eith...
4. `medquad_ans_0002788-1` (score=0.0001) - Hemophilia B is a bleeding disorder that slows the blood clotting process. People with this disorder experience prolonged bleeding or oozing following an injury or surgery. In severe cases of hemophilia, heavy bleedin...
5. `medquad_ans_0000069-5__dup1` (score=0.0001) - If you or your child appears to have a bleeding problem, your doctor will ask about your personal and family medical histories. This will reveal whether you or your family members, including women and girls, have blee...

**Answer**:

Yes, hemophilia is inherited. Hemophilia A and hemophilia B are inherited in an X-linked recessive pattern, meaning the genes associated with these conditions are located on the X chromosome [medquad_ans_0000450-4]. In males, who have only one X chromosome, one altered copy of the gene is sufficient to cause the condition. In females, who have two X chromosomes, a mutation would need to occur in both copies of the gene to cause the disorder, making it very rare for females to have hemophilia [medquad_ans_0000450-4]. The condition can be passed from mother to child, with carrier females having a 50% chance of passing the gene to their children [medquad_ans_0000069-3__dup1]. In some cases, mutations can occur randomly in the gene as it is passed to a child, even if

**Supported citations**: `medquad_ans_0000450-4`, `medquad_ans_0000069-3__dup1`
**Unsupported citations**: _none_

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.9994) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0000005-6__dup1` (score=0.0001) - Clinical trials are part of clinical research and at the heart of all treatment advances. Clinical trials look at new ways to prevent, detect, or treat disease. The National Institute of Mental Health at NIH supports ...
3. `medquad_ans_0000594-1__dup1` (score=0.0001) - Axenfeld-Rieger syndrome is a group of eye disorders that affects the development of the eye. Common eye symptoms include cornea defects, which is the clear covering on the front of the eye, and iris defects, which is...
4. `medquad_ans_0000593-1__dup1` (score=0.0001) - Axenfeld-Rieger syndrome is a group of eye disorders that affects the development of the eye. Common eye symptoms include cornea defects, which is the clear covering on the front of the eye, and iris defects, which is...
5. `medquad_ans_0000610-1__dup1` (score=0.0001) - Baller-Gerold syndrome is a rare condition characterized by the premature fusion of certain skull bones (craniosynostosis) and abnormalities of bones in the arms and hands, sometimes referred to as radial ray anomalie...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=0.833, groundedness=0.531, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004747-1` (score=0.9758) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People ...
2. `medquad_ans_0000775-1` (score=0.6112) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
3. `medquad_ans_0000093-5` (score=0.0009) - These resources address the diagnosis or management of ADNFLE: - Gene Review: Gene Review: Autosomal Dominant Nocturnal Frontal Lobe Epilepsy - Genetic Testing Registry: Epilepsy, nocturnal frontal lobe, type 1 - Gene...
4. `medquad_ans_0000775-2` (score=0.0005) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.
5. `medquad_ans_0000363-5` (score=0.0001) - These resources address the diagnosis or management of familial paroxysmal kinesigenic dyskinesia: - Gene Review: Gene Review: Familial Paroxysmal Kinesigenic Dyskinesia - Genetic Testing Registry: Dystonia 10 These r...

**Answer**:

Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. The disorder is characterized by sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to a deficiency of red blood cells (hemolytic anemia) and resulting in signs and symptoms such as fatigue, weakness, abnormally pale skin (pallor), shortness of breath, and an increased heart rate. Affected individuals may pass dark-colored urine due to the presence of hemoglobin, a

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.934, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.6004) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
2. `medquad_ans_0000221-1__dup2` (score=0.4409) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...
3. `medquad_ans_0000037_2-1` (score=0.0423) - Key Points - Rectal cancer is a disease in which malignant (cancer) cells form in the tissues of the rectum. - Health history affects the risk of developing rectal cancer. - Signs of rectal cancer include a change in ...
4. `medquad_ans_0000010-17` (score=0.0222) - Parents, siblings, or children of a person who has had colorectal cancer are somewhat more likely to develop this type of cancer themselves. This is especially true if the relative had the cancer at a young age. If ma...
5. `medquad_ans_0000037_4-1` (score=0.0212) - Key Points - Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or the rectum. - Colorectal cancer is the second leading cause of death from cancer in the United States. ...

**Answer**:

Colorectal cancer is a type of cancer that occurs when malignant tumors form in the lining of the large intestine, also known as the large bowel. The colon and rectum are part of the large intestine, which is part of the digestive system [medquad_ans_0000010-10]. It is common in both men and women and the risk of developing it rises after age 50 [medquad_ans_0000221-1__dup2]. Risk factors include having colorectal polyps, a family history of colorectal cancer, inflammatory bowel diseases such as ulcerative colitis or Crohn's disease, a diet high in fat, and smoking [medquad_ans_0000221-1__dup2]. Symptoms may include diarrhea or constipation, a feeling that the bowel does not empty completely, blood in the stool (either bright red or very dark), narrow

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

1. `medquad_ans_0000756-1` (score=0.3175) - Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The...
2. `medquad_ans_0000755-1` (score=0.0462) - Ornithine transcarbamylase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high...
3. `medquad_ans_0000755-4` (score=0.0067) - Ornithine transcarbamylase deficiency is an X-linked disorder. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes. A characte...
4. `medquad_ans_0000968-1__dup1` (score=0.0005) - Carnitine-acylcarnitine translocase deficiency is a condition that prevents the body from converting certain fats called long-chain fatty acids into energy, particularly during periods without food (fasting). Carnitin...
5. `medquad_ans_0000755-3` (score=0.0003) - Mutations in the OTC gene cause ornithine transcarbamylase deficiency. Ornithine transcarbamylase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reaction...

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

**Scores**: context_relevance=0.800, groundedness=0.984, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000258-2__dup3` (score=0.9836) - LCMV is most commonly recognized as causing neurological disease, as its name implies, though infection without symptoms or mild febrile illnesses are more common clinical manifestations. For infected persons who do b...
2. `medquad_ans_0004105-1` (score=0.8755) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
3. `medquad_ans_0000645-1__dup2` (score=0.0814) - The brain, spinal cord, and nerves make up the nervous system. Together they control all the workings of the body. When something goes wrong with a part of your nervous system, you can have trouble moving, speaking, s...
4. `medquad_ans_0000416-1__dup1` (score=0.0121) - Haemophilus is the name of a group of bacteria. There are several types of Haemophilus. They can cause different types of illnesses involving breathing, bones and joints, and the nervous system. One common type, Hib (...
5. `medquad_ans_0000115-4__dup1` (score=0.0039) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience other symptoms [medquad_ans_0004105-1].

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003319-3` (score=0.9871) - What causes juvenile amyotrophic lateral sclerosis? Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases the underlying gene abnormality...
2. `medquad_ans_0003331-3` (score=0.9829) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
3. `medquad_ans_0000309-3__dup1` (score=0.4638) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the...
4. `medquad_ans_0000551-4` (score=0.0001) - When caused by mutations in the ALS2 gene, juvenile primary lateral sclerosis is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individu...
5. `medquad_ans_0000551-5` (score=0.0001) - These resources address the diagnosis or management of juvenile primary lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Genetic Testing Registry: Juvenile primary lateral sclerosis These resour...

**Answer**:

Juvenile amyotrophic lateral sclerosis (ALS) is often genetic and may be caused by mutations in the ALS2 or SETX genes. In some cases, the underlying gene abnormality cannot be determined. Juvenile ALS may be inherited in an autosomal dominant (as in ALS type 4) or autosomal recessive (as in ALS type 2) fashion [medquad_ans_0003319-3].

**Supported citations**: `medquad_ans_0003319-3`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=1.000, groundedness=0.802, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000060-10` (score=0.0046) - You can keep your nervous system healthy by taking these steps: - Eat healthy meals and follow the meal plan that you and your doctor or dietitian have worked out. - If you drink alcoholic beverages, limit your intake...
2. `medquad_ans_0000219-4__dup3` (score=0.0007) - Erectile dysfunction affects men of all races and in all parts of the world. Men are more likely to have ED as they get older. For example, ED occurs in - about 12 percent of men younger than 60 - 22 percent of men ag...
3. `medquad_ans_0000193-6` (score=0.0006) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
4. `medquad_ans_0000591-1__dup1` (score=0.0003) - Summary : Medicines cure infectious diseases, prevent problems from chronic diseases, and ease pain. But medicines can also cause harmful reactions if not used correctly. Errors can happen in the hospital, at the doct...
5. `medquad_ans_0000032-3__dup2` (score=0.0003) - Don't Ignore Hearing Problems Some people have a hearing problem without realizing it. Others might think they have a problem, but are too embarrassed to tell their doctor, friends, or family. You can help identify a ...

**Answer**:

The provided context does not include specific information or recommendations on what to do for "What I need to know about Erectile Dysfunction." While it states that erectile dysfunction (ED) affects men of all races and ages, with higher prevalence in older men, and that ED occurs in 12% of men younger than 60, 22% of men aged 60 to 69, and 30% of men age 70 or older, there are no actionable steps or resources directed toward managing or understanding ED in detail [medquad_ans_0000219-4__dup3].

Therefore, based on the given context, it is not possible to provide specific guidance on "What I need to know about Erectile Dysfunction."

**Supported citations**: `medquad_ans_0000219-4__dup3`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.962, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.9702) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0000202-5` (score=0.0019) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Regist...
3. `medquad_ans_0000576-5` (score=0.0002) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
4. `medquad_ans_0000395-5` (score=0.0001) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
5. `medquad_ans_0001026-5` (score=0.0001) - These resources address the diagnosis or management of Walker-Warburg syndrome: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Genetic Testing Registry: Walker-Warburg congenital muscular dystrop...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often become necessary in the first or second decade of life. Prevention of chest infections may be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that people with Ullrich congenital muscular dystrophy may benefit from cyclosporin A, although more studies are needed to confirm its benefits [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=1.000, groundedness=0.783, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002834-4` (score=0.0002) - How is hereditary lymphedema type II inherited? Hereditary lymphedema type II appears to have an autosomal dominant pattern of inheritance, which means that one copy of an altered gene in each cell is sufficient to ca...
2. `medquad_ans_0004005-4` (score=0.0001) - How is microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) inherited? MOPD1 is thought to be inherited in an autosomal recessive manner. This means that affected individuals have abnormal gene changes (mu...
3. `medquad_ans_0000653-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcep...
4. `medquad_ans_0004006-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
5. `medquad_ans_0002973-1` (score=0.0001) - HEM (hydrops fetalis, ectopic calcifications, "moth-eaten" skeletal dysplasia) is a very rare type of lethal skeletal dysplasia. According to the reported cases of HEM in the medical literature, the condition's main f...

**Answer**:

The provided context does not specify the inheritance pattern of microcephalic osteodysplastic primordial dwarfism type II (MOPDII). While it describes the clinical features of MOPDII, such as short stature, microcephaly, and skeletal abnormalities, there is no information on how it is inherited.

Therefore, based on the given context, the inheritance pattern of MOPDII cannot be determined. [medquad_ans_0000653-1]

**Supported citations**: `medquad_ans_0000653-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.038, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002561-3` (score=0.3553) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
2. `medquad_ans_0000527-4` (score=0.0262) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
3. `medquad_ans_0001084-3__dup1` (score=0.0030) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
4. `medquad_ans_0003718-4` (score=0.0022) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
5. `medquad_ans_0000553-4` (score=0.0004) - When Kabuki syndrome is caused by mutations in the KMT2D gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. When Kabuki s...

**Answer**:

The provided context does not contain any information about Ochoa syndrome or its inheritance pattern. Therefore, it is not possible to determine based on the given text whether Ochoa syndrome is inherited.

Answer: Insufficient context [no_doc_id].

**Supported citations**: _none_
**Unsupported citations**: `no_doc_id`

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.810, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.9960) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
2. `medquad_ans_0000531-2` (score=0.0001) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
3. `medquad_ans_0000531-3` (score=0.0001) - Isodicentric chromosome 15 syndrome results from the presence of an abnormal extra chromosome, called an isodicentric chromosome 15, in each cell. An isodicentric chromosome contains mirror-image segments of genetic m...
4. `medquad_ans_0006554-1` (score=0.0001) - 15q11.2 microdeletion refers to a chromosome abnormality in which a tiny piece of genetic material on the long arm of chromosome 15 (at a location designated q11.2) is missing (deleted). The features of people with a ...
5. `medquad_ans_0006555-1` (score=0.0001) - 15q13.3 microdeletion syndrome is a type of contiguous gene deletion syndrome. Individuals with this microdeletion may have very different signs and symptoms from other affected individuals (even within the same famil...

**Answer**:

The provided context does not specify treatments for isodicentric chromosome 15 syndrome. While it mentions resources such as genetic counseling, drug therapy, surgery and rehabilitation, palliative care, and diagnostic tests as available for conditions like autism, these are general resources from MedlinePlus and not specifically tied to the treatment of isodicentric chromosome 15 syndrome [medquad_ans_0000531-5]. Therefore, based on the given information, the specific treatments for isodicentric chromosome 15 syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.887, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=0.9931) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0002414-3` (score=0.0014) - What causes Fowlers syndrome? The cause of Fowler's syndrome is not known. The association of Fowlers syndrome and polycystic ovaries in some patients raises the possibility that the syndrome is linked in some way to ...
3. `medquad_ans_0003414-3` (score=0.0004) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
4. `medquad_ans_0000272-4__dup1` (score=0.0002) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cau...
5. `medquad_ans_0003459-3` (score=0.0001) - What causes Lambert Eaton myasthenic syndrome? Lambert Eaton myasthenic syndrome is the result of an autoimmune process which causes a disruption of electrical impulses between nerve cells and muscle fibers. In cases ...

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene. This gene provides instructions for making a protein called spartin, which is present in many body tissues, including those of the nervous system. It is thought that the syndrome results from a loss of function of the spartin protein, although the exact role of this protein in cellular functions is not fully understood [medquad_ans_0006224-3].

**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.921, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000983-3` (score=0.9990) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
2. `medquad_ans_0000290-3` (score=0.0004) - Distal arthrogryposis type 1 can be caused by mutations in at least two genes: TPM2 and MYBPC1. These genes are active (expressed) in muscle cells, where they interact with other muscle proteins to help regulate the t...
3. `medquad_ans_0000869-3` (score=0.0004) - Restless legs syndrome likely results from a combination of genetic and environmental factors, many of which are unknown. Studies suggest that restless legs syndrome is related to a shortage (deficiency) of iron in ce...
4. `medquad_ans_0000592-3` (score=0.0002) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
5. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...

**Answer**:

Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. These mutations change the structure of CaV1.2 channels, causing them to stay open much longer than usual, which leads to an abnormal influx of calcium ions into cells. This calcium overload alters the way cardiac muscle cells beat and can cause arrhythmia [medquad_ans_0000983-3].

**Supported citations**: `medquad_ans_0000983-3`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000061-3__dup3` (score=0.9930) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
2. `medquad_ans_0000061-11` (score=0.9910) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your d...
3. `medquad_ans_0000061-10` (score=0.0139) - Rheumatoid arthritis is characterized by inflammation of the joint lining. This inflammation causes warmth, redness, swelling, and pain around the joints. The pain of rheumatoid arthritis varies greatly from person to...
4. `medquad_ans_0000061-13` (score=0.0024) - Medication, exercise, and, in some cases, surgery are common treatments for this disease. Most people who have rheumatoid arthritis take medications. Some drugs only provide relief for pain; others reduce inflammation...
5. `medquad_ans_0000061-4__dup3` (score=0.0007) - Most Symptoms Are Treatable Doctors use a variety of approaches to treat rheumatoid arthritis. The goals of treatment are to help relieve pain, reduce swelling, slow down or help prevent joint damage, increase the abi...

**Answer**:

The symptoms of rheumatoid arthritis include pain and stiffness in the joints, warmth, redness, swelling, and pain around the joints. People also often feel sick, tired, and may experience fever. Rheumatoid arthritis generally occurs in a symmetrical pattern, meaning if one knee or hand is affected, the other is also likely to be affected [medquad_ans_0000061-3__dup3].

**Supported citations**: `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=0.800, groundedness=0.522, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.9718) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.9287) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0003148-1` (score=0.0017) - Idiopathic thrombocytopenic purpura (ITP) is a bleeding disorder characterized by too few platelets in the blood. This is because platelets are being destroyed by the immune system. Symptoms may include bruising, nose...
4. `medquad_ans_0000131-8` (score=0.0012) - Both inherited and acquired thrombotic thrombocytopenic purpura (TTP) occur suddenly with no clear cause. You can't prevent either type. If you've had TTP, watch for signs and symptoms of a relapse (flareup). (For mor...
5. `medquad_ans_0000837-1` (score=0.0011) - Protein C deficiency is a disorder that increases the risk of developing abnormal blood clots; the condition can be mild or severe. Individuals with mild protein C deficiency are at risk of a type of blood clot known ...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It may develop as an immune response to an infection, though the exact cause is not fully understood. HSP is usually seen in children, but it may affect people of any age. Most cases go away on their own without treatment, and the main goal of treatment, when needed, is to relieve symptoms such as joint pain, abdominal pain, or swelling. In some patients with severe arthritis, prednisone, a steroid medicine, may be prescribed. The primary symptom is a rash that looks like many small raised bruises, caused by inflammation and leakage from small blood vessels in the body. HSP can also affect the kidneys, digestive tract, and joints. Most people recover completely, though kidney damage is the most likely

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

**Scores**: context_relevance=0.889, groundedness=0.959, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.9991) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000142-3` (score=0.0002) - C3 glomerulopathy is associated with changes in many genes. Most of these genes provide instructions for making proteins that help regulate a part of the body's immune response known as the complement system. This sys...
3. `medquad_ans_0001041-3` (score=0.0002) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
4. `medquad_ans_0000290-3` (score=0.0002) - Distal arthrogryposis type 1 can be caused by mutations in at least two genes: TPM2 and MYBPC1. These genes are active (expressed) in muscle cells, where they interact with other muscle proteins to help regulate the t...
5. `medquad_ans_0001078-3` (score=0.0002) - 1p36 deletion syndrome is caused by a deletion of genetic material from a specific region in the short (p) arm of chromosome 1. The signs and symptoms of 1p36 deletion syndrome are probably related to the loss of mult...

**Answer**:

The genetic changes related to Senior-Lken syndrome are mutations in one of at least five genes. These mutations are believed to affect proteins involved in the structure and function of cilia, which are important for cell signaling and sensory perception. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. In some individuals, no mutations in these known genes are identified, and the genetic cause of the disorder is unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.931, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.9988) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
2. `medquad_ans_0003700-1` (score=0.0001) - Lymphedema distichiasis syndrome is a condition that affects the normal function of the lymphatic system (part of the immune system that produces and transports fluids and immune cells throughout the body). People wit...
3. `medquad_ans_0000610-1` (score=0.0001) - Lymphedema-distichiasis syndrome is a condition that affects the normal function of the lymphatic system, which is a part of the circulatory and immune systems. The lymphatic system produces and transports fluids and ...
4. `medquad_ans_0000610-3` (score=0.0001) - Lymphedema-distichiasis syndrome is caused by mutations in the FOXC2 gene. The FOXC2 gene provides instructions for making a protein that plays a critical role in the formation of many organs and tissues before birth....
5. `medquad_ans_0000075-2` (score=0.0001) - The prevalence of Asperger syndrome is not well established. Estimates range from 1 in 250 to 1 in 5,000 children. Three to four times as many males are affected than females. Because of changes in the way development...

**Answer**:

The prevalence of lymphedema-distichiasis syndrome is unknown [medquad_ans_0000610-2].

**Supported citations**: `medquad_ans_0000610-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=1.000, groundedness=0.949, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0002853-4` (score=0.0003) - Is genetic testing available for hereditary sensory neuropathy type 1? At least four genes responsible for hereditary sensory neuropathy type 1 (HSN1) have been found: HSN1A (the most common form) is associated with m...
2. `medquad_ans_0001483-5` (score=0.0001) - Is genetic testing available for congenital bilateral absence of the vas deferens (CBAVD)? GeneTests lists the names of laboratories that are performing genetic testing for CBAVD. To view the contact information for t...
3. `medquad_ans_0000011-5__dup1` (score=0.0001) - Is genetic testing available for abetalipoproteinemia? Yes. The Genetic Testing Registry (GTR) provides information about the genetic tests available for abetalipoproteinemia. The intended audience for the GTR is heal...
4. `medquad_ans_0002550-4` (score=0.0001) - Is genetic testing available for Gilbert syndrome? The Genetic Testing Registry provides information about the genetic tests for this condition. The intended audience for the GTR is health care providers and researche...
5. `medquad_ans_0005415-4` (score=0.0001) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...

**Answer**:

Yes, information about genetic testing is available for certain conditions. For hereditary sensory neuropathy type 1 (HSN1), at least four genes have been identified (HSN1A, HSN1B, HSN1C, HSN1D, HSN1E), and the Genetic Testing Registry (GTR) provides information about genetic testing for HSN1A. However, clinical laboratories offering testing for other types of HSN1 are not currently known, and patients should consult a health care provider or genetics professional to locate testing options [medquad_ans_0002853-4].

For congenital bilateral absence of the vas deferens (CBAVD), GeneTests lists laboratories performing genetic testing, though most do not accept direct contact from patients; patients should work with a health care provider or genetics professional to access testing [medquad_ans_0001483-5].

For

**Supported citations**: `medquad_ans_0002853-4`, `medquad_ans_0001483-5`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.531, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.8681) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0004739-1` (score=0.0244) - Parkes Weber syndrome (PWS) is a rare congenital condition causing an individual to have a large number of abnormal blood vessels. The main characteristics of PWS typically include a capillary malformation on the skin...
3. `medquad_ans_0002728-1` (score=0.0054) - Hardikar syndrome is a very rare multiple congenital malformation syndrome characterized by obstructive liver and kidney disease, intestinal malrotation, genitourinary abnormalities, cleft lip and palate, pigmentary r...
4. `medquad_ans_0005994-1` (score=0.0049) - Systemic capillary leak syndrome is a condition in which fluid and proteins leak out of tiny blood vessels and flow into surrounding tissues, resulting in dangerously low blood pressure. Attacks frequently last for se...
5. `medquad_ans_0000641-1` (score=0.0032) - Megalencephaly-capillary malformation syndrome (MCAP) is a disorder characterized by overgrowth of several tissues in the body. Its primary features are a large brain (megalencephaly) and abnormalities of small blood ...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin. These malformations appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs, and may be visible at birth or during childhood. While capillary malformations alone usually do not cause health problems, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries. These abnormalities can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure, some of which can be life-threatening. Complications of AVMs and AV

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.9994) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
2. `medquad_ans_0001995-1` (score=0.0003) - What are the signs and symptoms of Dystonia 19? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 19. If the information is available, the table below includes how often the s...
3. `medquad_ans_0001999-1` (score=0.0001) - What are the signs and symptoms of Dystonia 6, torsion? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 6, torsion. If the information is available, the table below includes...
4. `medquad_ans_0000307-1__dup1` (score=0.0001) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
5. `medquad_ans_0003335-1` (score=0.0001) - Juvenile-onset dystonia is a form of dystonia, which is a movement disorder characterized by involuntary muscle contractions that cause repetitive movements and/or abnormal postures. The severity and frequency of the ...

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
3. `medquad_ans_0000689-5` (score=0.0002) - These resources address the diagnosis or management of multiple epiphyseal dysplasia: - Cedars-Sinai Medical Center - Gene Review: Gene Review: Multiple Epiphyseal Dysplasia, Dominant - Gene Review: Gene Review: Multi...
4. `medquad_ans_0000504-5` (score=0.0002) - These resources address the diagnosis or management of hypohidrotic ectodermal dysplasia: - Gene Review: Gene Review: Hypohidrotic Ectodermal Dysplasia - Genetic Testing Registry: Autosomal dominant hypohidrotic ectod...
5. `medquad_ans_0000144-5` (score=0.0001) - These resources address the diagnosis or management of campomelic dysplasia: - European Skeletal Dysplasia Network - Gene Review: Gene Review: Campomelic Dysplasia - Genetic Testing Registry: Camptomelic dysplasia - M...

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

Complications related to treatment include side effects from medications such as hives, rash, itching, shortness of breath, rapid or irregular heartbeat, painful erection lasting hours, swelling of the face, eyes, lips, tongue, throat, arms, hands, feet, ankles, or legs, difficulty breathing or swallowing, chest pain, dizziness or fainting when standing up suddenly, sudden decrease or loss of vision, blurred vision, sudden decrease or loss of hearing, chest

**Supported citations**: `medquad_ans_0000208-7`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.889, groundedness=0.114, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000072-1__dup2` (score=0.0001) - Holes in the heart are simple congenital (kon-JEN-ih-tal) heart defects. Congenital heart defects are problems with the heart's structure that are present at birth. These defects change the normal flow of blood throug...
2. `medquad_ans_0000215-3` (score=0.0001) - Mutations in the CFTR gene cause congenital bilateral absence of the vas deferens. More than half of all men with this condition have mutations in the CFTR gene. Mutations in this gene also cause cystic fibrosis. When...
3. `medquad_ans_0000001-3__dup1` (score=0.0001) - When the absence of the septum pellucidum is part of septo-optic dysplasia, the prognosis varies according to the presence and severity of associated symptoms. By itself, absence of the septum pellucidum is not life-t...
4. `medquad_ans_0003726-1` (score=0.0001) - What are the signs and symptoms of Macrocephaly-capillary malformation? The Human Phenotype Ontology provides the following list of signs and symptoms for Macrocephaly-capillary malformation. If the information is ava...
5. `medquad_ans_0000001-1__dup1` (score=0.0001) - The septum pellucidum (SP) is a thin membrane located at the midline of the brain between the two cerebral hemispheres, or halves of the brain.. It is connected to the corpus callosum -- a collection of nerve fibers t...

**Answer**:

The provided context does not mention the absence of the septum pellucidum or any treatments for this condition. Therefore, based on the given information, it is not possible to answer the question. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.517, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.9992) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0000266-2__dup2` (score=0.0083) - The symptoms of Lujo hemorrhagic fever, as described in the five patients in the original cluster outbreak, resemble those of severe Lassa Fever. After an incubation period of 7 to 13 days, the clinical course started...
3. `medquad_ans_0000128-2__dup1` (score=0.0065) - A child who has a febrile seizure usually doesn't need to be hospitalized. If the seizure is prolonged or is accompanied by a serious infection, or if the source of the infection cannot be determined, a doctor may rec...
4. `medquad_ans_0000128-3__dup1` (score=0.0031) - The vast majority of febrile seizures are short and harmless. There is no evidence that short febrile seizures cause brain damage. Multiple or prolonged seizures are a risk factor for epilepsy but most children who ex...
5. `medquad_ans_0000266-4__dup2` (score=0.0027) - During the acute febrile phase, Lujo virus was isolated from blood from days 2 to 13 after onset. Virus was also isolated from liver tissue obtained post-mortem. A subsequent complete genomic analysis of Lujo virus fa...

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

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.9994) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
2. `medquad_ans_0000389-1` (score=0.0001) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
3. `medquad_ans_0000389-3` (score=0.0001) - Mutations in the FLNA gene cause frontometaphyseal dysplasia. The FLNA gene provides instructions for producing the protein filamin A, which helps build the network of protein filaments (cytoskeleton) that gives struc...
4. `medquad_ans_0000598-1__dup1` (score=0.0001) - Axial spondylometaphyseal dysplasia is a genetic disorder of bone growth. The term axial means towards the center of the body. Sphondylos is a Greek term meaning vertebra. Metaphyseal dysplasia refers to abnormalities...
5. `medquad_ans_0000389-2` (score=0.0001) - Frontometaphyseal dysplasia is a rare disorder; only a few dozen cases have been reported worldwide.

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.909, groundedness=0.544, answer_relevance=0.764

**Top docs**:

1. `medquad_ans_0000128-7` (score=0.9832) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
2. `medquad_ans_0000193-6` (score=0.9410) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
3. `medquad_ans_0000128-10` (score=0.8487) - Eat a high-fiber diet to help prevent problems. Talk to your doctor about using fiber products like Benefiber, Citrucel, or Metamucil. Daily use can help you get the fiber you need if you do not get it through your di...
4. `medquad_ans_0000591-1__dup1` (score=0.0480) - Summary : Medicines cure infectious diseases, prevent problems from chronic diseases, and ease pain. But medicines can also cause harmful reactions if not used correctly. Errors can happen in the hospital, at the doct...
5. `medquad_ans_0000035-9` (score=0.0041) - People can manage their diabetes with meal planning, physical activity, and if needed, medications. More information about taking care of type 1 or type 2 diabetes is provided in the NIDDK health topics: - What I need...

**Answer**:

Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding, you need to go to the hospital so a doctor can find the part of your colon that is bleeding. The doctor may use a special drug that makes the bleeding stop. The doctor may also decide to operate and remove the part of the colon that is bleeding. Eating a high-fiber diet can help prevent problems and reduce symptoms. Recommended high-fiber foods include fruit (raw apples, peaches, pears, and tangerines), vegetables (fresh broccoli, squash, carrots, and brussels sprouts), starchy vegetables (potatoes, baked beans, kidney beans, and lima beans), and grains (whole-wheat bread, brown rice, bran flake cereal, and oatmeal). Talk to your doctor about using fiber

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.9995) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0001404-4` (score=0.0001) - How is Coffin-Siris syndrome diagnosed? Diagnosis of Coffin-Siris syndrome is largely based upon the presence or absence of common signs and symptoms in the individual. While formal diagnostic criteria have not been e...
3. `medquad_ans_0000570-1` (score=0.0001) - L1 syndrome is an inherited disorder that primarily affects the nervous system. L1 syndrome involves a variety of features that were once thought to be distinct disorders, but are now considered to be part of the same...
4. `medquad_ans_0000013_1-6` (score=0.0001) - Key Points - There are different types of treatment for patients with myelodysplastic syndromes. - Treatment for myelodysplastic syndromes includes supportive care, drug therapy, and stem cell transplantation. - Three...
5. `medquad_ans_0001102-5` (score=0.0001) - These resources address the diagnosis or management of 8p11 myeloproliferative syndrome: - Cancer.Net from the American Society of Clinical Oncology: Acute Myeloid Leukemia Diagnosis - Cancer.Net from the American Soc...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomeg

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000581-5` (score=0.9943) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
2. `medquad_ans_0003911-2` (score=0.9540) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a men...
3. `medquad_ans_0000084-2__dup1` (score=0.0318) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
4. `medquad_ans_0000803-5` (score=0.0007) - These resources address the diagnosis or management of Poland syndrome: - Children's Medical Center of Dallas - Great Ormond Street Hospital (UK): Treatment Options for Symbrachydactyly - St. Louis Children's Hospital...
5. `medquad_ans_0002796-4` (score=0.0002) - What treatments are available for Henoch-Schonlein purpura? Unfortunately, there is no cure for Henoch-Schonlein purpura (HSP). Treatments aim to relieve the symptoms of this condition. For example, non-steroidal anti...

**Answer**:

The treatment for meningiomas varies depending on the location of the tumor and the symptoms it causes. Careful observation may be the best course of action for some individuals. When treatment is necessary, surgery and radiation are the most common options. Radiation may be used if the meningioma cannot be operated on or if it is only partially removed by surgery. It may also be used in cases of malignant, atypical, or recurrent tumors. Other treatments that have been tried or are under investigation include hydroxyurea, epidermal growth factor receptor inhibitors, platelet-derived growth factor receptor inhibitors, vascular endothelial growth factor inhibitors, immunotherapy to stimulate the immune system, and somatostatin analogs which prevent the release of growth hormones [medquad_ans_0003911-2].

**Supported citations**: `medquad_ans_0003911-2`
**Unsupported citations**: _none_

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-8` (score=0.9738) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
2. `medquad_ans_0000015_1-8` (score=0.9636) - Key Points - There are different types of treatment for children with extracranial germ cell tumors. - Children with extracranial germ cell tumors should have their treatment planned by a team of health care providers...
3. `medquad_ans_0000006_6-8` (score=0.0341) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health...
4. `medquad_ans_0000006_4-8` (score=0.0284) - Key Points - There are different types of treatment for patients with central nervous system atypical teratoid/rhabdoid tumor. - Children with atypical teratoid/rhabdoid tumor should have their treatment planned by a ...
5. `medquad_ans_0000006_4-6` (score=0.0189) - Key Points - There is no standard staging system for central nervous system atypical teratoid/rhabdoid tumor. There is no standard staging system for central nervous system atypical teratoid/rhabdoid tumor. The extent...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and participation in clinical trials. Standard treatments are currently used, and new treatments are being tested in clinical trials. Treatment planning should be done by a team of health care providers who are experts in treating cancer in children, including a pediatric oncologist and/or a radiation oncologist, along with other specialists such as a pediatric neurosurgeon, neurologist, endocrinologist, ophthalmologist, pediatric nurse specialist, rehabilitation specialist, psychologist, and social worker [medquad_ans_0000006_7-8].

**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=0.889, groundedness=0.984, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0004401-1` (score=0.9127) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
2. `medquad_ans_0000729-1` (score=0.8874) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0000002-1__dup1` (score=0.0052) - Acid lipase disease or deficiency occurs when the enzyme needed to break down certain fats that are normally digested by the body is lacking or missing, resulting in the toxic buildup of these fats in the bodys cells ...
4. `medquad_ans_0000123-4__dup1` (score=0.0019) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
5. `medquad_ans_0000002-4__dup1` (score=0.0017) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge abut the brain and nervous system, and to use that knowledge to reduce the burden of neurological dis...

**Answer**:

Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene and is inherited in an autosomal recessive pattern. Other features may include a fatty liver, a weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. Signs and symptoms vary greatly among affected individuals [medquad_ans_0004401-1][medquad_ans_0000729-1].

**Supported citations**: `medquad_ans_0004401-1`, `medquad_ans_0000729-1`
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000981-1__dup1` (score=0.4161) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
2. `medquad_ans_0006546-1` (score=0.0613) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
3. `medquad_ans_0000092-1__dup1` (score=0.0066) - Cytomegalovirus (CMV) is a virus found throughout the world that infects between 50 to 80 percent of all adults in the United States by the age of 40. CMV is in the same family of viruses that causes cold sores (herpe...
4. `medquad_ans_0000266-1__dup4` (score=0.0009) - Like all arenaviruses, Lujo virus has a rodent host as its reservoir. Humans can contract LUHF through contact with an infected rodent. Contact can be direct or through inhalation of aerosolized Lujo virus from the ur...
5. `medquad_ans_0000193-1__dup2` (score=0.0005) - Microcephaly is a medical condition in which the circumference of the head is smaller than normal because the brain has not developed properly or has stopped growing. Microcephaly can be present at birth or it may dev...

**Answer**:

Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus has spread through blood transfusions. Most people who get the virus do not get sick. One in five people do get symptoms, which can include a fever, rash, joint pain, and conjunctivitis (pinkeye). Symptoms are usually mild, and start 2 to 7 days after being bitten by an infected mosquito. A blood test can tell whether you have the infection. There are no vaccines or medicines to treat it. Drinking lots of fluids, resting, and taking acetaminophen might help. Zika can cause microcephaly (a serious birth defect of the brain) and other problems in babies whose mothers were infected while pregnant. The Centers for Disease Control and Prevention recommends that pregnant women

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000019-3__dup2` (score=0.9850) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
2. `medquad_ans_0000060-3__dup2` (score=0.3281) - Research suggests thatcoronary heart disease(CHD) begins with damage to the lining and inner layers of the coronary (heart) arteries. Several factors contribute to this damage. They include: Smoking, including secondh...
3. `medquad_ans_0000039-5__dup1` (score=0.0339) - If your child has a congenital heart defect, you may think you did something wrong during your pregnancy to cause the problem. However, doctors often don't know why congenital heart defects occur. Heredity may play a ...
4. `medquad_ans_0000019-8` (score=0.0004) - Researchers are still learning about broken heart syndrome, and no treatments have been shown to prevent it. For people who have experienced the condition, the risk of recurrence is low. An emotionally upsetting or se...
5. `medquad_ans_0000019-4__dup2` (score=0.0003) - Broken heart syndrome affects women more often than men. Often, people who experience broken heart syndrome have previously been healthy. Research shows that the traditional risk factors for heart disease may not appl...

**Answer**:

Broken heart syndrome is believed to be caused by extreme emotional or physical stress, which triggers a surge of fight or flight hormones such as adrenaline and noradrenaline. This hormone surge overwhelms and "stuns" the heart muscle, leading to temporary changes in heart muscle cells or coronary blood vessels. Potential causes include a hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=1.000, groundedness=0.949, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000892-1__dup1` (score=0.8647) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...
2. `medquad_ans_0000005-1__dup5` (score=0.4499) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
3. `medquad_ans_0000448-1` (score=0.2328) - Hashimoto thyroiditis is a condition that affects the function of the thyroid, which is a butterfly-shaped gland in the lower neck. The thyroid makes hormones that help regulate a wide variety of critical body functio...
4. `medquad_ans_0000008-3__dup4` (score=0.0036) - Hyperthyroidism has several causes, including - Graves disease - thyroid nodules - thyroiditis, or inflammation of the thyroid - consuming too much iodine - overmedicating with synthetic thyroid hormone, which is used...
5. `medquad_ans_0000448-2` (score=0.0024) - Hashimoto thyroiditis affects 1 to 2 percent of people in the United States. It occurs more often in women than in men, which may be related to hormonal factors. The condition is the most common cause of thyroid under...

**Answer**:

Hashimoto thyroiditis is a condition that affects the function of the thyroid, a butterfly-shaped gland in the lower neck. It is a form of chronic inflammation that can damage the thyroid, reducing its ability to produce hormones. In Hashimoto thyroiditis, the immune system attacks the thyroid gland, causing inflammation and interfering with its ability to produce thyroid hormones. This immune attack is due to an autoimmune response where large numbers of white blood cells called lymphocytes accumulate in the thyroid and produce antibodies that initiate the autoimmune process. One of the first signs of Hashimoto thyroiditis is an enlargement of the thyroid called a goiter. Hashimoto thyroiditis is the most common cause of hypothyroidism in the United States [medquad_ans_0000005-1\_\_dup5][medquad_ans_0000448-1].

**Supported citations**: `medquad_ans_0000005-1__dup5`, `medquad_ans_0000448-1`
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=1.000, groundedness=0.497, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.7181) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
2. `medquad_ans_0006463-1` (score=0.6708) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
3. `medquad_ans_0000012-20` (score=0.0005) - More information on COPD is available at: What is COPD? and at the Learn More, Breathe Better Campaign For information on quitting smoking, visit http://www.surgeongeneral.gov/tobacco/ or Smokefree.gov. For informatio...
4. `medquad_ans_0001043-2` (score=0.0003) - Wolman disease is estimated to occur in 1 in 350,000 newborns.
5. `medquad_ans_0000051-1__dup4` (score=0.0001) - Enlarged and Misshapen Bones Paget's disease of bone causes affected bones to become enlarged and misshapen. Our bones are living tissue, and our bodies are constantly breaking down old bone and replacing it with new ...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in body organs such as the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are often healthy at birth but soon develop symptoms including enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor nutrient absorption. The disease is severe and life-threatening, with most children not surviving past early childhood. It occurs in approximately 1 in 350,000 newborns [medquad_ans_00010

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.9955) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0004813-3` (score=0.0082) - What causes periodic fever, aphthous stomatitis, pharyngitis, cervical adenitis (PFAPA)? The cause of PFAPA is unknown, although viral or autoimmune causes have been suggested.
3. `medquad_ans_0000058-14` (score=0.0043) - Scientists don't know exactly what causes prostate cancer. They cannot explain why one man gets prostate cancer and another does not. However, they have been able to identify some risk factors that are associated with...
4. `medquad_ans_0004177-3` (score=0.0016) - What causes multifocal choroiditis? Multifocal choroiditis occurs spontaneously and the cause is not currently known (idiopathic). It is possible that a bacterial or viral infection may trigger an immune response that...
5. `medquad_ans_0002483-3` (score=0.0016) - What causes gamma heavy chain disease? The causes or risk factors for gamma heavy chain disease are not known.

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.875, groundedness=0.947, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000655-3` (score=0.9980) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
2. `medquad_ans_0000593-3` (score=0.0024) - Mutations in at least two genes on the X chromosome are thought to be responsible for Lenz microphthalmia syndrome. Only one of these genes, BCOR, has been identified. The BCOR gene provides instructions for making a ...
3. `medquad_ans_0000142-3` (score=0.0002) - C3 glomerulopathy is associated with changes in many genes. Most of these genes provide instructions for making proteins that help regulate a part of the body's immune response known as the complement system. This sys...
4. `medquad_ans_0000836-3` (score=0.0001) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...
5. `medquad_ans_0000137-3` (score=0.0001) - Cancers occur when a buildup of mutations in critical genesthose that control cell growth and division or repair damaged DNAallow cells to grow and divide uncontrollably to form a tumor. In most cases of breast cancer...

**Answer**:

Genetic changes related to microphthalmia include mutations in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from chromosomal abnormalities affecting one or more genes. In the case of Lenz microphthalmia syndrome, mutations in at least two genes on the X chromosome are thought to be responsible, with only the BCOR gene currently identified. The BCOR gene provides instructions for making the BCL6 corepressor protein, which helps regulate the activity of other genes and appears to play an important role in early embryonic development. A mutation in the BCOR gene changes the structure of the BCL6 corepressor protein, disrupting normal eye and organ development before birth [medquad_ans_0000593-3]. Most genetic changes associated with isolated microphthalmia have been identified in only a small number of affected individuals [medquad_ans_0000655

**Supported citations**: `medquad_ans_0000593-3`
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.867, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000028-15` (score=0.0029) - Physicians often prescribe high doses of non-steroidal anti-inflammatory drugs (NSAIDs) or steroids for a sudden attack of gout. NSAIDs are taken by mouth and corticosteroids are either taken by mouth or injected into...
2. `medquad_ans_0000028-9` (score=0.0026) - Most people with gout have too much uric acid in their blood, a condition called hyperuricemia. Uric acid is a substance that results from the breakdown of purines, which are part of all human tissue and are found in ...
3. `medquad_ans_0000978-1__dup1` (score=0.0023) - Your wrist is made up of eight small bones known as carpals. They support a tube that runs through your wrist. That tube, called the carpal tunnel, has tendons and a nerve inside. It is covered by a ligament, which ho...
4. `medquad_ans_0000899-1__dup1` (score=0.0013) - Fourteen of the 26 bones in your feet are in your toes. The toes, particularly your big toe, help you move and keep your balance. Playing sports, running, and receiving a blow to the foot can damage your toes. Wearing...
5. `medquad_ans_0000028-4__dup3` (score=0.0005) - Symptoms Can Be Controlled With proper treatment, most people with gout are able to control their symptoms and live productive lives. The goals for treatment are to ease the pain that comes from sudden attacks, preven...

**Answer**:

The symptoms of gout include pain, swelling, and redness in the affected joint, most commonly in the big toe. Needle-like crystals of uric acid can build up in the joint space between two bones, leading to inflammation and severe pain. Patients often experience sudden and intense attacks of pain and swelling, with improvement beginning within a few hours of treatment and the attack usually resolving within a week [medquad_ans_0000028-15]. Wrist pain with bruising and swelling, as well as misshapen joints and inability to move the wrist, may also be signs of gout, though these are more commonly associated with wrist injuries or other conditions [medquad_ans_0000978-1__dup1]. Gout often causes pain in the big toe, and it is frequently linked to hyperuricemia, a condition of having too much uric acid in the blood [med

**Supported citations**: `medquad_ans_0000028-15`, `medquad_ans_0000978-1__dup1`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.667, groundedness=0.740, answer_relevance=0.200

**Top docs**:

1. `medquad_ans_0000860-4` (score=0.0004) - This condition is inherited in an autosomal dominant pattern, which means one copy of the recombinant chromosome 8 in each cell is sufficient to cause the disorder. Most people with recombinant 8 syndrome have at leas...
2. `medquad_ans_0000971-2` (score=0.0001) - Task-specific focal dystonia affects an estimated 7 to 69 per million people in the general population. Musician's dystonia that is severe enough to impact performance occurs in about 1 percent of musicians.
3. `medquad_ans_0000309-5` (score=0.0001) - These resources address the diagnosis or management of dystonia 6: - Gene Review: Gene Review: Dystonia Overview - Genetic Testing Registry: Dystonia 6, torsion These resources from MedlinePlus offer information about...
4. `medquad_ans_0000307-1__dup1` (score=0.0001) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...
5. `medquad_ans_0000860-3` (score=0.0001) - Recombinant 8 syndrome is caused by a rearrangement of chromosome 8 that results in a deletion of a piece of the short (p) arm and a duplication of a piece of the long (q) arm. The deletion and duplication result in t...

**Answer**:

The provided context does not mention "Dystonia 8" or its cause. While the context discusses dystonia in general and recombinant 8 syndrome, there is no information linking Dystonia 8 to a specific cause. Therefore, based on the available information, the cause of Dystonia 8 cannot be determined.

Answer: Insufficient context to determine the cause of Dystonia 8. [medquad_ans_0000307-1__dup1]

**Supported citations**: `medquad_ans_0000307-1__dup1`
**Unsupported citations**: _none_

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

**Scores**: context_relevance=1.000, groundedness=0.796, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0004373-2` (score=0.0001) - What are the signs and symptoms of Neurofibromatosis type 1? People affected by neurofibromatosis type 1 (NF1) have an increased risk of developing many different types of tumors (both cancerous and noncancerous). Alm...
2. `medquad_ans_0000170-8` (score=0.0001) - Treatment for RAS includes lifestyle changes, medications, and surgery and aims to - prevent RAS from getting worse - treat RVH - relieve the blockage of the renal arteries RAS that has not led to RVH or caused a sign...
3. `medquad_ans_0000170-10` (score=0.0001) - - Renal artery stenosis (RAS) is the narrowing of one or both renal arteries. The renal arteries are blood vessels that carry blood to the kidneys from the aortathe main blood vessel that carries blood from the heart ...
4. `medquad_ans_0000840-2__dup1` (score=0.0001) - What are the signs and symptoms of Branchiootorenal syndrome? Signs and symptoms of branchiootorenal syndrome can vary greatly from person to person and even between people within the same family. Hearing loss is the ...
5. `medquad_ans_0005294-1` (score=0.0001) - What are the signs and symptoms of Renal dysplasia-limb defects syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Renal dysplasia-limb defects syndrome. If the information is...

**Answer**:

The provided context does not contain information about the symptoms of Renal Artery Stenosis. [medquad_ans_0004373-2] discusses symptoms of Neurofibromatosis type 1, not Renal Artery Stenosis. Therefore, based on the given context, it is not possible to answer the question.

**Supported citations**: `medquad_ans_0004373-2`
**Unsupported citations**: _none_

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.700, groundedness=0.760, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.9908) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
2. `medquad_ans_0000728-5` (score=0.9897) - These resources address the diagnosis or management of NARP: - Gene Review: Gene Review: Mitochondrial DNA-Associated Leigh Syndrome and NARP - Gene Review: Gene Review: Mitochondrial Disorders Overview - Genetic Test...
3. `medquad_ans_0001010-5` (score=0.9875) - These resources address the diagnosis or management of Usher syndrome: - Gene Review: Gene Review: Usher Syndrome Type I - Gene Review: Gene Review: Usher Syndrome Type II - Genetic Testing Registry: Usher syndrome ty...
4. `medquad_ans_0000862-5` (score=0.0015) - These resources address the diagnosis or management of Refsum disease: - Gene Review: Gene Review: Refsum Disease - Gene Review: Gene Review: Retinitis Pigmentosa Overview - Genetic Testing Registry: Phytanic acid sto...
5. `medquad_ans_0000079-5` (score=0.0002) - These resources address the diagnosis or management of ataxia with vitamin E deficiency: - Gene Review: Gene Review: Ataxia with Vitamin E Deficiency - Genetic Testing Registry: Ataxia with vitamin E deficiency - Medl...

**Answer**:

The provided context does not specify treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While it mentions that the Genetic Testing Registry includes resources for these conditions and that MedlinePlus offers information on diagnosis and management of various health conditions—including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care—it does not detail specific treatments for these syndromes. Therefore, based on the given information, the treatments are not explicitly stated.

[medquad_ans_0000712-5]

**Supported citations**: `medquad_ans_0000712-5`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.9994) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0000429-2` (score=0.0001) - GM1 gangliosidosis is estimated to occur in 1 in 100,000 to 200,000 newborns. Type I is reported more frequently than the other forms of this condition. Most individuals with type III are of Japanese descent.
3. `medquad_ans_0000353-1` (score=0.0001) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control ...
4. `medquad_ans_0002593-1` (score=0.0001) - Glucocorticoid-remediable aldosteronism is one of three types of familial hyperaldosteronism and was first described in 1966. Aldosterone is a hormone manufactured by the adrenal glands which helps the body retain wat...
5. `medquad_ans_0000340-1` (score=0.0001) - Familial adenomatous polyposis (FAP) is an inherited disorder characterized by cancer of the large intestine (colon) and rectum. People with the classic type of familial adenomatous polyposis may begin to develop mult...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria, metabolic acidosis, polydipsia, polyuria, adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

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

**Scores**: context_relevance=1.000, groundedness=0.739, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000117-4` (score=0.9978) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
2. `medquad_ans_0002290-1` (score=0.0001) - Familial pemphigus vulgaris refers to a cluster of pemphigus vulgaris within a family. Pemphigus vulgaris is a rare autoimmune condition that is characterized by blisters and sores on the skin and mucus membranes. Alt...
3. `medquad_ans_0000115-1` (score=0.0001) - Benign chronic pemphigus, often called Hailey-Hailey disease, is a rare skin condition that usually appears in early adulthood. The disorder is characterized by red, raw, and blistered areas of skin that occur most of...
4. `medquad_ans_0004796-1` (score=0.0001) - Pemphigus vulgaris is an autoimmune disorder that involves blistering of the skin and mucous membranes. It occurs almost exclusively in middle-aged or older people. Many cases begin with blisters in the mouth, followe...
5. `medquad_ans_0000698-1__dup2` (score=0.0001) - Pemphigus is an autoimmune disorder. If you have it, your immune system attacks healthy cells in your skin and mouth, causing blisters and sores. No one knows the cause. Pemphigus does not spread from person to person...

**Answer**:

The provided context does not state whether benign chronic pemphigus is inherited. While it describes the condition's symptoms, progression, and associations (such as white lines on fingernails), there is no information about inheritance patterns. Therefore, based on the given context, it cannot be determined if benign chronic pemphigus is inherited.

[medquad_ans_0000115-1]

**Supported citations**: `medquad_ans_0000115-1`
**Unsupported citations**: _none_

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.312, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000916-4__dup1` (score=0.9533) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
2. `medquad_ans_0001389-4` (score=0.0007) - How is Coats disease inherited? In most cases, Coats disease is not inherited. Eighty to 90% of affected people have no evidence of a genetic predisposition to the condition and no affected family members. Rarely, Coa...
3. `medquad_ans_0003718-4` (score=0.0002) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
4. `medquad_ans_0000734-4__dup1` (score=0.0001) - Is Binswanger's disease an inherited condition? Although Binswanger's disease is not considered an inherited condition, genetics may play a role in many of the conditions and risk factors that are associated with the ...
5. `medquad_ans_0000915-1__dup1` (score=0.0001) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a mutation in only one copy of the responsible gene in each cell is enough to cause the condition. [medquad_ans_0000916-4]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000916-4`

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.900, groundedness=0.935, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000483-3` (score=0.9990) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family o...
2. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
3. `medquad_ans_0000483-2` (score=0.0001) - Histiocytosis-lymphadenopathy plus syndrome is a rare disorder, affecting approximately 100 individuals worldwide.
4. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000483-1` (score=0.0001) - Histiocytosis-lymphadenopathy plus syndrome (also known as SLC29A3 spectrum disorder) is a group of conditions with overlapping signs and symptoms that affect many parts of the body. This group of disorders includes H...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene [medquad_ans_0000483-3]. These mutations reduce or eliminate the activity of the equilibrative nucleoside transporter 3 (ENT3) protein, which is responsible for transporting nucleosides out of lysosomes and into mitochondria [medquad_ans_0000483-3]. This impairment in nucleoside transport is believed to lead to a buildup of nucleosides in lysosomes and a shortage in mitochondria, potentially affecting cellular energy production and contributing to the condition's symptoms [medquad_ans_0000483-3].

**Supported citations**: `medquad_ans_0000483-3`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.949, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004450-3` (score=0.9363) - How is Noonan syndrome inherited? Noonan syndrome is inherited in an autosomal dominant manner. This means that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. ...
2. `medquad_ans_0004457-1` (score=0.0001) - Noonan-like syndrome with loose anagen hair is characterized by facial features suggestive of Noonan syndrome (macrocephaly, high forehead, wide-set eyes or hypertelorism, palpebral ptosis, and low-set and posteriorly...
3. `medquad_ans_0004453-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004452-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004454-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Yes, Noonan syndrome is typically inherited in an autosomal dominant manner, meaning that having one changed (mutated) copy of the responsible gene in each cell is enough to cause the condition. Each child of a person with Noonan syndrome has a 50% (1 in 2) chance to inherit the condition. However, many cases are due to a new mutation that occurs for the first time in the affected person and are not inherited from an affected parent [medquad_ans_0004450-3]. It may also be caused by mutations in several genes, including PTPN11, KRAS, RAF1, SOS1, NRAS, and BRAF, and is sometimes referred to as a specific subtype based on the responsible gene [medquad_ans_0004453-1].

**Supported citations**: `medquad_ans_0004450-3`, `medquad_ans_0004453-1`
**Unsupported citations**: _none_

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.875, groundedness=0.817, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0001093-5` (score=0.9952) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
2. `medquad_ans_0001092-5` (score=0.9949) - These resources address the diagnosis or management of 3-MCC deficiency: - Baby's First Test - Genetic Testing Registry: 3 Methylcrotonyl-CoA carboxylase 1 deficiency - Genetic Testing Registry: 3-methylcrotonyl CoA c...
3. `medquad_ans_0000445-3__dup1` (score=0.0260) - How might arginase deficiency be treated? The treatment and management of arginase deficiency is generally focused on lowering arginine levels and preventing hyperammonemia (an accumulation of ammonia in the blood). T...
4. `medquad_ans_0003264-5` (score=0.0018) - How is isobutyryl-CoA dehydrogenase deficiency (IBD deficiency) treated? There is no standard treatment protocol for IBD deficiency. Infants diagnosed through newborn screening are encouraged to work with a metabolic ...
5. `medquad_ans_0004452-2` (score=0.0016) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain specific information about the treatments for 3-methylglutaconyl-CoA hydratase deficiency. While resources related to diagnosis and management are mentioned, such as genetic testing and information from MedlinePlus on diagnostic tests and genetic counseling, no details on treatment protocols are available [medquad_ans_0001093-5]. Therefore, based on the given context, the treatments for 3-methylglutaconyl-CoA hydratase deficiency cannot be determined.

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

**Scores**: context_relevance=0.800, groundedness=0.817, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.9944) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
2. `medquad_ans_0000228-5` (score=0.0016) - These resources address the diagnosis or management of congenital hepatic fibrosis: - Gene Review: Gene Review: Congenital Hepatic Fibrosis Overview - Genetic Testing Registry: Congenital hepatic fibrosis These resour...
3. `medquad_ans_0000225-3` (score=0.0001) - CFEOM1 and rare cases of CFEOM3 result from mutations in the KIF21A gene. This gene provides instructions for making a protein called a kinesin, which is essential for the transport of materials within cells. Research...
4. `medquad_ans_0000225-2` (score=0.0001) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few fami...
5. `medquad_ans_0000084-1__dup2` (score=0.0001) - All forms of myasthenia are due to problems in the communication between nerve cells and muscles. Most involve the activities of neurotransmitters. Neurotransmitters are chemicals that allow neurons to relay informati...

**Answer**:

The provided context does not specify treatments for congenital fibrosis of the extraocular muscles. While it mentions resources related to diagnosis and management, such as diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care from MedlinePlus, it does not detail specific treatments for congenital fibrosis of the extraocular muscles [medquad_ans_0000225-5]. Therefore, based on the given information, the exact treatments cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=1.000, groundedness=0.522, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.6731) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0000152-1__dup1` (score=0.4376) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressiv...
3. `medquad_ans_0000075-1__dup2` (score=0.0710) - Chorea is an abnormal involuntary movement disorder, one of a group of neurological disorders called dyskinesias, which are caused by overactivity of the neurotransmitter dopamine in the areas of the brain that contro...
4. `medquad_ans_0000645-1__dup2` (score=0.0111) - The brain, spinal cord, and nerves make up the nervous system. Together they control all the workings of the body. When something goes wrong with a part of your nervous system, you can have trouble moving, speaking, s...
5. `medquad_ans_0000013-2__dup2` (score=0.0023) - Diseases Can Have Various Causes Many things influence your overall health and likelihood of developing a disease. Sometimes, it's not clear what causes a disease. Many diseases are thought to be caused by a combinati...

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. It is caused by a mutation in the gene for a protein called huntingtin, which results in the cytosine, adenine, and guanine (CAG) building blocks of DNA repeating many more times than is normal. Symptoms, which progressively worsen, include uncontrolled movements (called chorea), abnormal body postures, impaired coordination, slurred speech, difficulty feeding and swallowing, and changes in behavior, emotion, judgment, and cognition. HD typically begins between ages 30 and 50, though an earlier onset form called juvenile HD occurs under age 20, with symptoms such as unsteadiness, rigidity, difficulty at school, and seizures. Individuals who inherit the HD gene will eventually develop the disease, and each child of a parent

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
2. `medquad_ans_0004452-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004450-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004454-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004456-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not specify the number of people affected by Noonan syndrome. [medquad_ans_0004453-1] [medquad_ans_0004452-1] [medquad_ans_0004450-1] [medquad_ans_0004454-1] describe the features, genetic causes, and inheritance pattern of Noonan syndrome but do not include epidemiological data on prevalence or affected population size. Therefore, the number of people affected by Noonan syndrome cannot be determined from the given information.

**Supported citations**: `medquad_ans_0004453-1`, `medquad_ans_0004452-1`, `medquad_ans_0004450-1`, `medquad_ans_0004454-1`
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

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.9688) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0000084-2__dup1` (score=0.1268) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
3. `medquad_ans_0000954-5` (score=0.0174) - These resources address the diagnosis or management of Stickler syndrome: - Gene Review: Gene Review: Stickler Syndrome - Genetic Testing Registry: Marshall syndrome - Genetic Testing Registry: Stickler syndrome - Med...
4. `medquad_ans_0000392-5` (score=0.0069) - These resources address the diagnosis or management of Fryns syndrome: - Children's Hospital of Philadelphia: Treatment of Congenital Diaphragmatic Hernia - Gene Review: Gene Review: Fryns Syndrome - Genetic Testing R...
5. `medquad_ans_0001094-5` (score=0.0051) - These resources address the diagnosis or management of 3MC syndrome: - Genetic Testing Registry: Carnevale syndrome - Genetic Testing Registry: Craniofacial-ulnar-renal syndrome - Genetic Testing Registry: Malpuech fa...

**Answer**:

Treatment options attempted for Shapiro syndrome include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, and sympathectomy. The effectiveness of these treatments varies among individuals. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.970, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.6838) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-1` (score=0.6256) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0000113-4__dup3` (score=0.0019) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
4. `medquad_ans_0000278-5` (score=0.0004) - These resources address the diagnosis or management of Denys-Drash syndrome: - Gene Review: Gene Review: Wilms Tumor Overview - Genetic Testing Registry: Drash syndrome - MedlinePlus Encyclopedia: Nephrotic Syndrome T...
5. `medquad_ans_0000136-5` (score=0.0003) - These resources address the diagnosis or management of branchiootorenal/branchiootic syndrome: - Gene Review: Gene Review: Branchiootorenal Spectrum Disorders - Genetic Testing Registry: Branchiootic syndrome - Geneti...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.9994) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0000593-1__dup1` (score=0.0001) - Axenfeld-Rieger syndrome is a group of eye disorders that affects the development of the eye. Common eye symptoms include cornea defects, which is the clear covering on the front of the eye, and iris defects, which is...
3. `medquad_ans_0000594-1__dup1` (score=0.0001) - Axenfeld-Rieger syndrome is a group of eye disorders that affects the development of the eye. Common eye symptoms include cornea defects, which is the clear covering on the front of the eye, and iris defects, which is...
4. `medquad_ans_0000225-3` (score=0.0001) - CFEOM1 and rare cases of CFEOM3 result from mutations in the KIF21A gene. This gene provides instructions for making a protein called a kinesin, which is essential for the transport of materials within cells. Research...
5. `medquad_ans_0001404-4` (score=0.0001) - How is Coffin-Siris syndrome diagnosed? Diagnosis of Coffin-Siris syndrome is largely based upon the presence or absence of common signs and symptoms in the individual. While formal diagnostic criteria have not been e...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, and syndactyly [medquad_ans_0006242-1]. The frequency of these symptoms is estimated based on data from the Human Phenotype Ontology, primarily derived from Orphanet, with some frequencies listed as percentages or fractions representing the proportion of patients observed to have the symptom [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.833, groundedness=0.125, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000581-5` (score=0.9955) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
2. `medquad_ans_0000203-2__dup1` (score=0.0702) - There is no cure for MSA. Currently, there are no treatments to delay the progress of neurodegeneration in the brain. But there are treatments available to help people cope with some of the more disabling symptoms of ...
3. `medquad_ans_0004155-5` (score=0.0221) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...
4. `medquad_ans_0000040_1-6` (score=0.0038) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000023_1-8` (score=0.0028) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...

**Answer**:

The provided context does not contain information about the treatments for mucolipidoses. Therefore, based on the given context, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.9994) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
2. `medquad_ans_0005084-1` (score=0.0001) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
3. `medquad_ans_0000525-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy is a liver disorder that occurs in pregnant women. Cholestasis is a condition that impairs the release of a digestive fluid called bile from liver cells. As a result, bile builds ...
4. `medquad_ans_0000525-3` (score=0.0001) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
5. `medquad_ans_0003240-2` (score=0.0001) - What are the signs and symptoms of Intrahepatic cholestasis of pregnancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Intrahepatic cholestasis of pregnancy. If the information is...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 (PFIC2) include severe itching (pruritus), jaundice, failure to thrive, portal hypertension, hepatosplenomegaly (enlarged liver and spleen), diarrhea, fat malabsorption, elevated alkaline phosphatase, conjugated hyperbilirubinemia, intermittent jaundice, intrahepatic cholestasis, and short stature. In some cases, patients may develop hepatocellular carcinoma and cirrhosis, with death in childhood being a possible outcome [medquad_ans_0005084-1]. The condition typically presents during infancy and progresses to liver failure in the first few years of life [medquad_ans_0005084-1].

**Supported citations**: `medquad_ans_0005084-1`
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

**Scores**: context_relevance=0.800, groundedness=0.544, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005415-1` (score=0.3207) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
2. `medquad_ans_0000882-1` (score=0.2402) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
3. `medquad_ans_0000113-4__dup3` (score=0.0016) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
4. `medquad_ans_0005415-4` (score=0.0002) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...
5. `medquad_ans_0000882-2` (score=0.0002) - Both the autosomal recessive and autosomal dominant forms of Robinow syndrome are rare. Fewer than 200 people with autosomal recessive Robinow syndrome have been described in the medical literature. This form of the c...

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

**Scores**: context_relevance=0.750, groundedness=0.887, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000554-1__dup2` (score=0.7928) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
2. `medquad_ans_0000040-1__dup3` (score=0.4616) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...
3. `medquad_ans_0000040-5__dup1` (score=0.0984) - Unlike other types of cancer, leukemia isn't a tumor that your doctor can surgically remove. Leukemia cells are produced in the bone marrow and travel throughout the body. The Goal of Treatment The goal of treatment f...
4. `medquad_ans_0000188-1__dup2` (score=0.0856) - Leukemia is cancer of the white blood cells. It is the most common type of childhood cancer. Your blood cells form in your bone marrow. White blood cells help your body fight infection. In leukemia, the bone marrow pr...
5. `medquad_ans_0000013_2-1__dup3` (score=0.0402) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treat...

**Answer**:

Leukemia is a cancer of the blood cells, specifically affecting the bone marrow where blood cells are formed. It begins in the bone marrow and results in the production of abnormal white blood cells, known as leukemia cells. These abnormal cells crowd out healthy blood cells, including white blood cells, red blood cells, and platelets, impairing the blood's ability to fight infection, carry oxygen, and control bleeding. There are four common types of leukemia: acute lymphocytic leukemia, acute myeloid leukemia, chronic lymphocytic leukemia, and chronic myeloid leukemia. Chronic forms develop slowly and are more common in older adults, while acute forms progress rapidly and are more frequently seen in children, particularly acute lymphocytic leukemia [medquad_ans_0000554-1\_\_dup2][medquad_ans_0000040-1__dup3].

**Supported citations**: `medquad_ans_0000554-1__dup2`, `medquad_ans_0000040-1__dup3`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002485-1` (score=0.7954) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
2. `medquad_ans_0000113-4__dup3` (score=0.0012) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
3. `medquad_ans_0002825-1` (score=0.0004) - Hereditary fructose intolerance (HFI) is a metabolic disease caused by the absence of an enzyme called aldolase B. In people with HFI, ingestion of fructose (fruit sugar) and sucrose (cane or beet sugar, table sugar) ...
4. `medquad_ans_0000038-16` (score=0.0003) - When you visit your doctor, here are questions to ask about your kidneys. - What is my GFR? - What is my urine albumin result? - What is my blood pressure? - What is my blood glucose (for people with diabetes)? What i...
5. `medquad_ans_0000032-6__dup1` (score=0.0002) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most commonly occurs in people between the ages of 10 and 30. The most common site is the temporal lobe of the brain, although it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are among the most frequent tumors associated with epilepsy. Signs and symptoms depend on the tumor's location and may include seizures (most commonly), increased brain pressure, endocrine disorders, and focal symptoms. These tumors are generally slow-growing and usually do not become malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.328, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.7352) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0004068-1` (score=0.0810) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. Characteristic features include underdeveloped cheek bones, a very small lower jaw, cleft lip and/or palate, abnormalities...
3. `medquad_ans_0000658-1` (score=0.0291) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. The severity of this disorder varies among affected individuals. Children with Miller syndrome are born with underdevelope...
4. `medquad_ans_0000130-4__dup1` (score=0.0011) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.
5. `medquad_ans_0000658-2` (score=0.0004) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barré syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Like Guillain-Barré syndrome, symptoms may be preceded by a viral illness. Additional symptoms include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000130-1`

### Example 78 (`0006570-3`)

**Question**: What causes 22q11.2 deletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.538, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006570-3` (score=0.9947) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
2. `medquad_ans_0006579-3` (score=0.0223) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at ...
3. `medquad_ans_0001084-3` (score=0.0001) - Most people with 22q11.2 deletion syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22 in each cell. This region contains 30 to 40 genes, many of which have ...
4. `medquad_ans_0001078-3` (score=0.0001) - 1p36 deletion syndrome is caused by a deletion of genetic material from a specific region in the short (p) arm of chromosome 1. The signs and symptoms of 1p36 deletion syndrome are probably related to the loss of mult...
5. `medquad_ans_0001084-2` (score=0.0001) - 22q11.2 deletion syndrome affects an estimated 1 in 4,000 people. However, the condition may actually be more common than this estimate because doctors and researchers suspect it is underdiagnosed due to its variable ...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell, specifically near the middle of the chromosome at a location designated q11.2. Most people with this syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22, which contains about 30 to 40 genes, many of which have not been well characterized. Some affected individuals have smaller deletions. The deletion of the gene TBX1 is probably responsible for many of the syndrome's characteristic signs, such as heart defects, cleft palate, distinctive facial features, hearing loss, and low calcium levels. Loss of the gene COMT may also contribute to an increased risk of behavioral problems and mental illness. The loss of additional genes in the deleted region likely contributes to the various features of the syndrome [medquad_ans_00

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.911, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-12` (score=0.9888) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained....
2. `medquad_ans_0000062-2__dup3` (score=0.9619) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...
3. `medquad_ans_0000148-3__dup2` (score=0.0060) - For most healthy people who receive treatment soon after the outbreak of blisters, the lesions heal, the pain subsides within 3 to 5 weeks, and the blisters often leave no scars. However, shingles is a serious threat ...
4. `medquad_ans_0000062-4__dup3` (score=0.0025) - Burning, Itching, Tingling, Then a Rash An outbreak of shingles usually begins with a burning, itching, or tingling sensation on the back, chest, or around the rib cage or waist. It is also common for the face or eye ...
5. `medquad_ans_0000062-1__dup5` (score=0.0012) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single s...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After recovering from chickenpox, the virus remains in the body for life, hiding in nerve cells, usually in the spine. It can become active again when the immune system weakens, which may happen due to aging, certain medical conditions (such as cancer, HIV/AIDS, or transplant operations), or stress. When reactivated, the virus travels along a nerve path (dermatome) to the skin, causing a painful rash [medquad_ans_0000062-12][medquad_ans_0000062-2__dup3].

**Supported citations**: `medquad_ans_0000062-12`, `medquad_ans_0000062-2__dup3`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.831, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.9994) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0000333-2` (score=0.0001) - Fabry disease affects an estimated 1 in 40,000 to 60,000 males. This disorder also occurs in females, although the prevalence is unknown. Milder, late-onset forms of the disorder are probably more common than the clas...
3. `medquad_ans_0002458-1` (score=0.0001) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...
4. `medquad_ans_0000392-1` (score=0.0001) - Fryns syndrome is a condition that affects the development of many parts of the body. The features of this disorder vary widely among affected individuals and overlap with the signs and symptoms of several other disor...
5. `medquad_ans_0006312-4` (score=0.0001) - Is genetic testing available for VACTERL association? Because there is no known cause of VACTERL association, clinical genetic testing is not available for the condition. If an individual has a specific diagnosis of a...

**Answer**:

The provided context does not describe "Fryns Hofkens Fabry syndrome" as a valid or recognized medical condition. The context references Fryns syndrome and Fabry disease separately, and there is no information that combines these into a single condition called "Fryns Hofkens Fabry syndrome." Therefore, based on the available information, the symptoms of Fryns Hofkens Fabry syndrome cannot be determined.

Note: Fryns syndrome includes symptoms such as congenital diaphragmatic hernia, abnormalities of fingers and toes, distinctive facial features, severe developmental delay, and abnormalities in the brain, cardiovascular system, gastrointestinal system, kidneys, and genitalia [medquad_ans_0002458-1]. Fabry disease includes symptoms related to enzyme deficiency, such as skin rashes, pain, and organ damage, but specific symptoms for a combined condition are not listed [medquad_ans_0000333-

**Supported citations**: `medquad_ans_0002458-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.800, groundedness=0.746, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000024_2-8` (score=0.0064) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000006_3-8` (score=0.0053) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000004_6-7` (score=0.0025) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000040_1-6` (score=0.0021) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000025_1-8` (score=0.0014) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for osteoporosis. [medquad_ans_0000024_2-8] and [medquad_ans_0000006_3-8] discuss clinical trials for cancer treatments, such as chemoprevention and radiosensitizers, and drug therapies like lenalidomide, but do not address osteoporosis. Therefore, no information is available in the context regarding research for osteoporosis.

**Supported citations**: `medquad_ans_0000024_2-8`, `medquad_ans_0000006_3-8`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=1.000, groundedness=0.904, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000430-1__dup1` (score=0.9966) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
2. `medquad_ans_0000362-1__dup2` (score=0.9960) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
3. `medquad_ans_0000286-1__dup2` (score=0.0033) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
4. `medquad_ans_0000925-1__dup1` (score=0.0015) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
5. `medquad_ans_0000006-18` (score=0.0001) - You can help your doctor make a diagnosis by writing down key information about your dizziness or balance problem beforehand and giving the information to your doctor during the visit. Write down answers to these ques...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. These scams often promise quick fixes, miracle cures, ancient remedies, secret ingredients, or scientific breakthroughs for common health issues like weight loss, memory loss, sexual performance, and joint pain. They target individuals with serious conditions such as cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. To protect yourself, recognize red flags and consult a doctor or healthcare professional before taking unproven treatments, especially when using prescription drugs. The Food and Drug Administration addresses these issues [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.917, groundedness=0.055, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0000164-5__dup1` (score=0.0016) - People with either type 1 or type 2 diabetes must choose foods carefully to control their blood glucose, the bodys main source of energy. Following a meal plan to keep blood glucose at a healthy level may prevent CKD ...
2. `medquad_ans_0000015-24` (score=0.0006) - People with type 1 diabetes control their blood sugar with insulin -- either with shots or an insulin pen. Many people with type 2 diabetes can control blood glucose levels with diet and exercise alone. Others require...
3. `medquad_ans_0000009-3__dup3` (score=0.0006) - There are several things you can do to lower your risk for cataract. They include - having regular eye exams - quitting smoking - wearing sunglasses - taking care of other health problems - maintaining a healthy weigh...
4. `medquad_ans_0000060-4__dup3` (score=0.0002) - Peripheral Neuropathy Peripheral neuropathy is the most common type of diabetic neuropathy, and it affects the sensory nerves of your feet, legs, hands, and arms. These areas of your body may feel - numb - weak - cold...
5. `medquad_ans_0000095-8` (score=0.0002) - Following a healthy lifestyle can help you prevent overweight and obesity. Many lifestyle habits begin during childhood. Thus, parents and families should encourage their children to make healthy choices, such as foll...

**Answer**:

The provided context does not mention symptoms related to preventing diabetes problems or specifically addressing eye health. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000527-4` (score=0.9916) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
2. `medquad_ans_0000001-4` (score=0.9843) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chro...
3. `medquad_ans_0002561-3` (score=0.6152) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
4. `medquad_ans_0001084-3__dup1` (score=0.0010) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
5. `medquad_ans_0000553-4` (score=0.0004) - When Kabuki syndrome is caused by mutations in the KMT2D gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. When Kabuki s...

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern [medquad_ans_0000001-4].

**Supported citations**: `medquad_ans_0000001-4`
**Unsupported citations**: _none_

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.800, groundedness=0.915, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0001062-3` (score=0.9992) - Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. The ALAS2 gene provides instructions for making an enzyme called erythroid ALA-synthase, which plays a critical role in the production of heme (a compon...
2. `medquad_ans_0001063-3` (score=0.9982) - Mutations in the ABCB7 gene cause X-linked sideroblastic anemia and ataxia. The ABCB7 gene provides instructions for making a protein that is critical for heme production. Heme is a component of the hemoglobin protein...
3. `medquad_ans_0001063-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
4. `medquad_ans_0001062-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0006525-1` (score=0.0001) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...

**Answer**:

Genetic changes related to X-linked sideroblastic anemia include mutations in the ALAS2 gene and mutations in the ABCB7 gene. Mutations in the ALAS2 gene impair the activity of erythroid ALA-synthase, disrupting heme production and leading to reduced hemoglobin synthesis and excess iron accumulation in erythroblasts [medquad_ans_0001062-3]. Mutations in the ABCB7 gene alter the structure of the ABCB7 protein, disrupting its role in heme production and iron homeostasis in developing red blood cells, which results in anemia [medquad_ans_0001063-3]. These conditions are inherited in an X-linked recessive pattern, meaning the gene is located on the X chromosome [medquad_ans_0001063-4]. Additionally, individuals with mutations in both ALAS2 and HFE genes may

**Supported citations**: `medquad_ans_0001062-3`, `medquad_ans_0001063-3`, `medquad_ans_0001063-4`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.958, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.3290) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
2. `medquad_ans_0003601-1` (score=0.1475) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0003610-1` (score=0.0108) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
4. `medquad_ans_0000576-1` (score=0.0054) - LAMA2-related muscular dystrophy is a disorder that causes weakness and wasting (atrophy) of muscles used for movement (skeletal muscles). This condition generally appears in one of two ways: as a severe, early-onset ...
5. `medquad_ans_0003604-1` (score=0.0029) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...

**Answer**:

Limb-girdle muscular dystrophy is a group of diseases that cause weakness and wasting of the muscles in the arms and legs, particularly affecting the proximal muscles closest to the body, such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0000602-1]. It is a progressive condition leading to a loss of muscle strength and bulk over time, with onset possibly occurring in childhood, adolescence, young adulthood, or later in life [medquad_ans_0003601-1]. The severity, age of onset, and features vary among the subtypes and may differ even within the same family [medquad_ans_0000602-1]. Most forms are inherited in an autosomal recessive manner, while several rare forms are inherited in an autosomal dominant pattern [medquad_ans_0003601-1].

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

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000038-1__dup4` (score=0.9859) - What the Kidneys Do You have two kidneys. They are bean-shaped and about the size of a fist. They are located in the middle of your back, on the left and right of your spine, just below your rib cage. The kidneys filt...
2. `medquad_ans_0000144-10` (score=0.1711) - Renal failure is any acute or chronic loss of kidney function and is the term used when some kidney function remains. Total kidney failure, sometimes called end-stage renal disease (ESRD), indicates permanent loss of ...
3. `medquad_ans_0000167-1__dup4` (score=0.0330) - Polycystic kidney disease is a genetic disorder that causes numerous cysts to grow in the kidneys. A kidney cyst is an abnormal sac filled with fluid. PKD cysts can greatly enlarge the kidneys while replacing much of ...
4. `medquad_ans_0000804-1` (score=0.0270) - Polycystic kidney disease is a disorder that affects the kidneys and other organs. Clusters of fluid-filled sacs, called cysts, develop in the kidneys and interfere with their ability to filter waste products from the...
5. `medquad_ans_0000038-3__dup2` (score=0.0255) - Kidney Disease Kidney disease is often called a "silent" disease, because most people have no symptoms with early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NO...

**Answer**:

Kidney disease means the kidneys are damaged and can no longer remove wastes and extra water from the blood as they should [medquad_ans_0000038-1__dup4]. It is most often caused by diabetes or high blood pressure, which damage the blood vessels in the kidneys' filtering units (glomeruli), leading to reduced filtering ability. This damage typically occurs slowly over many years and is called chronic kidney disease [medquad_ans_0000038-1__dup4]. Early kidney disease usually has no symptoms [medquad_ans_0000038-1__dup4].

**Supported citations**: `medquad_ans_0000038-1__dup4`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.901, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000036_1-7` (score=0.7404) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000036_1-6` (score=0.7046) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopha...
3. `medquad_ans_0000058-6__dup1` (score=0.3091) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade...
4. `medquad_ans_0000058-23` (score=0.1778) - There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade of the tumor. But ...
5. `medquad_ans_0000058-3__dup3` (score=0.0263) - Symptoms Most cancers in their early, most treatable stages don't cause any symptoms. Early prostate cancer usually does not cause symptoms. However, if prostate cancer develops and is not treated, it can cause these ...

**Answer**:

The treatments for prostate cancer include watchful waiting or active surveillance, surgery, radiation therapy, cryosurgery, high-intensity focused ultrasound therapy, proton beam radiation therapy, and participation in clinical trials [medquad_ans_0000036_1-6]. Cryosurgery uses freezing to destroy prostate cancer cells, with ultrasound guiding the treatment; high-intensity focused ultrasound therapy uses high-energy sound waves to destroy cancer cells via an endorectal probe; proton beam radiation therapy is a type of external radiation therapy that targets tumors with protons [medquad_ans_0000036_1-7]. Clinical trials are also an option and may involve receiving standard or new treatments, contributing to future cancer treatment advancements [medquad_ans_0000036_1-7].

**Supported citations**: `medquad_ans_0000036_1-6`, `medquad_ans_0000036_1-7`
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

**Scores**: context_relevance=0.833, groundedness=0.530, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.4641) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0000635-1` (score=0.4579) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
3. `medquad_ans_0000635-4` (score=0.0009) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...
4. `medquad_ans_0000635-3` (score=0.0006) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
5. `medquad_ans_0001101-5` (score=0.0001) - These resources address the diagnosis or management of 7q11.23 duplication syndrome: - Cardiff University (United Kingdom): Copy Number Variant Research - Gene Review: Gene Review: 7q11.23 Duplication Syndrome - Genet...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia, delayed motor milestones (e.g., sitting up, crawling), recurrent infections, poor or absent speech, seizures, and/or spasticity. The condition results from an extra copy (duplication) of the MECP2 gene in each cell, which is generally caused by a duplication of genetic material on the long (q) arm of the X chromosome. It is inherited in an X-linked manner. In males, a duplication of the only copy of the MECP2 gene in each cell is sufficient to cause the condition, whereas in females, a duplication of one of the two copies typically does not cause the disorder due to X-inactivation, which inactivates one X chromosome in each somatic cell, usually randomly [medquad_ans_0003860

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=1.000, groundedness=0.714, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-1__dup3` (score=0.0038) - A heart murmur is an extra or unusual sound heard during a heartbeat. Murmurs range from very faint to very loud. Sometimes they sound like a whooshing or swishing noise. Normal heartbeats make a "lub-DUPP" or "lub-DU...
2. `medquad_ans_0000028_2-3` (score=0.0017) - Signs of melanoma include a change in the way a mole or pigmented area looks. These and other signs and symptoms may be caused by melanoma or by other conditions. Check with your doctor if you have any of the followin...
3. `medquad_ans_0000063-8` (score=0.0002) - Many Areas of Research Scientists are constantly searching for new ways to detect skin cancer, assess risk, and predict patient outcomes. They are interested in finding new treatments and new ways to deliver drugs and...
4. `medquad_ans_0000028-3__dup1` (score=0.0001) - Arachnoiditis is adisorder that causes chronic pain and neurological deficits and does not improve significantly with treatment.Surgery may only provide temporary relief. The outlook for someone witharachnoiditis isco...
5. `medquad_ans_0000153-3__dup1` (score=0.0001) - The outlook for children with hydranencephaly is generally poor, and many children with this disorder die before age 1. However, in rare cases, children with hydranencephaly may survive for several years or more.

**Answer**:

The provided context does not contain information about the outlook for melanoma. [medquad_ans_0000028_2-3] describes signs of melanoma but does not address prognosis or outlook. Therefore, the outlook for melanoma cannot be determined from the given context.

**Supported citations**: `medquad_ans_0000028_2-3`
**Unsupported citations**: _none_

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

**Scores**: context_relevance=0.900, groundedness=0.890, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000466-3` (score=0.9991) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
2. `medquad_ans_0000836-3` (score=0.0001) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...
3. `medquad_ans_0000764-3` (score=0.0001) - Cancers occur when a buildup of mutations in critical genesthose that control cell growth and division or repair damaged DNAallow cells to grow and divide uncontrollably to form a tumor. Most cases of ovarian cancer a...
4. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
5. `medquad_ans_0002838-1` (score=0.0001) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...

**Answer**:

The genetic changes related to hereditary multiple osteochondromas are mutations in the EXT1 and EXT2 genes. These mutations lead to defective production of the exostosin-1 and exostosin-2 proteins, which form a complex in the Golgi apparatus responsible for modifying heparan sulfate. When this modification is impaired due to mutations in EXT1 or EXT2, heparan sulfate becomes nonfunctional, contributing to the development of osteochondromas. Mutations in the EXT1 gene cause hereditary multiple osteochondromas type 1, and mutations in the EXT2 gene cause type 2. EXT1 mutations are found in 55 to 75 percent of cases and are associated with more severe symptoms compared to EXT2 mutations. Approximately 15 percent of individuals with the condition have no identifiable mutation in either EXT1 or EXT2 [medquad_ans_0000466-3].

**Supported citations**: `medquad_ans_0000466-3`
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.760, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000613-1` (score=0.0001) - Mabry syndrome is a condition characterized by intellectual disability, distinctive facial features, increased levels of an enzyme called alkaline phosphatase in the blood (hyperphosphatasia), and other signs and symp...
2. `medquad_ans_0000746-1` (score=0.0001) - Oculofaciocardiodental (OFCD) syndrome is a condition that affects the development of the eyes (oculo-), facial features (facio-), heart (cardio-) and teeth (dental). This condition occurs only in females. The eye abn...
3. `medquad_ans_0000749-1` (score=0.0001) - The Say-Barber-Biesecker-Young-Simpson (SBBYS) variant of Ohdo syndrome is a rare condition characterized by genital abnormalities in males, missing or underdeveloped kneecaps (patellae), intellectual disability, dist...
4. `medquad_ans_0004510-1` (score=0.0001) - Oculofaciocardiodental syndrome is a genetic syndrome that affects the eyes, heart, face, and teeth. Common signs and symptoms include abnormally small deep-set eyes, cataracts, long narrow face, a broad nasal tip tha...
5. `medquad_ans_0000075-2` (score=0.0001) - The prevalence of Asperger syndrome is not well established. Estimates range from 1 in 250 to 1 in 5,000 children. Three to four times as many males are affected than females. Because of changes in the way development...

**Answer**:

The provided context does not specify the number of people affected by oculofaciocardiodental (OFCD) syndrome. [medquad_ans_0000746-1] mentions that the condition affects females and describes associated abnormalities but does not provide population statistics or prevalence data. Therefore, the number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000746-1`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.0631) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0000038-16` (score=0.0104) - When you visit your doctor, here are questions to ask about your kidneys. - What is my GFR? - What is my urine albumin result? - What is my blood pressure? - What is my blood glucose (for people with diabetes)? What i...
3. `medquad_ans_0001350-1` (score=0.0046) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
4. `medquad_ans_0000113-4__dup3` (score=0.0017) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
5. `medquad_ans_0000189-2` (score=0.0005) - Type I citrullinemia is the most common form of the disorder, affecting about 1 in 57,000 people worldwide. Type II citrullinemia is found primarily in the Japanese population, where it occurs in an estimated 1 in 100...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described: Type I citrullinemia (also known as classic citrullinemia) and Type II citrullinemia. Type I citrullinemia usually becomes evident in the first few days of life, with symptoms including lethargy, poor feeding, vomiting, seizures, and loss of consciousness; a milder, later-onset form may present with intense headaches, partial loss of vision, ataxia, and lethargy. Type II citrullinemia chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma, and may appear during adulthood, often triggered by medications, infections, surgery, or alcohol intake. In some cases, the signs and symptoms of adult-onset type II citrull

**Supported citations**: _none_
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

**Scores**: context_relevance=0.857, groundedness=0.969, answer_relevance=0.514

**Top docs**:

1. `medquad_ans_0000949-5` (score=0.9918) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
2. `medquad_ans_0000944-5` (score=0.9918) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic ...
3. `medquad_ans_0005860-4` (score=0.0968) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respi...
4. `medquad_ans_0000190-5` (score=0.0002) - These resources address the diagnosis or management of cleidocranial dysplasia: - Gene Review: Gene Review: Cleidocranial Dysplasia - Genetic Testing Registry: Cleidocranial dysostosis - MedlinePlus Encyclopedia: Clei...
5. `medquad_ans_0000944-2` (score=0.0001) - Spondylocostal dysostosis is a rare condition, although its exact prevalence is unknown.

**Answer**:

Treatment for spondylocostal dysostosis usually consists of intensive medical care, including treatment of respiratory infections, bone surgery, and orthopedic treatment [medquad_ans_0005860-4].

**Supported citations**: `medquad_ans_0005860-4`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.919, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.9994) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0000786-3` (score=0.0001) - Perry syndrome results from mutations in the DCTN1 gene. This gene provides instructions for making a protein called dynactin-1, which is involved in the transport of materials within cells. To move materials, dynacti...
3. `medquad_ans_0000786-2` (score=0.0001) - Perry syndrome is very rare; about 50 affected individuals have been reported worldwide.
4. `medquad_ans_0001404-4` (score=0.0001) - How is Coffin-Siris syndrome diagnosed? Diagnosis of Coffin-Siris syndrome is largely based upon the presence or absence of common signs and symptoms in the individual. While formal diagnostic criteria have not been e...
5. `medquad_ans_0000593-1__dup1` (score=0.0001) - Axenfeld-Rieger syndrome is a group of eye disorders that affects the development of the eye. Common eye symptoms include cornea defects, which is the clear covering on the front of the eye, and iris defects, which is...

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
3. `medquad_ans_0000024-3__dup1` (score=0.0072) - APS improves significantly with anticoagulation therapy, which reduces the risk of further clots in veins and arteries. Treatment should be lifelong, since there is a high risk of further clots in individuals who stop...
4. `medquad_ans_0000034-2__dup2` (score=0.0007) - Within a few hours after onset of a stroke, treatment with drugs or devices that dissolve or break up the clot can restore blood flow to the brain and lead to a better recovery. To prevent strokes related to AF, docto...
5. `medquad_ans_0001027-5` (score=0.0001) - These resources address the diagnosis or management of warfarin resistance: - American Society of Hematology: Antithrombotic Therapy - MedlinePlus Drugs & Supplements: Warfarin - PharmGKB These resources from MedlineP...

**Answer**:

The signs and symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet. However, specific symptom frequencies are not detailed in the provided context [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.964, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0004262-1` (score=0.7196) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
2. `medquad_ans_0000701-1` (score=0.6933) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
3. `medquad_ans_0000608-1__dup1` (score=0.0121) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...
4. `medquad_ans_0000113-4__dup3` (score=0.0019) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
5. `medquad_ans_0000701-2` (score=0.0005) - Myhre syndrome is a rare disorder. Only about 30 cases have been documented in the medical literature. For reasons that are unknown, most affected individuals have been males.

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. It is caused by mutations in the SMAD4 gene and is inherited in an autosomal dominant pattern, with most cases resulting from a new mutation [medquad_ans_0004262-1]. The condition affects multiple systems and functions of the body, with features such as delayed development of language and motor skills, hearing loss (sensorineural, conductive, or mixed), growth restriction beginning before birth and continuing through adolescence, skin stiffness, and a muscular appearance [medquad_ans_0000701-1]. Typical facial features include short palpebral fissures, a short philtrum, midface hypoplasia, a small mouth

**Supported citations**: `medquad_ans_0004262-1`, `medquad_ans_0000701-1`
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006057-2` (score=0.9994) - What are the signs and symptoms of Thanatophoric dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Thanatophoric dysplasia. If the information is available, the table below ...
2. `medquad_ans_0000976-4` (score=0.0001) - Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell is sufficient to cause the condition. Virtually all cases of thanatophoric dysplasia are cau...
3. `medquad_ans_0000976-2` (score=0.0001) - This condition occurs in 1 in 20,000 to 50,000 newborns. Type I thanatophoric dysplasia is more common than type II.
4. `medquad_ans_0000250-1` (score=0.0001) - Craniometaphyseal dysplasia is a rare condition characterized by progressive thickening of bones in the skull (cranium) and abnormalities at the ends of long bones in the limbs (metaphyseal dysplasia). Except in the m...
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
5. `medquad_ans_0000113-4__dup3` (score=0.0013) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...

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

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004155-3` (score=0.9959) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
2. `medquad_ans_0004155-5` (score=0.0001) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...
3. `medquad_ans_0001006-5` (score=0.0001) - These resources address the diagnosis or management of tyrosinemia: - Baby's First Test: Tyrosinemia, Type I - Baby's First Test: Tyrosinemia, Type II - Baby's First Test: Tyrosinemia, Type III - Gene Review: Gene Rev...
4. `medquad_ans_0004155-4` (score=0.0001) - How is mucopolysaccharidosis I (MPS I) inherited? MPS I is inherited from both parents in an autosomal recessive pattern.
5. `medquad_ans_0000022-7__dup1` (score=0.0001) - To find out your risk for type 2 diabetes, check each item that applies to you. - I am age 45 or older. - I am overweight or obese. - I have a parent, brother, or sister with diabetes. - My family background is Africa...

**Answer**:

Mutations in the IDUA gene cause mucopolysaccharidosis type I (MPS I). The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme. The lack of IDUA enzyme activity leads to the accumulation of GAGs within cells, specifically inside the lysosomes. Lysosomes are compartments in the cell that digest and recycle different types of molecules. Conditions that cause molecules to build up inside the lysosomes, including MPS I, are called lysosomal storage disorders. The accumulation of GAGs increases the size of the lysosomes, which is why many tissues and organs are enlarged in this disorder. [medquad_ans_0004155-3]

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.800, groundedness=0.294, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000825-3` (score=0.9992) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
2. `medquad_ans_0000825-4` (score=0.0001) - When this condition is caused by mutations in the FLCN gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, ...
3. `medquad_ans_0000825-2` (score=0.0001) - Primary spontaneous pneumothorax is more common in men than in women. This condition occurs in 7.4 to 18 per 100,000 men each year and 1.2 to 6 per 100,000 women each year.
4. `medquad_ans_0000825-1` (score=0.0001) - Primary spontaneous pneumothorax is an abnormal accumulation of air in the space between the lungs and the chest cavity (called the pleural space) that can result in the partial or complete collapse of a lung. This ty...
5. `medquad_ans_0000181-3__dup2` (score=0.0001) - Alagille syndrome is caused by a gene mutation, or defect. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the DNA sequence that makes up a gene. DNA, or deoxyribon...

**Answer**:

Genetic changes related to primary spontaneous pneumothorax include mutations in the FLCN gene. These mutations can cause primary spontaneous pneumothorax, although they are considered a very rare cause of the condition. The FLCN gene provides instructions for making a protein called folliculin, which is found in connective tissue cells and in cells lining the alveoli of the lungs. Researchers believe folliculin may help control cell growth and division and may play a role in repairing and re-forming lung tissue after damage. Mutations in the FLCN gene are inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder. However, only about 40 percent of individuals with these mutations go on to develop primary spontaneous pneumothorax [medquad_ans_0000825-3, medquad_ans_0000825-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000825-3, medquad_ans_0000825-4`

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.958, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000066-23` (score=0.9929) - One test that helps doctors judge the severity of a stroke is the standardized NIH Stroke Scale, developed by the National Institute of Neurological Disorders and Stroke at the National Institutes of Health, or NIH. H...
2. `medquad_ans_0000129-4__dup1` (score=0.9755) - The National Institute of Neurological Disorders and Stroke (NINDS), a component of the National Institutes of Health (NIH) within the U.S. Department of Health and Human Services, is the nations primary funding sourc...
3. `medquad_ans_0000527-1__dup2` (score=0.9744) - A stroke is a medical emergency. There are two types - ischemic and hemorrhagic. Ischemic stroke is the most common type. It is usually caused by a blood clot that blocks or plugs a blood vessel in the brain. This kee...
4. `medquad_ans_0000857-1__dup2` (score=0.9664) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by ...
5. `medquad_ans_0000066-13` (score=0.9599) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...

**Answer**:

A stroke is a medical emergency that occurs when blood flow to the brain stops. There are two main types: ischemic stroke, which is the most common and is caused by a blood clot that blocks or plugs a blood vessel in the brain, and hemorrhagic stroke, which occurs when a blood vessel in the brain breaks and bleeds into the brain. Transient ischemic attacks (TIAs), also known as "mini-strokes," occur when the blood supply to the brain is interrupted briefly and can indicate a risk for a more serious stroke [medquad_ans_0000527-1\_\_dup2][medquad_ans_0000857-1__dup2]. Symptoms include sudden numbness or weakness of the face, arm, or leg (especially on one side of the body), sudden confusion, trouble speaking or understanding speech, sudden trouble seeing in one or both eyes, sudden trouble walking, dizziness,

**Supported citations**: `medquad_ans_0000527-1__dup2`, `medquad_ans_0000857-1__dup2`
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-3` (score=0.9953) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been s...
2. `medquad_ans_0000145-2__dup3` (score=0.0019) - The causes of Goodpasture syndrome are not fully understood. People who smoke or use hair dyes appear to be at increased risk for this condition. Exposure to hydrocarbon fumes, metallic dust, and certain drugs, such a...
3. `medquad_ans_0006276-6` (score=0.0001) - How might uncombable hair syndrome be treated? There is no definitive treatment for uncombable hair syndrome, but the condition usually improves or resolves on its own with the onset of puberty. Gentle hair care is ge...
4. `medquad_ans_0006276-1` (score=0.0001) - Uncombable hair syndrome (UHS) is a rare disorder of the hair shaft of the scalp. It usually is characterized by silvery-blond or straw-colored hair that is disorderly; stands out from the scalp; and cannot be combed ...
5. `medquad_ans_0006276-4` (score=0.0001) - Is uncombable hair syndrome inherited? Uncombable hair syndrome (UHS) is thought to be inherited in an autosomal dominant manner with reduced penetrance. Autosomal dominant means that having a change (mutation) in onl...

**Answer**:

Uncombable hair syndrome (UHS) is likely caused by premature keratinization of the inner root sheath, which forms an irregular, rigid tube that alters the shape of the emerging hair due to the triangular shape of the hair shaft seen in cross section. The condition is thought to be inherited in an autosomal dominant manner with reduced penetrance, although no responsible gene has been identified [medquad_ans_0006276-3].

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

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.9971) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000286-1__dup2` (score=0.0008) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...
3. `medquad_ans_0000043-6__dup1` (score=0.0001) - - Reach and maintain a reasonable body weight. - Make wise food choices most of the time. - Be physically active every day. - Take your prescribed medicines. Doing these things can reduce your risk of developing type ...
4. `medquad_ans_0000022-9` (score=0.0001) - You can do a lot to reduce your risk of getting type 2 diabetes. Being more physically active, reducing fat and calorie intake, and losing a little weight can help you lower your chances of developing type 2 diabetes....
5. `medquad_ans_0000150-4__dup2` (score=0.0001) - Most people with high blood pressure do not have symptoms. In rare cases, high blood pressure can cause headaches. Kidney disease also does not have symptoms in the early stages. A person may have swelling called edem...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It involves different types of surgery that often limit the amount of food you can take in, and some affect how you digest and absorb nutrients. All types of surgery carry risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off, and lifelong medical follow-up is required. [medquad_ans_0000969-1__dup2]

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.783, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000889-5` (score=0.9947) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
2. `medquad_ans_0001094-5` (score=0.9893) - These resources address the diagnosis or management of 3MC syndrome: - Genetic Testing Registry: Carnevale syndrome - Genetic Testing Registry: Craniofacial-ulnar-renal syndrome - Genetic Testing Registry: Malpuech fa...
3. `medquad_ans_0000084-2__dup1` (score=0.0236) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
4. `medquad_ans_0000392-5` (score=0.0027) - These resources address the diagnosis or management of Fryns syndrome: - Children's Hospital of Philadelphia: Treatment of Congenital Diaphragmatic Hernia - Gene Review: Gene Review: Fryns Syndrome - Genetic Testing R...
5. `medquad_ans_0004453-3` (score=0.0007) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not specify the treatments for Saethre-Chotzen syndrome. While it lists resources that address the diagnosis or management of Saethre-Chotzen syndrome, such as the Genetic Testing Registry and MedlinePlus Encyclopedia entries, no detailed information on specific treatments is included [medquad_ans_0000889-5]. Therefore, based on the given context, the treatments for Saethre-Chotzen syndrome cannot be determined.

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

**Scores**: context_relevance=0.714, groundedness=0.878, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000221-8` (score=0.9934) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
2. `medquad_ans_0000221-4__dup2` (score=0.0001) - Doctors have found four main types of kidney stones: - The most common types of stones contain calcium. Calcium is a normal part of a healthy diet. Calcium that is not used by the bones and muscles goes to the kidneys...
3. `medquad_ans_0000203-3__dup2` (score=0.0001) - Anyone can get a kidney stone, but some people are more likely to get one. Men are affected more often than women, and kidney stones are more common in non-Hispanic white people than in non-Hispanic black people and M...
4. `medquad_ans_0000203-4__dup2` (score=0.0001) - Kidney stones can form when substances in the urinesuch as calcium, oxalate, and phosphorusbecome highly concentrated. Certain foods may promote stone formation in people who are susceptible, but scientists do not bel...
5. `medquad_ans_0000221-12` (score=0.0001) - - A kidney stone is a solid piece of material that forms in a kidney when there are high levels of certain substances in the urine. These substances are normally found in the urine and do not cause problems at lower l...

**Answer**:

To diagnose kidney stones in adults, a doctor will perform a physical exam and ask about medical history, including family history of kidney stones, diet, digestive problems, and other health issues. The doctor may conduct urine, blood, and imaging tests to complete the diagnosis. Urine tests can detect infections or substances in urine that form stones. Blood tests can identify underlying conditions that contribute to stone formation. Imaging tests are used to locate kidney stones and may also reveal issues that caused the stones to form [medquad_ans_0000221-8].

**Supported citations**: `medquad_ans_0000221-8`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.987, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000205-6` (score=0.4465) - The job of the sphincter muscles is to hold urine in the bladder by squeezing the urethra shut. If the urethral sphincter fails to stay closed, urine may leak out of the bladder. When nerve signals are coordinated pro...
2. `medquad_ans_0000041_1-1` (score=0.0700) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
3. `medquad_ans_0000929-1__dup1` (score=0.0185) - The urethra is the tube that allows urine to pass out of the body. In men, it's a long tube that runs through the penis. It also carries semen in men. In women, it's short and is just above the vagina. Urethral proble...
4. `medquad_ans_0000213-3__dup2` (score=0.0044) - Urinary retention can result from - obstruction of the urethra - nerve problems - medications - weakened bladder muscles Obstruction of the Urethra Obstruction of the urethra causes urinary retention by blocking the n...
5. `medquad_ans_0000041_1-5` (score=0.0036) - Key Points - After urethral cancer has been diagnosed, tests are done to find out if cancer cells have spread within the urethra or to other parts of the body. - There are three ways that cancer spreads in the body. -...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. It begins in cells that line the urethra. A history of bladder cancer can affect the risk of urethral cancer. Signs of urethral cancer include bleeding or trouble with urination. Tests that examine the urethra and bladder are used to detect and diagnose urethral cancer. Certain factors affect prognosis and treatment options [medquad_ans_0000041_1-1].

**Supported citations**: `medquad_ans_0000041_1-1`
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=1.000, groundedness=0.753, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004671-4` (score=0.0251) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible...
2. `medquad_ans_0000308-4` (score=0.0001) - Dyskeratosis congenita can have different inheritance patterns. When dyskeratosis congenita is caused by DKC1 gene mutations, it is inherited in an X-linked recessive pattern. The DKC1 gene is located on the X chromos...
3. `medquad_ans_0004287-1` (score=0.0001) - Myotonia congenita is a genetic condition characterized by the inability of the skeletal muscles to quickly relax after a voluntary movement. The symptoms associated with the condition typically appear in childhood an...
4. `medquad_ans_0004725-1` (score=0.0001) - Paramyotonia congenita is an inherited condition that affects muscles used for movement (skeletal muscles), mainly in the face, neck, arms, and hands. Symptoms begin in infancy or early childhood and include episodes ...
5. `medquad_ans_0000771-2` (score=0.0001) - Paramyotonia congenita is an uncommon disorder; it is estimated to affect fewer than 1 in 100,000 people.

**Answer**:

The provided context does not specify the inheritance pattern of paramyotonia congenita. While it mentions that paramyotonia congenita is an inherited condition affecting skeletal muscles, the specific mode of inheritance is not detailed in the given text. [medquad_ans_0004725-1]

**Supported citations**: `medquad_ans_0004725-1`
**Unsupported citations**: _none_

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.540, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.9987) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000125-2` (score=0.0001) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
3. `medquad_ans_0000125-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the BTD gene in each cell have mutations. The parents of an individual with biotinidase deficiency each carry one copy of the m...
4. `medquad_ans_0000735-1__dup1` (score=0.0001) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form ...
5. `medquad_ans_0000125-1` (score=0.0001) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. If this condition is not recognized and treated, its signs and symptoms typically appear within the first few ...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=1.000, groundedness=0.400, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000846-1__dup1` (score=0.0003) - Summary : Playing sports can be fun, but it can also be dangerous if you are not careful. You can help prevent injuries by - Getting a physical to make sure you are healthy before you start playing your sport - Wearin...
2. `medquad_ans_0000505-1__dup1` (score=0.0002) - Summary : We usually think of air pollution as being outdoors, but the air in your house or office could also be polluted. Sources of indoor pollution include - Mold and pollen - Tobacco smoke - Household products and...
3. `medquad_ans_0000019-1__dup3` (score=0.0001) - Summary : Air pollution is a mixture of solid particles and gases in the air. Car emissions, chemicals from factories, dust, pollen and mold spores may be suspended as particles. Ozone, a gas, is a major part of air p...
4. `medquad_ans_0000708-1__dup2` (score=0.0001) - Summary : Pets can add fun, companionship and a feeling of safety to your life. Before getting a pet, think carefully about which animal is best for your family. What is each family member looking for in a pet? Who wi...
5. `medquad_ans_0000219-7` (score=0.0001) - To find the cause of your ED, your doctor may - take your medical and sexual history - ask you questions about your mental health - give you a physical exam - test your blood - give you a nighttime erection test - per...

**Answer**:

No, there is no information about water pollution in the provided context.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=1.000, groundedness=0.923, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.9937) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000028-3__dup1` (score=0.0008) - Arachnoiditis is adisorder that causes chronic pain and neurological deficits and does not improve significantly with treatment.Surgery may only provide temporary relief. The outlook for someone witharachnoiditis isco...
3. `medquad_ans_0000132-2__dup1` (score=0.0002) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the mus...
4. `medquad_ans_0000132-1__dup1` (score=0.0001) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend t...
5. `medquad_ans_0000132-4__dup1` (score=0.0001) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to the neurological conditions that cause foot drop in its laboratories at the National Institutes of Health (NIH), and als...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1]. Treatment options include the use of ankle-foot orthotics, exercise therapy, electrical stimulation of the peroneal nerve, and in some cases, surgery [medquad_ans_0000132-2__dup1]. Foot drop is a symptom of an underlying condition and can be either temporary or permanent, depending on the cause, which includes neurodegenerative disorders, motor neuron disorders, nerve root injuries, peripheral nerve disorders, local nerve compression, or muscle disorders [medquad_ans_0000132-1__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`, `medquad_ans_0000132-2__dup1`, `medquad_ans_0000132-1__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000145-1` (score=0.7351) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
2. `medquad_ans_0000915-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000916-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
4. `medquad_ans_0000012-20` (score=0.0003) - More information on COPD is available at: What is COPD? and at the Learn More, Breathe Better Campaign For information on quitting smoking, visit http://www.surgeongeneral.gov/tobacco/ or Smokefree.gov. For informatio...
5. `medquad_ans_0000145-2` (score=0.0001) - The prevalence of Camurati-Engelmann disease is unknown. Approximately 200 cases have been reported worldwide.

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. The age at which affected individuals first experience symptoms varies greatly; however, most people with this condition develop pain or weakness by adolescence. Camurati-Engelmann disease is caused by a mutation in the TGFB1 gene which is inherited in an autosomal dominant fashion. In some instances, people have the gene mutation that causes Camurati-Engelmann disease but never develop the characteristic features of this condition. In others, features are present, but a mutation cannot be identified. These cases are referred to as Camurati-Engelmann disease type

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.9994) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
2. `medquad_ans_0000316-4` (score=0.0001) - Emery-Dreifuss muscular dystrophy can have several different patterns of inheritance. When this condition is caused by mutations in the EMD gene, it is inherited in an X-linked recessive pattern. A condition is consid...
3. `medquad_ans_0003612-1` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy, type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy, type 2G. If the informatio...
4. `medquad_ans_0000316-1` (score=0.0001) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...
5. `medquad_ans_0003610-2` (score=0.0001) - What are the signs and symptoms of Limb-girdle muscular dystrophy, type 2C? The Human Phenotype Ontology provides the following list of signs and symptoms for Limb-girdle muscular dystrophy, type 2C. If the informatio...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include Achilles tendon contracture, elbow flexion contracture, decreased cervical spine flexion due to contractures of posterior cervical muscles, pectus excavatum, elevated serum creatine phosphokinase, primary atrial arrhythmia, atrioventricular block, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, childhood onset, juvenile onset, and X-linked recessive inheritance [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.799, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000870-5` (score=0.9972) - These resources address the diagnosis or management of RAMSVPS: - Calgary Retina Consultants: Retinal Arterial Macroaneurysm - Genetic Testing Registry: Retinal arterial macroaneurysm with supravalvular pulmonic steno...
2. `medquad_ans_0000402-5` (score=0.9971) - These resources address the diagnosis or management of GACI: - Gene Review: Gene Review: Generalized Arterial Calcification of Infancy - Genetic Testing Registry: Generalized arterial calcification of infancy 2 These ...
3. `medquad_ans_0000072-5` (score=0.9945) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
4. `medquad_ans_0000054-17` (score=0.0006) - Here are links to more information about P.A.D. from the National Heart, Lung, and Blood Institute. - What Is Peripheral Arterial Disease? - Atherosclerosis - The DASH Eating Plan - Facts about P.A.D. - Facts about P....
5. `medquad_ans_0005177-3` (score=0.0005) - How might pulmonary arterial hypertension be treated? People with pulmonary arterial hypertension (PAH) benefit from receiving treatment at specialized centers. The Pulmonary Hypertension Association offers a Find a D...

**Answer**:

The provided context does not specify the treatments for arterial tortuosity syndrome. While it mentions resources related to the diagnosis and management of arterial tortuosity syndrome, including the Gene Review, Genetic Testing Registry, Johns Hopkins McKusick-Nathans Institute of Genetic Medicine, and the National Heart, Lung, and Blood Institute's information on aneurysm treatment, no explicit details on treatments are given [medquad_ans_0000072-5]. Therefore, based on the available information, the treatments for arterial tortuosity syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.318, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0001112-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...
3. `medquad_ans_0001100-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
4. `medquad_ans_0001104-5` (score=0.0001) - Is genetic testing available for Charcot-Marie-Tooth disease type 2F? Yes. GeneTests lists the names of laboratories that are performing clincial genetic testing for Charcot-Marie-Tooth disease type 2F. To view the co...
5. `medquad_ans_0005142-1` (score=0.0001) - Pseudohypoparathyroidism type 1C is a genetic disorder that is very similar to hypoparathyroidism (parathyroid hormone levels are too low). However, pseudohypoparathyroidism is caused by no response to parathyroid hor...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 1C include decreased motor nerve conduction velocity, distal amyotrophy, distal muscle weakness, distal sensory impairment, hypertrophic nerve changes, hyporeflexia, juvenile onset, onion bulb formation, pes cavus, and segmental peripheral demyelination/remyelination [medquad_ans_0001092-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0001092-1`

### Example 127 (`0005321-1`)

**Question**: What are the symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005321-1` (score=0.9994) - What are the signs and symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Retinal arterial macroaneurysm...
2. `medquad_ans_0006446-2` (score=0.0001) - What are the signs and symptoms of Williams syndrome? The signs and symptoms of Williams syndrome can be variable, but the disorder is generally characterized by mild to moderate intellectual disability a distinctive ...
3. `medquad_ans_0000870-1` (score=0.0001) - Retinal arterial macroaneurysm with supravalvular pulmonic stenosis (RAMSVPS) is a disorder that affects blood vessels in the eyes and heart. The condition generally becomes apparent in infancy or childhood. RAMSVPS d...
4. `medquad_ans_0000199-1` (score=0.0001) - COL4A1-related brain small-vessel disease is part of a group of conditions called the COL4A1-related disorders. The conditions in this group have a range of signs and symptoms that involve fragile blood vessels. COL4A...
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
2. `medquad_ans_0001006-5` (score=0.0001) - These resources address the diagnosis or management of tyrosinemia: - Baby's First Test: Tyrosinemia, Type I - Baby's First Test: Tyrosinemia, Type II - Baby's First Test: Tyrosinemia, Type III - Gene Review: Gene Rev...
3. `medquad_ans_0000434-5` (score=0.0001) - These resources address the diagnosis or management of GRACILE syndrome: - Genetic Testing Registry: GRACILE syndrome - MedlinePlus Encyclopedia: Aminoaciduria - MedlinePlus Encyclopedia: Cholestasis These resources f...
4. `medquad_ans_0000612-5` (score=0.0001) - These resources address the diagnosis or management of lysinuric protein intolerance: - Gene Review: Gene Review: Lysinuric Protein Intolerance - Genetic Testing Registry: Lysinuric protein intolerance - MedlinePlus E...
5. `medquad_ans_0004788-1` (score=0.0001) - What are the signs and symptoms of Pellagra like syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pellagra like syndrome. If the information is available, the table below in...

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

**Scores**: context_relevance=0.833, groundedness=0.304, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000542-1` (score=0.9244) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
2. `medquad_ans_0003296-1` (score=0.8756) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
3. `medquad_ans_0000608-1__dup1` (score=0.0017) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...
4. `medquad_ans_0000113-4__dup3` (score=0.0015) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
5. `medquad_ans_0000542-2` (score=0.0004) - Jervell and Lange-Nielsen syndrome is uncommon; it affects an estimated 1.6 to 6 per 1 million people worldwide. This condition has a higher prevalence in Denmark, where it affects at least 1 in 200,000 people.

**Answer**:

Jervell Lange-Nielsen syndrome is a rare condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is a form of long QT syndrome, which results in the heart muscle taking longer than usual to recharge between beats. This disorder increases the risk of fainting (syncope) and sudden death beginning in early childhood. The syndrome is inherited in an autosomal recessive fashion and has two types: type 1 and type 2. It affects an estimated 1.6 to 6 per 1 million people worldwide, with a higher prevalence in Denmark, where it affects at least 1 in 200,000 people [medquad_ans_0000542-1, medquad_ans_0003296-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000542-1, medquad_ans_0003296-1`

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=0.917, groundedness=0.745, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000218-1` (score=0.0001) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
2. `medquad_ans_0004456-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004452-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004450-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004453-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

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
3. `medquad_ans_0000127-3__dup2` (score=0.0001) - Your body makes three types of blood cells: red blood cells, white blood cells, and platelets (PLATE-lets). Red blood cells contain hemoglobin, an iron-rich protein that carries oxygen from your lungs to all parts of ...
4. `medquad_ans_0000040-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The ATRX gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the g...
5. `medquad_ans_0000040-2` (score=0.0001) - Alpha thalassemia X-linked intellectual disability syndrome appears to be a rare condition, although its exact prevalence is unknown. More than 200 affected individuals have been reported.

**Answer**:

The genetic changes related to alpha thalassemia X-linked intellectual disability syndrome are mutations in the ATRX gene. These mutations change the structure of the ATRX protein, which likely prevents it from effectively regulating gene expression. Reduced activity of the HBA1 and HBA2 genes, which are necessary for normal hemoglobin production, causes alpha thalassemia. Abnormal expression of other unidentified genes probably leads to developmental delay, distinctive facial features, and other signs and symptoms of the syndrome [medquad_ans_0000040-3].

**Supported citations**: `medquad_ans_0000040-3`
**Unsupported citations**: _none_

### Example 133 (`0002622-2`)

**Question**: What are the symptoms of Glycogen storage disease type 6 ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002622-2` (score=0.9994) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
2. `medquad_ans_0000427-1` (score=0.0001) - Glycogen storage disease type VI (also known as GSDVI or Hers disease) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in liver cells. A lack of glycogen breakdown interfe...
3. `medquad_ans_0000424-5` (score=0.0001) - These resources address the diagnosis or management of glycogen storage disease type IV: - Gene Review: Gene Review: Glycogen Storage Disease Type IV - Genetic Testing Registry: Glycogen storage disease, type IV - Med...
4. `medquad_ans_0002622-1` (score=0.0001) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
5. `medquad_ans_0002620-4` (score=0.0001) - How is glycogen storage disease type 4 inherited? Glycogen storage disease type 4 is inherited in an autosomal recessive manner. This means that an individual must have 2 abnormal copies of the GBE1 gene to be affecte...

**Answer**:

The symptoms of Glycogen storage disease type 6 include hypoglycemia (seen in approximately 90% of patients), short stature (seen in approximately 90% of patients), hepatomegaly (enlarged liver), increased hepatic glycogen content, and postnatal growth retardation. These symptoms typically begin in infancy to early childhood, with the first sign often being an enlarged liver. Hypoglycemia and other symptoms may improve with age, and most adults with this condition do not have significant health problems [medquad_ans_0002622-2][medquad_ans_0000427-1].

**Supported citations**: `medquad_ans_0002622-2`, `medquad_ans_0000427-1`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=1.000, groundedness=0.762, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000042-5__dup2` (score=0.9951) - Diabetes Medications Hypoglycemia can occur as a side effect of some diabetes medications, including insulin and oral diabetes medicationspillsthat increase insulin production, such as - chlorpropamide (Diabinese) - g...
2. `medquad_ans_0000042-6__dup1` (score=0.9951) - Diabetes Medications Hypoglycemia can occur as a side effect of some diabetes medications, including insulin and oral diabetes medicationspillsthat increase insulin production, such as - chlorpropamide (Diabinese) - g...
3. `medquad_ans_0000037-5__dup2` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...
4. `medquad_ans_0000071-6__dup1` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...
5. `medquad_ans_0000037-6` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...

**Answer**:

The provided context does not address the causes of diabetes. It discusses hypoglycemia as a side effect of certain diabetes medications but does not identify the underlying causes of diabetes. Therefore, based on the given information, the causes of diabetes cannot be determined. [medquad_ans_0000042-5__dup2] [medquad_ans_0000042-6__dup1]

**Supported citations**: `medquad_ans_0000042-5__dup2`, `medquad_ans_0000042-6__dup1`
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

**Scores**: context_relevance=1.000, groundedness=0.921, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.9100) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000977-1__dup1` (score=0.0014) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from m...
3. `medquad_ans_0000123-1__dup2` (score=0.0003) - The brachial plexus is a network of nerves that conducts signals from the spine to the shoulder, arm, and hand. Brachial plexus injuries are caused by damage to those nerves. Symptoms may include - A limp or paralyzed...
4. `medquad_ans_0000845-1__dup1` (score=0.0002) - Exercising is good for you, but sometimes you can injure yourself when you play sports or exercise. Accidents, poor training practices, or improper gear can cause them. Some people get hurt because they are not in sha...
5. `medquad_ans_0000419-1__dup1` (score=0.0001) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, it may worsen. The best prevention is to limit exposure, especially when smoke is detected or fires are nearby [medquad_ans_0000516-1__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.961, answer_relevance=0.475

**Top docs**:

1. `medquad_ans_0000134-4__dup2` (score=0.9982) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...
2. `medquad_ans_0000134-9` (score=0.0005) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing Zollinger-Ellison syndrome.
3. `medquad_ans_0000134-3__dup2` (score=0.0002) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes horm...
4. `medquad_ans_0000010_1-2` (score=0.0002) - Health history can affect the risk of gastrointestinal carcinoid tumors. Anything that increases a person's chance of developing a disease is called a risk factor. Having a risk factor does not mean that you will get ...
5. `medquad_ans_0000075-2` (score=0.0001) - The prevalence of Asperger syndrome is not well established. Estimates range from 1 in 250 to 1 in 5,000 children. Three to four times as many males are affected than females. Because of changes in the way development...

**Answer**:

Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2].

**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.740, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.9949) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
2. `medquad_ans_0001092-5` (score=0.0030) - These resources address the diagnosis or management of 3-MCC deficiency: - Baby's First Test - Genetic Testing Registry: 3 Methylcrotonyl-CoA carboxylase 1 deficiency - Genetic Testing Registry: 3-methylcrotonyl CoA c...
3. `medquad_ans_0000497-5` (score=0.0022) - These resources address the diagnosis or management of hypermethioninemia: - Baby's First Test - Genetic Testing Registry: Glycine N-methyltransferase deficiency - Genetic Testing Registry: Hepatic methionine adenosyl...
4. `medquad_ans_0001859-5` (score=0.0013) - How might dihydrolipoamide dehydrogenase deficiency be treated? There are currently no consensus recommendations for the management of dihydrolipoamide dehydrogenase (DLD) deficiency. Management can be hard because va...
5. `medquad_ans_0000415-5` (score=0.0002) - These resources address the diagnosis or management of GLUT1 deficiency syndrome: - G1D Registry - Gene Review: Gene Review: Glucose Transporter Type 1 Deficiency Syndrome - Genetic Testing Registry: Glucose transport...

**Answer**:

The provided context does not specify treatments for tetrahydrobiopterin deficiency. While it lists diagnostic resources and genetic testing options for tetrahydrobiopterin deficiency [medquad_ans_0000974-5], there is no information on therapeutic interventions or management strategies. Therefore, based on the given context, the treatments for tetrahydrobiopterin deficiency cannot be determined.

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

**Scores**: context_relevance=1.000, groundedness=0.812, answer_relevance=0.655

**Top docs**:

1. `medquad_ans_0000525-3` (score=0.0013) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
2. `medquad_ans_0000525-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy is a liver disorder that occurs in pregnant women. Cholestasis is a condition that impairs the release of a digestive fluid called bile from liver cells. As a result, bile builds ...
3. `medquad_ans_0000829-1` (score=0.0001) - Progressive familial intrahepatic cholestasis (PFIC) is a disorder that causes progressive liver disease, which typically leads to liver failure. In people with PFIC, liver cells are less able to secrete a digestive f...
4. `medquad_ans_0003240-2` (score=0.0001) - What are the signs and symptoms of Intrahepatic cholestasis of pregnancy? The Human Phenotype Ontology provides the following list of signs and symptoms for Intrahepatic cholestasis of pregnancy. If the information is...
5. `medquad_ans_0000525-5` (score=0.0001) - These resources address the diagnosis or management of intrahepatic cholestasis of pregnancy: - Gene Review: Gene Review: ATP8B1 Deficiency - Genetic Testing Registry: Cholestasis of pregnancy These resources from Med...

**Answer**:

The provided context does not mention progressive familial intrahepatic cholestasis or describe genetic changes related to this condition. The context discusses genetic changes in the ABCB11 and ABCB4 genes associated with intrahepatic cholestasis of pregnancy, which is a different condition from progressive familial intrahepatic cholestasis [medquad_ans_0000525-3]. Therefore, based on the given information, the genetic changes related to progressive familial intrahepatic cholestasis cannot be determined.

Answer: Insufficient context to determine genetic changes related to progressive familial intrahepatic cholestasis.

**Supported citations**: `medquad_ans_0000525-3`
**Unsupported citations**: _none_

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.990, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.9994) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0000284-1` (score=0.0001) - Diastrophic dysplasia is a disorder of cartilage and bone development. Affected individuals have short stature with very short arms and legs. Most also have early-onset joint pain (osteoarthritis) and joint deformitie...
3. `medquad_ans_0000598-1__dup1` (score=0.0001) - Axial spondylometaphyseal dysplasia is a genetic disorder of bone growth. The term axial means towards the center of the body. Sphondylos is a Greek term meaning vertebra. Metaphyseal dysplasia refers to abnormalities...
4. `medquad_ans_0006057-4` (score=0.0001) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of tha...
5. `medquad_ans_0000578-1` (score=0.0001) - Langer mesomelic dysplasia is a disorder of bone growth. Affected individuals typically have extreme shortening of the long bones in the arms and legs (mesomelia). As a result of the shortened leg bones, people with L...

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
4. `medquad_ans_0003163-1` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
5. `medquad_ans_0006510-1` (score=0.0001) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...

**Answer**:

The provided context does not contain information about hepatic veno-occlusive disease with immunodeficiency or its inheritance pattern. Therefore, it is not possible to determine based on the given text whether this condition is inherited.

Answer: Insufficient context [medquad_ans_0000228-4, medquad_ans_0000480-1__dup1, medquad_ans_0003166-1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000228-4, medquad_ans_0000480-1__dup1, medquad_ans_0003166-1`

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.9993) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0000056-24` (score=0.0524) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions....
3. `medquad_ans_0000057-4__dup3` (score=0.0117) - Symptoms Vary With Disorders There are several types of taste disorders depending on how the sense of taste is affected. People who have taste disorders usually lose their ability to taste or can no longer perceive ta...
4. `medquad_ans_0000006-3__dup4` (score=0.0001) - Some people may have a balance problem without realizing it. Others might think they have a problem, but are too embarrassed to tell their doctor, friends, or family. Here are common symtoms experienced by people with...
5. `medquad_ans_0000057-22` (score=0.0001) - You can help your doctor make a diagnosis by writing down important information about your problem beforehand and giving the information to your doctor during your visit. Write down answers to the following questions....

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles information on how often a feature occurs in individuals with this condition [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.937, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000159-7` (score=0.9732) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
2. `medquad_ans_0000159-8` (score=0.9253) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
3. `medquad_ans_0000159-4__dup2` (score=0.0088) - Lupus nephritis is diagnosed through urine and blood tests and a kidney biopsy: - Urinalysis. Urinalysis is testing of a urine sample. The urine sample is collected in a special container in a health care providers of...
4. `medquad_ans_0000165-3__dup2` (score=0.0015) - Kidney disease in children can be caused by - birth defects - hereditary diseases - infection - nephrotic syndrome - systemic diseases - trauma - urine blockage or reflux From birth to age 4, birth defects and heredit...
5. `medquad_ans_0000159-6` (score=0.0006) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent further damage to the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication such as cyclophosphamide or mycophenolate mofetil, typically used in combination with prednisone [medquad_ans_0000159-8]. If lupus nephritis is causing high blood pressure, medications to lower blood pressure may be necessary, as this can significantly slow the progression of kidney disease [medquad_ans_0000159-8]. Additionally, people with kidney disease may need to consult a health care provider about dietary changes, such as reducing sodium intake, especially if they have high blood pressure [medquad_ans_0000159-7]. Diagnosis is typically confirmed through urine and blood tests,

**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0000159-7`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.864, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000869-5` (score=0.9811) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndro...
2. `medquad_ans_0000776-1__dup2` (score=0.0006) - Restless legs syndrome (RLS) causes a powerful urge to move your legs. Your legs become uncomfortable when you are lying down or sitting. Some people describe it as a creeping, crawling, tingling, or burning sensation...
3. `medquad_ans_0000824-1__dup2` (score=0.0002) - Is it hard for you to fall asleep or stay asleep through the night? Do you wake up feeling tired or feel very sleepy during the day, even if you have had enough sleep? You might have a sleep disorder. The most common ...
4. `medquad_ans_0001104-6` (score=0.0002) - How might Charcot-Marie-Tooth disease type 2F be treated? Treatment for Charcot-Marie-Tooth disease type 2 mainly focuses on the specific symptoms present. Affected individuals are often managed by a team of various s...
5. `medquad_ans_0000114-2__dup3` (score=0.0001) - Faulty Use of Iron or Lack of Iron Research suggests that the main cause of restless legs syndrome (RLS) is a faulty use of iron or a lack of iron in the brain. The brain uses iron to make the chemical dopamine (DO-pa...

**Answer**:

Treatments for Restless Legs Syndrome (RLS) include lifestyle changes such as maintaining regular sleep habits, practicing relaxation techniques, and engaging in moderate exercise during the day. If these measures do not alleviate symptoms, medications may be used to reduce the symptoms of RLS [medquad_ans_0000776-1__dup2]. The National Heart, Lung, and Blood Institute provides information on how RLS is treated, and the Restless Leg Syndrome Foundation also outlines treatment options [medquad_ans_0000869-5]. Additionally, in some cases, RLS may be managed as part of a broader approach that includes addressing underlying conditions such as anemia or pregnancy, or medication side effects [medquad_ans_0000776-1__dup2].

**Supported citations**: `medquad_ans_0000776-1__dup2`, `medquad_ans_0000869-5`
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002573-1` (score=0.6410) - Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue th...
2. `medquad_ans_0000038-16` (score=0.0027) - When you visit your doctor, here are questions to ask about your kidneys. - What is my GFR? - What is my urine albumin result? - What is my blood pressure? - What is my blood glucose (for people with diabetes)? What i...
3. `medquad_ans_0000113-4__dup3` (score=0.0013) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
4. `medquad_ans_0002573-5` (score=0.0010) - Is genetic testing available for glioblastoma? Genetic testing is not available for many people with glioblastoma since most of these tumors occur sporadically (by chance) and are not caused by a genetic mutation. How...
5. `medquad_ans_0000032-6__dup1` (score=0.0006) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of...

**Answer**:

Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue that helps repair brain damage in response to injury. Glioblastomas are often very aggressive and grow into surrounding brain tissue. Signs and symptoms, such as headache, nausea, vomiting, and/or drowsiness, may develop when the tumor begins to put excess pressure on the brain. Affected people may also experience other features depending on the size and location of the tumor. In most cases, the exact underlying cause is unknown; however, they can rarely occur in people with certain genetic syndromes such as neurofibromatosis type 1, Turcot syndrome, and Li Fraumeni syndrome. There is currently no cure for glioblastoma. Treatment is palliative and may include surgery

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0005496-3` (score=0.9944) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, s...
2. `medquad_ans_0004813-3` (score=0.0003) - What causes periodic fever, aphthous stomatitis, pharyngitis, cervical adenitis (PFAPA)? The cause of PFAPA is unknown, although viral or autoimmune causes have been suggested.
3. `medquad_ans_0002483-3` (score=0.0002) - What causes gamma heavy chain disease? The causes or risk factors for gamma heavy chain disease are not known.
4. `medquad_ans_0000058-14` (score=0.0001) - Scientists don't know exactly what causes prostate cancer. They cannot explain why one man gets prostate cancer and another does not. However, they have been able to identify some risk factors that are associated with...
5. `medquad_ans_0004177-3` (score=0.0001) - What causes multifocal choroiditis? Multifocal choroiditis occurs spontaneously and the cause is not currently known (idiopathic). It is possible that a bacterial or viral infection may trigger an immune response that...

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, which supports a genetic cause in some cases. Schizencephaly has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.845, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000694-3` (score=0.0002) - Although the cause of multiple sclerosis is unknown, variations in dozens of genes are thought to be involved in multiple sclerosis risk. Changes in the HLA-DRB1 gene are the strongest genetic risk factors for develop...
2. `medquad_ans_0000309-5__dup1` (score=0.0001) - Is genetic testing available for amyotrophic lateral sclerosis? Yes. Clinical genetic testing is currently available for several genes in which mutations are known to cause ALS. Genetic testing on a research basis is ...
3. `medquad_ans_0000316-1__dup1` (score=0.0001) - What are the signs and symptoms of Amyotrophic lateral sclerosis type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis type 6. If the information is a...
4. `medquad_ans_0000309-1__dup1` (score=0.0001) - Amyotrophic lateral sclerosis (ALS), also referred to as "Lou Gehrig's disease," is a progressive motor neuron disease which leads to problems with muscle control and movement. There are various types of ALS, which ar...
5. `medquad_ans_0000688-1__dup1` (score=0.0001) - Paralysis is the loss of muscle function in part of your body. It happens when something goes wrong with the way messages pass between your brain and muscles. Paralysis can be complete or partial. It can occur on one ...

**Answer**:

The provided context does not contain information about genetic changes related to amyotrophic lateral sclerosis (ALS). [medquad_ans_0000694-3] discusses genetic factors in multiple sclerosis, not ALS. [medquad_ans_0000309-5__dup1] mentions that genetic testing is available for ALS and refers to research and clinical testing for specific genes, but it does not specify the genetic changes associated with ALS. Therefore, based on the given context, the genetic changes related to ALS cannot be determined.

**Supported citations**: `medquad_ans_0000694-3`, `medquad_ans_0000309-5__dup1`
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.940, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000581-5` (score=0.9964) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
2. `medquad_ans_0003473-6` (score=0.9531) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
3. `medquad_ans_0004453-3` (score=0.0024) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004450-4` (score=0.0024) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004454-3` (score=0.0024) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The only specific treatment available for Laron syndrome is subcutaneous injections of insulin-like growth factor 1 (IGF-1), a growth-promoting hormone. IGF-1 stimulates linear growth (height), improves brain growth, and addresses metabolic abnormalities caused by long-term IGF-1 deficiency. It also raises blood glucose levels, reduces cholesterol, and increases muscle growth. IGF-1 and growth hormone (GH) levels should be closely monitored during treatment because overdosage of IGF-1 can cause various health problems [medquad_ans_0003473-6].

**Supported citations**: `medquad_ans_0003473-6`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000055-7` (score=0.9745) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...
2. `medquad_ans_0000016-19` (score=0.9650) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
3. `medquad_ans_0000016-15` (score=0.9647) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...
4. `medquad_ans_0000016-4__dup2` (score=0.8793) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...
5. `medquad_ans_0000016-3__dup2` (score=0.1108) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year to detect the disease before it causes damage to your vision. Visio...

**Answer**:

The treatments for diabetic retinopathy include controlling blood glucose, blood pressure, cholesterol, and triglycerides [medquad_ans_0000055-7]. For specific conditions such as macular edema, treatments include injections of medicines into the eye that block a protein causing abnormal blood vessel growth and fluid leakage, and focal laser treatment that places small laser burns near the macula to reduce fluid leakage [medquad_ans_0000055-7]. Scatter laser treatment is used to treat new, weak blood vessels away from the macula by causing them to shrink, reducing the risk of blindness [medquad_ans_0000055-7]. In severe cases with significant bleeding, a vitrectomy surgery may be required, where the clouded vitreous gel is removed and replaced with a salt solution [medquad_ans_0000055-7]. For proliferative

**Supported citations**: `medquad_ans_0000055-7`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000116-8` (score=0.9757) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during ...
2. `medquad_ans_0001421-3` (score=0.9645) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
3. `medquad_ans_0000119-7` (score=0.0249) - Treatment of proctitis depends on its cause. The goal of treatment is to reduce inflammation, control symptoms, and eliminate infection, if it is present. Only a doctor can determine the cause of proctitis and the bes...
4. `medquad_ans_0001007-5` (score=0.0028) - These resources address the diagnosis or management of ulcerative colitis: - American Society of Colon and Rectal Surgeons - Cedars-Sinai - Crohn's & Colitis Foundation of America: Colitis Diagnosis and Testing - Croh...
5. `medquad_ans_0000122-9` (score=0.0003) - Complications of ulcerative colitis can include - rectal bleedingwhen ulcers in the intestinal lining open and bleed. Rectal bleeding can cause anemia, which health care providers can treat with diet changes and iron ...

**Answer**:

Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases, the condition may resolve on its own (spontaneous remission), although most people continue to have ongoing or occasional diarrhea. Dietary changes are usually tried first to alleviate symptoms, which may include a reduced-fat diet, eliminating foods that contain caffeine and lactose, and avoiding over-the-counter pain relievers such as ibuprofen or aspirin. If these changes alone are not enough, medications can be used to help control symptoms. Prescription anti-inflammatory medications such as mesalamine and sulfasalazine may help reduce swelling. Steroids including budesonide and prednisone can be used to reduce inflammation, but they are usually only used to control sudden attacks of diarrhea. Long-term use of steroids is typically avoided because of unwanted side effects. Anti-diarrheal medications such as bismuth subsalicylate, diphenoxyl

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.667, groundedness=0.969, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000418-7` (score=0.8627) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
2. `medquad_ans_0000094-11` (score=0.0397) - A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - a...
3. `medquad_ans_0000904-1__dup2` (score=0.0200) - Your teeth are made of a hard, bonelike material. Inside the tooth are nerves and blood vessels. You need your teeth for many activities you may take for granted. These include eating, speaking and even smiling. But t...
4. `medquad_ans_0000396-3` (score=0.0161) - The FH gene provides instructions for making an enzyme called fumarase (also known as fumarate hydratase). Fumarase participates in an important series of reactions known as the citric acid cycle or Krebs cycle, which...
5. `medquad_ans_0000418-2__dup1` (score=0.0075) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis), wash your hands with warm water and soap after handling raw meat. Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; therefore, these methods are not reliable. Freeze pork less than 6 inches thick for 20 days at 5°F (-15°C) to kill any worms. However, freezing wild game meats may not effectively kill all worms because some worm species that infect wild game animals are freeze-resistant. Clean meat grinders thoroughly after each use. To help prevent Trichinella infection in animal populations, do not allow pigs or wild animals to eat uncooked meat, scraps, or carcasses of any animals, including rats, which may be infected with Trichinella [medquad_ans_0000418-7].

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

**Scores**: context_relevance=0.800, groundedness=0.948, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.9580) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0000006_3-7` (score=0.0017) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
3. `medquad_ans_0000006_3-5` (score=0.0015) - Tests that examine the brain and spinal cord are used to detect (find) childhood astrocytomas. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general signs o...
4. `medquad_ans_0000006_3-4` (score=0.0003) - The signs and symptoms of astrocytomas are not the same in every child. Signs and symptoms depend on the following: - Where the tumor forms in the brain or spinal cord. - The size of the tumor. - How fast the tumor gr...
5. `medquad_ans_0000006_9-7` (score=0.0002) - Key Points - The area where the tumor is found and the childs age are used in place of a staging system to plan cancer treatment. - The information from tests and procedures done to detect (find) childhood ependymoma ...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.950, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-1__dup3` (score=0.8728) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
2. `medquad_ans_0000013_2-2__dup3` (score=0.0018) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.
3. `medquad_ans_0000013_2-1__dup3` (score=0.0004) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treat...
4. `medquad_ans_0000013_3-4__dup2` (score=0.0003) - Treatment of atypical chronic myelogenous leukemia (CML) may include chemotherapy. Check the list of NCI-supported cancer clinical trials that are now accepting patients with atypical chronic myeloid leukemia, BCR-ABL...
5. `medquad_ans_0000013_3-2__dup3` (score=0.0001) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check ...

**Answer**:

Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In this condition, the body tells too many blood stem cells to become a type of white blood cell called granulocytes, and some of these stem cells never become mature white blood cells; these immature cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. The leukemia cells in atypical CML and CML look alike under a microscope, but in atypical CML, the "Philadelphia chromosome" chromosome change is not present [medquad_ans_0000013_3-1__dup3].

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=1.000, groundedness=0.934, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.9960) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
2. `medquad_ans_0000996-4` (score=0.0003) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...
3. `medquad_ans_0000996-2` (score=0.0001) - Trisomy 13 occurs in about 1 in 16,000 newborns. Although women of any age can have a child with trisomy 13, the chance of having a child with this condition increases as a woman gets older.
4. `medquad_ans_0000997-3` (score=0.0001) - Most cases of trisomy 18 result from having three copies of chromosome 18 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
5. `medquad_ans_0003520-3` (score=0.0001) - What causes Leber hereditary optic neuropathy (LHON)? Leber hereditary optic neuropathy is a condition related to changes in mitochondrial DNA. Mutations in the MT-ND1, MT-ND4, MT-ND4L, and MT-ND6 genes cause LHON. Th...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development [medquad_ans_0000996-3]. This can occur due to nondisjunction during the formation of reproductive cells (eggs and sperm), resulting in a cell with an extra copy of chromosome 13 [medquad_ans_0000996-4]. In rare cases, trisomy 13 can result from a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or early fetal development [medquad_ans_0000996-3]. In these translocation cases, individuals have two normal copies of chromosome 13 plus an extra copy attached to another chromosome [medquad_ans_0000996-3].

**Supported citations**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.857, groundedness=0.936, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.9866) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - G...
2. `medquad_ans_0000218-2__dup1` (score=0.9733) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bo...
3. `medquad_ans_0004415-3` (score=0.8279) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
4. `medquad_ans_0000019_3-7` (score=0.0434) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
5. `medquad_ans_0000084-2__dup1` (score=0.0379) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...

**Answer**:

There is currently no cure for Niemann-Pick disease. Treatment is supportive and may include care from specialists such as pulmonologists for respiratory problems, cardiologists for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists [medquad_ans_0000218-2__dup1]. For Niemann-Pick disease type A, there is no specific treatment, and management focuses on supportive care [medquad_ans_0004415-3]. Bone marrow transplantation has been attempted in a few individuals with type B, and enzyme replacement and gene therapies may be helpful for those with type B [medquad_ans_0000218-2__dup1]. Restricting one's diet does not prevent the buildup of lipids in cells and tissues [medquad_ans_0000218-2__dup1]. New treatments are being

**Supported citations**: `medquad_ans_0000218-2__dup1`, `medquad_ans_0004415-3`
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

**Scores**: context_relevance=1.000, groundedness=0.885, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003789-3` (score=0.9941) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Man...
2. `medquad_ans_0000004_7-1` (score=0.0001) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major ...
3. `medquad_ans_0003789-1` (score=0.0001) - Mantle cell lymphoma (MCL) belongs to a group of diseases known as non-Hodgkins lymphomas (NHL), which are cancers that affect the the lymphatic system (part of the immune system). MCL accounts for 6% of all non-Hodgk...
4. `medquad_ans_0000004_6-1` (score=0.0001) - Key Points - Childhood Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - There are two types of childhood Hodgkin lymphoma. - Epstein-Barr virus infection increases the risk o...
5. `medquad_ans_0003789-4` (score=0.0001) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be perf...

**Answer**:

Mantle cell lymphoma is caused by DNA damage within cells, which can result from a combination of factors. Many cases are associated with a chromosomal translocation [medquad_ans_0003789-3]. Additionally, environmental factors such as hair dyes and pesticides, as well as immunodeficiencies, may contribute to the development of non-Hodgkin lymphomas, including mantle cell lymphoma [medquad_ans_0003789-3].

**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000836-4` (score=0.0254) - Many cases of prostate cancer are not related to inherited gene changes. These cancers are associated with somatic mutations that occur only in certain cells in the prostate. When prostate cancer is related to inherit...
2. `medquad_ans_0000611-4` (score=0.0230) - Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk. It is important to note that people inhe...
3. `medquad_ans_0000608-4` (score=0.0048) - Most cases of lung cancer are not related to inherited gene changes. These cancers are associated with somatic mutations that occur only in certain cells in the lung. When lung cancer is related to inherited gene chan...
4. `medquad_ans_0003718-4` (score=0.0007) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
5. `medquad_ans_0000734-4__dup1` (score=0.0001) - Is Binswanger's disease an inherited condition? Although Binswanger's disease is not considered an inherited condition, genetics may play a role in many of the conditions and risk factors that are associated with the ...

**Answer**:

The provided context does not contain any information about Legg-Calv-Perthes disease or its inheritance. Therefore, it is not possible to determine based on the given text whether Legg-Calv-Perthes disease is inherited.

Answer: Insufficient context [medquad_ans_0000836-4, medquad_ans_0000611-4, medquad_ans_0000608-4, medquad_ans_0003718-4, medquad_ans_0000734-4__dup1].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000836-4, medquad_ans_0000611-4, medquad_ans_0000608-4, medquad_ans_0003718-4, medquad_ans_0000734-4__dup1`

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000192-1__dup1` (score=0.9699) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage ...
2. `medquad_ans_0003965-1` (score=0.7330) - Metachromatic leukodystrophy is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially cells of the nervous system. This accumulation results in progressive destruction...
3. `medquad_ans_0000647-1` (score=0.2649) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
4. `medquad_ans_0003566-1` (score=0.1022) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
5. `medquad_ans_0000181-1__dup2` (score=0.0983) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...

**Answer**:

Metachromatic leukodystrophy (MLD) is an inherited condition characterized by the accumulation of fats called sulfatides in cells, especially in cells of the nervous system. This accumulation results in progressive destruction of the white matter of the brain, which consists of nerve fibers covered by myelin. Affected individuals experience progressive deterioration of intellectual functions and motor skills, such as the ability to walk, as well as loss of sensation in the extremities, incontinence, seizures, paralysis, and inability to speak [medquad_ans_0003965-1]. MLD is one of a group of genetic disorders called leukodystrophies, characterized by the toxic buildup of lipids and other storage materials in cells of the central nervous system and peripheral nerves, impairing the growth or development of the myelin sheath. It is caused by a deficiency of the enzyme arylsulfatase A, which is necessary to

**Supported citations**: `medquad_ans_0003965-1`
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.984, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.9994) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0000190-1` (score=0.0001) - Cleidocranial dysplasia is a condition that primarily affects the development of the bones and teeth. Signs and symptoms of cleidocranial dysplasia can vary widely in severity, even within the same family. Individuals...
3. `medquad_ans_0000284-1` (score=0.0001) - Diastrophic dysplasia is a disorder of cartilage and bone development. Affected individuals have short stature with very short arms and legs. Most also have early-onset joint pain (osteoarthritis) and joint deformitie...
4. `medquad_ans_0000154-14` (score=0.0001) - - Kidney dysplasia is a condition in which the internal structures of one or both of a fetus kidneys do not develop normally while in the womb. - Genetic factors can cause kidney dysplasia. - Genetic syndromes that af...
5. `medquad_ans_0006057-4` (score=0.0001) - Is thanatophoric dysplasia inherited? Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell causes the condition. However, almost all cases of tha...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000251-1` (score=0.8085) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
2. `medquad_ans_0000251-2` (score=0.0005) - Cri-du-chat syndrome occurs in an estimated 1 in 20,000 to 50,000 newborns. This condition is found in people of all ethnic backgrounds.
3. `medquad_ans_0000113-4__dup3` (score=0.0003) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
4. `medquad_ans_0000251-4` (score=0.0002) - Most cases of cri-du-chat syndrome are not inherited. The deletion occurs most often as a random event during the formation of reproductive cells (eggs or sperm) or in early fetal development. Affected people typicall...
5. `medquad_ans_0000251-3` (score=0.0001) - Cri-du-chat syndrome is caused by a deletion of the end of the short (p) arm of chromosome 5. This chromosomal change is written as 5p-. The size of the deletion varies among affected individuals; studies suggest that...

**Answer**:

Cri-du-chat syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that sounds like that of a cat. The disorder is characterized by intellectual disability and delayed development, small head size (microcephaly), low birth weight, and weak muscle tone (hypotonia) in infancy. Affected individuals also have distinctive facial features, including widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children with cri-du-chat syndrome are born with a heart defect. Most cases are not inherited and occur as a random event during the formation of reproductive cells or in early fetal development, with about 10 percent of cases resulting from inheritance of a chromosomal rearrangement from an unaffected parent who carries a balanced translocation [medquad*ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.9992) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0000722-2__dup1` (score=0.0003) - What are the signs and symptoms of Bilateral perisylvian polymicrogyria? The signs and symptoms of bilateral perisylvian polymicrogyria (BPP) vary but may include: Partial paralysis of muscles on both sides of the fac...
3. `medquad_ans_0000215-1` (score=0.0001) - Congenital bilateral absence of the vas deferens occurs in males when the tubes that carry sperm out of the testes (the vas deferens) fail to develop properly. Although the testes usually develop and function normally...
4. `medquad_ans_0000718-1__dup1` (score=0.0001) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
5. `medquad_ans_0000722-4__dup1` (score=0.0001) - Is bilateral perisylvian polymicrogyria inherited? In most cases, bilateral perisylvian polymicrogyria (BPP) occurs sporadically in people with no family history of the condition. Rarely, more than one family member m...

**Answer**:

The signs and symptoms of bilateral frontoparietal polymicrogyria may include: moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, cerebellar hypoplasia, cerebral dysmyelination, esotropia, exotropia, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, polymicrogyria with an anterior to posterior gradient, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.988, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003414-3` (score=0.9965) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
2. `medquad_ans_0000674-3__dup1` (score=0.0003) - What causes Behr syndrome? The exact cause of Behr syndrome is not known; however, a genetic cause is suspected based on the families identified, thus far.
3. `medquad_ans_0002414-3` (score=0.0001) - What causes Fowlers syndrome? The cause of Fowler's syndrome is not known. The association of Fowlers syndrome and polycystic ovaries in some patients raises the possibility that the syndrome is linked in some way to ...
4. `medquad_ans_0003402-3` (score=0.0001) - What causes Klippel Feil syndrome (KFS)? The specific underlying causes and mechanisms of Klippel Feil syndrome (KFS)are not well understood. In general medical researchers believe KFS happens when the tissue of the e...
5. `medquad_ans_0002747-3` (score=0.0001) - What causes HELLP syndrome? Doctors are still unclear on what exactly causes HELLP syndrome. Although it is more common in women who have preeclampsia or pregnancy induced hypertension (high blood pressure), there are...

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein. This is written as Arg445His or R445H. It is unclear why the R445H mutation causes both hearing and vision loss in affected individuals. [medquad_ans_0003414-3]

**Supported citations**: `medquad_ans_0003414-3`
**Unsupported citations**: _none_

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.507, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004584-2` (score=0.9992) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...
2. `medquad_ans_0004589-2` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...
3. `medquad_ans_0004584-1` (score=0.0001) - Orofaciodigital syndrome 1 (OFD1), also called orofaciodigital syndrome type 1, is a condition that affects the development of the oral cavity (the mouth and teeth), facial features, and digits (fingers and toes). Thi...
4. `medquad_ans_0004592-1` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 5. If the information is available, the table ...
5. `medquad_ans_0004593-1` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 8? The Human Phenotype Ontology provides the following list of signs and symptoms for Orofaciodigital syndrome 8. If the information is available, the table ...

**Answer**:

The symptoms of Orofaciodigital syndrome 1 include oral features such as a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other facial abnormalities. Finger and toe abnormalities may include shortness (brachydactyly), webbing or joining (syndactyly), abnormally curved fingers (clinodactyly), or other malformations. Brain abnormalities such as cysts and kidney problems like polycystic kidney disease may also occur. About half of individuals have some degree of learning disability, usually mild. Additionally, symptoms include bifid tongue (90%), broad alveolar ridges (90%), cleft upper lip (90%), frontal bossing (90%), hypert

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=1.000, groundedness=0.717, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.9965) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000694-1__dup1` (score=0.0276) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...
3. `medquad_ans_0000130-1__dup3` (score=0.0026) - Summary : If you need a mastectomy, you have a choice about whether or not to have surgery to rebuild the shape of the breast. Instead of breast reconstruction, you could choose to wear a breast form that replaces the...
4. `medquad_ans_0000925-1__dup1` (score=0.0003) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
5. `medquad_ans_0000286-1__dup2` (score=0.0001) - Summary : Dietary supplements are vitamins, minerals, herbs, and many other products. They can come as pills, capsules, powders, drinks, and energy bars. Supplements do not have to go through the testing that drugs do...

**Answer**:

Yes, the provided context includes information about surgery. Weight loss surgery is discussed, including its purpose, types, effects on food intake and nutrient absorption, risks such as infections, hernias, and blood clots, and the importance of long-term medical follow-up [medquad_ans_0000969-1__dup2]. Additionally, general advice on patient involvement in surgical decisions is provided, such as asking questions, getting a second opinion, and understanding treatment plans [medquad_ans_0000694-1__dup1]. Information is also given about mastectomy and breast reconstruction options, including timing and methods of reconstruction [medquad_ans_0000130-1__dup3].

These summaries cover various types of surgical procedures and patient considerations related to surgery.

**Supported citations**: `medquad_ans_0000969-1__dup2`, `medquad_ans_0000694-1__dup1`, `medquad_ans_0000130-1__dup3`
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

**Scores**: context_relevance=0.833, groundedness=0.543, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000522-1__dup1` (score=0.5592) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
2. `medquad_ans_0001152-1` (score=0.0115) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD d...
3. `medquad_ans_0000758-1__dup2` (score=0.0098) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000009-1__dup2` (score=0.0084) - Asbestos-related lung diseases are diseases caused by exposure to asbestos (as-BES-tos) fibers. Asbestos is a mineral that, in the past, was widely used in many industries. Asbestos is made up of tiny fibers that can ...
5. `medquad_ans_0000038-1__dup2` (score=0.0032) - Childhood interstitial (in-ter-STISH-al) lung disease, or chILD, is a broad term for a group of rare lung diseases that can affect babies, children, and teens. These diseases have some similar symptoms, such as chroni...

**Answer**:

Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing in dust or other particles in the air is responsible for some types of interstitial lung diseases. Specific types include black lung disease among coal miners from inhaling coal dust, farmer's lung from inhaling farm dust, asbestosis from inhaling asbestos fibers, siderosis from inhaling iron from mines or welding fumes, and silicosis from inhaling silica dust. Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types of interstitial lung disease have no known cause. Treatment depends on the type of exposure and the stage of the disease and may involve medicines, oxygen therapy, or a lung transplant in severe cases [medquad_ans_0000522-1\_\_dup

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.106, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000527-4` (score=0.3595) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
2. `medquad_ans_0002561-3` (score=0.1977) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
3. `medquad_ans_0003718-4` (score=0.0005) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
4. `medquad_ans_0001084-3__dup1` (score=0.0003) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
5. `medquad_ans_0000562-4` (score=0.0002) - When Klippel-Feil syndrome is caused by mutations in the GDF6 or GDF3 genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder....

**Answer**:

The provided context does not contain information about the inheritance pattern of Melnick-Needles syndrome. Therefore, based on the given documents, it is not possible to determine whether Melnick-Needles syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=1.000, groundedness=0.533, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006138-1` (score=0.9703) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
2. `medquad_ans_0004732-1` (score=0.0017) - Parapsoriasis describes a group of skin diseases that can be characterized by scaly patches or slightly elevated papules and/or plaques (red, scaly patches) that have a resemblance to psoriasis. However, this descript...
3. `medquad_ans_0000113-4__dup3` (score=0.0011) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
4. `medquad_ans_0000008-15` (score=0.0002) - One definition of cure is being alive and free of breast cancer for 5 years. If the cancer is found early, a woman's chances of survival are better. In fact, nearly 98 percent of women who discover their breast cancer...
5. `medquad_ans_0000032-6__dup1` (score=0.0001) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse. There are two forms: a congenital form (primary TBM) that typically develops during infancy or early childhood, and an acquired form (secondary TBM) usually seen in adults. Some individuals may initially have no signs or symptoms, but the condition is typically progressive and most people eventually develop symptoms such as shortness of breath, cough, sputum retention, and wheezing or stridor with breathing. Most cases of primary TBM are caused by genetic conditions that weaken the airway walls, while secondary TBM often results from trauma, chronic inflammation, or prolonged compression of the airways. Treatment is generally only required in those with symptoms and may include stenting, surgical correction, continuous positive airway pressure

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=0.800, groundedness=0.993, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000880-1__dup2` (score=0.9909) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
2. `medquad_ans_0000531-1__dup2` (score=0.8110) - Your jaw is a set of bones that holds your teeth. It consists of two main parts. The upper part is the maxilla. It doesn't move. The moveable lower part is called the mandible. You move it when you talk or chew. The t...
3. `medquad_ans_0006030-1` (score=0.4411) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
4. `medquad_ans_0000079-1__dup1` (score=0.0036) - While acute pain is a normal sensation triggered in the nervous system to alert you to possible injury and the need to take care of yourself, chronic pain is different. Chronic pain persists. Pain signals keep firing ...
5. `medquad_ans_0000608-1__dup1` (score=0.0019) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...

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
4. `medquad_ans_0000527-4` (score=0.3683) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
5. `medquad_ans_0001084-3__dup1` (score=0.0020) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).

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
5. `medquad_ans_0000816-2__dup1` (score=0.0001) - What are the signs and symptoms of Brachydactyly type B? The Human Phenotype Ontology provides the following list of signs and symptoms for Brachydactyly type B. If the information is available, the table below includ...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000232-1__dup2` (score=0.8652) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...
2. `medquad_ans_0000133-1__dup2` (score=0.7908) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
3. `medquad_ans_0000197-1__dup3` (score=0.1864) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
4. `medquad_ans_0000021-1__dup3` (score=0.0154) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Con...
5. `medquad_ans_0000008-1__dup3` (score=0.0084) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus and can also cause shortness of breath, wheezing, a low fever, and chest tightness. It is one type of COPD (chronic obstructive pulmonary disease). The inflamed bronchial tubes produce a lot of mucus, leading to coughing and difficulty breathing. Cigarette smoking is the most common cause, and long-term exposure to air pollution, fumes, or dust can also contribute. Chronic bronchitis is a long-term condition that keeps coming back or never goes away completely. If you smoke, it is important to quit. Treatment often includes medicines to open the airways and help clear mucus, and may also include oxygen therapy and pulmonary rehabilitation [medquad_ans_0000197-1__dup3].

**Supported citations**: `medquad_ans_0000197-1__dup3`
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.625, groundedness=0.725, answer_relevance=0.825

**Top docs**:

1. `medquad_ans_0000937-5` (score=0.9966) - These resources address the diagnosis or management of SMARD1: - Genetic Testing Registry: Spinal muscular atrophy with respiratory distress 1 These resources from MedlinePlus offer information about the diagnosis and...
2. `medquad_ans_0000936-5` (score=0.9910) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscula...
3. `medquad_ans_0000935-5` (score=0.9884) - These resources address the diagnosis or management of spinal muscular atrophy: - Gene Review: Gene Review: Spinal Muscular Atrophy - Gene Review: Gene Review: Spinal Muscular Atrophy, X-Linked Infantile - Genetic Tes...
4. `medquad_ans_0000934-5` (score=0.0224) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - Med...
5. `medquad_ans_0000173-4__dup1` (score=0.0182) - Between 2003 and 2012, the NINDS piloted the Spinal Muscular Atrophy Project to expedite therapeutics development for this hereditary neurodegenerative disease. The Project was designed to accelerate the research proc...

**Answer**:

The provided context does not specify the treatments for Spinal Muscular Atrophy (SMA). While it mentions resources related to diagnosis and management, such as diagnostic tests, drug therapy, genetic counseling, and palliative care, it does not detail specific treatments for SMA [medquad_ans_0000935-5]. Therefore, based on the given information, the exact treatments are not available.

**Supported citations**: `medquad_ans_0000935-5`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.913, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000001_1-7` (score=0.9738) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
2. `medquad_ans_0000013-5` (score=0.0349) - These resources address the diagnosis or management of acute promyelocytic leukemia: - American Cancer Society: Diagnosis of Acute Myeloid Leukemia - American Cancer Society: Treatment of Acute Promyelocytic (M3) Leuk...
3. `medquad_ans_0000001_2-7` (score=0.0111) - Key Points - There are different types of treatment for patients with adult acute myeloid leukemia. - The treatment of adult AML usually has 2 phases. - Four types of standard treatment are used: - Chemotherapy - Radi...
4. `medquad_ans_0000001_6-7` (score=0.0069) - Key Points - There are different types of treatment for childhood acute lymphoblastic leukemia (ALL). - Children with ALL should have their treatment planned by a team of doctors who are experts in treating childhood ...
5. `medquad_ans_0000001_2-2` (score=0.0002) - Smoking, previous chemotherapy treatment, and exposure to radiation may affect the risk of adult AML. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean ...

**Answer**:

The treatments for Adult Acute Lymphoblastic Leukemia (ALL) include chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and biologic therapy. The treatment is typically divided into two phases: remission induction therapy, which aims to kill leukemia cells in the blood and bone marrow to achieve remission, and post-remission therapy, which targets any remaining leukemia cells to prevent relapse. Central nervous system (CNS) sanctuary therapy is also commonly given during each phase to address leukemia cells that may hide in the brain and spinal cord. Additionally, patients may consider participating in clinical trials, which are research studies testing new or improved treatments. [medquad_ans_0000001_1-7]

**Supported citations**: `medquad_ans_0000001_1-7`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002439-3` (score=0.9899) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
2. `medquad_ans_0001042-3__dup1` (score=0.2452) - What causes cerebellar degeneration? Cerebellar degeneration can be caused by a variety of different conditions. Neurological diseases that can lead to cerebellar degeneration include: Acute and hemorrhagic stroke can...
3. `medquad_ans_0000032-1__dup1` (score=0.0001) - Ataxia often occurs when parts of the nervous system that control movement are damaged. People with ataxia experience a failure of muscle control in their arms and legs, resulting in a lack of balance and coordination...
4. `medquad_ans_0000078-5` (score=0.0001) - These resources address the diagnosis or management of ataxia with oculomotor apraxia: - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type 1 - Gene Review: Gene Review: Ataxia with Oculomotor Apraxia Type ...
5. `medquad_ans_0000388-3` (score=0.0001) - Mutations in the FXN gene cause Friedreich ataxia. This gene provides instructions for making a protein called frataxin. Although its role is not fully understood, frataxin appears to be important for the normal funct...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [medquad_ans_0002439-3].

**Supported citations**: `medquad_ans_0002439-3`
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.976, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000034-1__dup2` (score=0.9724) - Atrial fibrillation (AF) describes the rapid, irregular beating of the left atrium (upper chamber) of the heart. These rapid contractions of the heart are weaker than normal contractions, resulting in slow flow of blo...
2. `medquad_ans_0000013-4__dup2` (score=0.9705) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
3. `medquad_ans_0000341-1` (score=0.8466) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which cau...
4. `medquad_ans_0002230-1` (score=0.8236) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular respons...
5. `medquad_ans_0000341-2` (score=0.7779) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with...

**Answer**:

Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm, characterized by uncoordinated or erratic electrical activity in the heart's upper chambers (the atria). This leads to a fast and irregular heartbeat (arrhythmia) and can cause symptoms such as dizziness, chest pain, palpitations, shortness of breath, or fainting. If untreated, it increases the risk of stroke and sudden death. Complications can occur at any age, although some individuals may never experience health problems associated with the disorder [medquad_ans_0000341-1].

**Supported citations**: `medquad_ans_0000341-1`
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
3. `medquad_ans_0003801-1` (score=0.0001) - Marfan syndrome is a disorder of the connective tissue. Connective tissue provides strength and flexibility to structures throughout the body such as bones, ligaments, muscles, walls of blood vessels, and heart valves...
4. `medquad_ans_0000141-1` (score=0.0001) - Buschke-Ollendorff syndrome is a hereditary disorder of connective tissues, which are tissues that provide strength and flexibility to structures throughout the body. Specifically, the condition is characterized by sk...
5. `medquad_ans_0000259-1` (score=0.0001) - Cutis laxa is a disorder of connective tissue, which is the tissue that forms the body's supportive framework. Connective tissue provides structure and strength to the muscles, joints, organs, and skin. The term "cuti...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints, inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), neurologic abnormalities (e

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=1.000, groundedness=0.075, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000004-23` (score=0.6805) - Currently, no treatment can stop Alzheimer's disease. However, four medications are used to treat its symptoms. These medicines may help maintain thinking, memory, and speaking skills for a limited time. They work by ...
2. `medquad_ans_0000004-32` (score=0.4979) - People with Alzheimer's disease, those with mild cognitive impairment, those with a family history of Alzheimers, and healthy people with no memory problems who want to help scientists test new treatments may be able ...
3. `medquad_ans_0000004-24` (score=0.1477) - Currently, no medicines or treatments are known to prevent Alzheimer's disease, but scientists are studying many possibilities. These possibilities include lifestyle factors such as exercise and physical activity, a h...
4. `medquad_ans_0000004-21` (score=0.1315) - An early, accurate diagnosis of Alzheimer's disease helps people and their families plan for the future. It gives them time to discuss care options, find support, and make legal and financial arrangements while the pe...
5. `medquad_ans_0000004-5__dup2` (score=0.0969) - Medications Can Treat Symptoms There is no known cure for Alzheimer's disease, but there are medicines that can treat symptoms of the disease. Most Alzheimers medicines work best for people in the mild or moderate sta...

**Answer**:

The provided context does not mention treatments specifically for Alzheimer's caregiving. Therefore, based on the available information, no answer can be given. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.900, groundedness=0.896, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000511-3` (score=0.9972) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
2. `medquad_ans_0000293-3` (score=0.0024) - A mutation in the MATR3 gene has been identified in people with distal myopathy 2. This gene provides instructions for making a protein called matrin 3, which is found in the nucleus of the cell as part of the nuclear...
3. `medquad_ans_0000717-3` (score=0.0004) - Mutations in one of many genes can cause nemaline myopathy. These genes provide instructions for producing proteins that play important roles in skeletal muscles. Within skeletal muscle cells, these proteins are found...
4. `medquad_ans_0000547-3` (score=0.0003) - Juvenile idiopathic arthritis is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Its signs and symptoms res...
5. `medquad_ans_0000023-3` (score=0.0001) - The term "idiopathic" means that the cause of this condition is unknown. Adolescent idiopathic scoliosis probably results from a combination of genetic and environmental factors. Studies suggest that the abnormal spin...

**Answer**:

The genetic changes related to idiopathic inflammatory myopathy include variations in several genes, most commonly within the human leukocyte antigen (HLA) complex. These HLA gene variations are thought to influence the risk of developing the disorder by affecting how the immune system distinguishes between the body's own proteins and foreign invaders. Specific variations in these HLA genes have been associated with an increased risk of idiopathic inflammatory myopathy. Researchers are also studying other genes related to immune function to better understand their contribution to the disease. However, the exact mechanisms by which these genetic variations increase risk remain under investigation [medquad_ans_0000511-3].

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

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.9991) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0006562-1` (score=0.0001) - 17q23.1q23.2 microdeletion syndrome is a condition caused by a small deletion of genetic material from chromosome 17. The deletion occurs at a location encompassing bands 23.1 to 23.2 on the long (q) arm of the chromo...
3. `medquad_ans_0000566-3` (score=0.0001) - Koolen-de Vries syndrome is caused by genetic changes that eliminate the function of one copy of the KANSL1 gene in each cell. Most affected individuals are missing a small amount of genetic material, including the KA...
4. `medquad_ans_0001103-1` (score=0.0001) - 9q22.3 microdeletion is a chromosomal change in which a small piece of chromosome 9 is deleted in each cell. The deletion occurs on the long (q) arm of the chromosome in a region designated q22.3. This chromosomal cha...
5. `medquad_ans_0006562-4` (score=0.0001) - Is 17q23.2q23.2 microdeletion syndrome inherited? Parental FISH testing in most of the reported cases confirmed a de novo origin, meaning that the deletion was new to the family.

**Answer**:

The symptoms of 17q23.1q23.2 microdeletion syndrome include developmental delay, microcephaly, short stature, heart defects, and hand, foot, and limb abnormalities. Specific signs and symptoms reported in the literature include:

- Developmental delay, particularly delays in speech [medquad_ans_0006562-2]
- Microcephaly (seen in approximately 50% of patients) [medquad_ans_0006562-2]
- Short stature (seen in approximately 50% of patients) [medquad_ans_0006562-2]
- Heart defects, including patent ductus arteriosus or atrial septal defects (each seen in approximately 50% of patients) [medquad_ans_0006562-2]
- Limb abnormalities such as long, thin fingers and toes

**Supported citations**: `medquad_ans_0006562-2`
**Unsupported citations**: _none_

### Example 186 (`0003860-3`)

**Question**: What causes MECP2 duplication syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.9952) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0005347-3` (score=0.0079) - What causes Rett syndrome? Rett syndrome is typically caused by changes (mutations) in the MECP2 gene. This gene provides instructions for making a protein (MeCP2) needed for the development of the nervous system and ...
3. `medquad_ans_0000635-3` (score=0.0001) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
4. `medquad_ans_0000635-2` (score=0.0001) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...
5. `medquad_ans_0000635-4` (score=0.0001) - MECP2 duplication syndrome is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome...

**Answer**:

MECP2 duplication syndrome is caused by a duplication of genetic material located on the long (q) arm of the X chromosome, which results in an extra copy of the MECP2 gene in each cell. This duplication leads to the production of excess MeCP2 protein, which disrupts the normal regulation of other genes in the brain, causing irregular brain activity and the signs and symptoms of the condition. The size of the duplication can vary, ranging from 100,000 to 900,000 base pairs (100 to 900 kilobases), but the severity of the condition does not appear to be affected by the size of the duplication [medquad_ans_0003860-3]. The MECP2 gene encodes a protein critical for normal brain function, and its overexpression due to duplication impairs gene regulation in the brain [medquad_ans_00

**Supported citations**: `medquad_ans_0003860-3`
**Unsupported citations**: _none_

### Example 187 (`0004633-2`)

**Question**: What are the symptoms of Osteopetrosis autosomal dominant type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.536, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004633-2` (score=0.9994) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 2. If the informatio...
2. `medquad_ans_0004635-2` (score=0.0002) - What are the signs and symptoms of Osteopetrosis autosomal recessive 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal recessive 2. If the information is ava...
3. `medquad_ans_0004632-2` (score=0.0001) - What are the signs and symptoms of Osteopetrosis autosomal dominant type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Osteopetrosis autosomal dominant type 1. If the informatio...
4. `medquad_ans_0004637-1` (score=0.0001) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...
5. `medquad_ans_0004640-1` (score=0.0001) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...

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

**Scores**: context_relevance=0.667, groundedness=0.976, answer_relevance=0.578

**Top docs**:

1. `medquad_ans_0003459-4` (score=0.9687) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...
2. `medquad_ans_0000303-5` (score=0.2284) - These resources address the diagnosis or management of Down syndrome: - GeneFacts: Down Syndrome: Diagnosis - GeneFacts: Down Syndrome: Management - Genetic Testing Registry: Complete trisomy 21 syndrome - National Do...
3. `medquad_ans_0000277-2__dup2` (score=0.0692) - There is no cure for Zellweger syndrome, nor is there a standard course of treatment. Since the metabolic and neurological abnormalities that cause the symptoms of Zellweger syndrome are caused during fetal developmen...
4. `medquad_ans_0000954-5` (score=0.0222) - These resources address the diagnosis or management of Stickler syndrome: - Gene Review: Gene Review: Stickler Syndrome - Genetic Testing Registry: Marshall syndrome - Genetic Testing Registry: Stickler syndrome - Med...
5. `medquad_ans_0004456-3` (score=0.0078) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

Treatments for Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, plasmapheresis, IV immunoglobulins, steroids (e.g., prednisone), azathioprine or cyclosporine, and/or 3,4-diaminopyridine. While some evidence suggests that 3,4-diaminopyridine or IV immunoglobulin can improve muscle strength and nerve to muscle cell communication, the degree of benefit still needs to be determined [medquad_ans_0003459-4].

**Supported citations**: `medquad_ans_0003459-4`
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.933, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.9993) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
2. `medquad_ans_0000065-3` (score=0.0001) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This ...
3. `medquad_ans_0000065-1` (score=0.0001) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
4. `medquad_ans_0000445-1__dup1` (score=0.0001) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
5. `medquad_ans_0000572-1` (score=0.0001) - Lactate dehydrogenase deficiency is a condition that affects how the body breaks down sugar to use as energy in cells, primarily muscle cells. There are two types of this condition: lactate dehydrogenase-A deficiency ...

**Answer**:

The symptoms of Arginase deficiency include aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, EEG abnormality, hemiplegia/hemiparesis, hyperammonemia, seizures, anorexia, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1]. These symptoms are based on the Human Phenotype Ontology and are reported as approximate frequencies in patients with the condition [medquad_ans_0000445-2__dup1]. The underlying cause is the impaired function of arginase due to mutations in the ARG1 gene, leading to disrupted urea cycle and accumulation of ammonia and arginine [medquad_ans_0000065-3].

**Supported citations**: `medquad_ans_0000445-2__dup1`, `medquad_ans_0000065-3`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.889, groundedness=0.981, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006376-6` (score=0.9676) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
2. `medquad_ans_0000139-5__dup1` (score=0.0074) - Treatment for von Willebrand disease (VWD) is based on the type of VWD you have and how severe it is. Most cases of VWD are mild, and you may need treatment only if you have surgery, tooth extraction, or an accident. ...
3. `medquad_ans_0001020-5` (score=0.0060) - These resources address the diagnosis or management of von Hippel-Lindau syndrome: - Brigham and Women's Hospital - Gene Review: Gene Review: Von Hippel-Lindau Syndrome - Genetic Testing Registry: Von Hippel-Lindau sy...
4. `medquad_ans_0001021-5` (score=0.0007) - These resources address the diagnosis or management of von Willebrand disease: - Gene Review: Gene Review: von Willebrand Disease - Genetic Testing Registry: von Willebrand disorder - MedlinePlus Encyclopedia: von Wil...
5. `medquad_ans_0000961-1__dup1` (score=0.0001) - Von Hippel-Lindau disease (VHL) is a rare, genetic disease that causes tumors and cysts to grow in your body. The tumors can be either cancerous or benign. They can grow in your brain and spinal cord, kidneys, pancrea...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms but are still small to avoid permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All people with VHL disease should be carefully followed by a physician or medical team familiar with the disorder [medquad_ans_0006376-6].

**Supported citations**: `medquad_ans_0006376-6`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.701, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004454-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
2. `medquad_ans_0004452-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004456-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004451-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004453-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

The provided context does not mention "Noonan syndrome 3" or any specific subtype of Noonan syndrome beyond general references to subtypes based on the responsible gene. Therefore, based on the available information, Noonan syndrome 3 cannot be identified or described.

Answer: Insufficient context to determine what Noonan syndrome 3 is. [medquad_ans_0004454-1]

**Supported citations**: `medquad_ans_0004454-1`
**Unsupported citations**: _none_

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.9991) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0002966-1` (score=0.0004) - What are the signs and symptoms of Hydrocephalus, costovertebral dysplasia, and Sprengel anomaly? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrocephalus, costovertebral dyspla...
3. `medquad_ans_0005865-1` (score=0.0001) - Sprengel deformity is a congenital condition characterized by abnormal development and elevation of the shoulder blade (scapula). Severity can range considerably from being almost invisible when covered with clothes, ...
4. `medquad_ans_0000290-1` (score=0.0001) - Distal arthrogryposis type 1 is a disorder characterized by joint deformities (contractures) that restrict movement in the hands and feet. The term "arthrogryposis" comes from the Greek words for joint (arthro-) and c...
5. `medquad_ans_0005865-3` (score=0.0001) - How might Sprengel deformity be treated? Treatment of Sprengel deformity depends on the degree of functional impairment and/or cosmetic disfigurement. Many people with Sprengel deformity do not need surgery and may ha...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, hemivertebrae, neck muscle hypoplasia, rib segmentation abnormalities, cervical segmentation defects, spina bifida occulta, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2]. The condition may also be associated with limb length discrepancy or abnormalities in the spine or ribs [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.933, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000882-3` (score=0.9988) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...
2. `medquad_ans_0000869-3` (score=0.0003) - Restless legs syndrome likely results from a combination of genetic and environmental factors, many of which are unknown. Studies suggest that restless legs syndrome is related to a shortage (deficiency) of iron in ce...
3. `medquad_ans_0000290-3` (score=0.0002) - Distal arthrogryposis type 1 can be caused by mutations in at least two genes: TPM2 and MYBPC1. These genes are active (expressed) in muscle cells, where they interact with other muscle proteins to help regulate the t...
4. `medquad_ans_0001041-3` (score=0.0002) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000667-3` (score=0.0001) - The causes of Moebius syndrome are unknown, although the condition probably results from a combination of environmental and genetic factors. Researchers are working to identify and describe specific genes related to t...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene, which cause autosomal recessive Robinow syndrome by preventing the production of a functional ROR2 protein and disrupting developmental signaling pathways essential for the formation of the skeleton, heart, and genitals [medquad_ans_0000882-3]. Autosomal dominant Robinow syndrome can result from mutations in the WNT5A or DVL1 gene, with the osteosclerotic form specifically linked to DVL1 gene mutations. These mutations alter the production or function of their respective proteins, impairing chemical signaling critical for early development [medquad_ans_0000882-3]. In some cases, no mutation is identified in the ROR2, WNT5A, or DVL1 genes, and the cause remains unknown [medquad_ans_0000882-3].

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

**Scores**: context_relevance=0.750, groundedness=0.994, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.9368) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000159-6` (score=0.0019) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
3. `medquad_ans_0000718-1__dup2` (score=0.0004) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
4. `medquad_ans_0000712-1` (score=0.0002) - Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis (NFJS/DPR) represents a rare type of ectodermal dysplasia, a group of about 150 conditions characterized by abnormal development of ectoder...
5. `medquad_ans_0000017-10` (score=0.0002) - Most people with dry eye will not have serious problems, but severe dry eye may lead to inflammation, ulcers, or scars on the cornea, and some loss of vision. Permanent loss of vision from dry eye is uncommon.

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams [medquad_ans_0000795-1__dup1].

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.950, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0001045-3` (score=0.9992) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules call...
2. `medquad_ans_0001047-3` (score=0.0003) - X-linked chondrodysplasia punctata 1 is caused by genetic changes involving the ARSE gene. This gene provides instructions for making an enzyme called arylsulfatase E. The function of this enzyme is unknown, although ...
3. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
4. `medquad_ans_0001063-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0001062-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...

**Answer**:

Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. These mutations result in a deficiency of the adrenoleukodystrophy protein (ALDP), which is responsible for transporting very long-chain fatty acids (VLCFAs) into peroxisomes. The lack of this protein disrupts the transport and breakdown of VLCFAs, leading to their accumulation in the body, which may be toxic to the adrenal cortex and myelin [medquad_ans_0001045-3].

**Supported citations**: `medquad_ans_0001045-3`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.750, groundedness=0.984, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.9981) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
2. `medquad_ans_0000988-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
3. `medquad_ans_0000988-1` (score=0.0001) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most fre...
4. `medquad_ans_0000027-1__dup3` (score=0.0001) - Amyloidosis occurs when abnormal proteins called amyloids build up and form deposits. The deposits can collect in organs such as the kidney and heart. This can cause the organs to become stiff and unable to work the w...
5. `medquad_ans_0000988-3` (score=0.0001) - Mutations in the TTR gene cause transthyretin amyloidosis. The TTR gene provides instructions for producing a protein called transthyretin. Transthyretin transports vitamin A (retinol) and a hormone called thyroxine t...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form affects between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.886, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.9994) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0000063-29` (score=0.0010) - Yes. Some skin cancer patients take part in studies of new treatments. These studies, called clinical trials, are designed to find out whether a new treatment is both safe and effective. Clinical trials are research s...
3. `medquad_ans_0000004_1-6` (score=0.0002) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000001_3-6` (score=0.0002) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000005-6__dup1` (score=0.0002) - Clinical trials are part of clinical research and at the heart of all treatment advances. Clinical trials look at new ways to prevent, detect, or treat disease. The National Institute of Mental Health at NIH supports ...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include information compiled by the Human Phenotype Ontology (HPO), which provides a list of features observed in individuals with this condition. The frequency of these symptoms is often estimated as a percentage or as a fraction (number of patients with the symptom out of the total number examined). For example, a frequency of 25/25 indicates that all patients in a study had the symptom. However, the specific symptoms themselves are not detailed in the provided context [medquad_ans_0001780-1].

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

**Scores**: context_relevance=0.800, groundedness=0.460, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000306-1` (score=0.1412) - Muscular dystrophies are a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy). The Duchenne and Becker types of muscular dystrophy are two related conditions that primarily ...
2. `medquad_ans_0001935-1` (score=0.0195) - Duchenne muscular dystrophy (DMD) is a rapidly progressive form of muscular dystrophy that occurs primarily in boys. It is caused by a mutation in a gene, called the DMD gene, which encodes the muscle protein dystroph...
3. `medquad_ans_0000395-5` (score=0.0016) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
4. `medquad_ans_0002213-1` (score=0.0009) - Facioscapulohumeral muscular dystrophy is a disorder characterized by muscle weakness and wasting (atrophy). This condition gets its name from the areas of the body that are affected most often: muscles in the face (f...
5. `medquad_ans_0000576-5` (score=0.0005) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...

**Answer**:

Muscular dystrophy is a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy). Duchenne and Becker types are two related conditions that primarily affect skeletal muscles and heart muscle, occurring almost exclusively in males. These conditions are caused by different mutations in the same gene (the DMD gene), which encodes the muscle protein dystrophin. In Duchenne muscular dystrophy, boys do not produce the dystrophin protein, leading to rapid progression of muscle weakness that typically appears in early childhood and worsens rapidly, with affected children often becoming wheelchair-dependent by adolescence. Becker muscular dystrophy has a milder and more variable course, with muscle weakness appearing later in childhood or adolescence and progressing more slowly. Both conditions are associated with cardiomyopathy, which typically begins in adolescence and may progress to dilated cardiomyopathy, causing symptoms such as shortness of breath, fatigue, arrhythmia, and leg swelling. Males with Duchenne

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.8062) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0005020-1` (score=0.0002) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
3. `medquad_ans_0005022-1` (score=0.0001) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...
4. `medquad_ans_0002679-2` (score=0.0001) - What are the signs and symptoms of Greig cephalopolysyndactyly syndrome? The symptoms of Greig cephalopolysyndactyly syndrome (GCPS) are highly variable, ranging from mild to severe. People with this condition typical...
5. `medquad_ans_0002884-1` (score=0.0001) - What are the signs and symptoms of Hirschsprung disease polydactyly heart disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Hirschsprung disease polydactyly heart disease. If ...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=0.500, groundedness=0.075, answer_relevance=0.633

**Top docs**:

1. `medquad_ans_0000037_2-8` (score=0.0001) - Other types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, t...
2. `medquad_ans_0000005-6__dup1` (score=0.0001) - Clinical trials are part of clinical research and at the heart of all treatment advances. Clinical trials look at new ways to prevent, detect, or treat disease. The National Institute of Mental Health at NIH supports ...
3. `medquad_ans_0000033_1-8` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
4. `medquad_ans_0000006_9-6` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
5. `medquad_ans_0000032_2-8` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...

**Answer**:

The provided context does not mention the symptoms of Geniospasm. Therefore, no answer can be given based on the available information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

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

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000813-4` (score=0.9934) - Potocki-Shaffer syndrome follows an autosomal dominant inheritance pattern, which means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. In some cases, an affected per...
2. `medquad_ans_0000527-4` (score=0.7178) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
3. `medquad_ans_0002561-3` (score=0.7023) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
4. `medquad_ans_0001084-3__dup1` (score=0.0072) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).
5. `medquad_ans_0000840-4__dup1` (score=0.0002) - Is branchiootorenal syndrome inherited? Branchiootorenal syndrome may be inherited or occur sporadically. The inheritance pattern of branchiootorenal syndrome is autosomal dominant. Autosomal dominant inheritance is w...

**Answer**:

Yes, Potocki-Shaffer syndrome is inherited in an autosomal dominant pattern. A deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. This can occur when an affected person inherits the chromosome with a deleted segment from an affected parent or when the deletion occurs spontaneously during the formation of reproductive cells or in early fetal development [medquad_ans_0000813-4].

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

**Scores**: context_relevance=0.833, groundedness=0.941, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000159-3__dup2` (score=0.9977) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
2. `medquad_ans_0000159-6` (score=0.0062) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
3. `medquad_ans_0000186-2__dup1` (score=0.0014) - There is no cure for lupus. Treatment is symptomatic. With a combination of medication, rest, exercise, proper nutrition, and stress management, most individuals with lupus can often achieve remission or reduce their ...
4. `medquad_ans_0000966-5` (score=0.0004) - These resources address the diagnosis or management of systemic lupus erythematosus: - MedlinePlus Encyclopedia: Antinuclear Antibody Panel These resources from MedlinePlus offer information about the diagnosis and ma...
5. `medquad_ans_0000569-1__dup2` (score=0.0002) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus eryt...

**Answer**:

The symptoms of lupus may include joint pain or swelling, muscle pain, fever with no known cause, red rashes often on the face (called butterfly rashes due to their shape), high blood pressure, foamy urine, and edema—swelling usually in the legs, feet, or ankles and less often in the hands or face [medquad_ans_0000159-3__dup2]. Additionally, lupus can cause damage to joints, skin, blood vessels, and organs due to the immune system attacking healthy cells and tissues [medquad_ans_0000569-1__dup2].

**Supported citations**: `medquad_ans_0000159-3__dup2`, `medquad_ans_0000569-1__dup2`
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.531, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-1` (score=0.2765) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
2. `medquad_ans_0001398-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
3. `medquad_ans_0001395-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001397-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
5. `medquad_ans_0000073-1__dup2` (score=0.0065) - Cerebro-oculo-facio-skeletal syndrome (COFS) is a pediatric, genetic, degenerative disorder that involves the brain and the spinal cord. It is characterized by craniofacial and skeletal abnormalities, severely reduced...

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging (progeria). It is associated with severe photosensitivity, failure to thrive, microcephaly, and impaired development of the nervous system. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain visible on brain scans. The syndrome is divided into three subtypes: classical (type I or type A), which has an onset in early childhood (usually after age 1 year) and is sometimes called "moderate" or "classic" Cockayne syndrome; type II (type B), which is more severe and presents at birth with growth and developmental abnormalities, also known as cerebro-oculo-facio-skeletal (COFS) syndrome or Pena-Shokeir syndrome type II; and type III (type C), which is the mildest form and appears later in childhood. Cockayne

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.667, groundedness=0.890, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000029-9__dup1` (score=0.9863) - When gingivitis is not treated, it can advance to periodontitis (which means "inflammation around the tooth.") In periodontitis, gums pull away from the teeth and form "pockets" that become infected. The body's immune...
2. `medquad_ans_0000413-1__dup1` (score=0.9751) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the t...
3. `medquad_ans_0000029-6__dup1` (score=0.9748) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...
4. `medquad_ans_0000029-8__dup1` (score=0.9484) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regu...
5. `medquad_ans_0000029-19` (score=0.9084) - In deep cleaning, the dentist, periodontist, or dental hygienist removes the plaque through a method called scaling and root planing. Scaling means scraping off the tartar from above and below the gum line. Root plani...

**Answer**:

Gum (periodontal) disease is an infection of the tissues that hold your teeth in place. It ranges from mild inflammation of the gums, known as gingivitis, to serious damage involving the loss of bone and tissue that support the teeth. In gingivitis, the gums become red, swollen, and may bleed easily, but it does not involve loss of bone or tissue. If left untreated, gingivitis can progress to periodontitis, in which the gums pull away from the teeth and form infected pockets. The body's immune response to bacteria in the plaque leads to the breakdown of bone and connective tissue that support the teeth, potentially resulting in tooth loss [medquad_ans_0000029-6\_\_dup1][medquad_ans_0000413-1__dup1][medquad_ans_0000029-8__dup1].

**Supported citations**: `medquad_ans_0000029-6__dup1`, `medquad_ans_0000413-1__dup1`, `medquad_ans_0000029-8__dup1`
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

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.8399) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0000012-20` (score=0.0035) - More information on COPD is available at: What is COPD? and at the Learn More, Breathe Better Campaign For information on quitting smoking, visit http://www.surgeongeneral.gov/tobacco/ or Smokefree.gov. For informatio...
3. `medquad_ans_0000113-4__dup3` (score=0.0015) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
4. `medquad_ans_0002502-1` (score=0.0002) - Gaucher disease refers to a group of inherited conditions that affect many organs and tissues in the body. Signs and symptoms vary widely among affected individuals. There are different types of this condition: Gauche...
5. `medquad_ans_0000135-1__dup4` (score=0.0001) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles, platyspondyly (flattening of the spine bones), and cortical thinning. Cranial involvement is minimal, with some cases showing mild hyperostosis of the skull base and thickening of the frontal and occipital bones. Pyle disease is inherited in an autosomal recessive manner, and only about 30 cases have been reported in the literature [medquad_ans_0005212-1].

**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.800, groundedness=0.924, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000108-3__dup2` (score=0.0331) - Major Causes Pulmonary embolism (PE) usually begins as a blood clot in a deep vein of the leg. This condition is called deep vein thrombosis. The clot can break free, travel through the bloodstream to the lungs, and b...
2. `medquad_ans_0000108-4__dup2` (score=0.0002) - Pulmonary embolism (PE) occurs equally in men and women. The risk increases with age. For every 10 years after age 60, the risk of having PE doubles. Certain inherited conditions, such as factor V Leiden, increase the...
3. `medquad_ans_0000104-1__dup4` (score=0.0001) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
4. `medquad_ans_0000108-8` (score=0.0001) - Preventing pulmonary embolism (PE) begins with preventing deep vein thrombosis (DVT). Knowing whether you're at risk for DVT and taking steps to lower your risk are important. Exercise your lower leg muscles if you're...
5. `medquad_ans_0000456-1` (score=0.0001) - Hereditary antithrombin deficiency is a disorder of blood clotting. People with this condition are at higher than average risk for developing abnormal blood clots, particularly a type of clot that occurs in the deep v...

**Answer**:

Deep vein thrombosis (DVT) occurs when blood flow in the deep veins of the legs is restricted and slows down, which can lead to blood clot formation. This can happen due to prolonged periods of immobility, such as after some types of surgery, during long trips in a car or airplane, or when staying in bed for an extended time. Blood clots are also more likely to develop in veins that are damaged from surgery or injury [medquad_ans_0000108-3__dup2]. Additionally, other risk factors include having certain diseases or conditions such as stroke, paralysis, chronic heart disease, or high blood pressure, as well as being overweight or obese [medquad_ans_0000108-4__dup2].

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

**Scores**: context_relevance=0.857, groundedness=0.523, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.4373) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0000026-1__dup1` (score=0.0015) - Apraxia (called "dyspraxia" if mild) is a neurological disorder characterized by loss of the ability to execute or carry out skilled movements and gestures, despite having the desire and the physical ability to perfor...
3. `medquad_ans_0000495-1__dup1` (score=0.0007) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
4. `medquad_ans_0000026-2__dup1` (score=0.0003) - Generally, treatment for individuals with apraxia includes physical, speech,or occupational therapy. If apraxia is a symptom of another disorder, the underlying disorder should be treated.
5. `medquad_ans_0000078-2` (score=0.0002) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, which makes it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4, while in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Individuals

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
3. `medquad_ans_0002561-3` (score=0.2100) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
4. `medquad_ans_0001553-4` (score=0.0773) - How is congenital radio-ulnar synostosis inherited? Congenital radio-ulnar synostosis appears to be inherited in an autosomal dominant manner in some cases. This means that one mutated copy of the disease-causing gene...
5. `medquad_ans_0001084-3__dup1` (score=0.0015) - Is Chandler's syndrome inherited? While the cause of Chandler's syndrome is unknown, at this time there is no evidence that it is inherited (hereditary).

**Answer**:

Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases result from new mutations and occur in individuals with no family history of the disorder, but individuals with Apert syndrome can pass the condition to their offspring [medquad_ans_0000064-4].

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

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air that contains airborne barnyard dust contaminated by dried placental material, birth fluids, and excreta of infected animals. Other modes of transmission to humans, including tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission, are rare. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, the affected people with acute Q fever infection develop a chronic Q fever. Treatment of the acute form is

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000001-3__dup2` (score=0.9952) - Alpha-1 antitrypsin (AAT) deficiency occurs in all ethnic groups. However, the condition occurs most often in White people of European descent. AAT deficiency is an inherited condition. "Inherited" means the condition...
2. `medquad_ans_0000038-5__dup1` (score=0.9936) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
3. `medquad_ans_0000038-9` (score=0.0145) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...
4. `medquad_ans_0000032_2-2` (score=0.0011) - Smoking is the major risk factor for small cell lung cancer. Anything that increases your chance of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having ri...
5. `medquad_ans_0000009-1__dup2` (score=0.0008) - Asbestos-related lung diseases are diseases caused by exposure to asbestos (as-BES-tos) fibers. Asbestos is a mineral that, in the past, was widely used in many industries. Asbestos is made up of tiny fibers that can ...

**Answer**:

Children who have a family history of interstitial lung disease or childhood interstitial lung disease (chILD) are at increased risk. Other risk factors include having an inherited surfactant disorder or a family history of such disorders, problems with aspiration, immune system disorders, exposure to environmental lung irritants such as molds and chemicals, having a systemic or autoimmune disease (e.g., collagen vascular disease or inflammatory bowel disease), undergoing cancer treatments like radiation or chemotherapy, or having a bone marrow or lung transplant [medquad_ans_0000038-5__dup1]. Additionally, certain types of chILD are more common in infants and young children, while others can occur in children of any age [medquad_ans_0000038-5__dup1].

**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.9011) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000172-1__dup1` (score=0.0171) - Klver-Bucy syndrome is a rare behavioral impairment that is associated with damage to both of the anterior temporal lobes of the brain. It causes individuals to put objects in their mouths and engage in inappropriate ...
3. `medquad_ans_0003405-1` (score=0.0039) - Kluver Bucy syndrome is a rare behavioral impairment characterized by inappropriate sexual behaviors and mouthing of objects. Other signs and symptoms, include a diminished ability to visually recognize objects, loss ...
4. `medquad_ans_0000441-3` (score=0.0010) - GRN-related frontotemporal dementia results from mutations in the GRN gene. This gene provides instructions for making a protein called granulin (also known as progranulin). Granulin is active in many different tissue...
5. `medquad_ans_0000391-3` (score=0.0009) - FTDP-17 is caused by mutations in the MAPT gene. This gene is located on chromosome 17, which is how the disease got its name. The MAPT gene provides instructions for making a protein called tau. This protein is found...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries located at the lower sides of the skull, underneath the temple. Affected individuals often have no signs or symptoms other than a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically without a family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.518, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.9565) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
2. `medquad_ans_0001082-1` (score=0.9285) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0001082-2` (score=0.0004) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
4. `medquad_ans_0000286-1` (score=0.0001) - Dihydrolipoamide dehydrogenase deficiency is a severe condition that can affect several body systems. Signs and symptoms of this condition usually appear shortly after birth, and they can vary widely among affected in...
5. `medquad_ans_0000413-1` (score=0.0001) - Glucose-6-phosphate dehydrogenase deficiency is a genetic disorder that occurs most often in males. This condition mainly affects red blood cells, which carry oxygen from the lungs to tissues throughout the body. In a...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of the enzyme 2-methylbutyryl-CoA dehydrogenase, which is necessary for the proper processing of the amino acid isoleucine. This deficiency leads to the buildup of isoleucine and organic acids in the body, which can be toxic and cause a range of health problems. Symptoms vary widely, from severe and life-threatening to mild or absent, and may include poor feeding, lack of energy (lethargy), vomiting, irritability, difficulty breathing, seizures, coma, poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills such as standing and walking. Symptoms can be triggered by fasting, infections, or high-protein diets. The disorder is more common among individuals of Hmong ancestry, particularly in Hmong populations in southeast Asia and Hmong Americans,

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
3. `medquad_ans_0000120-3__dup1` (score=0.0003) - While epilepsy cannot be cured, for some people the seizures can be controlled with medication, diet, devices, and/or surgery. Most seizures do not cause brain damage, but ongoing uncontrolled seizures may cause brain...
4. `medquad_ans_0000853-2` (score=0.0001) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
5. `medquad_ans_0000004-4__dup1` (score=0.0001) - The NINDS supports broad and varied programs of research on epilepsy and developmental disorders. This research is aimed at discovering new ways to prevent, diagnose, and treat epilepsy and developmental disorders and...

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The ALDH7A1 gene provides instructions for making an enzyme called α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the protein building block (amino acid) lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. Pyridoxine plays a role in many processes in the body, such as the breakdown of amino acids and the production of chemicals that transmit signals in the brain (neurotransmitters). It is unclear how a lack of pyridoxine causes the seizures that are characteristic of this condition. Some individuals with pyridox

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.9949) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
2. `medquad_ans_0000004_6-5` (score=0.0027) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...
3. `medquad_ans_0000004_1-5` (score=0.0003) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The patient's signs and symptoms. - The stage of the can...
4. `medquad_ans_0000483-1__dup1` (score=0.0001) - Hodgkin disease is a type of lymphoma. Lymphoma is a cancer of a part of the immune system called the lymph system. The first sign of Hodgkin disease is often an enlarged lymph node. The disease can spread to nearby l...
5. `medquad_ans_0000004_3-1` (score=0.0001) - Key Points - AIDS-related lymphoma is a disease in which malignant (cancer) cells form in the lymph system of patients who have acquired immunodeficiency syndrome (AIDS). - There are many different types of lymphoma. ...

**Answer**:

Children who are infected with the Epstein-Barr virus, those infected with the human immunodeficiency virus (HIV), individuals with certain diseases of the immune system, children with a personal history of mononucleosis ("mono"), and those with a parent or sibling who has a personal history of Hodgkin lymphoma are at risk for childhood Hodgkin lymphoma [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.8953) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0001795-1` (score=0.7929) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0000088-11` (score=0.0098) - Most people with celiac disease have a significant improvement in symptoms when they follow a gluten-free diet. Health care providers typically refer people to a dietitian who specializes in treating people with the d...
4. `medquad_ans_0000768-1__dup1` (score=0.0079) - A rash is an area of irritated or swollen skin. Many rashes are itchy, red, painful, and irritated. Some rashes can also lead to blisters or patches of raw skin. Rashes are a symptom of many different medical problems...
5. `medquad_ans_0000084-2` (score=0.0050) - Atopic dermatitis is a common disorder that affects 10 to 20 percent of children and 5 to 10 percent of adults.

**Answer**:

Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These lesions are more common on the knees, elbows, buttocks, and shoulder blades. The condition is often associated with the inability to digest gluten and may present with symptoms such as fluid-filled sores, red lesions resembling hives, and itchiness, redness, and burning. A burning sensation may precede lesion formation, and the lesions are usually scratched off by the time a patient presents for a physical exam, appearing as erosions and excoriations. It is also associated with celiac disease, and patients may experience dental enamel defects in permanent teeth. Less than 20 percent of people with dermatitis herpetiformis have symptoms of celiac disease [medquad_ans_0001795-1]. [medquad_ans_0000095-1

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.857, groundedness=0.970, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000709-5` (score=0.9926) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
2. `medquad_ans_0000040-2__dup1` (score=0.9899) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...
3. `medquad_ans_0000040-3__dup1` (score=0.9090) - Most individuals with myotonia congenita lead long, productive lives. Although muscle stiffness may interfere with walking, grasping, chewing, and swallowing, it is usually relieved with exercise.
4. `medquad_ans_0000378-3__dup1` (score=0.0061) - How might anonychia congenita be treated? There is limited information regarding anonychia congenita because it is very rare. After a careful review of the medical literature, we did not find any information about tre...
5. `medquad_ans_0000308-5` (score=0.0002) - These resources address the diagnosis or management of dyskeratosis congenita: - Gene Review: Gene Review: Dyskeratosis Congenita - Genetic Testing Registry: Dyskeratosis congenita - Genetic Testing Registry: Dyskerat...

**Answer**:

Most people with myotonia congenita do not require special treatments; stiff muscles usually resolve with exercise or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors have had some success with medications such as quinine or anticonvulsant drugs like phenytoin. Physical therapy and other rehabilitative therapies are also sometimes used to improve muscle function [medquad_ans_0000040-2__dup1].

**Supported citations**: `medquad_ans_0000040-2__dup1`
**Unsupported citations**: _none_

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=1.000, groundedness=0.463, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000133-13` (score=0.8860) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
2. `medquad_ans_0000133-11` (score=0.8679) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the i...
3. `medquad_ans_0000133-12` (score=0.0097) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
4. `medquad_ans_0001037-5` (score=0.0008) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Hu...
5. `medquad_ans_0000276-4__dup1` (score=0.0004) - The National Institute of Neurological Disorders and Stroke, the Eunice Kennedy Shriver National Institute of Child Health and Human Development, and other institutes of the National Institutes of Health (NIH) conduct...

**Answer**:

For Wilson disease, treatment involves a lifelong effort to reduce and control copper levels in the body. This may include medications, changes in eating, diet, and nutrition, and in some cases, a liver transplant. People should reduce dietary copper intake by avoiding high-copper foods such as shellfish, liver, mushrooms, nuts, and chocolate, especially during initial treatment; however, they should consult a health care provider to determine if these foods can be consumed in moderation during maintenance treatment. Additionally, individuals should check the copper levels in their tap water, as water from copper pipes or wells may contain copper residue; running water helps reduce copper levels. People should not use copper containers or cookware to store or prepare food or drinks. It is important to discuss the use of complementary and alternative medical practices, including vitamins and dietary supplements, with a health care provider, as some supplements contain copper. If a supplement is recommended, a pharmacist can suggest copper-free options. Finally

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.969, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.9980) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0001078-3` (score=0.0002) - 1p36 deletion syndrome is caused by a deletion of genetic material from a specific region in the short (p) arm of chromosome 1. The signs and symptoms of 1p36 deletion syndrome are probably related to the loss of mult...
3. `medquad_ans_0000290-3` (score=0.0001) - Distal arthrogryposis type 1 can be caused by mutations in at least two genes: TPM2 and MYBPC1. These genes are active (expressed) in muscle cells, where they interact with other muscle proteins to help regulate the t...
4. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000592-4` (score=0.0001) - Most cases of Lennox-Gastaut syndrome are sporadic, which means they occur in people with no history of the disorder in their family. However, 3 to 30 percent of people with this condition have a family history of som...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. [medquad_ans_0000592-3]

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

**Scores**: context_relevance=0.909, groundedness=0.919, answer_relevance=0.618

**Top docs**:

1. `medquad_ans_0000418-2__dup1` (score=0.9752) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
2. `medquad_ans_0000124-5__dup1` (score=0.0034) - Certain traits, conditions, and habits can raise your risk of having a stroke or transient ischemic attack (TIA). These traits, conditions, and habits are known as risk factors. The more risk factors you have, the mor...
3. `medquad_ans_0000269-2__dup2` (score=0.0017) - There are three different filarial species that can cause lymphatic filariasis in humans. Most of the infections worldwide are caused by Wuchereria bancrofti. In Asia, the disease can also be caused by Brugia malayi a...
4. `medquad_ans_0000031-3__dup3` (score=0.0008) - The major risk factors for carotid artery disease, listed below, also are the major risk factors for coronary heart disease (also called coronary artery disease) and peripheral artery disease. Diabetes. With this dise...
5. `medquad_ans_0000050-5__dup1` (score=0.0002) - Who Should Be Tested? The United States Preventive Service Task Force recommends that women aged 65 and older be screened (tested) for osteoporosis, as well as women aged 60 and older who are at increased risk for an ...

**Answer**:

People at risk for Trichinellosis include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating even small amounts of undercooked meat during preparation or cooking puts individuals at risk. Those who consume undercooked wild game, especially bear meat, are at increased risk. The risk from commercially raised and properly prepared pork is very low, but it remains a concern when meat is not adequately cooked [medquad_ans_0000418-2__dup1].

**Supported citations**: `medquad_ans_0000418-2__dup1`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.919, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.9994) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0000108-3__dup1` (score=0.0019) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of co...
3. `medquad_ans_0000108-2__dup2` (score=0.0016) - Seizures in Dravet syndrome are difficult to control, but can be reduced by anticonvulsant drugs. A ketogenic diet, high in fats and low in carbohydrates, also may be beneficial.
4. `medquad_ans_0000108-1__dup2` (score=0.0001) - Dravet syndrome, also called severe myoclonic epilepsy of infancy (SMEI), is a severe form of epilepsy. It appears during the first year of life with frequent febrile seizures fever-related seizures that, by definitio...
5. `medquad_ans_0000610-1__dup1` (score=0.0001) - Baller-Gerold syndrome is a rare condition characterized by the premature fusion of certain skull bones (craniosynostosis) and abnormalities of bones in the arms and hands, sometimes referred to as radial ray anomalie...

**Answer**:

The symptoms of Dravet syndrome include absence seizures, ataxia, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2]. Additionally, children with Dravet syndrome experience poor development of language and have seizures that are difficult to control, often beginning with frequent febrile seizures during the first year of life and later progressing to other types of seizures such as myoclonus [medquad_ans_0000108-1__dup2]. Cognitive impairment correlates with the frequency of seizures, and while cognitive decline may stabilize or slightly improve with age, most teenagers remain dependent on caregivers [medquad_ans_0000

**Supported citations**: `medquad_ans_0001924-2`, `medquad_ans_0000108-1__dup2`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000826-3` (score=0.9974) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, whi...
2. `medquad_ans_0000011-13` (score=0.0010) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
3. `medquad_ans_0000832-3` (score=0.0001) - In most cases, the genetic cause of progressive supranuclear palsy is unknown. Rarely, the disease results from mutations in the MAPT gene. Certain normal variations (polymorphisms) in the MAPT gene have also been ass...
4. `medquad_ans_0000142-3` (score=0.0001) - C3 glomerulopathy is associated with changes in many genes. Most of these genes provide instructions for making proteins that help regulate a part of the body's immune response known as the complement system. This sys...
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
2. `medquad_ans_0006470-1` (score=0.0001) - What are the signs and symptoms of WT limb blood syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for WT limb blood syndrome. If the information is available, the table below in...
3. `medquad_ans_0000901-1` (score=0.0001) - Sheldon-Hall syndrome, also known as distal arthrogryposis type 2B, is a disorder characterized by joint deformities (contractures) that restrict movement in the hands and feet. The term "arthrogryposis" comes from th...
4. `medquad_ans_0005127-2` (score=0.0001) - What are the signs and symptoms of Pseudoachondroplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudoachondroplasia. If the information is available, the table below includ...
5. `medquad_ans_0000817-2__dup1` (score=0.0001) - What are the signs and symptoms of Brachydactyly type C? The Human Phenotype Ontology provides the following list of signs and symptoms for Brachydactyly type C. If the information is available, the table below includ...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.910, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.9994) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0000056-11` (score=0.0282) - People who experience smell disorders either have a decrease in their ability to smell or changes in the way they perceive odors. Total smell loss is relatively rare, but a decrease in the sense of smell occurs more o...
3. `medquad_ans_0000818-1__dup1` (score=0.0010) - Summary : Your skin changes as you age. You might notice wrinkles, age spots and dryness. Your skin also becomes thinner and loses fat, making it less plump and smooth. It might take longer to heal, too. Sunlight is a...
4. `medquad_ans_0004524-1` (score=0.0001) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
5. `medquad_ans_0000056-26` (score=0.0001) - The National Institute on Deafness and Other Communication Disorders (NIDCD) supports basic and clinical investigations of smell and taste disorders at its laboratories in Bethesda, Md. and at universities and chemose...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is estimated based on the Human Phenotype Ontology, which compiles data from sources like Orphanet, with frequencies often expressed as percentages or fractions [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.817, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000979-1` (score=0.0001) - Thrombocytopenia-absent radius (TAR) syndrome is characterized by the absence of a bone called the radius in each forearm. Affected individuals also have a shortage (deficiency) of blood cells involved in clotting (pl...
2. `medquad_ans_0006007-1` (score=0.0001) - TAR syndrome is characterized by the absence of a bone called the radius in each forearm, short stature, and thrombocytopenia. The thrombocytopenia often appears first in infancy but becomes less severe or returns to ...
3. `medquad_ans_0000075-2` (score=0.0001) - The prevalence of Asperger syndrome is not well established. Estimates range from 1 in 250 to 1 in 5,000 children. Three to four times as many males are affected than females. Because of changes in the way development...
4. `medquad_ans_0006312-2` (score=0.0001) - What are the signs and symptoms of VACTERL association? The Human Phenotype Ontology provides the following list of signs and symptoms for VACTERL association. If the information is available, the table below includes...
5. `medquad_ans_0001929-2` (score=0.0001) - What are the signs and symptoms of Duane syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Duane syndrome type 2. If the information is available, the table below incl...

**Answer**:

The provided context does not include information on the prevalence or number of people affected by thrombocytopenia-absent radius (TAR) syndrome. Therefore, it is not possible to determine how many people are affected based on the given text. [medquad_ans_0000979-1][medquad_ans_0006007-1]

**Supported citations**: `medquad_ans_0000979-1`, `medquad_ans_0006007-1`
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

**Scores**: context_relevance=1.000, groundedness=0.538, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.4357) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000608-1__dup1` (score=0.0006) - Metabolic syndrome is a group of conditions that put you at risk for heart disease and diabetes. These conditions are - High blood pressure - High blood glucose, or blood sugar, levels - High levels of triglycerides, ...
3. `medquad_ans_0000038-16` (score=0.0006) - When you visit your doctor, here are questions to ask about your kidneys. - What is my GFR? - What is my urine albumin result? - What is my blood pressure? - What is my blood glucose (for people with diabetes)? What i...
4. `medquad_ans_0000008-2` (score=0.0004) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
5. `medquad_ans_0000008-15` (score=0.0004) - One definition of cure is being alive and free of breast cancer for 5 years. If the cancer is found early, a woman's chances of survival are better. In fact, nearly 98 percent of women who discover their breast cancer...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. Achromatopsia also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals may also have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green [medquad_ans_0

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=1.000, groundedness=0.869, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000039-9` (score=0.9869) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
2. `medquad_ans_0000039-11` (score=0.9823) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
3. `medquad_ans_0000039-26` (score=0.9592) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circl...
4. `medquad_ans_0000049-4__dup3` (score=0.0870) - Treatment Goals: Manage Pain, Improve Function Osteoarthritis treatment plans often include ways to manage pain and improve function. Such plans can include exercise, rest and joint care, pain relief, weight control, ...
5. `medquad_ans_0000039-13` (score=0.0358) - Knee replacement may be either total or partial/unicompartmental. In total knee replacement, as the name suggests, the entire knee joint is replaced. You will likely need a total knee replacement if you have damage to...

**Answer**:

The provided context does not list treatments for knee replacement surgery. Instead, it outlines treatments recommended _before_ knee replacement, such as exercises to strengthen muscles, weight loss, walking aids, shoe inserts, and pain medications [medquad_ans_0000039-11]. It also mentions that the main reason for knee replacement is to ease pain and disability caused by arthritis or other joint problems while preserving movement [medquad_ans_0000039-9]. However, specific treatments for knee replacement itself are not described in the context.

Therefore, based on the given information, there is no explicit detail about treatments for knee replacement surgery.

**Supported citations**: `medquad_ans_0000039-11`, `medquad_ans_0000039-9`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.845, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000607-1__dup1` (score=0.0022) - Summary : A metabolic panel is a group of tests that measures different chemicals in the blood. These tests are usually done on the fluid (plasma) part of blood. The tests provide information about your body's chemica...
2. `medquad_ans_0000090-6` (score=0.0004) - Your doctor will diagnose metabolic syndrome based on the results of a physical exam and blood tests. You must have at least three of the five metabolic risk factors to be diagnosed with metabolic syndrome. Metabolic ...
3. `medquad_ans_0000090-1__dup2` (score=0.0001) - Metabolicsyndrome is the name for a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. The term "metabolic" refers to the biochemical processes involv...
4. `medquad_ans_0000090-5__dup1` (score=0.0001) - Metabolic syndrome is a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. These risk factors can increase your risk for health problems even if they'...
5. `medquad_ans_0000058-4__dup2` (score=0.0001) - Certain risk factors make it more likely that you'll develop coronary heart disease (CHD) and have a heart attack. You can control many of these risk factors. Risk Factors You Can Control The major risk factors for a ...

**Answer**:

The provided context does not mention any symptoms of metabolic syndrome. It describes the diagnostic criteria, including the five metabolic risk factors (large waistline, high triglyceride level, low HDL cholesterol level, high blood pressure, and high fasting blood sugar), but does not list symptoms. Therefore, based on the given information, it is not possible to answer the question about symptoms.

[medquad_ans_0000090-6]

**Supported citations**: `medquad_ans_0000090-6`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=0.500, groundedness=0.746, answer_relevance=0.567

**Top docs**:

1. `medquad_ans_0000037_2-8` (score=0.0002) - Other types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, t...
2. `medquad_ans_0000028_3-8` (score=0.0001) - New types of treatment are being tested in clinical trials. Patients may want to think about taking part in a clinical trial. For some patients, taking part in a clinical trial may be the best treatment choice. Clinic...
3. `medquad_ans_0000006_7-7` (score=0.0001) - Check the list of NCI-supported cancer clinical trials that are now accepting patients with childhood central nervous system germ cell tumor. For more specific results, refine the search by using other search features...
4. `medquad_ans_0000032_2-8` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
5. `medquad_ans_0000001_4-7` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...

**Answer**:

The provided context does not contain any information about the symptoms of Pyknoachondrogenesis. [medquad_ans_0000037_2-8] and [medquad_ans_0000028_3-8] discuss clinical trials for cancer treatments and are not relevant to the symptoms of Pyknoachondrogenesis.

Answer: Insufficient context.

**Supported citations**: `medquad_ans_0000037_2-8`, `medquad_ans_0000028_3-8`
**Unsupported citations**: _none_

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003965-2` (score=0.9993) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
2. `medquad_ans_0003966-1` (score=0.0163) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
3. `medquad_ans_0003965-4` (score=0.0002) - Who might consider genetic carrier testing for a family history of metachromatic leukodystrophy? If someone has a family history of metachromatic leukodystrophy (MLD) or someone is known to be a carrier for MLD, indiv...
4. `medquad_ans_0000181-1__dup2` (score=0.0001) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...
5. `medquad_ans_0001077-5` (score=0.0001) - These resources address the diagnosis or management of 18q deletion syndrome: - Gene Review: Gene Review: Leukodystrophy Overview - University of Texas Chromosome 18 Clinical Research Center These resources from Medli...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, limitation of joint mobility, muscular hypotonia, nystagmus, optic atrophy, aganglionic megacolon, ataxia, Babinski sign, bulbar palsy, cholecystitis, chorea, delusions, dysarthria, dystonia, EMG: neuropathic changes, emotional lability, gallbladder dysfunction, hallucinations, hyporeflexia, increased CSF protein, intellectual disability, loss of speech, mental deterioration, peripheral demyelination, progressive peripheral neuropathy, spastic tetraplegia, tetraplegia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000186-4` (score=0.9678) - When chronic granulomatous disease is caused by mutations in the CYBB gene, the condition is inherited in an X-linked recessive pattern. The CYBB gene is located on the X chromosome, which is one of the two sex chromo...
2. `medquad_ans_0001323-4` (score=0.0029) - Is chronic inflammatory demyelinating polyneuropathy (CIDP) inherited? CIDP is not known to be inherited and is considered an acquired disorder. No clear genetic predisposition or other predisposing factors for CIDP h...
3. `medquad_ans_0004341-3` (score=0.0004) - Is nephrocalcinosis inherited? Nephrocalcinosis may be caused by a large variety of things, including underlying disorders, certain medications and supplements, and infections. Nephrocalcinosis itself is not inherited...
4. `medquad_ans_0000186-1` (score=0.0001) - Chronic granulomatous disease is a disorder that causes the immune system to malfunction, resulting in a form of immunodeficiency. Immunodeficiencies are conditions in which the immune system is not able to protect th...
5. `medquad_ans_0000047-2__dup3` (score=0.0001) - Coughing occurs when the nerve endings in your airways become irritated. Certain irritants and allergens, medical conditions, and medicines can irritate these nerve endings. Irritants and Allergens An irritant is some...

**Answer**:

Yes, chronic granulomatous disease can be inherited. When caused by mutations in the CYBB gene, it is inherited in an X-linked recessive pattern. In this pattern, males are more frequently affected because they have only one X chromosome, and one altered copy of the gene is sufficient to cause the condition. Females require mutations in both copies of the gene to be affected, which is rare. Additionally, when caused by mutations in the CYBA, NCF1, NCF2, or NCF4 genes, the condition is inherited in an autosomal recessive pattern, meaning both copies of the gene in each cell must be mutated [medquad_ans_0000186-4].

**Supported citations**: `medquad_ans_0000186-4`
**Unsupported citations**: _none_

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.977, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000061-3__dup1` (score=0.9886) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
2. `medquad_ans_0000061-2__dup1` (score=0.0046) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...
3. `medquad_ans_0000176-3__dup1` (score=0.0012) - The outlook for someone with Wallenbergs syndrome depends upon the size and location of the area of the brain stem damaged by the stroke. Some individuals may see a decrease in their symptoms within weeks or months. O...
4. `medquad_ans_0000060-1__dup4` (score=0.0002) - Your nervous system carries signals between your brain and other parts of your body through your spinal cord. Nerves are bundles of special tissues that transmit these signals. The signals share information between yo...
5. `medquad_ans_0000061-4__dup1` (score=0.0001) - Our understanding of central cord syndrome has increased greatly in recent decades as a result of research funded conducted by the National Institute of Neurological Disorders and Stroke (NINDS). Much of this research...

**Answer**:

The outlook for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict the likelihood that neurological recovery may occur naturally. Those who receive medical intervention soon after their injury often have good outcomes. Many people with the disorder recover substantial function after their initial injury, and the ability to walk is recovered in most cases, although some impairment may remain. Improvement occurs first in the legs, then the bladder, and may be seen in the arms. Hand function recovers last, if at all. Recovery is generally better in younger patients, compared to those over the age of 50 [medquad_ans_0000061-3__dup1]. Additionally, there is no cure, but some people recover near-normal function, and early medical intervention or surgery may improve recovery chances [medquad_ans_0000061-2\_\_

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

**Scores**: context_relevance=1.000, groundedness=0.906, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000180-1__dup2` (score=0.6764) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
2. `medquad_ans_0000875-1__dup1` (score=0.0002) - Summary : Being a teenager is hard. You're under stress to be liked, do well in school, get along with your family, and make big decisions. You can't avoid most of these pressures, and worrying about them is normal. B...
3. `medquad_ans_0000006-4__dup1` (score=0.0001) - Several components of the NIH support research on developmental disorders such as ADHD. Research programs of the NINDS, the National Institute of Mental Health (NIMH), and the National Institute of Child Health and Hu...
4. `medquad_ans_0000432-1__dup2` (score=0.0001) - Summary : Health literacy refers to how well a person can get the health information and services that they need, and how well they understand them. It is also about using them to make good health decisions. It involv...
5. `medquad_ans_0000071-3__dup3` (score=0.0001) - Symptoms of a urinary tract infection (UTI) in the bladder may include - cloudy, bloody, or foul-smelling urine - pain or burning during urination - strong and frequent need to urinate, even right after emptying the b...

**Answer**:

Yes, there is information about child mental health. Early recognition and treatment of mental illnesses in children are important because once mental illness develops, it becomes a regular part of the child's behavior and is more difficult to treat. Warning signs that may indicate a more serious problem include problems in more than one setting (at school, at home, with peers), changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to younger behaviors such as bedwetting, signs of being upset like sadness or tearfulness, self-destructive behaviors like head-banging or sudden injuries, and repeated thoughts of death. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history. Treatments may include medicines and talk therapy [medquad_ans_0000180-1__dup2]. Additionally, mental health problems in teenagers can also be concerning, with warning signs such as persistent sadness, hopelessness, or worth

**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001618-2` (score=0.9976) - What are the signs and symptoms of Cowden syndrome? Cowden syndrome is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. Approximately 99% of people affected b...
2. `medquad_ans_0000246-1` (score=0.0001) - Cowden syndrome is a disorder characterized by multiple noncancerous, tumor-like growths called hamartomas and an increased risk of developing certain cancers. Almost everyone with Cowden syndrome develops hamartomas....
3. `medquad_ans_0004303-3` (score=0.0001) - How is Naegeli syndrome diagnosed? In most cases of Naegeli syndrome, a diagnosis is made based on the typical clinical features of this condition. The clinical diagnosis may be confirmed by genetic testing of the KRT...
4. `medquad_ans_0001618-1` (score=0.0001) - Cowden syndrome is an inherited condition that is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. It is considered part of the PTEN Hamartoma Tumor Syndrome ...
5. `medquad_ans_0000246-4` (score=0.0001) - Cowden syndrome and Cowden-like syndrome are inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the condition and increase the risk of developing c...

**Answer**:

The symptoms of Cowden syndrome include multiple noncancerous growths (hamartomas) on the skin and/or in the mouth, with approximately 99% of affected individuals having these by the third decade of life. A majority also develop hamartomatous polyps in the gastrointestinal tract. Other common symptoms include benign diseases of the breast, thyroid, and endometrium; a rare noncancerous brain tumor called Lhermitte-Duclos disease; enlarged head (macrocephaly); autism spectrum disorder; intellectual disability; and vascular abnormalities. Specific signs and symptoms reported in the Human Phenotype Ontology include abnormality of the pupil, abnormality of the tongue, aplasia/hypoplasia of the cerebellum, arteriovenous malformation, cognitive impairment, conjunctival hamartoma, dental malocclusion, epibulbar dermoid, exostoses, foot polydactyly,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.745, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.9929) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000334-5` (score=0.0936) - These resources address the diagnosis or management of facioscapulohumeral muscular dystrophy: - Gene Review: Gene Review: Facioscapulohumeral Muscular Dystrophy - Genetic Testing Registry: Facioscapulohumeral muscula...
3. `medquad_ans_0000576-5` (score=0.0004) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
4. `medquad_ans_0006267-3` (score=0.0003) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
5. `medquad_ans_0001026-5` (score=0.0001) - These resources address the diagnosis or management of Walker-Warburg syndrome: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Genetic Testing Registry: Walker-Warburg congenital muscular dystrop...

**Answer**:

The provided context does not specify the treatments for Fukuyama congenital muscular dystrophy. While it lists resources related to the diagnosis or management of Fukuyama congenital muscular dystrophy, such as Gene Review and Genetic Testing Registry entries, and references to general MedlinePlus resources on diagnosis and management (e.g., diagnostic tests, drug therapy, genetic counseling, palliative care), no specific treatments are detailed [medquad_ans_0000395-5]. Therefore, based on the available information, the treatments for Fukuyama congenital muscular dystrophy cannot be determined.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.855, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000004_1-4` (score=0.0148) - Tests that examine the lymph nodes are used to detect (find) and diagnose adult Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general sign...
2. `medquad_ans_0000004_6-3` (score=0.0001) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with ...
3. `medquad_ans_0000004_6-2` (score=0.0001) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
4. `medquad_ans_0000573-1__dup1` (score=0.0001) - Lymphoma is a cancer of a part of the immune system called the lymph system. There are many types of lymphoma. One type is Hodgkin disease. The rest are called non-Hodgkin lymphomas. Non-Hodgkin lymphomas begin when a...
5. `medquad_ans_0000004_6-6` (score=0.0001) - Key Points - After childhood Hodgkin lymphoma has been diagnosed, tests are done to find out if cancer cells have spread within the lymph system or to other parts of the body. - There are three ways that cancer spread...

**Answer**:

The provided context does not contain specific information on how to diagnose childhood Hodgkin lymphoma. While it details tests used to detect and diagnose adult Hodgkin lymphoma, such as physical exam and history, complete blood count, blood chemistry studies, sedimentation rate, lymph node biopsy, and immunophenotyping, there is no explicit mention of diagnostic procedures for childhood Hodgkin lymphoma. Therefore, based on the given context, the diagnosis of childhood Hodgkin lymphoma cannot be fully addressed.

[medquad_ans_0000004_1-4]

**Supported citations**: `medquad_ans_0000004_1-4`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.057, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002561-3` (score=0.6756) - How is Glass-Chapman-Hockley syndrome inherited? Based on the only family that has been reported in the medical literature, to date, the syndrome is believed to be inherited in an autosomal dominant manner.
2. `medquad_ans_0003718-4` (score=0.2713) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
3. `medquad_ans_0000840-4__dup1` (score=0.0321) - Is branchiootorenal syndrome inherited? Branchiootorenal syndrome may be inherited or occur sporadically. The inheritance pattern of branchiootorenal syndrome is autosomal dominant. Autosomal dominant inheritance is w...
4. `medquad_ans_0000562-4` (score=0.0204) - When Klippel-Feil syndrome is caused by mutations in the GDF6 or GDF3 genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder....
5. `medquad_ans_0000527-4` (score=0.0131) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...

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
2. `medquad_ans_0000194-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000196-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000195-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000197-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

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

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.9778) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0000113-4__dup3` (score=0.0004) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
3. `medquad_ans_0000038-16` (score=0.0001) - When you visit your doctor, here are questions to ask about your kidneys. - What is my GFR? - What is my urine albumin result? - What is my blood pressure? - What is my blood glucose (for people with diabetes)? What i...
4. `medquad_ans_0000015-29` (score=0.0001) - Heart disease and stroke are the leading causes of death for people with diabetes. Controlling the ABCs of diabetes -- your blood glucose, your blood pressure, and your cholesterol, as well as stopping smoking -- can ...
5. `medquad_ans_0000032-6__dup1` (score=0.0001) - Metabolic syndrome is a grouping of traits and medical conditions that puts people at risk for both heart disease and type 2 diabetes. It is defined by the National Cholesterol Education Program as having any three of...

**Answer**:

Drowning occurs when people get too much water in their lungs. It can happen quickly and silently, and even as little as an inch or two of water can be sufficient. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at risk in the water. Drowning precautions include fencing around pools, supervising children near any body of water, not swimming or boating under the influence of alcohol or sedatives, wearing life jackets when boating, and learning CPR [medquad_ans_0000299-1__dup2].

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

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.9994) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0000898-2` (score=0.0001) - Sepiapterin reductase deficiency appears to be a rare condition. At least 30 cases have been described in the scientific literature.
3. `medquad_ans_0000438-3` (score=0.0001) - Mutations in the LBR gene cause Greenberg dysplasia. This gene provides instructions for making a protein called the lamin B receptor. One region of this protein, called the sterol reductase domain, plays an important...
4. `medquad_ans_0001860-1` (score=0.0001) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
5. `medquad_ans_0001612-1` (score=0.0001) - What are the signs and symptoms of Cortisone reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Cortisone reductase deficiency. If the information is available, th...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment, microcephaly, cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, and tremor. Variable expressivity is also noted [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.857, groundedness=0.921, answer_relevance=0.286

**Top docs**:

1. `medquad_ans_0000669-2` (score=0.9987) - The prevalence of monilethrix is unknown.
2. `medquad_ans_0000075-2` (score=0.0001) - The prevalence of Asperger syndrome is not well established. Estimates range from 1 in 250 to 1 in 5,000 children. Three to four times as many males are affected than females. Because of changes in the way development...
3. `medquad_ans_0002491-2` (score=0.0001) - What are the signs and symptoms of Gardner-Diamond syndrome? People with Gardner-Diamond syndrome have reported that bruises occur either spontaneously or after trauma or surgery (even at other sites of the body). Som...
4. `medquad_ans_0000714-1` (score=0.0001) - Nail-patella syndrome is characterized by abnormalities of the nails, knees, elbows, and pelvis. The features of nail-patella syndrome vary in severity between affected individuals, even among members of the same fami...
5. `medquad_ans_0000875-1` (score=0.0001) - Rheumatoid arthritis is a disease that causes chronic abnormal inflammation, primarily affecting the joints. The most common signs and symptoms are pain, swelling, and stiffness of the joints. Small joints in the hand...

**Answer**:

The prevalence of monilethrix is unknown. [medquad_ans_0000669-2]

**Supported citations**: `medquad_ans_0000669-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.3190) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0000113-4__dup3` (score=0.0018) - The symptoms of IBS include abdominal pain or discomfort and changes in bowel habits. To meet the definition of IBS, the pain or discomfort should be associated with two of the following three symptoms: - start with b...
3. `medquad_ans_0000012-20` (score=0.0006) - More information on COPD is available at: What is COPD? and at the Learn More, Breathe Better Campaign For information on quitting smoking, visit http://www.surgeongeneral.gov/tobacco/ or Smokefree.gov. For informatio...
4. `medquad_ans_0000135-1__dup4` (score=0.0001) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...
5. `medquad_ans_0000671-5` (score=0.0001) - These resources address the diagnosis or management of moyamoya disease: - Barrow Neurological Institute: What Medical Therapies Are Used To Treat Moyamoya Disease? - Boston Children's Hospital: Learn More About Treat...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria and is caused by mutations in the SLC6A19 gene, inherited in an autosomal recessive manner. Most people with Hartnup disease have no symptoms (asymptomatic), though some may develop symptoms starting between the ages of 3 and 9, occasionally in adulthood. Symptoms may include skin photosensitivity, neurologic findings, psychiatric symptoms, and ocular findings. Mental development is usually normal, though a few cases with intellectual impairment have been reported. Treatment may include a high-protein diet, protection from sunlight, vitamin supplementation, avoidance of certain drugs/medications, and in some cases, nicotinamide supplements and tryptophan ethyl ester [medquad_ans_0002733-1].

**Supported citations**: `medquad_ans_0002733-1`
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=1.000, groundedness=0.764, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000428-1` (score=0.0001) - Glycogen storage disease type VII (GSDVII) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in muscle cells. A lack of glycogen breakdown interferes with the function of mu...
2. `medquad_ans_0000427-1` (score=0.0001) - Glycogen storage disease type VI (also known as GSDVI or Hers disease) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in liver cells. A lack of glycogen breakdown interfe...
3. `medquad_ans_0002615-3` (score=0.0001) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an impor...
4. `medquad_ans_0000424-5` (score=0.0001) - These resources address the diagnosis or management of glycogen storage disease type IV: - Gene Review: Gene Review: Glycogen Storage Disease Type IV - Genetic Testing Registry: Glycogen storage disease, type IV - Med...
5. `medquad_ans_0002617-1` (score=0.0001) - Glycogen storage disease type 1B (GSD1B) is an inherited condition in which the body is unable to break down a complex sugar called glycogen. As a result, glycogen accumulates in cells throughout the body. In GSD1B, s...

**Answer**:

The provided context does not specify the genetic changes related to glycogen storage disease type VII. [medquad_ans_0000428-1] describes the clinical features and types of GSDVII but does not mention the underlying genetic mutations or changes. Therefore, based on the given information, the genetic changes are not available.

**Supported citations**: `medquad_ans_0000428-1`
**Unsupported citations**: _none_

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.756, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000977-5` (score=0.9933) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
2. `medquad_ans_0000513-3` (score=0.0001) - Mutations in the AMN or CUBN gene can cause Imerslund-Grsbeck syndrome. The AMN gene provides instructions for making a protein called amnionless, and the CUBN gene provides instructions for making a protein called cu...
3. `medquad_ans_0000977-2` (score=0.0001) - Thiamine-responsive megaloblastic anemia syndrome has been reported in approximately 30 families worldwide. Its prevalence is unknown.
4. `medquad_ans_0000283-5` (score=0.0001) - These resources address the diagnosis or management of Diamond-Blackfan anemia: - Gene Review: Gene Review: Diamond-Blackfan Anemia - Genetic Testing Registry: Aase syndrome - Genetic Testing Registry: Diamond-Blackfa...
5. `medquad_ans_0000977-3` (score=0.0001) - Mutations in the SLC19A2 gene cause thiamine-responsive megaloblastic anemia syndrome. This gene provides instructions for making a protein called thiamine transporter 1, which transports thiamine into cells. Thiamine...

**Answer**:

The provided context does not specify the treatments for thiamine-responsive megaloblastic anemia syndrome. While it mentions resources related to the diagnosis and management of the condition, such as genetic testing, diagnostic tests, drug therapy, and genetic counseling, it does not detail specific treatment options [medquad_ans_0000977-5]. Therefore, based on the given information, the treatments for thiamine-responsive megaloblastic anemia syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.772, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000173-5__dup1` (score=0.0240) - People with a solitary kidney do not need to eat a special diet. However, people with reduced kidney function may need to make changes to their diet to slow the progression of kidney disease. More information about re...
2. `medquad_ans_0000201-9` (score=0.0019) - - Hematuria is blood in the urine. - Most people with microscopic hematuria do not have symptoms. People with gross hematuria have urine that is pink, red, or cola-colored due to the presence of red blood cells (RBCs)...
3. `medquad_ans_0000038-22` (score=0.0012) - During your next health care visit, talk to your provider about your test results and how to manage your kidney disease. Below is a list of questions you may want to ask. Add any questions you think are missing, and m...
4. `medquad_ans_0000135-2__dup2` (score=0.0006) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
5. `medquad_ans_0000135-1__dup4` (score=0.0003) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...

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

The provided context does not mention any research or clinical trials being conducted for Rheumatoid Arthritis. Therefore, based on the given information, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.965, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0001094-5` (score=0.9968) - These resources address the diagnosis or management of 3MC syndrome: - Genetic Testing Registry: Carnevale syndrome - Genetic Testing Registry: Craniofacial-ulnar-renal syndrome - Genetic Testing Registry: Malpuech fa...
2. `medquad_ans_0000392-5` (score=0.9955) - These resources address the diagnosis or management of Fryns syndrome: - Children's Hospital of Philadelphia: Treatment of Congenital Diaphragmatic Hernia - Gene Review: Gene Review: Fryns Syndrome - Genetic Testing R...
3. `medquad_ans_0002561-4` (score=0.9770) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
4. `medquad_ans_0000084-2__dup1` (score=0.1724) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
5. `medquad_ans_0004451-3` (score=0.0133) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

Surgery is typically the treatment for craniosynostosis associated with Glass-Chapman-Hockley syndrome and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.788, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000392-5` (score=0.9968) - These resources address the diagnosis or management of Fryns syndrome: - Children's Hospital of Philadelphia: Treatment of Congenital Diaphragmatic Hernia - Gene Review: Gene Review: Fryns Syndrome - Genetic Testing R...
2. `medquad_ans_0001094-5` (score=0.9965) - These resources address the diagnosis or management of 3MC syndrome: - Genetic Testing Registry: Carnevale syndrome - Genetic Testing Registry: Craniofacial-ulnar-renal syndrome - Genetic Testing Registry: Malpuech fa...
3. `medquad_ans_0000868-5` (score=0.9961) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
4. `medquad_ans_0004452-2` (score=0.0911) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004456-3` (score=0.0911) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain specific information about the treatments for Renpenning syndrome. While it lists resources related to Renpenning syndrome, such as the Genetic Testing Registry: Renpenning syndrome 1 and Greenwood Genetics Center: X-Linked Intellectual Disability, and references to general resources like Genetic Counseling and Palliative Care from MedlinePlus, no detailed treatment information is available [medquad_ans_0000868-5]. Therefore, based on the given context, the treatments for Renpenning syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.734, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0001094-1__dup1` (score=0.0001) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
3. `medquad_ans_0001100-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2B2? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2B2. If the information is a...
4. `medquad_ans_0001093-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1D? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1D. If the information is ava...
5. `medquad_ans_0001112-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...

**Answer**:

The provided context does not contain information about the symptoms of Charcot-Marie-Tooth disease type 2G. Therefore, a specific answer cannot be given based on the available content. [medquad_ans_0001100-1__dup1] only mentions Charcot-Marie-Tooth disease type 2B2, and no details are provided for type 2G.

**Supported citations**: `medquad_ans_0001100-1__dup1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.633

**Top docs**:

1. `medquad_ans_0000005-6__dup1` (score=0.0001) - Clinical trials are part of clinical research and at the heart of all treatment advances. Clinical trials look at new ways to prevent, detect, or treat disease. The National Institute of Mental Health at NIH supports ...
2. `medquad_ans_0000014_1-8` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
3. `medquad_ans_0000014_2-8` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
4. `medquad_ans_0000032_2-8` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...
5. `medquad_ans_0000003_2-7` (score=0.0001) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI website. Patients may want to think about taking part in a clinical trial. For some patients, tak...

**Answer**:

The provided context does not mention the symptoms of coccygodynia. Therefore, the information required cannot be derived from the given text. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

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
4. `medquad_ans_0000161-5__dup2` (score=0.0006) - Urine samples are taken to diagnose people suspected of having nephrotic syndrome. Nephrotic syndrome is diagnosed when large amounts of protein are found in the urine. The blood protein albumin makes up much of the p...
5. `medquad_ans_0000089-5__dup1` (score=0.0001) - Your doctor will diagnose Marfan syndrome based on your medical and family histories, a physical exam, and test results. He or she also will consult a set of guidelines called Ghent criteria, which are used to diagnos...

**Answer**:

Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. KFS can be associated with a wide range of other abnormalities involving many parts of the body. Therefore, other initial exams are needed to detect additional physical abnormalities or underlying conditions. These include: examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present in each person with KFS [medquad_ans_0003402

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.536, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001497-1` (score=0.9537) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
2. `medquad_ans_0000222-1` (score=0.0661) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
3. `medquad_ans_0002458-1` (score=0.0068) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...
4. `medquad_ans_0000222-2` (score=0.0004) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
5. `medquad_ans_0000222-3` (score=0.0002) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...

**Answer**:

Congenital diaphragmatic hernia (CDH) is a defect in the diaphragm caused by abnormal development before birth. The diaphragm, composed of muscle and other fibrous tissue, normally separates the organs in the abdomen from those in the chest. CDH ranges in severity from a thinned area in the diaphragm to its complete absence, resulting in an abnormal opening (hernia) that allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This crowding can lead to underdevelopment of the lungs (pulmonary hypoplasia), potentially causing life-threatening breathing difficulties apparent from birth. CDH may occur with other malformations and is classified by position, such as Bochdalek hernia (80–90% of cases, involving the side or back of the diaphragm) and Morgagni hernia (about 2% of cases

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001088-1__dup1` (score=0.0761) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
2. `medquad_ans_0000173-1` (score=0.0431) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
3. `medquad_ans_0001094-1__dup1` (score=0.0020) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
4. `medquad_ans_0000173-2` (score=0.0014) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence o...
5. `medquad_ans_0001104-5` (score=0.0003) - Is genetic testing available for Charcot-Marie-Tooth disease type 2F? Yes. GeneTests lists the names of laboratories that are performing clincial genetic testing for Charcot-Marie-Tooth disease type 2F. To view the co...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Defects in many different genes cause different forms of this disease. Common symptoms may include foot drop, foot deformity, loss of lower leg muscle, numbness in the foot or leg, slapping gait, and weakness of the hips, legs, or feet. The disease usually becomes apparent in adolescence or early adulthood, but onset may occur anytime from early childhood through late adulthood. Symptoms vary in severity, even among family members, and may include balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities such as high arches or flat feet, and decreased sensitivity to touch. While there is currently no cure, management options include physical therapy, occupational therapy, braces, orthopedic devices, pain medication, and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=1.000, groundedness=0.934, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000276-4` (score=0.9969) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000527-4` (score=0.9932) - The inheritance of IMAGe syndrome is complex. The condition is described as having an autosomal dominant inheritance pattern because one copy of the altered CDKN1C gene in each cell is sufficient to cause the disorder...
3. `medquad_ans_0000669-4` (score=0.9908) - Monilethrix can have multiple patterns of inheritance. When the condition is caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene...
4. `medquad_ans_0003860-4` (score=0.0107) - Is MECP2 duplication syndrome inherited? MECP2 duplication syndrome is inherited in an X-linked manner. A condition is considered X-linked if the genetic change that causes the condition is located on the X chromosome...
5. `medquad_ans_0000836-4` (score=0.0014) - Many cases of prostate cancer are not related to inherited gene changes. These cancers are associated with somatic mutations that occur only in certain cells in the prostate. When prostate cancer is related to inherit...

**Answer**:

Yes, monilethrix can be inherited, depending on the cause. When caused by a mutation in one of the keratin genes, it is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder [medquad_ans_0000669-4]. In rare cases, monilethrix results from a new mutation and is not inherited. When caused by mutations in the DSG4 gene, it is inherited in an autosomal recessive pattern, requiring mutations in both copies of the gene [medquad_ans_0000669-4]. Therefore, monilethrix can be inherited in either an autosomal dominant or autosomal recessive pattern.

**Supported citations**: `medquad_ans_0000669-4`
**Unsupported citations**: _none_

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.895, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000215-3__dup2` (score=0.0924) - Most UTIs are caused by bacteria that live in the bowel. The bacterium Escherichia coli (E. coli) causes the vast majority of UTIs. The urinary tract has several systems to prevent infection. The points where the uret...
2. `medquad_ans_0000223-1__dup3` (score=0.0533) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. Bacteria \* are the most common cause of UTIs. Normally, bacteria that enter the urinary...
3. `medquad_ans_0000215-1__dup3` (score=0.0212) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
4. `medquad_ans_0000223-2__dup2` (score=0.0116) - Most UTIs are caused by bacteria that live in the bowel, the part of the digestive tract where stool is changed from liquid to solid. The bacterium Escherichia coli (E. coli) causes most UTIs. The urinary tract has se...
5. `medquad_ans_0000215-4__dup2` (score=0.0038) - Urinary tract infections affect about 3 percent of children in the United States every year. UTIs account for more than 1 million visits to pediatricians offices every year.1

**Answer**:

Urinary Tract Infections (UTIs) in children are infections in the urinary tract caused primarily by bacteria, most commonly Escherichia coli (E. coli), which live in the bowel. These infections occur when bacteria overcome the body's natural defenses and enter the urinary tract. The urinary tract has protective mechanisms, such as one-way valves at the ureters that prevent urine reflux and urination that flushes microbes out of the body. However, factors such as delaying urination, inadequate fluid intake, chronic constipation (fewer than two bowel movements per week), and the physical pressure from full bowel contents on the bladder can increase the risk of bacterial growth and infection. Children who are prone to such infections are more likely to develop UTIs, similar to how some children are prone to other infections like coughs or ear infections [medquad_ans_0000215-3__dup2]medquad_ans_00

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
5. `medquad_ans_0000344-1__dup1` (score=0.0001) - Your eyes can get infections from bacteria, fungi, or viruses. Eye infections can occur in different parts of the eye and can affect just one eye or both. Two common eye infections are - Conjunctivitis - also known as...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, a group of related viruses. Infection leads to an inflammation of the stomach and intestines, with common symptoms including diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms are fever, headache, and body aches. The illness can spread from person to person, through contaminated food or water, or by touching a contaminated surface. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in adults, often responsible for epidemics, especially from October to April. Symptoms typically appear 1 to 2 days after exposure and last for 1 to 3 days. There is no specific medicine to treat norovirus infections; treatment includes bed rest and plenty of fluids to prevent dehydration. Prevention can be achieved through proper hand washing and safe food preparation [medquad_ans_0000649-

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.900, groundedness=0.958, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000392-3` (score=0.0234) - The cause of Fryns syndrome is unknown. The disorder is thought to be genetic because it tends to run in families and has features similar to those of other genetic disorders. Duplications and deletions in several chr...
2. `medquad_ans_0000222-3` (score=0.0110) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...
3. `medquad_ans_0000222-2` (score=0.0001) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
4. `medquad_ans_0000254-3` (score=0.0001) - Crohn disease is related to chromosomes 5 and 10. Variations of the ATG16L1, IRGM, and NOD2 genes increase the risk of developing Crohn disease. The IL23R gene is associated with Crohn disease. A variety of genetic an...
5. `medquad_ans_0000222-1` (score=0.0001) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...

**Answer**:

The genetic changes related to congenital diaphragmatic hernia include duplications and deletions in several chromosome regions, which have been associated with the condition and some of its features, such as those seen in Fryns syndrome [medquad_ans_0000392-3]. Some syndromes in which congenital diaphragmatic hernia may occur, like Donnai-Barrow, Fryns, and Pallister-Killian mosaic syndrome, are caused by changes in single genes or chromosomal abnormalities affecting multiple genes [medquad_ans_0000222-3]. Researchers are also studying changes in several genes involved in diaphragm development, including transcription factors that regulate gene expression and genes involved in cell structure or cell migration during embryonic development [medquad_ans_0000222-3]. However, no specific genetic change has been found to cause all of the signs and symptoms of Fryns

**Supported citations**: `medquad_ans_0000392-3`, `medquad_ans_0000222-3`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004534-3` (score=0.9950) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
2. `medquad_ans_0000674-3__dup1` (score=0.0005) - What causes Behr syndrome? The exact cause of Behr syndrome is not known; however, a genetic cause is suspected based on the families identified, thus far.
3. `medquad_ans_0003540-3` (score=0.0005) - What causes Lemierre syndrome? In about 90% of cases, Lemierre syndrome is caused by Fusobacterium necrophorum; however, the syndrome has also been reported with other bacteria, including Stapylococcus aureus, Bactero...
4. `medquad_ans_0002747-3` (score=0.0004) - What causes HELLP syndrome? Doctors are still unclear on what exactly causes HELLP syndrome. Although it is more common in women who have preeclampsia or pregnancy induced hypertension (high blood pressure), there are...
5. `medquad_ans_0000765-3__dup1` (score=0.0003) - What causes blue rubber bleb nevus syndrome? Currently the cause of blue rubber bleb syndrome is not known.

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by mutations in several genes, including RAG-1 and RAG-2 (mapped to chromosome band 11p13), ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4. Some cases have also been associated with 22q11 microdeletion syndrome [medquad_ans_0004534-3].

**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: _none_

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=1.000, groundedness=0.977, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.9864) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0000032-1__dup1` (score=0.0264) - Ataxia often occurs when parts of the nervous system that control movement are damaged. People with ataxia experience a failure of muscle control in their arms and legs, resulting in a lack of balance and coordination...
3. `medquad_ans_0000495-1__dup1` (score=0.0014) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
4. `medquad_ans_0000078-2` (score=0.0005) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
5. `medquad_ans_0000496-1__dup1` (score=0.0001) - What are the signs and symptoms of Ataxia with Oculomotor Apraxia Type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Ataxia with Oculomotor Apraxia Type 2. If the information is...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive movement problems due to a loss (atrophy) of nerve cells in the cerebellum, the part of the brain that coordinates movement. Signs and symptoms first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both, as well as difficulty with movements that involve judging distance or scale (dysmetria). Other features include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive and may eventually result in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.750, groundedness=0.890, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000049-2__dup1` (score=0.9721) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
2. `medquad_ans_0000050-2__dup1` (score=0.9721) - Some brachial plexus injuries may heal without treatment. Many children who are injured during birth improve or recover by 3 to 4 months of age. Treatment for brachial plexus injuries includes physical therapy and, in...
3. `medquad_ans_0000034-8` (score=0.0011) - Proximal neuropathy, sometimes called lumbosacral plexus neuropathy, femoral neuropathy, or diabetic amyotrophy, starts with pain in the thighs, hips, buttocks, or legs, usually on one side of the body. This type of n...
4. `medquad_ans_0001025-5__dup1` (score=0.0005) - How might celiac artery compression syndrome be treated? Surgery is currently the only treatment option for celiac artery compression syndrome. Surgery typically involves decompression of the celiac artery by dividing...
5. `medquad_ans_0000468-3` (score=0.0005) - Mutations in the SEPT9 gene cause hereditary neuralgic amyotrophy. The SEPT9 gene provides instructions for making a protein called septin-9, which is part of a group of proteins called septins. Septins are involved i...

**Answer**:

The treatments for brachial plexus injuries include physical therapy and, in some cases, surgery [medquad_ans_0000049-2__dup1].

**Supported citations**: `medquad_ans_0000049-2__dup1`
**Unsupported citations**: _none_

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.927, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000216-8` (score=0.9936) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
2. `medquad_ans_0000216-7` (score=0.0299) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
3. `medquad_ans_0000216-2__dup1` (score=0.0293) - Many types of defects in the urinary tract can cause urine blockage: - Vesicoureteral reflux (VUR). Most children with VUR are born with a ureter that did not grow long enough during development in the womb. The valve...
4. `medquad_ans_0000216-4__dup1` (score=0.0033) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...
5. `medquad_ans_0000217-6` (score=0.0020) - The most common tests used to diagnose VUR include - Voiding cystourethrogram (VCUG). VCUG is an x-ray image of the bladder and urethra taken during urination, also called voiding. The bladder and urethra are filled w...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), ureterocele, and genetic conditions like prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects. Additionally, spina bifida and other birth defects affecting the spinal cord can lead to urine blockage [medquad_ans_0000216-8]. These defects can result in urine backing up, causing hydroureter and hydronephrosis. Defects may be detected before or after birth through prenatal tests such as ultrasound, amniocentesis, or chorionic villus sampling, and further evaluated using imaging techniques like ultrasound, voiding cystourethrogram (VCUG

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

**Scores**: context_relevance=0.833, groundedness=0.879, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000058-3__dup4` (score=0.9849) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
2. `medquad_ans_0000036-12` (score=0.9806) - Abnormal blood pressure is higher than 120/80 mmHg. If either your systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hy...
3. `medquad_ans_0000036-9` (score=0.9619) - Blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If your blood pressure rises and stays high over time, its called high blood pressure. High blood pressure ...
4. `medquad_ans_0000471-1__dup1` (score=0.9464) - Blood pressure is the force of your blood pushing against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries. Your blood pressure is highest when your heart beats, pumping the blo...
5. `medquad_ans_0000150-1__dup3` (score=0.9007) - Blood pressure is the force of blood pushing against blood vessel walls as the heart pumps out blood, and high blood pressure, also called hypertension, is an increase in the amount of force that blood places on blood...

**Answer**:

High blood pressure is defined as a blood pressure reading of 140/90 mmHg or higher. It is characterized by consistently elevated pressure in the blood vessels, which forces the heart to work harder and can lead to damage in arteries and organs such as the heart, kidneys, brain, and eyes. High blood pressure is dangerous because it increases the risk of serious health problems including stroke, heart attack, heart failure, and kidney failure [medquad_ans_0000058-3\_\_dup4][medquad_ans_0000036-9][medquad_ans_0000471-1__dup1].

**Supported citations**: `medquad_ans_0000058-3__dup4`, `medquad_ans_0000036-9`, `medquad_ans_0000471-1__dup1`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=0.750, groundedness=0.012, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0001058-4` (score=0.9962) - This condition is inherited in an X-linked pattern. A condition is considered X-linked if the mutated gene that causes the disorder is located on the X chromosome, one of the two sex chromosomes in each cell. In males...
2. `medquad_ans_0001053-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
3. `medquad_ans_0000110-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
4. `medquad_ans_0000570-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0001052-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...

**Answer**:

The provided context does not mention X-linked dystonia-parkinsonism or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether X-linked dystonia-parkinsonism is inherited. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.526, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000702-1__dup1` (score=0.9065) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
2. `medquad_ans_0000120-1` (score=0.8601) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
3. `medquad_ans_0000120-2` (score=0.0002) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.
4. `medquad_ans_0003472-1` (score=0.0001) - L-arginine:glycine amidinotransferase (AGAT) deficiency is a rare condition that primarily affects the brain. People with AGAT deficiency generally have mild to moderate intellectual disability. Other signs and sympto...
5. `medquad_ans_0000415-5` (score=0.0001) - These resources address the diagnosis or management of GLUT1 deficiency syndrome: - G1D Registry - Gene Review: Gene Review: Glucose Transporter Type 1 Deficiency Syndrome - Genetic Testing Registry: Glucose transport...

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months and include episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, known as ketoacidotic attacks, sometimes lead to coma and are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. The condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene. It is estimated to affect fewer than 1 in 1 million newborns [medquad_ans_0000702-1\_\_dup1, medquad

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

**Scores**: context_relevance=0.857, groundedness=0.539, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000842-1__dup2` (score=0.9903) - Your backbone, or spine, is made up of 26 bone discs called vertebrae. The vertebrae protect your spinal cord and allow you to stand and bend. A number of problems can change the structure of the spine or damage the v...
2. `medquad_ans_0000266-1__dup2` (score=0.3112) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
3. `medquad_ans_0000060-1__dup4` (score=0.1780) - Your nervous system carries signals between your brain and other parts of your body through your spinal cord. Nerves are bundles of special tissues that transmit these signals. The signals share information between yo...
4. `medquad_ans_0000061-1__dup1` (score=0.0023) - Central cord syndrome is the most common form of incomplete spinal cord injury characterized by impairment in the arms and hands and to a lesser extent in the legs. The brain's ability to send and receive signals to a...
5. `medquad_ans_0000839-1__dup1` (score=0.0017) - Your spinal cord is a bundle of nerves that runs down the middle of your back. It carries signals back and forth between your body and your brain. A spinal cord injury disrupts the signals. Spinal cord injuries usuall...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course of the disorder. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is often linked to improper growth of the neural tube during fetal development and is closely associated with spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to the degree of strain on the spinal

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.732, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000034-9` (score=0.0009) - Focal neuropathy appears suddenly and affects specific nerves, most often in the head, torso, or leg. Focal neuropathy may cause - inability to focus the eye - double vision - aching behind one eye - paralysis on one ...
2. `medquad_ans_0000136-5__dup2` (score=0.0001) - The symptoms of dialysis-related amyloidosis may include - pain, stiffness, and fluid in the joints. - abnormal, fluid-containing sacs, called cysts, in some bones. - carpal tunnel syndrome, caused by unusual buildup ...
3. `medquad_ans_0000058-1__dup1` (score=0.0001) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
4. `medquad_ans_0000034-2__dup4` (score=0.0001) - The causes are probably different for different types of diabetic neuropathy. Researchers are studying how prolonged exposure to high blood glucose causes nerve damage. Nerve damage is likely due to a combination of f...
5. `medquad_ans_0000419-1__dup1` (score=0.0001) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...

**Answer**:

The provided context does not contain information about the outlook or prognosis for carpal tunnel syndrome. While it describes symptoms, causes, and contributing factors, there is no mention of recovery timeline, long-term outcomes, or prognosis. Therefore, based on the given context, the outlook for carpal tunnel syndrome cannot be determined. [medquad_ans_0000136-5\_\_dup2][medquad_ans_0000058-1__dup1]

**Supported citations**: `medquad_ans_0000136-5__dup2`, `medquad_ans_0000058-1__dup1`
**Unsupported citations**: _none_

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.783, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0001094-5` (score=0.9966) - These resources address the diagnosis or management of 3MC syndrome: - Genetic Testing Registry: Carnevale syndrome - Genetic Testing Registry: Craniofacial-ulnar-renal syndrome - Genetic Testing Registry: Malpuech fa...
2. `medquad_ans_0000296-5` (score=0.9939) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphrag...
3. `medquad_ans_0000392-5` (score=0.0506) - These resources address the diagnosis or management of Fryns syndrome: - Children's Hospital of Philadelphia: Treatment of Congenital Diaphragmatic Hernia - Gene Review: Gene Review: Fryns Syndrome - Genetic Testing R...
4. `medquad_ans_0004452-2` (score=0.0057) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004454-3` (score=0.0057) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not specify the treatments for Donnai-Barrow syndrome. While it lists resources related to diagnosis and management, such as genetic counseling, diagnostic tests, drug therapy, surgery and rehabilitation, and palliative care, it does not detail specific treatments for Donnai-Barrow syndrome [medquad_ans_0000296-5]. Therefore, based on the available information, the exact treatments cannot be determined.

**Supported citations**: `medquad_ans_0000296-5`
**Unsupported citations**: _none_

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=0.750, groundedness=0.533, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000189-3__dup2` (score=0.6440) - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Experts often classify porphyrias as acute or cutaneous based on the symptoms a person experiences:...
2. `medquad_ans_0000729-1__dup1` (score=0.4636) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. Th...
3. `medquad_ans_0000811-1` (score=0.3049) - Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marr...
4. `medquad_ans_0000189-5__dup1` (score=0.0841) - Most porphyrias are inherited disorders. Scientists have identified genes for all eight enzymes in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, f...
5. `medquad_ans_0004987-1` (score=0.0456) - Porphyria cutanea tarda (PCT) is a form of porphyria that primarily affects the skin. People affected by this condition generally experience "photosensitivity," which causes painful, blistering lesions to develop on s...

**Answer**:

Porphyria is a group of genetic disorders caused by problems with how the body makes a substance called heme. Heme is found throughout the body, especially in the blood and bone marrow, where it carries oxygen. There are two main types of porphyrias: one that affects the skin and another that affects the nervous system. Skin-type porphyrias cause blisters, itching, and swelling of the skin when exposed to sunlight. Acute porphyria, which affects the nervous system, causes symptoms such as pain in the chest, abdomen, limbs, or back; muscle numbness, tingling, paralysis, or cramping; vomiting; constipation; and personality changes or mental disorders. These symptoms come and go and can be triggered by factors such as certain medicines, smoking, alcohol, infections, stress, or sun exposure. Attacks develop over hours or days and can last for days or weeks [medquad_ans_00007

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=0.917, groundedness=0.986, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0001900-1` (score=0.9606) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual...
2. `medquad_ans_0000585-1__dup1` (score=0.9521) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
3. `medquad_ans_0000674-1__dup1` (score=0.0385) - Behr syndrome is a disorder mainly characterized by early-onset optic atrophy, ataxia, and spasticity. Other signs and symptoms may be present and vary from person to person. Although the exact cause is unknown, the s...
4. `medquad_ans_0003414-1` (score=0.0250) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, whic...
5. `medquad_ans_0000229-1__dup1` (score=0.0031) - Peripheral neuropathy describes damage to the peripheral nervous system, which transmits information from the brain and spinal cord to every other part of the body. More than 100 types of peripheral neuropathy have be...

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

**Scores**: context_relevance=0.800, groundedness=0.103, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000608-3` (score=0.0125) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...
2. `medquad_ans_0000836-3` (score=0.0019) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...
3. `medquad_ans_0000849-3` (score=0.0008) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in thi...
4. `medquad_ans_0000142-3` (score=0.0002) - C3 glomerulopathy is associated with changes in many genes. Most of these genes provide instructions for making proteins that help regulate a part of the body's immune response known as the complement system. This sys...
5. `medquad_ans_0001041-3` (score=0.0002) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...

**Answer**:

The provided context does not mention pulmonary veno-occlusive disease or any genetic changes related to it. Therefore, no answer can be given based on the available information. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.875, groundedness=0.844, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0001006-2` (score=0.0001) - Worldwide, tyrosinemia type I affects about 1 in 100,000 individuals. This type is more common in Norway where 1 in 60,000 to 74,000 individuals are affected. Tyrosinemia type I is even more common in Quebec, Canada w...
2. `medquad_ans_0001006-1` (score=0.0001) - Tyrosinemia is a genetic disorder characterized by disruptions in the multistep process that breaks down the amino acid tyrosine, a building block of most proteins. If untreated, tyrosine and its byproducts build up i...
3. `medquad_ans_0006260-1` (score=0.0001) - Tyrosinemia type 1 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a shortage of the enzyme fumarylacetoacetate h...
4. `medquad_ans_0006262-1` (score=0.0001) - Tyrosinemia type 3 is a genetic disorder characterized by elevated blood levels of the amino acid tyrosine, a building block of most proteins. This condition is caused by a deficiency of the enzyme 4-hydroxyphenylpyru...
5. `medquad_ans_0000836-3` (score=0.0001) - Cancers occur when genetic mutations build up in critical genes, specifically those that control cell growth and division or the repair of damaged DNA. These changes allow cells to grow and divide uncontrollably to fo...

**Answer**:

The provided context does not specify the exact genetic changes related to tyrosinemia. It mentions that there are three types of tyrosinemia, each distinguished by their symptoms and genetic cause [medquad_ans_0001006-1], and that tyrosinemia type III is the rarest with features including intellectual disability, seizures, and ataxia [medquad_ans_0001006-1], but it does not detail the specific genetic mutations or changes associated with each type. Therefore, based on the given information, the genetic changes related to tyrosinemia cannot be determined.

**Supported citations**: `medquad_ans_0001006-1`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.857, groundedness=0.841, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000436-5` (score=0.9887) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
2. `medquad_ans_0000084-2__dup1` (score=0.8171) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
3. `medquad_ans_0000203-2__dup1` (score=0.5399) - There is no cure for MSA. Currently, there are no treatments to delay the progress of neurodegeneration in the brain. But there are treatments available to help people cope with some of the more disabling symptoms of ...
4. `medquad_ans_0000019_3-7` (score=0.0273) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
5. `medquad_ans_0000581-5` (score=0.0010) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...

**Answer**:

The treatments for Graves' disease include options provided by the Graves' Disease & Thyroid Foundation, which outlines treatment options for the condition [medquad_ans_0000436-5]. Additionally, diagnostic and management resources from the American Thyroid Association, National Institute of Diabetes and Digestive and Kidney Diseases, and Thyroid Disease Manager cover diagnosis and treatment of Graves' disease [medquad_ans_0000436-5]. However, specific drug therapies or treatment modalities are not detailed in the provided context.

**Supported citations**: `medquad_ans_0000436-5`
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

**Scores**: context_relevance=0.900, groundedness=0.990, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.9991) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000941-3` (score=0.9805) - SCA36 is caused by mutations in the NOP56 gene. The NOP56 gene provides instructions for making a protein called nucleolar protein 56, which is primarily found in the nucleus of nerve cells (neurons), particularly tho...
3. `medquad_ans_0000416-2` (score=0.0001) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
4. `medquad_ans_0002601-1` (score=0.0001) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
5. `medquad_ans_0000416-1` (score=0.0001) - Glutamate formiminotransferase deficiency is an inherited disorder that affects physical and mental development. There are two forms of this condition, which are distinguished by the severity of symptoms. People with ...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the last two steps of histidine metabolism and in the production of folate [medquad_ans_0000416-3]. However, it is unclear how these changes are related to the specific health problems associated with the mild and severe forms of the disorder or why individuals are affected by one form or the other [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001169-2` (score=0.9994) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
2. `medquad_ans_0001169-1` (score=0.0001) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
3. `medquad_ans_0004472-1` (score=0.0001) - What are the signs and symptoms of Nystagmus 1, congenital, X- linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Nystagmus 1, congenital, X- linked. If the information is avail...
4. `medquad_ans_0001170-1` (score=0.0001) - X-linked dominant chondrodysplasia punctata (CDPX2), also known as Conradi-Hnermann-Happle syndrome, is a rare form of skeletal dysplasia characterized by skeletal malformations, skin abnormalities, cataracts and shor...
5. `medquad_ans_0001047-1` (score=0.0001) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include abnormality of the vertebral column, anosmia, cataract, depressed nasal bridge, epiphyseal stippling, hearing impairment, hypogonadism, ichthyosis, microcephaly, short distal phalanx of finger, short nasal septum, short nose, short stature, and X-linked recessive inheritance. Additional common features include breathing abnormalities, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development. Epiphyseal stippling is visible on x-ray and typically disappears between ages 2 and 3. The condition is caused by changes in the ARSE gene on the X chromosome and is inherited in an X-linked recessive manner, occurring almost exclusively in males [medquad_ans_0001169-1]. [medquad_ans_0

**Supported citations**: `medquad_ans_0001169-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.857, groundedness=0.465, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000132-10` (score=0.9879) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
2. `medquad_ans_0000132-8` (score=0.9844) - The health care provider prescribes antibiotics to destroy the T. whipplei bacteria and treat Whipple disease. Health care providers choose antibiotics that treat the infection in the small intestine and cross the blo...
3. `medquad_ans_0000084-2__dup1` (score=0.0586) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
4. `medquad_ans_0000019_3-7` (score=0.0492) - New types of treatment are being tested in clinical trials. Information about clinical trials is available from the NCI Web site. Patients may want to think about taking part in a clinical trial. For some patients, ta...
5. `medquad_ans_0000203-2__dup1` (score=0.0426) - There is no cure for MSA. Currently, there are no treatments to delay the progress of neurodegeneration in the brain. But there are treatments available to help people cope with some of the more disabling symptoms of ...

**Answer**:

The treatments for Whipple disease include antibiotics to destroy T. whipplei bacteria. Initially, intravenous (IV) antibiotics are prescribed for the first 2 weeks of treatment and may include ceftriaxone (Rocephin), meropenem (Merrem I.V.), penicillin G (Pfizerpen), and streptomycin (Streptomycin). After completing IV antibiotics, patients receive long-term oral antibiotics for at least 1 to 2 years, which may include trimethoprim/sulfamethoxazole (Septra, Bactrim) or doxycycline (Vibramycin). Patients must finish the prescribed course to ensure all bacteria are eliminated. In cases with neurologic symptoms, treatment may be more aggressive and include a combination of antibiotics, hydroxychloroquine (Plaquenil), weekly injections of interferon gamma, and corticosteroids to reduce inflammation. Follow

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000042-2__dup2` (score=0.9856) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as...
2. `medquad_ans_0000676-4__dup1` (score=0.9747) - How might Bell's palsy be treated?
3. `medquad_ans_0000676-2__dup1` (score=0.0061) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0005695-6` (score=0.0003) - How might spastic diplegia cerebral palsy be treated? Treatment of spastic diplegia cerebral palsy varies based on the signs and symptoms present in each person and the severity of the condition. Affected people are o...
5. `medquad_ans_0000042-4__dup2` (score=0.0002) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge of the brain and nervous system and to use that knowledge to reduce the burden of neurological diseas...

**Answer**:

Steroids such as prednisone, which reduce inflammation and swelling, are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as aspirin, acetaminophen, or ibuprofen may relieve pain. Keeping the eye moist and protecting it from debris and injury, especially at night, is important, and lubricating eye drops can help. Other therapies such as physical therapy, facial massage, or acupuncture may provide a potential small improvement in facial nerve function and pain [medquad_ans_0000042-2__dup2].

**Supported citations**: `medquad_ans_0000042-2__dup2`
**Unsupported citations**: _none_
