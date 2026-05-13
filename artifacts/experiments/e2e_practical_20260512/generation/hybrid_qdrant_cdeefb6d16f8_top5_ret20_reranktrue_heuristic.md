# Generation Evaluation

## Summary

- `avg_answer_chars`: 596.2800
- `avg_answer_relevance`: 0.7508
- `avg_citation_presence_rate`: 0.8267
- `avg_context_relevance`: 0.8432
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 8910.1843
- `avg_generation_latency_ms`: 6265.6924
- `avg_groundedness`: 0.7609
- `avg_insufficient_context`: 0.0300
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 2330.6935
- `avg_retrieval_latency_ms`: 313.5577
- `avg_supported_citation_rate`: 0.7233
- `avg_unsupported_citation_rate`: 0.1033
- `num_examples`: 300
- `reranker_enabled_rate`: 1.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006510-1` (score=0.9994) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
2. `medquad_ans_0000001_1-2` (score=0.0024) - Signs and symptoms of adult ALL include fever, feeling tired, and easy bruising or bleeding. The early signs and symptoms of ALL may be like the flu or other common diseases. Check with your doctor if you have any of ...
3. `medquad_ans_0006509-2` (score=0.0004) - What are the signs and symptoms of X-linked lymphoproliferative syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome. If the information is...
4. `medquad_ans_0006511-1` (score=0.0001) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 2? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 2. If the informatio...
5. `medquad_ans_0002901-1` (score=0.0001) - What are the signs and symptoms of Hodgkin lymphoma? The Human Phenotype Ontology provides the following list of signs and symptoms for Hodgkin lymphoma. If the information is available, the table below includes how o...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency (90%), decreased antibody level in blood (50%), hepatomegaly (50%), lymphadenopathy (50%), lymphoma (50%), splenomegaly (50%), anemia (7.5%), encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=0.400, groundedness=0.562, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000585-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000095-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000362-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000673-4` (score=0.9989) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000786-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not specify whether early-onset primary dystonia is inherited. While several documents mention an autosomal dominant inheritance pattern for certain conditions, none explicitly state that early-onset primary dystonia follows this pattern or confirm its inheritance status. Therefore, based on the given information, it is not possible to determine if early-onset primary dystonia is inherited.

[medquad_ans_0000585-4] [medquad_ans_0000095-4] [medquad_ans_0000362-4] [medquad_ans_0000673-4] [medquad_ans_0000786-4]

**Supported citations**: `medquad_ans_0000585-4`, `medquad_ans_0000095-4`, `medquad_ans_0000362-4`, `medquad_ans_0000673-4`, `medquad_ans_0000786-4`
**Unsupported citations**: _none_

### Example 3 (`0000397-2__dup1`)

**Question**: Who is at risk for Parasites - Taeniasis? ?

**Scores**: context_relevance=0.714, groundedness=0.963, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000397-2__dup1` (score=0.9876) - The tapeworms that cause taeniasis (Taenia saginata, T. solium, and T. asiatica) are found worldwide. Eating raw or undercooked beef or pork is the primary risk factor for acquiring taeniasis. Persons who don't eat ra...
2. `medquad_ans_0000397-1__dup2` (score=0.0824) - Taeniasis in humans is a parasitic infection caused by the tapeworm species Taenia saginata (beef tapeworm), Taenia solium (pork tapeworm), and Taenia asiatica (Asian tapeworm). Humans can become infected with these t...
3. `medquad_ans_0000415-1__dup2` (score=0.0051) - A single-celled parasite called Toxoplasma gondii causes a disease known as toxoplasmosis. While the parasite is found throughout the world, more than 60 million people in the United States may be infected with the To...
4. `medquad_ans_0000121-2__dup3` (score=0.0040) - Cysticercosis is an infection caused by the larvae of the parasite Taenia solium. This infection occurs after a person swallows tapeworm eggs. The larvae get into tissues such as muscle and brain, and form cysts there...
5. `medquad_ans_0000121-3__dup2` (score=0.0032) - Cysticercosis is an infection caused by the larvae of the tapeworm, Taenia solium. A person with an adult tapeworm, which lives in the person’s gut, sheds eggs in the stool. The infection with the adult tapeworm is ca...

**Answer**:

People at risk for taeniasis include those who eat raw or undercooked beef or pork. Infections with Taenia saginata occur where contaminated raw beef is consumed, particularly in Eastern Europe, Russia, eastern Africa, and Latin America. Taenia solium infections are more common in underdeveloped communities with poor sanitation and where raw or undercooked pork is consumed, with higher rates seen in Latin America, Eastern Europe, sub-Saharan Africa, India, and Asia. Taenia solium taeniasis is also seen in the United States, typically among Latin American immigrants. Taenia asiatica is limited to Asia and is found in countries such as the Republic of Korea, China, Taiwan, Indonesia, and Thailand. Persons who do not eat raw or undercooked beef or pork are not likely to get taeniasis [medquad_ans_0000397-2__dup1].

**Supported citations**: `medquad_ans_0000397-2__dup1`
**Unsupported citations**: _none_

### Example 4 (`0000194-2__dup1`)

**Question**: What are the symptoms of Aicardi-Goutieres syndrome type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.955, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.9994) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0000197-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000196-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000195-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000193-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1]. These symptoms are based on the Human Phenotype Ontology, which compiles data on the frequency of signs and symptoms in this condition [medquad_ans_0000194-2__dup1].

**Supported citations**: `medquad_ans_0000194-2__dup1`
**Unsupported citations**: _none_

### Example 5 (`0003421-1`)

**Question**: What are the symptoms of Kowarski syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.945, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003421-1` (score=0.9993) - What are the signs and symptoms of Kowarski syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kowarski syndrome. If the information is available, the table below includes how...
2. `medquad_ans_0003418-1` (score=0.0002) - What are the signs and symptoms of Kosztolanyi syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kosztolanyi syndrome. If the information is available, the table below includ...
3. `medquad_ans_0003339-2` (score=0.0001) - What are the signs and symptoms of Kallmann syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kallmann syndrome. If the information is available, the table below includes how...
4. `medquad_ans_0003423-1` (score=0.0001) - What are the signs and symptoms of Kozlowski Celermajer Tink syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Kozlowski Celermajer Tink syndrome. If the information is avail...
5. `medquad_ans_0003414-2` (score=0.0001) - What are the signs and symptoms of Konigsmark Knox Hussels syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Konigsmark Knox Hussels syndrome. If the information is available...

**Answer**:

The signs and symptoms of Kowarski syndrome include abnormality of metabolism/homeostasis, autosomal recessive inheritance, delayed skeletal maturation, and pituitary dwarfism [medquad_ans_0003421-1].

**Supported citations**: `medquad_ans_0003421-1`
**Unsupported citations**: _none_

### Example 6 (`0000450-4`)

**Question**: Is hemophilia inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000801-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000300-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000990-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000405-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000683-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether hemophilia is inherited. While the context repeatedly describes autosomal recessive inheritance patterns, it does not mention hemophilia or link the condition to this pattern. Therefore, based on the given information, it is not possible to determine if hemophilia is inherited.

Answer: Insufficient context to determine if hemophilia is inherited. [medquad_ans_0000801-4, medquad_ans_0000300-4, medquad_ans_0000990-4, medquad_ans_0000405-4, medquad_ans_0000683-4]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000801-4, medquad_ans_0000300-4, medquad_ans_0000990-4, medquad_ans_0000405-4, medquad_ans_0000683-4`

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.9994) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0003103-2` (score=0.0002) - What are the signs and symptoms of IBIDS syndrome? The most common symptoms of Tay syndrome are brittle hair (trichothiodystrophy); dry, thickened, scaling skin (ichthyosis); photosensitivity (abnormal light sensitivi...
3. `medquad_ans_0006355-2` (score=0.0002) - What are the signs and symptoms of Vici syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Vici syndrome. If the information is available, the table below includes how often t...
4. `medquad_ans_0004152-1` (score=0.0001) - What are the signs and symptoms of Mucolipidosis III alpha/beta? The Human Phenotype Ontology provides the following list of signs and symptoms for Mucolipidosis III alpha/beta. If the information is available, the ta...
5. `medquad_ans_0004163-1` (score=0.0001) - Mucopolysaccharidosis type IVA (MPS IVA, also called Morquio syndrome, type A) is a metabolic condition that primarily affects the skeleton. The severity, age of onset, and associated symptoms vary significantly from ...

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
5. `medquad_ans_0000039-1` (score=0.0001) - Alpha thalassemia is a blood disorder that reduces the production of hemoglobin. Hemoglobin is the protein in red blood cells that carries oxygen to cells throughout the body. In people with the characteristic feature...

**Answer**:

Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. The disorder is characterized by sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to a deficiency of red blood cells (hemolytic anemia) and resulting in signs and symptoms such as fatigue, weakness, abnormally pale skin (pallor), shortness of breath, and an increased heart rate. Affected individuals may pass dark-colored urine due to the presence of hemoglobin, a

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 9 (`0000010-10`)

**Question**: What is (are) Colorectal Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.916, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000010-10` (score=0.6004) - Cancer of the colon or rectum is called colorectal cancer. The colon and the rectum are part of the large intestine, which is part of the digestive system. Colorectal cancer occurs when malignant tumors form in the li...
2. `medquad_ans_0000221-1__dup2` (score=0.4409) - The colon and rectum are part of the large intestine. Colorectal cancer occurs when tumors form in the lining of the large intestine. It is common in both men and women. The risk of developing colorectal cancer rises ...
3. `medquad_ans_0000037_1-1` (score=0.0653) - Key Points - Colon cancer is a disease in which malignant (cancer) cells form in the tissues of the colon. - Health history affects the risk of developing colon cancer. - Signs of colon cancer include blood in the sto...
4. `medquad_ans_0000037_2-1` (score=0.0423) - Key Points - Rectal cancer is a disease in which malignant (cancer) cells form in the tissues of the rectum. - Health history affects the risk of developing rectal cancer. - Signs of rectal cancer include a change in ...
5. `medquad_ans_0000010-23` (score=0.0277) - Several types of surgery are available for someone with colorectal cancer. If the cancer is found at a very early stage, the doctor may remove it without cutting through the abdominal wall. Instead, the doctor may put...

**Answer**:

Colorectal cancer is a disease in which malignant (cancer) cells form in the tissues of the colon or rectum, which are part of the large intestine. It occurs when tumors form in the lining of the large intestine, also known as the large bowel [medquad_ans_0000010-10]. The condition is common in both men and women and the risk increases after age 50 [medquad_ans_0000221-1__dup2]. Factors that increase the risk include a family history of colorectal cancer, having colorectal polyps, ulcerative colitis or Crohn's disease, eating a diet high in fat, or smoking [medquad_ans_0000221-1__dup2]. Symptoms may include diarrhea or constipation, a feeling that the bowel does not empty completely, blood in the stool, narrow stools, gas pains or cr

**Supported citations**: `medquad_ans_0000010-10`, `medquad_ans_0000221-1__dup2`
**Unsupported citations**: _none_

### Example 10 (`0000544-1`)

**Question**: What is (are) junctional epidermolysis bullosa ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003317-1` (score=0.9712) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...
2. `medquad_ans_0002121-1` (score=0.0042) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
3. `medquad_ans_0002122-1` (score=0.0017) - Epidermolysis bullosa acquisita (EBA) is a rare autoimmune disorder that causes the skin to blister in response to minor injury. Common areas of blistering include the hands, feet, knees, elbows, and buttocks. It can ...
4. `medquad_ans_0000544-2` (score=0.0003) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
5. `medquad_ans_0002002-1` (score=0.0001) - Dystrophic epidermolysis bullosa (DEB) is one of the major forms of epidermolysis bullosa. The signs and symptoms can vary widely among affected people. In mild cases, blistering may primarily affect the hands, feet, ...

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe, and individuals with this condition often do not survive infancy. The Non-Herlitz type includes several subtypes that cause mild to severe blistering of the skin present at birth or shortly thereafter. JEB is inherited in an autosomal recessive pattern and is caused by mutations in the LAMB3, COL17A1, or LAMC2, and LAMA3 genes. There is no cure for JEB; treatment is focused on management of blistering and prevention of secondary infections. Both types of JEB are rare, affecting fewer than 1 per million people

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.943, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000668-3` (score=0.9991) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
2. `medquad_ans_0000651-3` (score=0.0016) - Methylmalonic acidemia with homocystinuria can be caused by mutations in one of several genes: MMACHC, MMADHC, LMBRD1, ABCD4, or HCFC1. Mutations in these genes account for the different types of the disorder, which a...
3. `medquad_ans_0000671-4` (score=0.0003) - Up to 15 percent of Japanese people with moyamoya disease have one or more family members with the condition, indicating that the condition can be passed through generations in families; however, the inheritance patte...
4. `medquad_ans_0000486-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. Most often, the parents of an individual with an autosomal recessive condition each carry...
5. `medquad_ans_0000668-2` (score=0.0001) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought tha...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. Mutations in the MOCS1 gene cause type A, mutations in the MOCS2 gene cause type B, and mutations in the GPHN gene cause type C [medquad_ans_0000668-3]. These mutations reduce or eliminate the function of the associated protein, impairing molybdenum cofactor biosynthesis [medquad_ans_0000668-3].

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
5. `medquad_ans_0003472-1` (score=0.0013) - L-arginine:glycine amidinotransferase (AGAT) deficiency is a rare condition that primarily affects the brain. People with AGAT deficiency generally have mild to moderate intellectual disability. Other signs and sympto...

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is especially sensitive to the effects of excess ammonia. Ornithine translocase deficiency varies widely in its severity and age of onset. An infant with ornithine translocase deficiency may be lacking in energy (lethargic) or refuse to eat, or have poorly controlled breathing or body temperature. Some babies with this disorder may experience seizures or unusual body movements, or go into a coma. Episodes of illness may coincide with the introduction of high-protein formulas or solid foods into the diet. In most affected individuals, signs and symptoms of ornithine translocase deficiency do not appear until later in life. Later-onset forms of ornithine translocase deficiency are usually less severe than the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.857, groundedness=0.890, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.9970) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000362-1__dup2` (score=0.9954) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
3. `medquad_ans_0000062-1__dup4` (score=0.9898) - Summary : Assisted living is for adults who need help with everyday tasks. They may need help with dressing, bathing, eating, or using the bathroom, but they don't need full-time nursing care. Some assisted living fac...
4. `medquad_ans_0000655-1__dup1` (score=0.0108) - Summary : Nutritional support is therapy for people who cannot get enough nourishment by eating or drinking. You may need it if you - Can't swallow - Have problems with your appetite - Are severely malnourished - Can'...
5. `medquad_ans_0000153-1__dup2` (score=0.0080) - Summary : Caregivers provide help to another person in need. The person receiving care may be an adult - often a parent or a spouse - or a child with special medical needs. Some caregivers are family members. Others a...

**Answer**:

Yes, there is information about financial assistance. Financial assistance may be available if you don't have health insurance or need help with costs that aren't covered. Certain government programs and nonprofit organizations can provide such assistance. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Additionally, Medicaid is a government health insurance program that helps low-income individuals pay their medical bills, and each state may have specific rules regarding cost-sharing and eligibility based on income, age, disability, pregnancy, and immigration status [medquad_ans_0000586-1__dup2].

**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000586-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=0.800, groundedness=0.987, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0004105-1` (score=0.8755) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
2. `medquad_ans_0003916-1` (score=0.0028) - Meningoencephalocele is a type of encephalocele, which is an abnormal sac of fluid, brain tissue, and meninges (membranes that cover the brain and spinal cord) that extends through a defect in the skull. There are two...
3. `medquad_ans_0000653-2` (score=0.0024) - MOPDII appears to be a rare condition, although its prevalence is unknown.
4. `medquad_ans_0005611-1` (score=0.0002) - Sialadenitis is an infection of the salivary glands. It is usually caused by a virus or bacteria. The parotid (in front of the ear) and submandibular (under the chin) glands are most commonly affected. Sialadenitis ma...
5. `medquad_ans_0000598-1__dup2` (score=0.0001) - Meningitis is inflammation of the thin tissue that surrounds the brain and spinal cord, called the meninges. There are several types of meningitis. The most common is viral meningitis, which you get when a virus enter...

**Answer**:

Mollaret meningitis is a rare type of meningitis characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches, separated by weeks or months of no symptoms. About half of affected individuals may also experience long-term abnormalities of the nervous system that come and go, such as seizures, double vision, abnormal reflexes, paralysis of a cranial nerve (paresis), hallucinations, or coma. The exact cause remains unknown, although recent data suggests that herpes simplex virus (HSV-2 and, less frequently, HSV-1) may cause some, if not most cases. Other potential causes include trauma and viral infections other than herpes simplex [medquad_ans_0004105-1].

**Supported citations**: `medquad_ans_0004105-1`
**Unsupported citations**: _none_

### Example 15 (`0003319-3`)

**Question**: What causes Juvenile amyotrophic lateral sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.825, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003331-3` (score=0.9829) - What causes juvenile primary lateral sclerosis? Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene. The ALS2 gene provides instructions for making a protein called alsin. Alsin is abundant in m...
2. `medquad_ans_0000309-3__dup1` (score=0.4638) - What causes amyotrophic lateral sclerosis? In approximately 90-95% of cases the cause of amyotrophic lateral sclerosis (ALS) is unknown and is sporadic (occurring in individuals with no history of the condition in the...
3. `medquad_ans_0000551-5` (score=0.0001) - These resources address the diagnosis or management of juvenile primary lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Genetic Testing Registry: Juvenile primary lateral sclerosis These resour...
4. `medquad_ans_0000551-3` (score=0.0001) - Mutations in the ALS2 gene cause most cases of juvenile primary lateral sclerosis. This gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but its function is not fully u...
5. `medquad_ans_0003319-4` (score=0.0001) - How might juvenile amyotrophic lateral sclerosis be treated? Treatments and therapies are available to relieve symptoms and improve the quality of life of people with juvenile ALS. Medications, such as those that redu...

**Answer**:

The provided context does not contain information about the causes of Juvenile amyotrophic lateral sclerosis. While it discusses juvenile primary lateral sclerosis and amyotrophic lateral sclerosis (ALS), it does not specify the causes of juvenile ALS. Therefore, based on the available context, the cause of Juvenile amyotrophic lateral sclerosis cannot be determined.

[medquad_ans_0000309-3__dup1] mentions that in approximately 90-95% of ALS cases, the cause is unknown (sporadic), and in 5-10% of cases, it is genetic, with mutations in genes such as C9orf72, SOD1, TARDBP, FUS, ANG, ALS2, SETX, and VAPB. However, this information pertains to ALS in general, not specifically juvenile ALS. There is no explicit information linking genetic mutations or other factors to juvenile amyot

**Supported citations**: `medquad_ans_0000309-3__dup1`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.667, groundedness=0.927, answer_relevance=0.578

**Top docs**:

1. `medquad_ans_0000219-11` (score=0.9849) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
2. `medquad_ans_0000219-1__dup4` (score=0.9725) - Erectile dysfunction\* is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long ...
3. `medquad_ans_0000219-5__dup2` (score=0.5170) - Having ED can cause you to feel depressed or anxious. ED may also cause low self-esteem. When you have ED, you may not have a satisfying sex life. You may not feel as close with your sexual partner, which may strain y...
4. `medquad_ans_0000809-1__dup2` (score=0.0055) - Summary : Many men have sexual problems. They become more common as men age. Problems can include - Erectile dysfunction - Reduced or lost interest in sex - Problems with ejaculation - Low testosterone Stress, illness...
5. `medquad_ans_0000067-31` (score=0.0017) - For many older adults, intimacy remains an important need. Cancer and cancer treatment can have a major impact on intimacy and sexual functions for both men and women. Problems are often caused by physical changes, su...

**Answer**:

To learn more about erectile dysfunction (ED), you should understand that it is when you cannot get or keep an erection firm enough to have sex. You may have ED if you can get an erection sometimes but not every time, can get an erection that does not last long enough for sex, or are unable to get an erection at all [medquad_ans_0000219-11]. An erection occurs due to increased blood flow into the penis, involving hormones, blood vessels, nerves, and muscles working together [medquad_ans_0000219-11]. ED often has multiple causes, and many diseases can damage the nerves, arteries, and muscles involved [medquad_ans_0000219-11].

To determine the cause of ED, your doctor may take your medical and sexual history, ask about your mental health, perform a physical exam, test your blood, conduct a

**Supported citations**: `medquad_ans_0000219-11`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.916, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000084-2__dup1` (score=0.9873) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
2. `medquad_ans_0006267-3` (score=0.9702) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
3. `medquad_ans_0000202-5` (score=0.0019) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Regist...
4. `medquad_ans_0000334-5` (score=0.0002) - These resources address the diagnosis or management of facioscapulohumeral muscular dystrophy: - Gene Review: Gene Review: Facioscapulohumeral Muscular Dystrophy - Genetic Testing Registry: Facioscapulohumeral muscula...
5. `medquad_ans_0000576-5` (score=0.0002) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...

**Answer**:

The main focus of treatment for Ullrich congenital muscular dystrophy is supportive care, including physical therapy such as early mobilization, regular stretching, and splinting. Respiratory support and nighttime ventilation may become necessary in the first or second decade of life. Prevention of chest infections can be achieved with antibiotics. Feeding difficulties leading to failure to thrive may be managed with gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that cyclosporin A may benefit individuals with Ullrich congenital muscular dystrophy, although more studies are needed to confirm its effectiveness [medquad_ans_0006267-3].

**Supported citations**: `medquad_ans_0006267-3`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=1.000, groundedness=0.733, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0004005-4` (score=0.0001) - How is microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) inherited? MOPD1 is thought to be inherited in an autosomal recessive manner. This means that affected individuals have abnormal gene changes (mu...
2. `medquad_ans_0000653-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcep...
3. `medquad_ans_0004006-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
4. `medquad_ans_0004550-1` (score=0.0001) - Opsismodysplasia is a rare skeletal dysplasia characterized by congenital short stature and characteristic craniofacial abnormalities. Clinical signs observed at birth include short limbs, small hands and feet, relati...
5. `medquad_ans_0004005-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type 1 (MOPD1) is a genetic condition that is mainly characterized by intrauterine and post-natal growth retardation; an abnormally small head size (microcephaly); abn...

**Answer**:

The provided context does not specify how microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is inherited. [medquad_ans_0000653-1] describes the clinical features of MOPDII but does not mention inheritance patterns. Therefore, based on the given information, the inheritance of MOPDII cannot be determined.

**Supported citations**: `medquad_ans_0000653-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000556-4` (score=0.9985) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
2. `medquad_ans_0000283-4` (score=0.9984) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...
3. `medquad_ans_0000883-4` (score=0.9977) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...
4. `medquad_ans_0000324-4` (score=0.9969) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
5. `medquad_ans_0006597-4` (score=0.1926) - Can 48,XXYY syndrome be inherited?

**Answer**:

The provided context does not mention Ochoa syndrome or its inheritance pattern. Therefore, it is not possible to determine based on the given information whether Ochoa syndrome is inherited.

Answer: Insufficient context [medquad_ans_0000556-4, medquad_ans_0000283-4, medquad_ans_0000883-4, medquad_ans_0000324-4, medquad_ans_0006597-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000556-4, medquad_ans_0000283-4, medquad_ans_0000883-4, medquad_ans_0000324-4, medquad_ans_0006597-4`

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.806, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.9960) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
2. `medquad_ans_0001276-6` (score=0.1448) - How might chromosome 3p- syndrome be treated? Because chromosome 3p- syndrome affects many different systems of the body, medical management is often provided by a team of doctors and other healthcare professionals. T...
3. `medquad_ans_0004132-4` (score=0.0114) - How might mosaic trisomy 14 be treated? Treatment for signs and symptoms of mosaic trisomy 14 focuses on the specific features present in each individual. Infants with congenital heart defects may need surgery or othe...
4. `medquad_ans_0001282-6` (score=0.0071) - How might chromosome 4q deletion be treated? Because chromosome 4q deletion affects many different systems of the body, medical management is often provided by a team of doctors and other healthcare professionals. Tre...
5. `medquad_ans_0004137-6` (score=0.0028) - How might mosaic trisomy 9 be treated? Because mosaic trisomy 9 affects many different systems of the body, medical management is often provided by a team of doctors and other healthcare professionals. Treatment for t...

**Answer**:

The provided context does not contain specific information about the treatments for isodicentric chromosome 15 syndrome. While it mentions resources related to the diagnosis and management of various health conditions, including genetic counseling, palliative care, diagnostic tests, drug therapy, surgery and rehabilitation, and other services, no direct details about treatment for isodicentric chromosome 15 syndrome are given [medquad_ans_0000531-5]. Therefore, based on the available context, the treatments for isodicentric chromosome 15 syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.872, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=0.9931) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0000115-2__dup2` (score=0.0118) - Scientists are unsure about what causes Mntriers disease; however, researchers think that most people acquire, rather than inherit, the disease. In extremely rare cases, siblings have developed Mntriers disease as chi...
3. `medquad_ans_0000115-1__dup4` (score=0.0051) - Mntriers disease causes the ridges along the inside of the stomach wallcalled rugaeto enlarge, forming giant folds in the stomach lining. The rugae enlarge because of an overgrowth of mucous cells in the stomach wall....
4. `medquad_ans_0004414-3` (score=0.0003) - What causes Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is caused by mutations in the SMARCA2 gene, which is located on the small arm of chromosome 9. All mutations that have been identified i...
5. `medquad_ans_0000272-4__dup1` (score=0.0002) - The NINDS supports research on genetic disorders such as the hereditary spastic paraplegias. A gene for Troyer syndrome has been identified and others may be identified in the future. Understanding how these genes cau...

**Answer**:

Troyer syndrome is caused by mutations in the SPG20 gene, which lead to a loss of function of the spartin protein. This protein is involved in cellular functions, though its exact role is not fully understood. [medquad_ans_0006224-3]

**Supported citations**: `medquad_ans_0006224-3`
**Unsupported citations**: _none_

### Example 22 (`0000983-3`)

**Question**: What are the genetic changes related to Timothy syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.910, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000983-3` (score=0.9990) - Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. This gene provides instructions for making a protein that acts as a channel across cell membranes. This channel, known as CaV1....
2. `medquad_ans_0000560-3` (score=0.0007) - Kleefstra syndrome is caused by the loss of the EHMT1 gene or by mutations that disable its function. The EHMT1 gene provides instructions for making an enzyme called euchromatic histone methyltransferase 1. Histone m...
3. `medquad_ans_0000659-3` (score=0.0004) - Miller-Dieker syndrome is caused by a deletion of genetic material near the end of the short (p) arm of chromosome 17. The signs and symptoms of Miller-Dieker syndrome are probably related to the loss of multiple gene...
4. `medquad_ans_0000541-3` (score=0.0001) - Jacobsen syndrome is caused by a deletion of genetic material at the end of the long (q) arm of chromosome 11. The size of the deletion varies among affected individuals, with most affected people missing 5 million to...
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
4. `medquad_ans_0000061-4__dup3` (score=0.0007) - Most Symptoms Are Treatable Doctors use a variety of approaches to treat rheumatoid arthritis. The goals of treatment are to help relieve pain, reduce swelling, slow down or help prevent joint damage, increase the abi...
5. `medquad_ans_0000057-1__dup3` (score=0.0007) - If you feel pain and stiffness in your body or have trouble moving around, you might have arthritis. Most kinds of arthritis cause pain and swelling in your joints. Joints are places where two bones meet, such as your...

**Answer**:

The symptoms of rheumatoid arthritis include pain and stiffness in the joints, warmth, redness, swelling, and pain around the joints. People also often feel sick, tired, and may experience fever. Rheumatoid arthritis generally occurs in a symmetrical pattern, meaning if one knee or hand is affected, the other is also likely to be affected [medquad_ans_0000061-3__dup3].

**Supported citations**: `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.9718) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.9287) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0005502-1` (score=0.2490) - Schnitzler syndrome is a rare autoinflammatory condition. Signs and symptoms of the condition vary but may include urticaria; recurrent fevers; joint pain and inflammation; organomegaly (abnormally enlarged organs); a...
4. `medquad_ans_0003148-1` (score=0.0017) - Idiopathic thrombocytopenic purpura (ITP) is a bleeding disorder characterized by too few platelets in the blood. This is because platelets are being destroyed by the immune system. Symptoms may include bruising, nose...
5. `medquad_ans_0000837-1` (score=0.0011) - Protein C deficiency is a disorder that increases the risk of developing abnormal blood clots; the condition can be mild or severe. Individuals with mild protein C deficiency are at risk of a type of blood clot known ...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It may develop as an immune response to an infection, and while the exact cause is not fully understood, it is usually seen in children but can affect people of any age. Most cases go away on their own without treatment, and the main goal of treatment, when needed, is to relieve symptoms such as joint pain, abdominal pain, or swelling. Over-the-counter medicines can be used in many cases, and in some patients with severe arthritis, prednisone, a steroid medicine, may be prescribed. HSP can also cause inflammation of small blood vessels, leading to a rash that appears as many small raised bruises. In adults, HSP can lead to chronic kidney disease (CKD) and kidney failure, described

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.9994) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0003162-2` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition ...
3. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
4. `medquad_ans_0003163-1` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...
5. `medquad_ans_0000579-2__dup1` (score=0.0001) - What are the signs and symptoms of Autosomal recessive hyper IgE syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Autosomal recessive hyper IgE syndrome. If the information ...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.925, answer_relevance=0.844

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.9991) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000786-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
3. `medquad_ans_0000149-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000516-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000575-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. These mutations lead to defects in the structure and function of cilia, which are microscopic projections on cells involved in signaling pathways. The proteins produced from these genes are known or suspected to play roles in cilia function, and defects in cilia are believed to disrupt important chemical signaling pathways within cells. However, it remains unclear how these defects specifically lead to nephronophthisis and Leber congenital amaurosis. In some individuals, no mutation is identified in these five genes, and the genetic cause remains unknown [medquad_ans_0000896-3]. The condition is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder, with most cases resulting from inheritance from an affected parent and some due to new mutations [medquad_ans_000078

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=0.250, groundedness=0.817, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
4. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
5. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

The exact incidence of lymphedema-distichiasis syndrome is unknown, but it is estimated to affect 1 in 87,000 people [medquad_ans_0000484-2], 1 in 40,000 to 50,000 people [medquad_ans_0000195-2], 1 in 50,000 people [medquad_ans_0000989-2], or 1 to 3 per million people worldwide [medquad_ans_0000770-2]. The prevalence is also noted as unknown, with only a few affected individuals described in the medical literature [medquad_ans_0000955-2].

**Supported citations**: `medquad_ans_0000484-2`, `medquad_ans_0000195-2`, `medquad_ans_0000989-2`, `medquad_ans_0000770-2`, `medquad_ans_0000955-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=0.857, groundedness=0.840, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000399-1__dup1` (score=0.9977) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...
2. `medquad_ans_0000397-1__dup1` (score=0.1860) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourse...
3. `medquad_ans_0000013-25` (score=0.0037) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for ...
4. `medquad_ans_0000013-22` (score=0.0006) - In a genetic test, a small sample of blood, saliva, or tissue is taken to examine a person's genes. Sometimes, genetic testing can detect diseases that may be preventable or treatable. This type of testing is availabl...
5. `medquad_ans_0002853-4` (score=0.0003) - Is genetic testing available for hereditary sensory neuropathy type 1? At least four genes responsible for hereditary sensory neuropathy type 1 (HSN1) have been found: HSN1A (the most common form) is associated with m...

**Answer**:

Yes, I have information about genetic testing. Genetic tests are conducted on blood and other tissues to identify genetic disorders. Over 2000 such tests are available, and doctors use them for various purposes, including detecting genetic diseases in unborn babies, determining if individuals carry a gene that could be passed to their children, screening embryos for disease, testing adults before symptoms appear, making a diagnosis in someone with symptoms, and determining the most suitable type or dose of medication for a person [medquad_ans_0000399-1__dup1]. Test results can provide relief by reducing uncertainty about a health condition, assist doctors in making treatment or monitoring recommendations, support individuals in making health and family-related decisions, encourage preventive actions such as earlier screenings or lifestyle changes, and help people make informed choices about future decisions like having a child [medquad_ans_0000013-25]. Genetic counseling is often provided to

**Supported citations**: `medquad_ans_0000399-1__dup1`, `medquad_ans_0000013-25`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.544, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.8681) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0000030-1__dup2` (score=0.0268) - Arteriovenous malformations (AVMs) are abnormal, snarled tangles of blood vessels that cause multiple irregular connections between the arteries and veins. These malformations most often occur in the spinal cord and i...
3. `medquad_ans_0004739-1` (score=0.0244) - Parkes Weber syndrome (PWS) is a rare congenital condition causing an individual to have a large number of abnormal blood vessels. The main characteristics of PWS typically include a capillary malformation on the skin...
4. `medquad_ans_0000047-1` (score=0.0072) - Alveolar capillary dysplasia with misalignment of pulmonary veins (ACD/MPV) is a disorder affecting the development of the lungs and their blood vessels. The disorder affects the millions of small air sacs (alveoli) i...
5. `medquad_ans_0005994-1` (score=0.0049) - Systemic capillary leak syndrome is a condition in which fluid and proteins leak out of tiny blood vessels and flow into surrounding tissues, resulting in dangerously low blood pressure. Attacks frequently last for se...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin and appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs. These spots may be visible at birth or develop during childhood. In most cases, capillary malformations do not cause health problems and may be the only sign of the disorder. However, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries that can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure. These complications may appear in infancy or early

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.9994) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
2. `medquad_ans_0001994-1` (score=0.0010) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the s...
3. `medquad_ans_0001995-1` (score=0.0003) - What are the signs and symptoms of Dystonia 19? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 19. If the information is available, the table below includes how often the s...
4. `medquad_ans_0001903-2` (score=0.0003) - What are the signs and symptoms of Dopa-responsive dystonia? The most common form of dopa-responsive dystonia (DRD) is autosomal dominant DRD (caused by a mutation in the GCH1 gene). This form of DRD is usually charac...
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
3. `medquad_ans_0004291-4` (score=0.0003) - How might myotonic dystrophy type 1 associated vision problems be treated? Treatment of eye and vision problems must be individually tailored. Refractive error and astigmatism can be corrected with eyeglasses, contact...
4. `medquad_ans_0001461-3` (score=0.0002) - How might cone-rod dystrophy be treated? Currently, there is no therapy that stops the evolution of cone-rod dystrophy or restores vision. There are a few treatment options, such as light avoidance and the use of low-...
5. `medquad_ans_0000899-2` (score=0.0001) - Septo-optic dysplasia has a reported incidence of 1 in 10,000 newborns.

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

**Scores**: context_relevance=0.556, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000029-18` (score=0.9948) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
2. `medquad_ans_0000052-2__dup1` (score=0.9910) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
3. `medquad_ans_0004784-5` (score=0.1152) - How might Pelizaeus-Merzbacher disease be treated?
4. `medquad_ans_0004770-2` (score=0.0459) - How might patulous eustacian tube be treated? While no standard treatment has been found to work for every patient, there are several options that have been used to successfully manage the symptoms in a number of case...
5. `medquad_ans_0006236-5` (score=0.0034) - How might tubular aggregate myopathy be treated?

**Answer**:

The provided context does not contain information about the treatments for Absence of the Septum Pellucidum. Therefore, based on the given documents, no treatments can be identified. [No relevant context]

**Supported citations**: _none_
**Unsupported citations**: `No relevant context`

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.523, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.9992) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0000128-3__dup1` (score=0.0031) - The vast majority of febrile seizures are short and harmless. There is no evidence that short febrile seizures cause brain damage. Multiple or prolonged seizures are a risk factor for epilepsy but most children who ex...
3. `medquad_ans_0003540-2` (score=0.0001) - What are the symptoms reported in children who have Lemierre syndrome? In children and adolescents, Lemierre syndrome usually begins with a severe sore throat, persistent fever, and possibly chills. Some cases begin w...
4. `medquad_ans_0002322-3` (score=0.0001) - What causes febrile ulceronecrotic Mucha-Habermann disease? The cause of FUMHD is not known (idiopathic). A hypersensitivity to an infectious agent is suggested to be the main cause. Single cases of people with FUMHD ...
5. `medquad_ans_0002152-2` (score=0.0001) - What are the signs and symptoms of Erdheim-Chester disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Erdheim-Chester disease. If the information is available, the table below ...

**Answer**:

The signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease (FUMHD) include red scaly skin lesions (papules) that ulcerate, break down, form open sores, and then develop a red-brown crust (i.e., PLEVA). The lesions can progress to large, destructive ulcers with extensive, painful loss of skin tissue. Skin lesions may become infected, leading to pus and a putrid odor. FUMHD is often associated with high fever (up to 104°F), which may be persistent or intermittent. Other symptoms may include feeling ill, sore throat, congestion, muscle soreness or pain, joint pain, diarrhea, central nervous system symptoms, abdominal pain, enlarged spleen, arthritis, megaloblastic anemia, interstitial pneumonitis, lymphocytic (viral) myocarditis, and sepsis. FUMHD can become

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 35 (`0000465-4`)

**Question**: Is hereditary leiomyomatosis and renal cell cancer inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000801-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000891-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000784-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000998-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001090-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether hereditary leiomyomatosis and renal cell cancer is inherited. While the context repeatedly describes autosomal recessive inheritance patterns, it does not mention hereditary leiomyomatosis and renal cell cancer specifically. Therefore, based on the given information, the inheritance pattern of this condition cannot be determined.

Answer: Insufficient context to determine if hereditary leiomyomatosis and renal cell cancer is inherited. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 36 (`0002445-1`)

**Question**: What are the symptoms of Frontometaphyseal dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002445-1` (score=0.9994) - What are the signs and symptoms of Frontometaphyseal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontometaphyseal dysplasia. If the information is available, the tabl...
2. `medquad_ans_0002444-1` (score=0.0003) - What are the signs and symptoms of Frontofacionasal dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Frontofacionasal dysplasia. If the information is available, the table ...
3. `medquad_ans_0000389-1` (score=0.0001) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
4. `medquad_ans_0005849-1` (score=0.0001) - What are the signs and symptoms of Spondylometaphyseal dysplasia corner fracture type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylometaphyseal dysplasia corner fracture t...
5. `medquad_ans_0002374-2` (score=0.0001) - What are the symptoms of fibrous dysplasia? Fibrous dysplasia may cause no symptoms, mild symptoms, or severe symptoms. The most common symptoms are bone pain, bone deformities, fractures, and skin pigmentation differ...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=1.000, groundedness=0.882, answer_relevance=0.564

**Top docs**:

1. `medquad_ans_0000128-7` (score=0.9832) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
2. `medquad_ans_0000128-9` (score=0.9815) - A doctor may prescribe antibiotics and recommend following a liquid diet. Most people get better with this treatment. Some people may need surgery and other treatments. - Surgery. Serious problems from diverticulitis ...
3. `medquad_ans_0000097-8` (score=0.9577) - A health care provider may treat the symptoms of diverticulosis with a high-fiber diet or fiber supplements, medications, and possibly probiotics. Treatment for diverticular disease varies, depending on whether a pers...
4. `medquad_ans_0000193-6` (score=0.9410) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
5. `medquad_ans_0000128-11` (score=0.9136) - - Diverticular disease is more common in people as they grow older. - A low-fiber diet is the most likely cause of the disease. - Most people are treated with a high-fiber diet and pain medication. - Add whole grain f...

**Answer**:

Treatment for diverticular disease depends on whether the condition is diverticulosis or diverticulitis and the severity of symptoms. For diverticulosis, a high-fiber diet is recommended to help prevent further issues; this may be accompanied by fiber supplements such as methylcellulose or psyllium, taken with at least 8 ounces of water, and possibly probiotics [medquad_ans_0000097-8]. A slow increase in dietary fiber can help minimize gas and abdominal discomfort [medquad_ans_0000097-8].

For diverticulitis, treatment may include antibiotics and a liquid diet, with most people recovering with this approach [medquad_ans_0000128-9]. In more serious cases, surgery may be required. This can involve cleaning the abdomen, removing bleeding pouches or fistulas, or performing a colon resection if diverticul

**Supported citations**: `medquad_ans_0000097-8`, `medquad_ans_0000128-9`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.9995) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0003103-2` (score=0.0102) - What are the signs and symptoms of IBIDS syndrome? The most common symptoms of Tay syndrome are brittle hair (trichothiodystrophy); dry, thickened, scaling skin (ichthyosis); photosensitivity (abnormal light sensitivi...
3. `medquad_ans_0004414-2` (score=0.0002) - What are the signs and symptoms of Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is typically characterized by intellectual disability, seizures, short stature, sparse hair, distinctive facial f...
4. `medquad_ans_0002441-2` (score=0.0001) - What are the signs and symptoms of Froelich syndrome? Signs and symptoms of Froelich syndrome include obesity, small testes, delay in the onset of puberty, short stature (compared to other family members of the same s...
5. `medquad_ans_0002652-2` (score=0.0001) - What are the signs and symptoms of Gordon syndrome? Gordon syndrome belongs to a group of conditions known as the distal arthrogryposes, which are characterized by stiffness and impaired mobility of certain joints of ...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomeg

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=0.833, groundedness=0.977, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9958) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000029-18` (score=0.9942) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
3. `medquad_ans_0000001-8__dup1` (score=0.9895) - No single treatment is effective for all patients. Treatment should be individualized, and often combined, depending on patient characteristics such as age and tumor size. If the tumor has not yet invaded surrounding ...
4. `medquad_ans_0000205-2__dup1` (score=0.9863) - There is no specific treatment to stop or reverse any form of MD. Treatment may include physical therapy, respiratory therapy, speech therapy, orthopedic appliances used for support, and corrective orthopedic surgery....
5. `medquad_ans_0003911-2` (score=0.9540) - How might meningiomas be treated? The treatment varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a men...

**Answer**:

The treatment for meningioma varies depending on the location of the meningioma and the symptoms caused by the tumor. Careful observation is sometimes the best course of action for people with a meningioma. When treatment is necessary, surgery and radiation are the most common forms of treatment. Radiation may be used if the meningioma cannot be operated on or if the meningioma is only partially removed by surgery [medquad_ans_0003911-2]. The three most commonly used treatments are surgery, radiation, and chemotherapy [medquad_ans_0000052-2__dup1].

**Supported citations**: `medquad_ans_0003911-2`, `medquad_ans_0000052-2__dup1`
**Unsupported citations**: _none_

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.960, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-8` (score=0.9738) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
2. `medquad_ans_0000006_2-8` (score=0.6526) - Key Points - There are different types of treatment for children with brain and spinal cord tumors. - Children with brain or spinal cord tumors should have their treatment planned by a team of health care providers wh...
3. `medquad_ans_0000006_6-8` (score=0.0341) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health...
4. `medquad_ans_0000006_4-8` (score=0.0284) - Key Points - There are different types of treatment for patients with central nervous system atypical teratoid/rhabdoid tumor. - Children with atypical teratoid/rhabdoid tumor should have their treatment planned by a ...
5. `medquad_ans_0000006_4-6` (score=0.0189) - Key Points - There is no standard staging system for central nervous system atypical teratoid/rhabdoid tumor. There is no standard staging system for central nervous system atypical teratoid/rhabdoid tumor. The extent...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and participation in clinical trials for new treatments. Standard treatments are currently used, and some treatments are being tested in clinical trials. Treatment planning should be done by a team of health care providers who are experts in treating cancer in children, including a pediatric oncologist and/or a radiation oncologist, along with other specialists such as a pediatric neurosurgeon, neurologist, endocrinologist, ophthalmologist, pediatric nurse specialist, rehabilitation specialist, psychologist, and social worker [medquad_ans_0000006_7-8].

**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=0.889, groundedness=0.986, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0004401-1` (score=0.9127) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
2. `medquad_ans_0000729-1` (score=0.8874) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0000065-1__dup1` (score=0.0113) - Farbers disease, also known as Farber's lipogranulomatosis, describes a group of inherited metabolic disorders called lipid storage diseases, in which excess amounts of lipids (oils, fatty acids, and related compounds...
4. `medquad_ans_0000002-1__dup1` (score=0.0052) - Acid lipase disease or deficiency occurs when the enzyme needed to break down certain fats that are normally digested by the body is lacking or missing, resulting in the toxic buildup of these fats in the bodys cells ...
5. `medquad_ans_0000002-4__dup1` (score=0.0017) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge abut the brain and nervous system, and to use that knowledge to reduce the burden of neurological dis...

**Answer**:

Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (myopathy). This condition is caused by mutations in the PNPLA2 gene and is inherited in an autosomal recessive pattern. There is currently no treatment to correct the underlying metabolic problem. Other features may include a fatty liver, a weakened and enlarged heart (cardiomyopathy), inflammation of the pancreas (pancreatitis), reduced thyroid activity (hypothyroidism), and type 2 diabetes mellitus. Signs and symptoms vary greatly among affected individuals [medquad_ans_0004401-1][medquad_ans_0000729-1].

**Supported citations**: `medquad_ans_0004401-1`, `medquad_ans_0000729-1`
**Unsupported citations**: _none_

### Example 42 (`0000981-1__dup1`)

**Question**: What is (are) Zika Virus ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000981-1__dup1` (score=0.4161) - Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been repo...
2. `medquad_ans_0006546-1` (score=0.0613) - Zika virus infection is spread to people primarily through the bite of an infected mosquito. Zika virus can also be spread from a pregnant mother to her child and through sexual contact with an affected male partner. ...
3. `medquad_ans_0000092-1__dup1` (score=0.0066) - Cytomegalovirus (CMV) is a virus found throughout the world that infects between 50 to 80 percent of all adults in the United States by the age of 40. CMV is in the same family of viruses that causes cold sores (herpe...
4. `medquad_ans_0000092-1__dup4` (score=0.0013) - Like all arenaviruses, Chapare virus has a rodent host as its reservoir. Humans can contract CHHF through contact with an infected rodent. Contact can be direct or through inhalation of aerosolized Chapare virus from ...
5. `medquad_ans_0000254-1__dup3` (score=0.0010) - Cytomegalovirus (CMV) is a virus found around the world. It is related to the viruses that cause chickenpox and infectious mononucleosis (mono). Between 50 percent and 80 percent of adults in the United States have ha...

**Answer**:

Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus has spread through blood transfusions. Most people who get the virus do not get sick. One in five people do get symptoms, which can include a fever, rash, joint pain, and conjunctivitis (pinkeye). Symptoms are usually mild, and start 2 to 7 days after being bitten by an infected mosquito. A blood test can tell whether you have the infection. There are no vaccines or medicines to treat it. Drinking lots of fluids, resting, and taking acetaminophen might help. Zika can cause microcephaly (a serious birth defect of the brain) and other problems in babies whose mothers were infected while pregnant. The Centers for Disease Control and Prevention recommends that pregnant women

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000019-3__dup2` (score=0.9850) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
2. `medquad_ans_0000033-3__dup2` (score=0.9790) - Most heart attacks are caused by a blood clot that blocks one of the coronary arteries, the blood vessels that bring blood and oxygen to the heart muscle. When blood cannot reach part of your heart, that area starves ...
3. `medquad_ans_0000060-3__dup2` (score=0.3281) - Research suggests thatcoronary heart disease(CHD) begins with damage to the lining and inner layers of the coronary (heart) arteries. Several factors contribute to this damage. They include: Smoking, including secondh...
4. `medquad_ans_0000034-12` (score=0.0337) - Heart failure is caused by other diseases and conditions that damage the heart muscle. It is most commonly caused by coronary artery disease, including heart attack. Diabetes and high blood pressure also contribute to...
5. `medquad_ans_0000034-2__dup3` (score=0.0293) - Heart failure is caused by other diseases or conditions that damage the heart muscle such as coronary artery disease (including heart attacks), diabetes, and high blood pressure. Treating these problems can prevent or...

**Answer**:

Broken heart syndrome is believed to be caused by extreme emotional or physical stress, which triggers a surge of fight or flight hormones such as adrenaline and noradrenaline. This hormone surge overwhelms and "stuns" the heart muscle, leading to temporary changes in heart muscle cells or coronary blood vessels. Potential causes include a hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.800, groundedness=0.951, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000892-1__dup1` (score=0.8647) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...
2. `medquad_ans_0000008-1__dup6` (score=0.8143) - Hyperthyroidism is a disorder that occurs when the thyroid gland makes more thyroid hormone than the body needs. Hyperthyroidism is sometimes called thyrotoxicosis, the technical term for too much thyroid hormone in t...
3. `medquad_ans_0000498-1__dup2` (score=0.5367) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...
4. `medquad_ans_0000005-1__dup5` (score=0.4499) - Hashimotos disease, also called chronic lymphocytic thyroiditis or autoimmune thyroiditis, is an autoimmune disease. An autoimmune disease is a disorder in which the bodys immune system attacks the bodys own cells and...
5. `medquad_ans_0000448-1` (score=0.2328) - Hashimoto thyroiditis is a condition that affects the function of the thyroid, which is a butterfly-shaped gland in the lower neck. The thyroid makes hormones that help regulate a wide variety of critical body functio...

**Answer**:

Hashimoto thyroiditis is an autoimmune disorder in which the immune system attacks the thyroid gland, leading to chronic inflammation and reduced thyroid function. It is also known as chronic lymphocytic thyroiditis or autoimmune thyroiditis [medquad_ans_0000498-1__dup2]. This condition is a common cause of hypothyroidism [medquad_ans_0000498-1__dup2].

**Supported citations**: `medquad_ans_0000498-1__dup2`
**Unsupported citations**: _none_

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=0.800, groundedness=0.502, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.7181) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
2. `medquad_ans_0006463-1` (score=0.6708) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
3. `medquad_ans_0000708-2` (score=0.0012) - The prevalence of this condition is unknown.
4. `medquad_ans_0000955-2` (score=0.0009) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.
5. `medquad_ans_0000188-1__dup1` (score=0.0004) - Machado-Joseph disease (MJD), which is also called spinocerebellar ataxia type 3, is a rare hereditary ataxia (ataxia is a medical term meaning lack of muscle control). The disease is characterized by slowly progressi...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and is inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in body organs such as the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are often healthy at birth but soon develop symptoms including enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor absorption of nutrients from food. The disease is severe and life-threatening, with most affected children not surviving past early childhood. New therapies, such as bone marrow transplantation and enzyme replacement therapy, are being explored to improve outcomes [medquad

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.9955) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0004053-3` (score=0.0003) - What causes microscopic polyangiitis (MPA)? The cause of MPA is unknown. It is not contagious, does not usually run in families, and is not a form of cancer. The immune system is thought to play a critical role in the...
3. `medquad_ans_0002753-4` (score=0.0001) - What treatment is available for meningeal hemangiopericytoma? Radical surgical resection with removal of all meningeal attachments is typically the preferred treatment. However this treatment option is generally possi...
4. `medquad_ans_0002753-1` (score=0.0001) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...
5. `medquad_ans_0002826-1` (score=0.0001) - Hereditary hemorrhagic telangiectasia (HHT) is an inherited disorder of the blood vessels that can cause excessive bleeding. People with this condition can develop abnormal blood vessels called arteriovenous malformat...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.875, groundedness=0.987, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000621-3` (score=0.9991) - Mutations in the MLYCD gene cause malonyl-CoA decarboxylase deficiency. The MLYCD gene provides instructions for making an enzyme called malonyl-CoA decarboxylase. Within cells, this enzyme helps regulate the formatio...
2. `medquad_ans_0000588-3` (score=0.9988) - Mutations in the MT-ND1, MT-ND4, MT-ND4L, or MT-ND6 gene can cause LHON. These genes are found in the DNA of cellular structures called mitochondria, which convert the energy from food into a form that cells can use. ...
3. `medquad_ans_0000655-3` (score=0.9980) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
4. `medquad_ans_0000398-1__dup1` (score=0.9419) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...
5. `medquad_ans_0000396-1__dup2` (score=0.0149) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...

**Answer**:

The genetic changes related to microphthalmia include changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or more genes. Most genetic changes associated with isolated microphthalmia have been identified only in very small numbers of affected individuals [medquad_ans_0000655-3].

**Supported citations**: `medquad_ans_0000655-3`
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.752, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000028-18` (score=0.0379) - The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH has more information about gout. Check out the information here. Also, see "Key Words" related to gout.
2. `medquad_ans_0000028-14` (score=0.0125) - To confirm a diagnosis of gout, the doctor inserts a needle into the inflamed joint and draws a sample of synovial fluid, the substance that lubricates a joint. A laboratory technician places some of the fluid on a sl...
3. `medquad_ans_0000093-5__dup1` (score=0.0043) - The most common signs and symptoms of Crohn's disease are - diarrhea - abdominal cramping and pain - weight loss Other general signs and symptoms include - feeling tired - nausea or loss of appetite - fever - anemiaa ...
4. `medquad_ans_0000028-15` (score=0.0029) - Physicians often prescribe high doses of non-steroidal anti-inflammatory drugs (NSAIDs) or steroids for a sudden attack of gout. NSAIDs are taken by mouth and corticosteroids are either taken by mouth or injected into...
5. `medquad_ans_0000028-9` (score=0.0026) - Most people with gout have too much uric acid in their blood, a condition called hyperuricemia. Uric acid is a substance that results from the breakdown of purines, which are part of all human tissue and are found in ...

**Answer**:

The symptoms of gout include sudden and severe joint pain, swelling, redness, and tenderness in the affected joint, often occurring in the big toe but can affect other joints as well. The pain typically begins suddenly and is often described as intense. In addition, the presence of needle-like uric acid crystals in the joint space or connective tissue, resulting from hyperuricemia (excess uric acid in the blood), contributes to the development of gout [medquad_ans_0000028-9]. A diagnosis is confirmed by detecting uric acid crystals in synovial fluid drawn from the inflamed joint under a microscope [medquad_ans_0000028-14]. While the provided context does not explicitly list symptoms in a general list format, the clinical presentation of acute gout attacks includes pain, swelling, and redness, which are consistent with the described condition [medquad

**Supported citations**: `medquad_ans_0000028-9`, `medquad_ans_0000028-14`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.667, groundedness=0.709, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000971-2` (score=0.0001) - Task-specific focal dystonia affects an estimated 7 to 69 per million people in the general population. Musician's dystonia that is severe enough to impact performance occurs in about 1 percent of musicians.
2. `medquad_ans_0003598-1` (score=0.0001) - Limb dystonia is characterized by excessive pulling of the muscles of a limb, such as the hand or foot. The arm or leg might also be involved. Specific symptoms depend on the combinations of muscles involved and how h...
3. `medquad_ans_0000309-5` (score=0.0001) - These resources address the diagnosis or management of dystonia 6: - Gene Review: Gene Review: Dystonia Overview - Genetic Testing Registry: Dystonia 6, torsion These resources from MedlinePlus offer information about...
4. `medquad_ans_0000312-5` (score=0.0001) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Mov...
5. `medquad_ans_0000307-1__dup1` (score=0.0001) - Dystonia is a movement disorder that causes involuntary contractions of your muscles. These contractions result in twisting and repetitive movements. Sometimes they are painful. Dystonia can affect just one muscle, a ...

**Answer**:

The provided context does not specify the cause of Dystonia 8. [medquad_ans_0000309-5] and [medquad_ans_0000312-5] mention resources for diagnosing and managing dystonia, including genetic testing and gene reviews, but no information is given about the specific cause of Dystonia 8. Therefore, the cause of Dystonia 8 cannot be determined from the provided context.

**Supported citations**: `medquad_ans_0000309-5`, `medquad_ans_0000312-5`
**Unsupported citations**: _none_

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.444, groundedness=0.978, answer_relevance=0.356

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0001101-2` (score=0.9995) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
3. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000612-2` (score=0.9993) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
5. `medquad_ans_0000136-2` (score=0.9992) - Researchers estimate that BOR/BO syndrome affects about 1 in 40,000 people.

**Answer**:

Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=0.750, groundedness=0.075, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000150-4__dup2` (score=0.9987) - Most people with high blood pressure do not have symptoms. In rare cases, high blood pressure can cause headaches. Kidney disease also does not have symptoms in the early stages. A person may have swelling called edem...
2. `medquad_ans_0000071-5__dup1` (score=0.9951) - Because diagnosis is based on blood pressure readings, this condition can go undetected for years, as symptoms do not usually appear until the body is damaged from chronic high blood pressure. Complications of High Bl...
3. `medquad_ans_0000012-5__dup1` (score=0.4271) - Atherosclerosis usually doesn't cause signs and symptoms until it severely narrows or totally blocks an artery. Many people don't know they have the disease until they have a medical emergency, such as a heart attack ...
4. `medquad_ans_0000145-3__dup3` (score=0.0522) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breat...
5. `medquad_ans_0000144-5__dup2` (score=0.0020) - The signs and symptoms of glomerular disease include - albuminuria: large amounts of protein in the urine - hematuria: blood in the urine - reduced glomerular filtration rate: inefficient filtering of wastes from the ...

**Answer**:

The provided context does not mention Renal Artery Stenosis or its symptoms. Therefore, based on the given information, it is not possible to answer the question. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.800, groundedness=0.843, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.9908) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
2. `medquad_ans_0004303-4` (score=0.9591) - Is there a treatment for Naegeli syndrome? Treatment for Naegeli syndrome is based on an individual's symptoms. Dry skin can be moisturized with creams. To avoid overheating, affected individuals should wear appropria...
3. `medquad_ans_0000618-5` (score=0.9550) - These resources address the diagnosis or management of mal de Meleda: - Foundation for Ichthyosis and Related Skin Types: Palmoplantar Keratodermas - Genetic Testing Registry: Acroerythrokeratoderma These resources fr...
4. `medquad_ans_0000273-2__dup2` (score=0.0673) - There is no cure for TSC, although treatment is available for a number of the symptoms. Rapamycin and related drugs are not yet approved by the U.S. Food and Drug Administration (FDA) for any purpose in individuals wi...
5. `medquad_ans_0004428-3` (score=0.0018) - How might nodular nonsuppurative panniculitis be treated? Treatment for nodular nonsuppurative panniculitis (NNP) generally aims at controlling and relieving the symptoms that an individual has. Before treatment is in...

**Answer**:

The provided context does not specify treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While it mentions that treatment for Naegeli syndrome is based on an individual's symptoms, including moisturizing dry skin with creams, using appropriate clothing to avoid overheating, and dental care for cavities and tooth loss, this information is specific to Naegeli syndrome and not directly applicable to Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis [medquad_ans_0004303-4]. Therefore, based on the available context, no specific treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis can be identified.

**Supported citations**: `medquad_ans_0004303-4`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.9994) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0002269-1` (score=0.0001) - What are the signs and symptoms of Familial hypocalciuric hypercalcemia type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hypocalciuric hypercalcemia type 1. If the in...
3. `medquad_ans_0005138-2` (score=0.0001) - What are the signs and symptoms of Pseudohypoaldosteronism type 2? The most common symptom of pseudohypoaldosteronism type 2 is high blood pressure in adolescents or young adults. In its most severe form, it is associ...
4. `medquad_ans_0002270-1` (score=0.0001) - What are the signs and symptoms of Familial hypocalciuric hypercalcemia type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hypocalciuric hypercalcemia type 2. If the in...
5. `medquad_ans_0000353-1` (score=0.0001) - Familial hyperaldosteronism is a group of inherited conditions in which the adrenal glands, which are small glands located on top of each kidney, produce too much of the hormone aldosterone. Aldosterone helps control ...

**Answer**:

The symptoms of Familial hyperaldosteronism type III include hypercalciuria (5%), metabolic acidosis (5%), polydipsia (5%), polyuria (5%), adrenal hyperplasia, autosomal dominant inheritance, decreased circulating renin level, hyperaldosteronism, hypertension, and hypokalemia [medquad_ans_0002259-1].

**Supported citations**: `medquad_ans_0002259-1`
**Unsupported citations**: _none_

### Example 54 (`0005761-3`)

**Question**: Is Spinocerebellar ataxia 11 inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000575-4` (score=0.9987) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
2. `medquad_ans_0000786-4` (score=0.9987) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
3. `medquad_ans_0000700-4` (score=0.9987) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000673-4` (score=0.9987) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000149-4` (score=0.9987) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not specify whether Spinocerebellar ataxia 11 is inherited. While several documents describe autosomal dominant inheritance patterns for a condition, none explicitly identify the condition as Spinocerebellar ataxia 11. Therefore, based on the given information, it is not possible to confirm the inheritance pattern of Spinocerebellar ataxia 11. [insufficient context]

**Supported citations**: _none_
**Unsupported citations**: `insufficient context`

### Example 55 (`0003040-1`)

**Question**: What are the symptoms of Hypochromic microcytic anemia with iron overload ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0003040-1` (score=0.9993) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
2. `medquad_ans_0000503-1` (score=0.0001) - Hypochromic microcytic anemia with iron overload is a condition that impairs the normal transport of iron in cells. Iron is an essential component of hemoglobin, which is the substance that red blood cells use to carr...
3. `medquad_ans_0000503-2` (score=0.0001) - Hypochromic microcytic anemia with iron overload is likely a rare disorder; at least five affected families have been reported in the scientific literature.
4. `medquad_ans_0000185-1__dup3` (score=0.0001) - Hemochromatosis is the most common form of iron overload disease. Too much iron in the body causes hemochromatosis. Iron is important because it is part of hemoglobin, a molecule in the blood that transports oxygen fr...
5. `medquad_ans_0000529-1` (score=0.0001) - Iron-refractory iron deficiency anemia is one of many types of anemia, which is a group of conditions characterized by a shortage of healthy red blood cells. This shortage prevents the blood from carrying an adequate ...

**Answer**:

The symptoms of Hypochromic microcytic anemia with iron overload include pale skin (pallor), tiredness (fatigue), and slow growth. Additionally, the condition leads to abnormally small and pale red blood cells (microcytic and hypochromic), and iron accumulates in the liver, which can impair its function over time, typically becoming apparent in adolescence or early adulthood [medquad_ans_0000503-1].

**Supported citations**: `medquad_ans_0000503-1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000784-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000685-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000349-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000101-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000298-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention benign chronic pemphigus or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether benign chronic pemphigus is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.948, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000585-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000095-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000673-4` (score=0.9988) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0000916-4__dup1` (score=0.9533) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...
5. `medquad_ans_0004784-4` (score=0.7147) - How is Pelizaeus-Merzbacher disease inherited?

**Answer**:

Yes, Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is enough to cause features of the condition. In some cases, an affected person inherits the mutated gene from an affected parent, and in other cases, the mutation occurs for the first time in a person with no family history of the condition, known as a de novo mutation [medquad_ans_0000916-4__dup1].

**Supported citations**: `medquad_ans_0000916-4__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.900, groundedness=0.925, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000483-3` (score=0.9990) - Histiocytosis-lymphadenopathy plus syndrome is caused by mutations in the SLC29A3 gene, which provides instructions for making a protein called equilibrative nucleoside transporter 3 (ENT3). ENT3 belongs to a family o...
2. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
3. `medquad_ans_0000178-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000575-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
5. `medquad_ans_0001030-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene and occur in pe...

**Answer**:

The genetic changes related to histiocytosis-lymphadenopathy plus syndrome are mutations in the SLC29A3 gene [medquad_ans_0000483-3]. These mutations reduce or eliminate the activity of the equilibrative nucleoside transporter 3 (ENT3) protein, which is responsible for transporting nucleosides out of lysosomes and into mitochondria [medquad_ans_0000483-3]. This impairment in nucleoside transport is believed to lead to a buildup of nucleosides in lysosomes and a shortage in mitochondria, potentially affecting cellular energy production and contributing to the condition's symptoms [medquad_ans_0000483-3].

**Supported citations**: `medquad_ans_0000483-3`
**Unsupported citations**: _none_

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000101-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000683-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000770-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000370-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000405-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Noonan syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Noonan syndrome is inherited. [No supporting context available]

**Supported citations**: _none_
**Unsupported citations**: `No supporting context available`

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.875, groundedness=0.801, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0001093-5` (score=0.9952) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
2. `medquad_ans_0006372-6` (score=0.9421) - How might VLCAD deficiency be treated? Management of VLCAD deficiency depends on many factors, including the form of the condition and the specific signs and symptoms present. For example, people affected by the sever...
3. `medquad_ans_0006585-5` (score=0.0107) - How might 3-hydroxyisobutyric aciduria be treated? There is no cure for 3-hydroxyisobutyric aciduria. Because it is so rare, there is limited evidence to support the effectiveness of treatment. However, affected peopl...
4. `medquad_ans_0003264-5` (score=0.0018) - How is isobutyryl-CoA dehydrogenase deficiency (IBD deficiency) treated? There is no standard treatment protocol for IBD deficiency. Infants diagnosed through newborn screening are encouraged to work with a metabolic ...
5. `medquad_ans_0004451-3` (score=0.0016) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain information about the treatments for 3-methylglutaconyl-CoA hydratase deficiency. [medquad_ans_0001093-5] mentions resources related to the diagnosis or management of this condition, but no specific treatments are described. Therefore, based on the available context, the treatments for 3-methylglutaconyl-CoA hydratase deficiency cannot be determined.

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

**Scores**: context_relevance=0.800, groundedness=0.799, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000233-2__dup1` (score=0.9974) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...
2. `medquad_ans_0000225-5` (score=0.9944) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
3. `medquad_ans_0002381-3` (score=0.0099) - How might Filippi syndrome be treated? The treatment of Filippi syndrome is directed toward the specific symptoms that are apparent in each individual. Treatment may require the coordinated efforts of a team of medica...
4. `medquad_ans_0000873-5` (score=0.0007) - These resources address the diagnosis or management of retroperitoneal fibrosis: - Johns Hopkins Medicine These resources from MedlinePlus offer information about the diagnosis and management of various health conditi...
5. `medquad_ans_0000225-3` (score=0.0001) - CFEOM1 and rare cases of CFEOM3 result from mutations in the KIF21A gene. This gene provides instructions for making a protein called a kinesin, which is essential for the transport of materials within cells. Research...

**Answer**:

The provided context does not specify the treatments for congenital fibrosis of the extraocular muscles. While it lists resources related to the diagnosis and management of this condition, such as Gene Review, Genetic Testing Registry entries, MedlinePlus Encyclopedia entries on extraocular muscle function testing and strabismus, and general categories like diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, no specific treatments are detailed [medquad_ans_0000225-5]. Therefore, based on the given information, the exact treatments cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.492, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.6731) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0002944-1` (score=0.4746) - Huntington disease (HD) is an inherited condition that causes progressive degeneration of neurons in the brain. Signs and symptoms usually develop between ages 35 to 44 years and may include uncontrolled movements, lo...
3. `medquad_ans_0000152-1__dup1` (score=0.4376) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressiv...
4. `medquad_ans_0000489-1` (score=0.1633) - Huntington disease is a progressive brain disorder that causes uncontrolled movements, emotional problems, and loss of thinking ability (cognition). Adult-onset Huntington disease, the most common form of this disorde...
5. `medquad_ans_0000075-1__dup2` (score=0.0710) - Chorea is an abnormal involuntary movement disorder, one of a group of neurological disorders called dyskinesias, which are caused by overactivity of the neurotransmitter dopamine in the areas of the brain that contro...

**Answer**:

Huntington's disease (HD) is an inherited disorder that causes progressive degeneration of neurons in the brain, particularly in motor control regions and other areas. It is caused by a mutation in the HTT gene, leading to an abnormal repetition of cytosine, adenine, and guanine (CAG) building blocks in DNA. This mutation results in the production of a defective huntingtin protein, which causes brain cell degeneration. Symptoms typically begin between ages 30 and 50, and may include uncontrolled movements (chorea), abnormal body postures, impaired coordination, slurred speech, difficulty feeding and swallowing, changes in behavior, emotion, judgment, and cognition. In some cases, an early-onset form called juvenile HD occurs before age 20, with symptoms such as unsteadiness, rigidity, difficulty at school, and seizures. The disease is inherited in an autosomal dominant manner, meaning each child of

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.950, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000419-2` (score=0.9993) - Glutathione synthetase deficiency is very rare. This disorder has been described in more than 70 people worldwide.
2. `medquad_ans_0000416-2` (score=0.9992) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
3. `medquad_ans_0000442-2` (score=0.9986) - Guanidinoacetate methyltransferase deficiency is a very rare disorder. About 80 affected individuals have been described in the medical literature. Of these, approximately one-third are of Portuguese origin.
4. `medquad_ans_0000417-2` (score=0.9449) - Glutaric acidemia type I occurs in approximately 1 of every 30,000 to 40,000 individuals. It is much more common in the Amish community and in the Ojibwa population of Canada, where up to 1 in 300 newborns may be affe...
5. `medquad_ans_0002601-2` (score=0.0001) - What are the signs and symptoms of Glutamate formiminotransferase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Glutamate formiminotransferase deficiency. If the inform...

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.931, answer_relevance=0.475

**Top docs**:

1. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
2. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.
4. `medquad_ans_0000738-2` (score=0.9992) - Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people.
5. `medquad_ans_0004450-1` (score=0.0001) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Noonan syndrome occurs in approximately 1 in 1,000 to 2,500 people [medquad_ans_0000738-2].

**Supported citations**: `medquad_ans_0000738-2`
**Unsupported citations**: _none_

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.571, groundedness=0.989, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000053-5__dup2` (score=0.9966) - Body lice infestation is diagnosed by finding eggs and crawling lice in the seams of clothing. Sometimes a body louse can be seen crawling or feeding on the skin. Although body lice and nits can be large enough to be ...
2. `medquad_ans_0000339-5__dup1` (score=0.9943) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
3. `medquad_ans_0000344-5__dup1` (score=0.9895) - If you suspect you have been infected, consult your health care provider immediately. Be sure to tell your health care provider if you have recently been exposed to raccoons or their feces. Diagnosis is difficult beca...
4. `medquad_ans_0000214-5__dup1` (score=0.0870) - Misdiagnosis of head lice infestation is common. The diagnosis of head lice infestation is best made by finding a live nymph or adult louse on the scalp or hair of a person. Because adult and nymph lice are very small...
5. `medquad_ans_0000339-1__dup3` (score=0.0001) - Also called crab lice or "crabs," pubic lice are parasitic insects found primarily in the pubic or genital area of humans. Pubic lice infestation is found worldwide and occurs in all races, ethnic groups, and levels o...

**Answer**:

Pubic "crab" lice infestation is diagnosed by finding a "crab" louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp) [medquad_ans_0000339-5__dup1]. Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.925, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9961) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0005571-3` (score=0.9688) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
3. `medquad_ans_0000860-2__dup1` (score=0.9474) - How might Brown-Sequard syndrome be treated?
4. `medquad_ans_0002533-6` (score=0.0058) - How might Gerstmann-Straussler-Scheinker disease be treated? The treatment of Gerstmann-Straussler-Scheinker disease (GSS) is based on the signs and symptoms present in each person. There is currently no cure for the ...
5. `medquad_ans_0005502-5` (score=0.0033) - How might Schnitzler syndrome be treated? The treatment of Schnitzler syndrome is aimed at alleviating the signs and symptoms associated with the condition. The following medications have been used with variable succe...

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, or sympathectomy. Responses to these treatments have varied among individuals. It is recommended that treatment options be discussed with a health care provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

**Supported citations**: `medquad_ans_0005571-3`
**Unsupported citations**: _none_

### Example 68 (`0002359-1`)

**Question**: What is (are) FG syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.970, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0002359-1` (score=0.6838) - FG syndrome (FGS) is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first individuals diagnosed with the disorder. People wi...
2. `medquad_ans_0000373-1` (score=0.6256) - FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. FG syndrome affects ...
3. `medquad_ans_0002381-1` (score=0.0073) - Filippi syndrome is an extremely rare genetic condition characterized by a small head (microcephaly), webbing of the fingers and toes (syndactyly), intellectual disability, growth delay, and distinctive facial feature...
4. `medquad_ans_0000356-2` (score=0.0022) - FIBGC is thought to be a rare disorder; about 60 affected families have been described in the medical literature. However, because brain imaging tests are needed to recognize the calcium deposits, this condition is be...
5. `medquad_ans_0000373-2` (score=0.0002) - The prevalence of FG syndrome is unknown, although several hundred cases have been reported worldwide. Researchers suspect that FG syndrome may be overdiagnosed because many of its signs and symptoms are also seen wit...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.9994) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0000676-2__dup1` (score=0.0007) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0003459-2` (score=0.0001) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0005502-2` (score=0.0001) - What are the signs and symptoms of Schnitzler syndrome? The signs and symptoms of Schnitzler syndrome vary but may include: Red raised patches of skin (urticaria) that may become itchy Recurrent fevers Join pain and i...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, and syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.167, groundedness=0.049, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000674-5` (score=0.9859) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
2. `medquad_ans_0000675-5` (score=0.8807) - These resources address the diagnosis or management of mucolipidosis III alpha/beta: - Gene Review: Gene Review: Mucolipidosis III Alpha/Beta - Genetic Testing Registry: Pseudo-Hurler polydystrophy - MedlinePlus Encyc...
3. `medquad_ans_0004784-5` (score=0.1533) - How might Pelizaeus-Merzbacher disease be treated?
4. `medquad_ans_0000676-5` (score=0.0573) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloud...
5. `medquad_ans_0000677-5` (score=0.0223) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastr...

**Answer**:

The provided context does not specify specific treatments for mucolipidoses. However, it indicates that resources from MedlinePlus offer information on the diagnosis and management of various health conditions, including diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000674-5, medquad_ans_0000675-5, medquad_ans_0000676-5, medquad_ans_0000677-5]. These general management approaches may apply to mucolipidoses, but no specific treatments are detailed in the context.

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000674-5, medquad_ans_0000675-5, medquad_ans_0000676-5, medquad_ans_0000677-5`

### Example 71 (`0005084-2`)

**Question**: What are the symptoms of Progressive familial intrahepatic cholestasis type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0005084-2` (score=0.9994) - What are the signs and symptoms of Progressive familial intrahepatic cholestasis type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Progressive familial intrahepatic cholestasis...
2. `medquad_ans_0003240-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile ...
3. `medquad_ans_0005084-1` (score=0.0001) - Progressive familial intrahepatic cholestasis type 2 (PFIC2) is a rare condition that affects the liver. People with this condition generally develop signs and symptoms during infancy, which may include severe itching...
4. `medquad_ans_0003239-1` (score=0.0001) - Intrahepatic cholangiocarcinoma is a cancer that develops in the cells within the bile ducts; both inside and outside the liver. The terms cholangiocarinoma and bile duct cancer are often used to refer to the same con...
5. `medquad_ans_0000525-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy is a liver disorder that occurs in pregnant women. Cholestasis is a condition that impairs the release of a digestive fluid called bile from liver cells. As a result, bile builds ...

**Answer**:

The symptoms of Progressive familial intrahepatic cholestasis type 2 include cirrhosis, conjugated hyperbilirubinemia, death in childhood, diarrhea, elevated alkaline phosphatase, failure to thrive, fat malabsorption, hepatocellular carcinoma, hepatomegaly, infantile onset, intermittent jaundice, intrahepatic cholestasis, pruritus, short stature, and splenomegaly [medquad_ans_0005084-2].

**Supported citations**: `medquad_ans_0005084-2`
**Unsupported citations**: _none_

### Example 72 (`0000211-6`)

**Question**: How many people are affected by Urinary Incontinence in Men ?

**Scores**: context_relevance=0.800, groundedness=0.817, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.9968) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
2. `medquad_ans_0000213-4__dup2` (score=0.0061) - Urinary retention in men becomes more common with age. - In men 40 to 83 years old, the overall incidence of urinary retention is 4.5 to 6.8 per 1,000 men.2 - For men in their 70s, the overall incidence increases to 1...
3. `medquad_ans_0000214-4__dup2` (score=0.0005) - Urinary tract infections are the second most common type of infection in the body, accounting for about 8.1 million visits to health care providers each year.1 Women are especially prone to UTIs for anatomical reasons...
4. `medquad_ans_0000070-7` (score=0.0001) - Urinary incontinence means a person leaks urine by accident. Urinary incontinence is a common bladder problem as people age. Women are more likely than men to leak urine. If this problem is happening to you, there is ...
5. `medquad_ans_0000211-1__dup3` (score=0.0001) - Urinary incontinence is the loss of bladder control, resulting in the accidental leakage of urine from the body. For example, a man may feel a strong, sudden need, or urgency, to urinate just before losing a large amo...

**Answer**:

Urinary incontinence occurs in 11 to 34 percent of older men [medquad_ans_0000211-6]. Two to 11 percent of older men report daily urinary incontinence [medquad_ans_0000211-6]. These statistics indicate that a significant portion of older men are affected by urinary incontinence, though the exact number varies by age and severity. [medquad_ans_0000211-6]

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005415-1` (score=0.3207) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
2. `medquad_ans_0000882-1` (score=0.2402) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
3. `medquad_ans_0000420-1__dup1` (score=0.0009) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...
4. `medquad_ans_0002652-1` (score=0.0003) - Gordon Syndrome is a rare, inherited type of distal arthrogryposis typically characterized by a combination of camptodactyly (a permanent fixation of several fingers in a flexed position), clubfoot (abnormal bending i...
5. `medquad_ans_0005415-4` (score=0.0002) - Is genetic testing available for Robinow syndrome? Genetic testing for autosomal recessive Robinow syndrome and autosomal dominant Robinow syndrome is available. However, not all people diagnosed with either type of R...

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

**Scores**: context_relevance=0.750, groundedness=0.991, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-9` (score=0.9663) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
2. `medquad_ans_0000554-1__dup2` (score=0.7928) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
3. `medquad_ans_0000010-1__dup4` (score=0.7665) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...
4. `medquad_ans_0000040-20` (score=0.7260) - Chemotherapy is a cancer treatment that uses drugs to kill cancer cells. This is the most common treatment for most types of leukemia. Chemotherapy may be taken by mouth in pill form, by injection directly into a vein...
5. `medquad_ans_0000040-1__dup3` (score=0.4616) - Leukemia is a cancer of the blood cells. It is the most common type of blood cancer and affects 10 times as many adults as children. Most people diagnosed with leukemia are over 50 years old. Leukemia Starts in Bone M...

**Answer**:

Leukemia is a cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells crowd out the healthy blood cells, making it hard for blood to do its work [medquad_ans_0000554-1__dup2]. There are different types of leukemia, including acute lymphocytic leukemia, acute myeloid leukemia, chronic lymphocytic leukemia, and chronic myeloid leukemia [medquad_ans_0000554-1__dup2]. In acute leukemia, the cells are very abnormal and their number increases rapidly, while chronic leukemia grows slowly [medquad_ans_0000554-1__dup2]. Acute myeloid leukemia (AML) is the most common type of acute leukemia in adults and usually gets worse quickly if not

**Supported citations**: `medquad_ans_0000554-1__dup2`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=0.750, groundedness=0.992, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002485-1` (score=0.7954) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
2. `medquad_ans_0001809-1` (score=0.0100) - Desmoplastic infantile gangliomas (DIGs) are rare brain tumors that are normally located in the frontal or parietal lobes of the brain. They are usually diagnosed before 18 months of age with most infants presenting w...
3. `medquad_ans_0000737-1` (score=0.0013) - Paraganglioma is a type of noncancerous (benign) tumor that occurs in structures called paraganglia. Paraganglia are groups of cells that are found near nerve cell bunches called ganglia. Paragangliomas are usually fo...
4. `medquad_ans_0002574-1` (score=0.0011) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
5. `medquad_ans_0000135-1__dup1` (score=0.0007) - The gangliosidoses are a group of inherited metabolic diseases caused by a deficiency of the different proteins needed to break down fatty substances called lipids. Excess buildup of these fatty materials (oils, waxes...

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most commonly occurs in people between the ages of 10 and 30. The most common site is the temporal lobe of the brain, although it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are among the most frequent tumors associated with epilepsy and are generally slow-growing and usually do not become malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

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

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006570-3` (score=0.9947) - What causes 22q11.2 deletion syndrome? 22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location desig...
2. `medquad_ans_0006579-3` (score=0.0223) - What causes 2q37 deletion syndrome? 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at ...
3. `medquad_ans_0001087-3` (score=0.0003) - 2q37 deletion syndrome is caused by a deletion of genetic material from a specific region in the long (q) arm of chromosome 2. The deletion occurs near the end of the chromosome at a location designated 2q37. The size...
4. `medquad_ans_0006562-3` (score=0.0002) - What causes 17q23.2q23.2 microdeletion syndrome? The syndrome is caused by an interstitial deletion (a deletion that does not involve the ends of a chromosome) encompassing bands 23.1 to 23.2 on the long (q) arm of ch...
5. `medquad_ans_0001084-3` (score=0.0001) - Most people with 22q11.2 deletion syndrome are missing a sequence of about 3 million DNA building blocks (base pairs) on one copy of chromosome 22 in each cell. This region contains 30 to 40 genes, many of which have ...

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location designated q11.2. Most people with 22q11.2 deletion syndrome are missing a piece of the chromosome that contains about 30 to 40 genes, many of which have not been well characterized. Some affected people have smaller deletions. The deletion of a particular gene, TBX1, is probably responsible for many of the syndrome's characteristic signs (such as heart defects, a cleft palate, distinctive facial features, hearing loss, and low calcium levels). Loss of this gene may also contribute to behavioral problems. The loss of another gene, COMT, may also cause increased risk of behavioral problems and mental illness in affected people. The other genes that are deleted likely contribute to the various features of

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 79 (`0000062-2__dup3`)

**Question**: What causes Shingles ?

**Scores**: context_relevance=1.000, groundedness=0.920, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000062-12` (score=0.9888) - Shingles is caused by a virus called the varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores and other symptoms healed, but the virus remained....
2. `medquad_ans_0000062-2__dup3` (score=0.9619) - Caused By A Virus Shingles is caused by a virus called varicella-zoster virus -- the one that gave you chickenpox when you were a child. As you recovered from chickenpox, the sores healed and the other symptoms went a...
3. `medquad_ans_0000148-3__dup2` (score=0.0060) - For most healthy people who receive treatment soon after the outbreak of blisters, the lesions heal, the pain subsides within 3 to 5 weeks, and the blisters often leave no scars. However, shingles is a serious threat ...
4. `medquad_ans_0000062-4__dup3` (score=0.0025) - Burning, Itching, Tingling, Then a Rash An outbreak of shingles usually begins with a burning, itching, or tingling sensation on the back, chest, or around the rib cage or waist. It is also common for the face or eye ...
5. `medquad_ans_0000062-1__dup5` (score=0.0012) - Shingles is a painful rash that develops on one side of the face or body. The rash forms blisters that typically scab over in 7 to 10 days and clear up within 2 to 4 weeks. Most commonly, the rash occurs in a single s...

**Answer**:

Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. After recovering from chickenpox, the virus remains in the body for life, hiding in nerve cells, usually in the spine. It can become active again when the immune system weakens, which may occur due to aging, certain medical conditions (such as cancer, HIV/AIDS, or transplant operations), or stress. When reactivated, the virus travels along a nerve path (dermatome) to the skin, causing a painful rash [medquad_ans_0000062-12][medquad_ans_0000062-2__dup3].

**Supported citations**: `medquad_ans_0000062-12`, `medquad_ans_0000062-2__dup3`
**Unsupported citations**: _none_

### Example 80 (`0002456-1`)

**Question**: What are the symptoms of Fryns Hofkens Fabry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.9994) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0002458-2` (score=0.9970) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0000619-1__dup1` (score=0.0001) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
4. `medquad_ans_0002694-1` (score=0.0001) - What are the signs and symptoms of Grubben de Cock Borghgraef syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Grubben de Cock Borghgraef syndrome. If the information is ava...
5. `medquad_ans_0000333-2` (score=0.0001) - Fabry disease affects an estimated 1 in 40,000 to 60,000 males. This disorder also occurs in females, although the prevalence is unknown. Milder, late-onset forms of the disorder are probably more common than the clas...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (90%), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand [medquad_ans_0002456-1].

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.900, groundedness=0.971, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000050-8` (score=0.9897) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research foc...
2. `medquad_ans_0000050-28` (score=0.9646) - Consider talking to your doctor about being evaluated for osteoporosis if - you are a man or woman over age 50 or a postmenopausal woman and you break a bone - you are a woman age 65 or older - you are a woman younger...
3. `medquad_ans_0000050-36` (score=0.0099) - Millions of Americans are able to lead healthy, productive lives while living with osteoporosis. If you have been feeling symptoms of depressionsuch as loss of appetite, hopelessness, feeling useless and helpless, or ...
4. `medquad_ans_0000006_3-8` (score=0.0053) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000050-29` (score=0.0004) - Diagnosing osteoporosis involves several steps, starting with a physical exam and a careful medical history, blood and urine tests, and possibly a bone mineral density assessment. When recording information about your...

**Answer**:

Research on osteoporosis is focused on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and even enhance bone density and reduce fracture risk, and exploring the roles of factors such as genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8]. For more information, one can refer to the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH [medquad_ans_0000050-8].

**Supported citations**: `medquad_ans_0000050-8`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=0.857, groundedness=0.910, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000338-1__dup1` (score=0.9973) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
2. `medquad_ans_0000430-1__dup1` (score=0.9966) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
3. `medquad_ans_0000693-1__dup2` (score=0.9966) - Summary : As a patient, you have certain rights. Some are guaranteed by federal law, such as the right to get a copy of your medical records, and the right to keep them private. Many states have additional laws protec...
4. `medquad_ans_0000694-1__dup1` (score=0.9945) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...
5. `medquad_ans_0000033-26` (score=0.9301) - More detailed information on heart attacks is available at www.nhlbi.nih.gov/health/dci.

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. It often includes red flags such as "miracle cure," "quick fix," "ancient remedy," "secret ingredient," or "scientific breakthrough." These scams can be dangerous, waste money, and may prevent individuals from receiving necessary treatments. The Food and Drug Administration (FDA) warns that such practices are common, especially for conditions like cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Before taking unproven treatments, it is recommended to consult a doctor or healthcare professional, especially when considering prescription drugs [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.917, groundedness=0.784, answer_relevance=0.583

**Top docs**:

1. `medquad_ans_0000016-10` (score=0.9985) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year. Learn more about a comprehensive dilated eye exam.
2. `medquad_ans_0000055-3__dup2` (score=0.9946) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
3. `medquad_ans_0000016-24` (score=0.9885) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...
4. `medquad_ans_0000275-1__dup1` (score=0.0254) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
5. `medquad_ans_0000279-1__dup1` (score=0.0035) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage the covering on your nerves or the blood vessels that bring oxygen to your nerves. Damaged nerves may stop send...

**Answer**:

The provided context does not contain information about symptoms of preventing diabetes problems or specifically about "keeping your eyes healthy" in terms of preventive symptoms. The context discusses symptoms of diabetic retinopathy, such as blurry or double vision, rings or flashing lights, dark or floating spots, pain or pressure in the eyes, and trouble seeing out of the corners of the eyes, but these are symptoms of the condition, not preventive measures or symptoms of preventing diabetes problems [medquad_ans_0000055-3\_\_dup2][medquad_ans_0000275-1__dup1].

Therefore, based on the given context, there is no information to answer the question about symptoms of preventing diabetes problems or keeping eyes healthy.

Answer: Insufficient context.

**Supported citations**: `medquad_ans_0000055-3__dup2`, `medquad_ans_0000275-1__dup1`
**Unsupported citations**: _none_

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.900, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000557-4` (score=0.9954) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this cond...
2. `medquad_ans_0000001-4` (score=0.9843) - Aarskog-Scott syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chro...
3. `medquad_ans_0006460-4` (score=0.9803) - Is Wolff-Parkinson-White syndrome inherited?
4. `medquad_ans_0000661-4` (score=0.0077) - Mitochondrial complex III deficiency is usually inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive co...
5. `medquad_ans_0000646-4` (score=0.0024) - Menkes syndrome is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome)...

**Answer**:

Yes, Aarskog-Scott syndrome is inherited in an X-linked recessive pattern [medquad_ans_0000001-4].

**Supported citations**: `medquad_ans_0000001-4`
**Unsupported citations**: _none_

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.800, groundedness=0.903, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001062-3` (score=0.9992) - Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. The ALAS2 gene provides instructions for making an enzyme called erythroid ALA-synthase, which plays a critical role in the production of heme (a compon...
2. `medquad_ans_0001063-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
3. `medquad_ans_0001062-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
4. `medquad_ans_0006525-1` (score=0.0001) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin. People with X-linked sideroblastic anemia have mature red blood cells that...
5. `medquad_ans_0001062-1` (score=0.0001) - X-linked sideroblastic anemia is an inherited disorder that prevents developing red blood cells (erythroblasts) from making enough hemoglobin, which is the protein that carries oxygen in the blood. People with X-linke...

**Answer**:

Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. These mutations impair the activity of erythroid ALA-synthase, disrupting normal heme production and leading to reduced hemoglobin synthesis in erythroblasts. This results in a buildup of excess iron in erythroblasts because almost all transported iron is normally incorporated into heme. Additionally, the body compensates for hemoglobin shortage by absorbing more iron from the diet, contributing to iron overload and organ damage. In some cases, mutations in both the ALAS2 and HFE genes can lead to a more severe form of the disorder due to increased iron absorption and accumulation [medquad_ans_0001062-3]. The condition is inherited in an X-linked recessive pattern, meaning the gene is located on the X chromosome, and males are more frequently affected due to having only one X chromosome [medquad_ans_0001

**Supported citations**: `medquad_ans_0001062-3`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=0.833, groundedness=0.958, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.3290) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
2. `medquad_ans_0003601-1` (score=0.1475) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0003610-1` (score=0.0108) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
4. `medquad_ans_0004225-1` (score=0.0042) - The congenital muscle dystrophies are currently classified according to the genetic defects. Historically, congenital muscular dystrophies were classified in two broad groups: Classic CMD (which included the Merosin-d...
5. `medquad_ans_0003604-1` (score=0.0029) - Limb-girdle muscular dystrophy type 2A (LGMD2A) is an autosomal recessive limb-girdle muscular dystrophy characterized by progressive, symmetrical weakness of the proximal limb and girdle muscles (mainly those around ...

**Answer**:

Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs, particularly affecting the proximal muscles closest to the body, such as those in the shoulders, upper arms, pelvic area, and thighs [medquad_ans_0000602-1]. It is a group of disorders that affect the voluntary muscles around the hips and shoulders, characterized by progressive loss of muscle strength and bulk over time [medquad_ans_0003601-1]. The condition may onset in childhood, adolescence, young adulthood, or later in life, and affects males and females equally [medquad_ans_0003601-1]. Most forms are inherited in an autosomal recessive manner, while several rare forms are inherited in an autosomal dominant pattern [medquad_ans_0003601-1]. There

**Supported citations**: `medquad_ans_0000602-1`, `medquad_ans_0003601-1`
**Unsupported citations**: _none_

### Example 87 (`0000275-1__dup1`)

**Question**: What is (are) Diabetic Eye Problems ?

**Scores**: context_relevance=0.833, groundedness=0.897, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000016-26` (score=0.9879) - National Eye Institute National Institutes of Health 2020 Vision Place Bethesda, MD 20892-3655 301-496-5248 E-mail: 2020@nei.nih.gov www.nei.nih.gov Find eye health organizations that address diabetic eye disease.
2. `medquad_ans_0000016-24` (score=0.9810) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...
3. `medquad_ans_0000042-2__dup3` (score=0.8272) - Eye Diseases and Health Conditions Most people develop low vision because of eye diseases and health conditions like macular degeneration, cataracts, glaucoma, and diabetes. Your eye care professional can tell the dif...
4. `medquad_ans_0000275-1__dup1` (score=0.4466) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your eyes. The most common problem is diabetic retinopathy. It is a leading cause of blindness in American adul...
5. `medquad_ans_0000343-1__dup1` (score=0.2204) - Some eye problems are minor and don't last long. But some can lead to a permanent loss of vision. Common eye problems include - Refractive errors - Cataracts - clouded lenses - Glaucoma - a disorder caused by damage t...

**Answer**:

Diabetic eye problems include diabetic retinopathy, macular edema, cataracts, and glaucoma. Diabetic retinopathy is the most common problem and occurs when high blood glucose levels damage the tiny blood vessels in the retina. It can progress without symptoms and may lead to vision loss or blindness. Macular edema can develop at any stage of diabetic retinopathy and also occurs without symptoms. Cataracts involve a clouding of the eye's lens and can be treated with surgery. Glaucoma results from increased pressure in the eye and can also affect vision [medquad_ans_0000042-2\_\_dup3][medquad_ans_0000275-1__dup1][medquad_ans_0000016-24].

**Supported citations**: `medquad_ans_0000042-2__dup3`, `medquad_ans_0000275-1__dup1`, `medquad_ans_0000016-24`
**Unsupported citations**: _none_

### Example 88 (`0006449-1`)

**Question**: What is (are) Wilson disease ?

**Scores**: context_relevance=0.800, groundedness=0.538, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000276-1__dup1` (score=0.6807) - Wilson disease (WD) is a rare inherited disorder of copper metabolism in which excessive amounts of copper accumulate in the body. The buildup of copper leads to damage in the liver, brain, and eyes. Although copper a...
2. `medquad_ans_0000973-1__dup1` (score=0.2974) - Wilson disease is a rare inherited disorder that prevents your body from getting rid of extra copper. You need a small amount of copper from food to stay healthy. Too much copper is poisonous. Normally, your liver rel...
3. `medquad_ans_0001037-1` (score=0.2708) - Wilson disease is an inherited disorder in which excessive amounts of copper accumulate in the body, particularly in the liver, brain, and eyes. The signs and symptoms of Wilson disease usually first appear between th...
4. `medquad_ans_0000133-1__dup3` (score=0.2184) - Wilson disease is a genetic disease that prevents the body from removing extra copper. The body needs a small amount of copper from food to stay healthy; however, too much copper is poisonous. Normally, the liver filt...
5. `medquad_ans_0006449-1` (score=0.0779) - Wilson disease is a rare inherited disorder that is characterized by the accumulation of copper in the body. Because high levels of copper are toxic to tissues and organs, this buildup can lead to damage of the liver,...

**Answer**:

Wilson disease (WD) is a rare inherited disorder of copper metabolism in which excessive amounts of copper accumulate in the body. The buildup of copper leads to damage in the liver, brain, and eyes. Although copper accumulation begins at birth, symptoms of the disorder only appear later in life. The most characteristic sign of WD is the Kayser-Fleisher ring, a rusty brown ring around the cornea of the eye that can best be viewed using an ophthalmologist's slit lamp. The primary consequence for most individuals with WD is liver disease, appearing in late childhood or early adolescence as acute hepatitis, liver failure, or progressive chronic liver disease in the form of chronic active hepatitis or cirrhosis of the liver. In others, the first symptoms are neurological, occurring later in adulthood, and commonly include slurred speech (dysarthria), difficulty swallowing (dysphagia), and drooling. Other symptoms may include tremor of the head,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 89 (`0001503-2`)

**Question**: What are the symptoms of Congenital dyserythropoietic anemia type 2 ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001503-2` (score=0.9990) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
2. `medquad_ans_0003748-2` (score=0.0001) - What are the signs and symptoms of Majeed syndrome? Majeed syndrome is characterized by recurrent episodes of fever and inflammation in the bones and skin. There are two main features of Majeed syndrome: Chronic recur...
3. `medquad_ans_0001503-1` (score=0.0001) - Congenital dyserythropoietic anemia type 2 (CDA II) is an inherited blood disorder characterized by mild to severe anemia. It is usually diagnosed in adolescence or early adulthood. Many affected individuals have yell...
4. `medquad_ans_0000223-1` (score=0.0001) - Congenital dyserythropoietic anemia (CDA) is an inherited blood disorder that affects the development of red blood cells. This disorder is one of many types of anemia, which is a condition characterized by a shortage ...
5. `medquad_ans_0005562-1` (score=0.0001) - What are the signs and symptoms of Severe congenital neutropenia autosomal dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Severe congenital neutropenia autosomal dominant....

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen (splenomegaly), anemia of inadequate production, endopolyploidy on chromosome studies of bone marrow, reduced activity of N-acetylglucosaminyltransferase II, reticulocytosis, and iron overload. Iron overload can lead to complications such as an abnormal heart rhythm (arrhythmia), congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=0.800, groundedness=0.933, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000140-2__dup2` (score=0.9286) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to bui...
2. `medquad_ans_0000038-11` (score=0.9072) - Kidney disease -- also known as chronic kidney disease (CKD) -- occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should. According to the Centers for Di...
3. `medquad_ans_0000144-10` (score=0.1711) - Renal failure is any acute or chronic loss of kidney function and is the term used when some kidney function remains. Total kidney failure, sometimes called end-stage renal disease (ESRD), indicates permanent loss of ...
4. `medquad_ans_0004939-1` (score=0.0389) - Polycystic kidney disease refers to a group of inherited kidney disorders characterized by the presence of multiple cysts in both kidneys. Normal kidney tissue is replaced by fluid-filled sacs that interfere with the ...
5. `medquad_ans_0000167-1__dup4` (score=0.0330) - Polycystic kidney disease is a genetic disorder that causes numerous cysts to grow in the kidneys. A kidney cyst is an abnormal sac filled with fluid. PKD cysts can greatly enlarge the kidneys while replacing much of ...

**Answer**:

Kidney disease, also known as chronic kidney disease (CKD), occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should [medquad_ans_0000038-11]. Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure, and once damaged, the kidneys cannot filter blood as they should, leading to the buildup of wastes in the body and other health problems [medquad_ans_0000140-2__dup2]. Kidney disease can be acute or chronic, with acute renal failure involving a rapid deterioration of kidney function, often reversible, while chronic kidney disease involves a slow, gradual loss of kidney function that cannot be cured and may eventually lead to total kidney failure [medquad_ans_0000144-10].

**Supported citations**: `medquad_ans_0000038-11`, `medquad_ans_0000140-2__dup2`, `medquad_ans_0000144-10`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.975, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000058-4__dup3` (score=0.9707) - If tests show that you have cancer, you should talk with your doctor in order to make treatment decisions. Working With a Team of Specialists A team of specialists often treats people with cancer. The team will keep t...
2. `medquad_ans_0000058-30` (score=0.8510) - Through research, doctors are trying to find new, more effective ways to treat prostate cancer. Cryosurgery -- destroying cancer by freezing it -- is under study as an alternative to surgery and radiation therapy. To ...
3. `medquad_ans_0000036_1-7` (score=0.7404) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000036_1-6` (score=0.7046) - Key Points - There are different types of treatment for patients with prostate cancer. - Seven types of standard treatment are used: - Watchful waiting or active surveillance - Surgery - Radiation therapy and radiopha...
5. `medquad_ans_0000058-6__dup1` (score=0.3091) - Choosing Treatment There are a number of ways to treat prostate cancer, and the doctor will develop a treatment to fit each man's needs. The choice of treatment mostly depends on the stage of the disease and the grade...

**Answer**:

Treatments for prostate cancer include surgery, radiation therapy, hormonal therapy, cryosurgery (destroying cancer by freezing it), chemotherapy, and biological therapy. A medical oncologist, surgeon, or radiation oncologist may be part of a team that treats patients. Hormonal therapy given after radiation therapy can help certain men whose cancer has spread to nearby tissues. In cases where cancer does not respond or stops responding to hormonal therapy, chemotherapy and biological therapy are being tested. The FDA approved a therapeutic cancer vaccine, Provenge, for use in some men with metastatic prostate cancer based on clinical trial results. Clinical trials are also studying new ways to schedule and combine treatments, such as using hormonal therapy to shrink tumors before surgery or radiation, or combining hormone therapy with vaccines to prevent recurrence [medquad_ans_0000058-4\_\_dup3][medquad_ans_0000058-30].

**Supported citations**: `medquad_ans_0000058-4__dup3`, `medquad_ans_0000058-30`
**Unsupported citations**: _none_

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.714, groundedness=0.861, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.9973) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000431-1__dup1` (score=0.9972) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
3. `medquad_ans_0000362-1__dup2` (score=0.9971) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
4. `medquad_ans_0000589-1__dup2` (score=0.9958) - Summary : Medicare is the U.S. government's health insurance program for people age 65 or older. Some people under age 65 can qualify for Medicare, too. They include those with disabilities, permanent kidney failure, ...
5. `medquad_ans_0000590-1__dup2` (score=0.9927) - Summary : Part D is the name of Medicare's prescription drug coverage. It's insurance that helps people pay for prescription drugs. It is available to everyone who has Medicare. It provides protection if you pay high ...

**Answer**:

Yes, there is information about health insurance. Health insurance is a contract between you and your insurance company, where you buy a plan and the company agrees to pay part of your medical expenses when you need care [medquad_ans_0000431-1__dup1]. Many people obtain health insurance through their employers, and employer-sponsored plans are often managed care plans that contract with healthcare providers to offer care at reduced costs [medquad_ans_0000431-1__dup1]. Additionally, government health insurance programs such as Medicare and Medicaid are available to eligible individuals. Medicare serves people age 65 or older and some under 65 with disabilities or chronic conditions [medquad_ans_0000589-1__dup2]. Medicaid is a government health insurance program that assists low-income individuals, with eligibility based on factors such as age, pregnancy, disability, income, and immigration status

**Supported citations**: `medquad_ans_0000431-1__dup1`, `medquad_ans_0000589-1__dup2`
**Unsupported citations**: _none_

### Example 93 (`0003860-1`)

**Question**: What is (are) MECP2 duplication syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.538, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0003860-1` (score=0.4641) - MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Other signs and symptoms include infantile hypotonia; delay...
2. `medquad_ans_0000635-1` (score=0.4579) - MECP2 duplication syndrome is a condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. Most people with this condition also have weak muscle tone in infan...
3. `medquad_ans_0000814-4` (score=0.0008) - More than 99 percent of PPM-X syndrome cases occur in people with no history of the disorder in their family. Many of these cases result from new mutations in the MECP2 gene. A few families with more than one affected...
4. `medquad_ans_0000635-3` (score=0.0006) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
5. `medquad_ans_0005347-1` (score=0.0001) - Rett syndrome is a progressive, neuro-developmental condition that primarily affects girls. Affected girls appear to have normal psychomotor development during the first 6 to 18 months of life, followed by a developme...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. It is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome. The size of the duplication varies from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included in this duplication, with other genes possibly involved. Other signs and symptoms include infantile hypotonia, delayed motor milestones (such as sitting up or crawling), recurrent infections, poor or absent speech, seizures, and/or spasticity. Some individuals experience developmental regression, and approximately one third cannot walk without assistance. Recurrent respiratory tract infections are a major cause of death, with almost half

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.667, groundedness=0.916, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000017_1-5` (score=0.9940) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - How the melanoma cells look under a microscope. - The si...
2. `medquad_ans_0000063-15` (score=0.0225) - Melanoma is the fastest growing and most invasive type of skin cancer. This cancer arises from overgrowth of melanocytes. Melanocytes are specialized skin cells that produce a pigment called melanin.
3. `medquad_ans_0000063-6__dup1` (score=0.0050) - Many Treatment Options There are many treatment options for people with melanoma and non-melanoma skin cancer. The choice of treatment depends on your age and general health, the site of the cancer, the stage of the c...
4. `medquad_ans_0000028_2-3` (score=0.0017) - Signs of melanoma include a change in the way a mole or pigmented area looks. These and other signs and symptoms may be caused by melanoma or by other conditions. Check with your doctor if you have any of the followin...
5. `medquad_ans_0000028_2-6` (score=0.0010) - Key Points - After melanoma has been diagnosed, tests are done to find out if cancer cells have spread within the skin or to other parts of the body. - There are three ways that cancer spreads in the body. - Cancer ma...

**Answer**:

The outlook for melanoma depends on several factors, including how the melanoma cells look under a microscope, the size and thickness of the tumor, the part of the eye the tumor is in (such as the iris, ciliary body, or choroid), whether the tumor has spread within the eye or to other parts of the body, genetic changes linked to intraocular melanoma, the patient's age and general health, and whether the tumor has recurred after treatment [medquad_ans_0000017_1-5]. Early treatment is associated with better chances for successful outcomes [medquad_ans_0000063-6__dup1]. However, melanoma is the fastest growing and most invasive type of skin cancer, arising from overgrowth of melanocytes, which produce melanin [medquad_ans_0000063-15]. Due to its aggressive nature, the prognosis can

**Supported citations**: `medquad_ans_0000017_1-5`, `medquad_ans_0000063-6__dup1`, `medquad_ans_0000063-15`
**Unsupported citations**: _none_

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=0.500, groundedness=0.095, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000685-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000349-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000784-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000205-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000122-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention "boomerang dysplasia" or specify its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether boomerang dysplasia is inherited.

Answer: Insufficient context to determine if boomerang dysplasia is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.900, groundedness=0.499, answer_relevance=0.660

**Top docs**:

1. `medquad_ans_0000466-3` (score=0.9991) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
2. `medquad_ans_0000813-3` (score=0.0004) - Potocki-Shaffer syndrome (also known as proximal 11p deletion syndrome) is caused by a deletion of genetic material from the short (p) arm of chromosome 11 at a position designated 11p11.2. The size of the deletion va...
3. `medquad_ans_0000579-3` (score=0.0002) - Langer-Giedion syndrome is caused by the deletion or mutation of at least two genes on chromosome 8. Researchers have determined that the loss of a functional EXT1 gene is responsible for the multiple osteochondromas ...
4. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
5. `medquad_ans_0002838-1` (score=0.0001) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...

**Answer**:

Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. These proteins bind together to form a complex in the Golgi apparatus, where they modify heparan sulfate. When there is a mutation in exostosin-1 or exostosin-2, heparan sulfate cannot be processed correctly and becomes nonfunctional. A mutation in the EXT1 gene results in hereditary multiple osteochondromas type 1, while a mutation in the EXT2 gene causes hereditary multiple osteochondromas type 2. Mutations in the EXT1 gene are likely responsible for 55 to 75 percent of all cases, and symptoms tend to be more severe in type 1. Approximately 15 percent of individuals with the condition have no mutation

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=0.375, groundedness=0.704, answer_relevance=0.375

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9996) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
4. `medquad_ans_0000136-2` (score=0.9995) - Researchers estimate that BOR/BO syndrome affects about 1 in 40,000 people.
5. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

The exact incidence of oculofaciocardiodental syndrome is unknown, but it is estimated to affect 1 in 40,000 to 50,000 people [medquad_ans_0000195-2]. Another estimate suggests it affects 1 to 3 per million people worldwide [medquad_ans_0000770-2]. Additionally, one source specifically notes that BOR/BO syndrome (which may be related or confused with oculofaciocardiodental syndrome) affects about 1 in 40,000 people [medquad_ans_0000136-2]. However, without explicit confirmation that BOR/BO syndrome is oculofaciocardiodental syndrome, the most accurate estimate based on available context is that the condition affects 1 in 40,000 to 50,00

**Supported citations**: `medquad_ans_0000195-2`, `medquad_ans_0000770-2`, `medquad_ans_0000136-2`
**Unsupported citations**: _none_

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=0.750, groundedness=0.976, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.0631) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0001350-1` (score=0.0046) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
3. `medquad_ans_0000966-1__dup1` (score=0.0027) - Carnitine palmitoyltransferase 2 (CPT2) deficiency is a condition that prevents the body from using certain fats for energy, particularly during periods without food (fasting). There are three main types of CPT2 defic...
4. `medquad_ans_0004334-1` (score=0.0025) - Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is a liver condition is also known as neonatal-onset type II citrullinemia. NICCD blocks the flow of bile (a digestive fluid produced by the liver)...
5. `medquad_ans_0000232-2` (score=0.0023) - CIPA is a rare condition; however, the prevalence is unknown.

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described: Type I (also known as classic citrullinemia), which is caused by mutations in the ASS1 gene and inherited in an autosomal recessive pattern, and typically becomes evident in the first few days of life with symptoms such as lethargy, poor feeding, vomiting, seizures, and loss of consciousness; and Type II, which chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma, and may appear during adulthood, often triggered by medications, infections, surgery, or alcohol intake. Type II can also develop in individuals who had neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) as infants [medquad_ans_0000189-1]medquad_ans

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

**Scores**: context_relevance=0.857, groundedness=0.969, answer_relevance=0.514

**Top docs**:

1. `medquad_ans_0000949-5` (score=0.9918) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
2. `medquad_ans_0000944-5` (score=0.9918) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic ...
3. `medquad_ans_0005860-4` (score=0.0968) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respi...
4. `medquad_ans_0000370-4__dup1` (score=0.0002) - How might ankylosing spondylitis be treated? The main goal of treatment for people with ankylosing spondylitis (AS) is to maximize long-term quality of life. This may involve easing symptoms of pain and stiffness; ret...
5. `medquad_ans_0000949-2` (score=0.0001) - Spondylothoracic dysostosis affects about one in 200,000 people worldwide. However, it is much more common in people of Puerto Rican ancestry, affecting approximately one in 12,000 people.

**Answer**:

Treatment for spondylocostal dysostosis usually consists of intensive medical care, including treatment of respiratory infections, bone surgery, and orthopedic treatment. [medquad_ans_0005860-4]

**Supported citations**: `medquad_ans_0005860-4`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.9994) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0004102-2` (score=0.0001) - What are the signs and symptoms of Moebius syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Moebius syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0000786-3` (score=0.0001) - Perry syndrome results from mutations in the DCTN1 gene. This gene provides instructions for making a protein called dynactin-1, which is involved in the transport of materials within cells. To move materials, dynacti...
4. `medquad_ans_0002480-2` (score=0.0001) - What are the signs and symptoms of Galloway-Mowat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Galloway-Mowat syndrome. If the information is available, the table below ...
5. `medquad_ans_0003746-2` (score=0.0001) - What are the signs and symptoms of Maffucci syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Maffucci syndrome. If the information is available, the table below includes how...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function, respiratory insufficiency, sleep disturbance, tremor, weight loss, developmental regression, hallucinations, hypotension, abnormality of metabolism/homeostasis, apathy, autosomal dominant inheritance, bradykinesia, central hypoventilation, dysarthria, hypoventilation, inappropriate behavior, insomnia, mask-like facies, parkinsonism, rapidly progressive disease, rigidity, short stepped shuffling gait, vertical supranuclear gaze palsy, and weak voice. These symptoms occur in approximately 90% of patients, except for developmental regression, hallucinations, hypotension, and a few others which occur in about 7.5% of patients [medquad_ans_0004825-1].

**Supported citations**: `medquad_ans_0004825-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.904, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0006398-2` (score=0.9992) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
2. `medquad_ans_0006399-1` (score=0.9876) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...
3. `medquad_ans_0001028-2` (score=0.0001) - The prevalence of warfarin sensitivity is unknown. However, it appears to be more common in people who are older, those with lower body weights, and individuals of Asian ancestry. Of the approximately 2 million people...
4. `medquad_ans_0000005-7` (score=0.0001) - Antiphospholipid antibody syndrome (APS) has no cure. However, medicines can help prevent complications. The goals of treatment are to prevent blood clots from forming and keep existing clots from getting larger. You ...
5. `medquad_ans_0004947-2` (score=0.0001) - What are the signs and symptoms of Polycythemia vera? The Human Phenotype Ontology provides the following list of signs and symptoms for Polycythemia vera. If the information is available, the table below includes how...

**Answer**:

The signs and symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet. However, specific symptom frequencies are not detailed in the provided context [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.970, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0004262-1` (score=0.7196) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
2. `medquad_ans_0000701-1` (score=0.6933) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
3. `medquad_ans_0001030-1__dup1` (score=0.0020) - Central core disease (CCD) is an inherited condition that involves muscle weakness, skeletal abnormalities, and an increased chance of having a severe reaction to some anesthesia medications. Muscle weakness ranges fr...
4. `medquad_ans_0000208-1__dup2` (score=0.0012) - The myopathies are neuromuscular disorders in which the primary symptom is muscle weakness due to dysfunction of muscle fiber. Other symptoms of myopathy can include include muscle cramps, stiffness, and spasm. Myopat...
5. `medquad_ans_0000148-2` (score=0.0011) - Cap myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. It is caused by mutations in the SMAD4 gene and inherited in an autosomal dominant pattern, with most cases resulting from a new mutation [medquad_ans_0004262-1]. The condition affects multiple systems and functions of the body, with features such as delayed development of language and motor skills, hearing loss (sensorineural, conductive, or mixed), growth restriction beginning before birth and continuing through adolescence, skin stiffness, and a muscular appearance [medquad_ans_0000701-1]. Typical facial features include short palpebral fissures, a short philtrum, midface hypoplasia, a small mouth with

**Supported citations**: `medquad_ans_0004262-1`, `medquad_ans_0000701-1`
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006057-2` (score=0.9994) - What are the signs and symptoms of Thanatophoric dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Thanatophoric dysplasia. If the information is available, the table below ...
2. `medquad_ans_0000976-4` (score=0.0001) - Thanatophoric dysplasia is considered an autosomal dominant disorder because one mutated copy of the FGFR3 gene in each cell is sufficient to cause the condition. Virtually all cases of thanatophoric dysplasia are cau...
3. `medquad_ans_0002523-1` (score=0.0001) - What are the signs and symptoms of Genitopatellar syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Genitopatellar syndrome. If the information is available, the table below ...
4. `medquad_ans_0000673-2__dup1` (score=0.0001) - What are the signs and symptoms of Behcet's disease? Symptoms of Behcet's disease include recurrent ulcers in the mouth (resembling canker sores) and on the genitals, and eye inflammation (uveitis). The disorder may a...
5. `medquad_ans_0002946-1` (score=0.0001) - What are the signs and symptoms of HurlerScheie syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for HurlerScheie syndrome . If the information is available, the table below in...

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
4. `medquad_ans_0001051-1__dup1` (score=0.0563) - Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy, commonly known as CARASIL, is an inherited condition that causes stroke and other impairments. This progressive condition is...
5. `medquad_ans_0000251-1__dup3` (score=0.0062) - Summary : Computed tomography (CT) is a type of imaging. It uses special x-ray equipment to make cross-sectional pictures of your body. Doctors use CT scans to look for - Broken bones - Cancers - Blood clots - Signs o...

**Answer**:

CADASIL (Cerebral Autosomal Dominant Arteriopathy with Sub-cortical Infarcts and Leukoencephalopathy) is an inherited form of cerebrovascular disease caused by a mutation in the NOTCH3 gene, which leads to thickening of the blood vessel walls and blocks blood flow to the brain, particularly affecting small blood vessels in the white matter of the brain [medquad_ans_0000056-1__dup1]. It is characterized by migraine headaches, multiple strokes, cognitive deterioration, seizures, vision problems, psychiatric symptoms such as severe depression and behavioral changes, and an increased risk of heart attack [medquad_ans_0000056-1__dup1]. The disease progresses to dementia and typically presents with symptoms in the mid-30s, though onset can vary widely and some individuals may not show signs until later in life. CADASIL is inherited

**Supported citations**: `medquad_ans_0000056-1__dup1`
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=0.833, groundedness=0.542, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004185-1` (score=0.9647) - Multiple endocrine neoplasia, type 1 (MEN1) is an inherited condition that causes tumors of the endocrine system (the body's network of hormone-producing glands). People affected by MEN1 typically develop tumors of th...
2. `medquad_ans_0000010-1__dup6` (score=0.9398) - MEN1 is an inherited disorder that causes tumors in the endocrine glands and the duodenum, the first part of the small intestine. MEN1 is sometimes called multiple endocrine adenomatosis or Wermer's syndrome, after on...
3. `medquad_ans_0000688-1` (score=0.6657) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...
4. `medquad_ans_0000014-5__dup1` (score=0.0088) - In about 80 percent of people with primary hyperparathyroidism, a benign, or noncancerous, tumor called an adenoma has formed in one of the parathyroid glands.2 The tumor causes the gland to become overactive. In most...
5. `medquad_ans_0000358-1` (score=0.0030) - Familial isolated pituitary adenoma (FIPA) is an inherited condition characterized by development of a noncancerous tumor in the pituitary gland (called a pituitary adenoma). The pituitary gland, which is found at the...

**Answer**:

Multiple endocrine neoplasia (MEN) is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate the function of cells and tissues throughout the body. Multiple endocrine neoplasia typically involves tumors (neoplasia) in at least two endocrine glands; tumors can also develop in other organs and tissues. These growths can be noncancerous (benign) or cancerous (malignant). If the tumors become cancerous, the condition can be life-threatening. The major forms of multiple endocrine neoplasia are called type 1, type 2, and type 4. These types are distinguished by the genes involved, the types of hormones made, and the characteristic signs and symptoms. Many different types of tumors are associated with MEN [medquad_ans_000068

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004155-3` (score=0.9959) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
2. `medquad_ans_0004155-5` (score=0.0001) - What treatment is available for mucopolysaccharidosis I (MPS I)? The two main treatments for MPS I are enzyme replacement therapy (ERT) and bone marrow transplant. Both of these treatments work by replacing the missin...
3. `medquad_ans_0004155-4` (score=0.0001) - How is mucopolysaccharidosis I (MPS I) inherited? MPS I is inherited from both parents in an autosomal recessive pattern.
4. `medquad_ans_0000200-3__dup1` (score=0.0001) - The mucopolysaccharidoses syndromes share many clinical features but have varying degrees of severity. Most individuals with a mucopolysaccharidosis syndrome generally experience a period of normal development followe...
5. `medquad_ans_0000682-5` (score=0.0001) - These resources address the diagnosis or management of mucopolysaccharidosis type VI: - Emory University Lysosomal Storage Disease Center - Genetic Testing Registry: Mucopolysaccharidosis type VI - MedlinePlus Encyclo...

**Answer**:

Mutations in the IDUA gene cause mucopolysaccharidosis type I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme. The lack of IDUA enzyme activity leads to the accumulation of GAGs within cells, specifically inside the lysosomes. Lysosomes are compartments in the cell that digest and recycle different types of molecules. Conditions that cause molecules to build up inside the lysosomes, including MPS I, are called lysosomal storage disorders. The accumulation of GAGs increases the size of the lysosomes, which is why many tissues and organs are enlarged in this disorder [medquad_ans_0004155-3].

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.700, groundedness=0.933, answer_relevance=0.540

**Top docs**:

1. `medquad_ans_0000825-3` (score=0.9992) - Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition. The FLCN gene provides instructions for making a protein called follicu...
2. `medquad_ans_0000135-2__dup2` (score=0.0050) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
3. `medquad_ans_0000167-2__dup3` (score=0.0001) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
4. `medquad_ans_0000107-3__dup2` (score=0.0001) - Primary ciliary dyskinesia (PCD) is a rare, inherited disease. "Inherited" means the disease is passed from parents to children through genes. With PCD, this process is very complex. Researchers are still learning how...
5. `medquad_ans_0000825-4` (score=0.0001) - When this condition is caused by mutations in the FLCN gene, it is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, ...

**Answer**:

Mutations in the FLCN gene can cause primary spontaneous pneumothorax, although these mutations appear to be a very rare cause of this condition [medquad_ans_0000825-3]. The FLCN gene provides instructions for making a protein called folliculin, which is found in lung connective tissue cells and in cells lining the alveoli. Researchers believe folliculin may help control cell growth and division and may play a role in repairing and re-forming lung tissue after damage. However, the exact mechanism by which FLCN gene mutations lead to bleb formation and increased risk of primary spontaneous pneumothorax is not fully understood [medquad_ans_0000825-3]. In most cases, primary spontaneous pneumothorax occurs in individuals without an identified gene mutation, and the cause remains unknown [medquad_ans_0000825-3].

**Supported citations**: `medquad_ans_0000825-3`
**Unsupported citations**: _none_

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.479, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000066-23` (score=0.9929) - One test that helps doctors judge the severity of a stroke is the standardized NIH Stroke Scale, developed by the National Institute of Neurological Disorders and Stroke at the National Institutes of Health, or NIH. H...
2. `medquad_ans_0000057-9` (score=0.9882) - A stroke happens when part of your brain is not getting enough blood and stops working. Depending on the part of the brain that is damaged, a stroke can cause - sudden weakness or numbness of your face, arm, or leg on...
3. `medquad_ans_0000527-1__dup2` (score=0.9744) - A stroke is a medical emergency. There are two types - ischemic and hemorrhagic. Ischemic stroke is the most common type. It is usually caused by a blood clot that blocks or plugs a blood vessel in the brain. This kee...
4. `medquad_ans_0000857-1__dup2` (score=0.9664) - A stroke is a medical emergency. Strokes happen when blood flow to your brain stops. Within minutes, brain cells begin to die. There are two kinds of stroke. The more common kind, called ischemic stroke, is caused by ...
5. `medquad_ans_0000066-13` (score=0.9599) - There are two kinds of stroke. The most common kind of stroke is called ischemic stroke. It accounts for approximately 80 percent of all strokes. An ischemic stroke is caused by a blood clot that blocks or plugs a blo...

**Answer**:

A stroke is a medical emergency that occurs when part of the brain is not getting enough blood and stops working. It can be caused by an ischemic stroke, which is the most common type and results from a blood clot blocking or plugging a blood vessel in the brain, or by a hemorrhagic stroke, which involves bleeding in the brain. Depending on the area of the brain affected, a stroke can cause sudden weakness or numbness of the face, arm, or leg on one side of the body; sudden confusion, trouble talking, or trouble understanding; sudden dizziness, loss of balance, or trouble walking; sudden trouble seeing in one or both eyes or double vision; or sudden severe headache with no known cause. A transient ischemic attack (TIA), or "mini-stroke," is a brief interruption of blood supply to the brain and can be a warning sign of a more serious stroke [medquad_ans_000052

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0006276-3` (score=0.9953) - What causes uncombable hair syndrome? The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft that is seen in cross section in affected people. It has been s...
2. `medquad_ans_0000145-2__dup3` (score=0.0019) - The causes of Goodpasture syndrome are not fully understood. People who smoke or use hair dyes appear to be at increased risk for this condition. Exposure to hydrocarbon fumes, metallic dust, and certain drugs, such a...
3. `medquad_ans_0000247-3__dup1` (score=0.0011) - What causes alopecia universalis? The exact underlying cause of alopecia universalis (AU) is not currently known. AU is an advanced form of alopecia areata (AA), a condition that leads to round patches of hair loss. A...
4. `medquad_ans_0000417-1__dup1` (score=0.0002) - You lose up to 100 hairs from your scalp every day. That's normal, and in most people, those hairs grow back. But many men -- and some women -- lose hair as they grow older. You can also lose your hair if you have cer...
5. `medquad_ans_0002442-3` (score=0.0001) - What causes frontal fibrosing alopecia? The exact underlying cause of frontal fibrosing alopecia (FFA) is unknown. FFA is thought to be an autoimmune condition in which an affected person's immune system mistakenly at...

**Answer**:

The stiffness of the hair in uncombable hair syndrome (UHS) is likely due to the triangular shape of the hair shaft seen in cross section, which results from premature keratinization of the inner root sheath. This inner root sheath conforms to the abnormal outline of the hair shaft, forming an irregular, rigid tube that alters the shape of the emerging hair. While the condition is assumed to be autosomal dominant and thus due to mutations in a gene, no responsible gene has been identified [medquad_ans_0006276-3].

**Supported citations**: `medquad_ans_0006276-3`
**Unsupported citations**: _none_

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.897, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000706-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000371-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000557-4` (score=0.9943) - This condition is generally not inherited but arises from mutations in the body's cells that occur after conception. This alteration is called a somatic mutation and is present only in certain cells. Rarely, this cond...
4. `medquad_ans_0000629-4` (score=0.9884) - MIDD is inherited in a mitochondrial pattern, which is also known as maternal inheritance. This pattern of inheritance applies to genes contained in mtDNA. Because egg cells, but not sperm cells, contribute mitochondr...
5. `medquad_ans_0000887-4` (score=0.9860) - Most cases of Russell-Silver syndrome are sporadic, which means they occur in people with no history of the disorder in their family. Less commonly, Russell-Silver syndrome can run in families. In some affected famili...

**Answer**:

Russell-Silver syndrome can be inherited, but the pattern varies among families. In some affected families, it appears to have an autosomal dominant pattern of inheritance, where one copy of a genetic change in each cell is sufficient to cause the disorder. In other families, it has an autosomal recessive pattern of inheritance, which means both copies of the gene in each cell have mutations [medquad_ans_0000887-4]. Most cases, however, are sporadic and occur in people with no history of the disorder in their family [medquad_ans_0000887-4]. Therefore, inheritance of Russell-Silver syndrome is possible but not universal and depends on the family context.

**Supported citations**: `medquad_ans_0000887-4`
**Unsupported citations**: _none_

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=0.750, groundedness=0.958, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.9971) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000863-1__dup1` (score=0.0131) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
3. `medquad_ans_0000718-1__dup2` (score=0.0126) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
4. `medquad_ans_0000018-1__dup2` (score=0.0003) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects t...
5. `medquad_ans_0000049-22` (score=0.0002) - For many people, surgery helps relieve the pain and disability of osteoarthritis. Surgery may be performed to achieve one or more of the following. - Removal of loose pieces of bone and cartilage from the joint if the...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are different types of weight loss surgery, which often limit the amount of food you can take in. Some types also affect how you digest food and absorb nutrients. All types have risks and complications, such as infections, hernias, and blood clots. Many people who have the surgery lose weight quickly, but some regain weight later. Following diet and exercise recommendations can help keep most of the weight off. Lifelong medical follow-up is required [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.767, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000889-5` (score=0.9947) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
2. `medquad_ans_0000009-2__dup1` (score=0.9874) - There is no standard course of treatment for ACC. Treatment usually involves management of symptoms and seizures if they occur. Associated difficulties are much more manageable with early recognition and therapy, espe...
3. `medquad_ans_0000860-2__dup1` (score=0.0309) - How might Brown-Sequard syndrome be treated?
4. `medquad_ans_0000894-5` (score=0.0012) - These resources address the diagnosis or management of Schinzel-Giedion syndrome: - Genetic Testing Registry: Schinzel-Giedion syndrome These resources from MedlinePlus offer information about the diagnosis and manage...
5. `medquad_ans_0000075-2__dup2` (score=0.0006) - There is no standard course of treatment for chorea. Treatment depends on the type of chorea and the associated disease. Treatment for Huntington's disease is supportive, while treatment for Syndenham's chorea usually...

**Answer**:

The provided context does not specify the treatments for Saethre-Chotzen syndrome. It lists resources that address the diagnosis or management of Saethre-Chotzen syndrome, including the Genetic Testing Registry and MedlinePlus Encyclopedia entries on craniosynostosis and skull of a newborn, as well as general management resources like diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care [medquad_ans_0000889-5]. However, no specific treatments are detailed in the context. Therefore, based on the available information, the exact treatments for Saethre-Chotzen syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.480, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.9214) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
2. `medquad_ans_0000512-1` (score=0.2314) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
3. `medquad_ans_0000758-1__dup2` (score=0.1210) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000253-1__dup2` (score=0.0162) - Cystic fibrosis (CF) is an inherited disease of the mucus and sweat glands. It affects mostly your lungs, pancreas, liver, intestines, sinuses, and sex organs. CF causes your mucus to be thick and sticky. The mucus cl...
5. `medquad_ans_0000522-1__dup1` (score=0.0140) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...

**Answer**:

Idiopathic pulmonary fibrosis (IPF) is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, making them unable to transport oxygen into the bloodstream effectively. It usually affects people between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, gradual weight loss, and in some cases, widened and rounded tips of the fingers and toes (clubbing) due to oxygen shortage. IPF has no known cause in most cases and is referred to as "idiopathic." When more than one family member has the disease, it is called familial IPF. The disease progresses over time, leading to reduced oxygen supply to the body's organs and tissues, and it is associated with serious complications such as respiratory failure, pulmonary hypertension, heart failure, pulmonary embolism, pneumonia, and lung cancer. Currently,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.882, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000269-4__dup1` (score=0.9983) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes of the National Institutes of Health (NIH) conduct research in laboratories at the NIH and support additional research through g...
2. `medquad_ans_0000213-4__dup1` (score=0.9974) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
3. `medquad_ans_0000216-4` (score=0.9970) - The National Institute of Neurological Disorders and Stroke supports and conducts research on neurodegenerative disorders, such as neurosyphilis, in an effort to find ways to prevent, treat, and ultimately cure these ...
4. `medquad_ans_0000190-4__dup1` (score=0.9234) - The NINDS supports research on neurological disorders such as Melkersson-Rosenthal syndrome. Much of this research is aimed at increasing knowledge of these disorders and finding ways to treat, prevent, and ultimately...
5. `medquad_ans_0000273-4__dup1` (score=0.0951) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts TSC research in its laboratories at the National Institutes of Health (NIH) and also supports TSC research through grants to major medical i...

**Answer**:

Research on neuroleptic malignant syndrome is being conducted by the National Institute of Neurological Disorders and Stroke (NINDS), which supports research focused on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1].

**Supported citations**: `medquad_ans_0000213-4__dup1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.857, groundedness=0.898, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000221-8` (score=0.9934) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
2. `medquad_ans_0000221-10` (score=0.0012) - To prevent kidney stones, you need to know what caused your kidney stone. Your doctor may ask you to try to catch the kidney stone as it passes in your urine. The kidney stone can then be sent to a lab to find out wha...
3. `medquad_ans_0000221-4__dup2` (score=0.0001) - Doctors have found four main types of kidney stones: - The most common types of stones contain calcium. Calcium is a normal part of a healthy diet. Calcium that is not used by the bones and muscles goes to the kidneys...
4. `medquad_ans_0000204-9` (score=0.0001) - To prevent kidney stones, health care providers and their patients must understand what is causing the stones to form. Especially in children with suspected metabolic abnormalities or with recurrent stones, a 24-hour ...
5. `medquad_ans_0000203-4__dup2` (score=0.0001) - Kidney stones can form when substances in the urinesuch as calcium, oxalate, and phosphorusbecome highly concentrated. Certain foods may promote stone formation in people who are susceptible, but scientists do not bel...

**Answer**:

To diagnose kidney stones in adults, your doctor will perform a physical exam and ask about your medical history, including family history of kidney stones, diet, digestive problems, and other health issues. The doctor may conduct urine, blood, and imaging tests to complete the diagnosis. Urine tests can detect infections or substances in the urine that form stones. Blood tests can identify underlying conditions that lead to kidney stones. Imaging tests are used to locate kidney stones in the body and may also reveal issues that caused the stones to form [medquad_ans_0000221-8].

**Supported citations**: `medquad_ans_0000221-8`
**Unsupported citations**: _none_

### Example 118 (`0000041_1-1`)

**Question**: What is (are) Urethral Cancer ?

**Scores**: context_relevance=0.800, groundedness=0.536, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000041_1-1` (score=0.0700) - Key Points - Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. - There are different types of urethral cancer that begin in cells that line the urethra. - A history of ...
2. `medquad_ans_0000929-1__dup1` (score=0.0185) - The urethra is the tube that allows urine to pass out of the body. In men, it's a long tube that runs through the penis. It also carries semen in men. In women, it's short and is just above the vagina. Urethral proble...
3. `medquad_ans_0000036_1-1` (score=0.0131) - Key Points - Prostate cancer is a disease in which malignant (cancer) cells form in the tissues of the prostate. - Signs of prostate cancer include a weak flow of urine or frequent urination. - Tests that examine the ...
4. `medquad_ans_0000099-1__dup2` (score=0.0077) - The bladder is a hollow organ in your lower abdomen that stores urine. Bladder cancer occurs in the lining of the bladder. It is the sixth most common type of cancer in the United States. Symptoms include - Blood in y...
5. `medquad_ans_0000213-3__dup2` (score=0.0044) - Urinary retention can result from - obstruction of the urethra - nerve problems - medications - weakened bladder muscles Obstruction of the Urethra Obstruction of the urethra causes urinary retention by blocking the n...

**Answer**:

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. It is a rare cancer that occurs more often in men than in women. The urethra is the tube that carries urine from the bladder to outside the body; in women, it is about 1 inch long and is just above the vagina, while in men, it is about 8 inches long and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. There are different types of urethral cancer that begin in cells that line the urethra, including squamous cell carcinoma (the most common type), transitional cell carcinoma, and adenocarcinoma. Risk factors include a history of bladder cancer and chronic inflammation in the urethra due to conditions such as sexually transmitted diseases (including human papillomavirus type 16)

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000095-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000673-4` (score=0.9988) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
3. `medquad_ans_0006446-4` (score=0.9294) - Is Williams syndrome inherited?
4. `medquad_ans_0004784-4` (score=0.8859) - How is Pelizaeus-Merzbacher disease inherited?
5. `medquad_ans_0004671-4` (score=0.0251) - How is pachyonychia congenita inherited? Pachyonychia congenita (PC) is inherited in an autosomal dominant manner. This means that to be affected, a person only needs a change (mutation) in one copy of the responsible...

**Answer**:

The provided context does not contain information about the inheritance pattern of paramyotonia congenita. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.540, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.9987) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000125-2` (score=0.0001) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
3. `medquad_ans_0000125-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the BTD gene in each cell have mutations. The parents of an individual with biotinidase deficiency each carry one copy of the m...
4. `medquad_ans_0000735-1__dup1` (score=0.0001) - Biotinidase deficiency is an inherited disorder in which the body is unable to recycle the vitamin biotin. The disorder may become apparent in the first few months of life, or later in childhood. The more severe form ...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.714, groundedness=0.900, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000966-1__dup2` (score=0.9971) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, stre...
2. `medquad_ans_0000298-1__dup1` (score=0.0371) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...
3. `medquad_ans_0000331-1__dup1` (score=0.0012) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in th...
4. `medquad_ans_0000846-1__dup1` (score=0.0003) - Summary : Playing sports can be fun, but it can also be dangerous if you are not careful. You can help prevent injuries by - Getting a physical to make sure you are healthy before you start playing your sport - Wearin...
5. `medquad_ans_0000505-1__dup1` (score=0.0002) - Summary : We usually think of air pollution as being outdoors, but the air in your house or office could also be polluted. Sources of indoor pollution include - Mold and pollen - Tobacco smoke - Household products and...

**Answer**:

Yes, there is information about water pollution. The three most common pollutants that can harm rivers, streams, lakes, and oceans are soil, nutrients, and bacteria. Rain washes soil into water bodies, which can kill fish eggs, clog fish gills, and block light, causing plants to die. Nutrients from fertilizers cause algae growth, turning water green, especially in lakes, ponds, and reservoirs. Bacteria from sewage spills can pollute both fresh and salt water. To help protect water supplies, one should not pour household products like cleansers, medicines, or grease down the drain, and instead dispose of them at hazardous waste collection sites. Excess household grease, diapers, condoms, and personal hygiene products should be thrown in the garbage. Additionally, pet waste, which contains nutrients and germs, should be cleaned up after pets [medquad_ans_0000966-1__dup2].

**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.857, groundedness=0.973, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.9937) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000132-2__dup1` (score=0.0002) - Treatment depends on the specific cause of foot drop. The most common treatment is to support the foot with light-weight leg braces and shoe inserts, called ankle-foot orthotics. Exercise therapy to strengthen the mus...
3. `medquad_ans_0006002-1` (score=0.0001) - Talipes equinovarus is a congenital (present from birth) condition where the foot turns inward and downward. The cause of this condition is not known, although it may be passed down through families in some cases. Thi...
4. `medquad_ans_0000132-1__dup1` (score=0.0001) - Foot drop describes the inability to raise the front part of the foot due to weakness or paralysis of the muscles that lift the foot. As a result, individuals with foot drop scuff their toes along the ground or bend t...
5. `medquad_ans_0000132-4__dup1` (score=0.0001) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts research related to the neurological conditions that cause foot drop in its laboratories at the National Institutes of Health (NIH), and als...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000145-1` (score=0.7351) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
2. `medquad_ans_0000916-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
3. `medquad_ans_0000915-1__dup1` (score=0.3966) - Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the sku...
4. `medquad_ans_0000708-2` (score=0.0016) - The prevalence of this condition is unknown.
5. `medquad_ans_0000439-2` (score=0.0014) - This condition is very rare; its prevalence is unknown.

**Answer**:

Camurati-Engelmann disease is a genetic condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and hip bones are also affected. The thickened bones can lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. The age at which affected individuals first experience symptoms varies greatly; however, most people with this condition develop pain or weakness by adolescence. Camurati-Engelmann disease is caused by a mutation in the TGFB1 gene which is inherited in an autosomal dominant fashion. In some instances, people have the gene mutation that causes Camurati-Engelmann disease but never develop the characteristic features of this condition. In others, features are present, but a mutation cannot be identified. These cases are referred to as Camurati-Engelmann disease type

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.9994) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
2. `medquad_ans_0000316-4` (score=0.0001) - Emery-Dreifuss muscular dystrophy can have several different patterns of inheritance. When this condition is caused by mutations in the EMD gene, it is inherited in an X-linked recessive pattern. A condition is consid...
3. `medquad_ans_0000631-1__dup1` (score=0.0001) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
4. `medquad_ans_0002213-2` (score=0.0001) - What are the signs and symptoms of Facioscapulohumeral muscular dystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Facioscapulohumeral muscular dystrophy. If the information ...
5. `medquad_ans_0000316-1` (score=0.0001) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include Achilles tendon contracture, elbow flexion contracture, decreased cervical spine flexion due to contractures of posterior cervical muscles, pectus excavatum, elevated serum creatine phosphokinase, primary atrial arrhythmia, atrioventricular block, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, childhood onset, juvenile onset, and X-linked recessive inheritance [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.791, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000072-5` (score=0.9945) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
2. `medquad_ans_0004454-3` (score=0.0047) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
3. `medquad_ans_0004452-2` (score=0.0047) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
4. `medquad_ans_0004456-3` (score=0.0047) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...
5. `medquad_ans_0004451-3` (score=0.0047) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not contain information about the treatments for arterial tortuosity syndrome. Therefore, a specific answer cannot be generated based on the given resources. [medquad_ans_0000072-5] mentions resources for diagnosis or management of arterial tortuosity syndrome, such as Gene Review, Genetic Testing Registry, and Johns Hopkins McKusick-Nathans Institute of Genetic Medicine, but does not detail the actual treatments. Without explicit treatment information, a definitive response cannot be provided.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0001088-2__dup1` (score=0.0005) - What are the signs and symptoms of Charcot-Marie-Tooth disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease. If the information is available, the tabl...
3. `medquad_ans_0001105-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
4. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
5. `medquad_ans_0001112-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2O? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2O. If the information is ava...

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
4. `medquad_ans_0000962-1` (score=0.0001) - Supravalvular aortic stenosis (SVAS) is a heart defect that develops before birth. This defect is a narrowing (stenosis) of the large blood vessel that carries blood from the heart to the rest of the body (the aorta)....
5. `medquad_ans_0006376-2` (score=0.0001) - What are the signs and symptoms of Von Hippel-Lindau disease? Symptoms of Von Hippel-Lindau (VHL) disease vary among patients and depend on the size and location of the tumors. Hemangioblastomas that develop in the br...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis include exudative retinal detachment and pulmonic stenosis [medquad_ans_0005321-1].

**Supported citations**: `medquad_ans_0005321-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001838-1` (score=0.3892) - Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. Symptoms have varied greatly among the few reported cases. Dicarboxylic aminoaciduria is...
2. `medquad_ans_0000168-1__dup3` (score=0.0026) - Proteinuriaalso called albuminuria or urine albuminis a condition in which urine contains an abnormal amount of protein. Albumin is the main protein in the blood. Proteins are the building blocks for all body parts, i...
3. `medquad_ans_0002601-1` (score=0.0003) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...
4. `medquad_ans_0001860-1` (score=0.0003) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
5. `medquad_ans_0000434-5` (score=0.0001) - These resources address the diagnosis or management of GRACILE syndrome: - Genetic Testing Registry: GRACILE syndrome - MedlinePlus Encyclopedia: Aminoaciduria - MedlinePlus Encyclopedia: Cholestasis These resources f...

**Answer**:

Dicarboxylic aminoaciduria is a rare metabolic disorder characterized by the excessive loss of aspartate and glutamate in urine. It is caused by mutations in the SLC1A1 gene and is inherited in an autosomal recessive fashion [medquad_ans_0001838-1].

**Supported citations**: `medquad_ans_0001838-1`
**Unsupported citations**: _none_

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.870, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0005859-1` (score=0.0001) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevel...
2. `medquad_ans_0001034-2` (score=0.0001) - Weyers acrofacial dysostosis appears to be a rare disorder. Only a few affected families have been identified worldwide.
3. `medquad_ans_0000949-2` (score=0.0001) - Spondylothoracic dysostosis affects about one in 200,000 people worldwide. However, it is much more common in people of Puerto Rican ancestry, affecting approximately one in 12,000 people.
4. `medquad_ans_0000949-1` (score=0.0001) - Spondylothoracic dysostosis is a condition characterized by the malformation of the bones of the spine and ribs. The bones of the spine (vertebrae) do not develop properly, which causes them to be misshapen and abnorm...
5. `medquad_ans_0000949-5` (score=0.0001) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...

**Answer**:

Yes, spondylothoracic dysostosis is inherited in an autosomal recessive manner [medquad_ans_0005859-1]. This is supported by the context stating that spondylospinal thoracic dysostosis, which is similar to spondylothoracic dysostosis, is believed to be inherited in an autosomal recessive manner [medquad_ans_0005859-1]. Additionally, the condition is referred to as spondylocostal dysostosis, and gene review resources indicate it is inherited in an autosomal recessive pattern [medquad_ans_0000949-5].

**Supported citations**: `medquad_ans_0005859-1`, `medquad_ans_0000949-5`
**Unsupported citations**: _none_

### Example 130 (`0003296-1`)

**Question**: What is (are) Jervell Lange-Nielsen syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.950, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000334-1__dup1` (score=0.9409) - Andermann syndrome (AS) is a disorder that damages the nerves used for muscle movement and sensation (motor and sensory neuropathy). Agenesis or malformation of the corpus callosum also occurs in most people with this...
2. `medquad_ans_0000542-1` (score=0.9244) - Jervell and Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). This disorder is a form of long QT syndrome, which is a heart ...
3. `medquad_ans_0003296-1` (score=0.8756) - Jervell Lange-Nielsen syndrome is a form of long QT syndrome. Symptoms include deafness from birth, arrhythmia, fainting, and sudden death. There are two different types, Jervell Lange-Nielsen syndrome type 1 and 2. I...
4. `medquad_ans_0006442-1` (score=0.0129) - Wiedemann-Steiner syndrome is a rare genetic condition characterized by distinctive facial features, hairy elbows, short stature, and intellectual disability. This condition is caused by changes (mutations) in the KMT...
5. `medquad_ans_0002533-1` (score=0.0026) - Gerstmann-Straussler-Scheinker disease (GSS) is a type of prion disease, which is a group of conditions that affect the nervous system. Signs and symptoms generally develop between ages 35 and 50 years and may include...

**Answer**:

Jervell Lange-Nielsen syndrome is a condition that causes profound hearing loss from birth and a disruption of the heart's normal rhythm (arrhythmia). It is a form of long QT syndrome, which causes the heart's cardiac muscle to take longer than usual to recharge between beats. This leads to an increased risk of fainting (syncope) and sudden death. The syndrome is inherited in an autosomal recessive fashion and has two types: type 1 and type 2 [medquad_ans_0000542-1][medquad_ans_0003296-1].

**Supported citations**: `medquad_ans_0000542-1`, `medquad_ans_0003296-1`
**Unsupported citations**: _none_

### Example 131 (`0000218-2`)

**Question**: How many people are affected by congenital cataracts, facial dysmorphism, and neuropathy ?

**Scores**: context_relevance=0.917, groundedness=0.778, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000507-2` (score=0.9991) - The prevalence of hypomyelination and congenital cataract is unknown.
2. `medquad_ans_0000218-1` (score=0.0001) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
3. `medquad_ans_0001075-1__dup1` (score=0.0001) - Cerulean cataracts are opaque areas that develop in the lens of the eye that often have a bluish or whitish color. They may be present at birth or develop in very early childhood, but may not be diagnosed until adulth...
4. `medquad_ans_0000507-1` (score=0.0001) - Hypomyelination and congenital cataract is an inherited condition that affects the nervous system and the eyes. This disease is one of a group of genetic disorders called leukoencephalopathies. Leukoencephalopathies i...
5. `medquad_ans_0000034-6__dup1` (score=0.0001) - Peripheral neuropathy, also called distal symmetric neuropathy or sensorimotor neuropathy, is nerve damage in the arms and legs. Feet and legs are likely to be affected before hands and arms. Many people with diabetes...

**Answer**:

The number of people affected by congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is unknown based on the provided context. [medquad_ans_0000218-1] states that CCFDN is a rare disorder, but it does not provide specific prevalence data. [medquad_ans_0000507-2] notes that the prevalence of hypomyelination and congenital cataract is unknown, but this does not directly address the prevalence of CCFDN. Therefore, a specific number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000218-1`, `medquad_ans_0000507-2`
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

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002622-2` (score=0.9994) - What are the signs and symptoms of Glycogen storage disease type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 6. If the information is available, ...
2. `medquad_ans_0000427-1` (score=0.0001) - Glycogen storage disease type VI (also known as GSDVI or Hers disease) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in liver cells. A lack of glycogen breakdown interfe...
3. `medquad_ans_0002617-2` (score=0.0001) - What are the signs and symptoms of Glycogen storage disease type 1B? The Human Phenotype Ontology provides the following list of signs and symptoms for Glycogen storage disease type 1B. If the information is available...
4. `medquad_ans_0002622-1` (score=0.0001) - Glycogen storage disease type 6 is a genetic disease in which the liver cannot process sugar properly. Symptoms usually begin in infancy or childhood and include low blood sugar (hypoglycemia), an enlarged liver (hepa...
5. `medquad_ans_0000422-1` (score=0.0001) - Glycogen storage disease type I (also known as GSDI or von Gierke disease) is an inherited disorder caused by the buildup of a complex sugar called glycogen in the body's cells. The accumulation of glycogen in certain...

**Answer**:

The symptoms of Glycogen storage disease type 6 (GSD VI) include hypoglycemia in approximately 90% of patients, short stature in approximately 90% of patients, hepatomegaly (enlarged liver), increased hepatic glycogen content, and postnatal growth retardation. These symptoms typically begin in infancy to early childhood, with the first sign often being hepatomegaly. Hypoglycemia and lactic acidosis may occur during prolonged fasting. The signs and symptoms tend to improve with age, and most adults with GSD VI do not have significant health problems [medquad_ans_0002622-2][medquad_ans_0000427-1].

**Supported citations**: `medquad_ans_0002622-2`, `medquad_ans_0000427-1`
**Unsupported citations**: _none_

### Example 134 (`0000070-5__dup3`)

**Question**: What causes Causes of Diabetes ?

**Scores**: context_relevance=0.750, groundedness=0.878, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000027-5__dup1` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...
2. `medquad_ans_0000037-5__dup2` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...
3. `medquad_ans_0000027-6` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...
4. `medquad_ans_0000070-5__dup3` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...
5. `medquad_ans_0000070-6__dup1` (score=0.9911) - Other types of diabetes have a variety of possible causes. Genetic Mutations Affecting Beta Cells, Insulin, and Insulin Action Some relatively uncommon forms of diabetes known as monogenic diabetes are caused by mutat...

**Answer**:

Diabetes can be caused by several factors, including:

- Genetic mutations affecting beta cells, insulin, and insulin action: Monogenic diabetes, such as neonatal diabetes mellitus (NDM) and maturity-onset diabetes of the young (MODY), is caused by mutations in a single gene, often inherited or occurring spontaneously. These mutations reduce the beta cells' ability to produce insulin or impair the quality of insulin or insulin receptor function [medquad_ans_0000027-5__dup1].

- Other genetic diseases: Diabetes occurs at higher rates in individuals with Down syndrome, Klinefelter syndrome, and Turner syndrome. Cystic fibrosis and hemochromatosis are also linked to diabetes; cystic fibrosis leads to thick mucus blocking the pancreas, increasing diabetes risk with age, while hemochromatosis causes iron buildup that can damage the pancreas if untreated [medquad_ans_00

**Supported citations**: `medquad_ans_0000027-5__dup1`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=0.636, groundedness=0.269, answer_relevance=0.709

**Top docs**:

1. `medquad_ans_0000016-21` (score=0.9970) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
2. `medquad_ans_0000016-19` (score=0.0032) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
3. `medquad_ans_0000016-9` (score=0.0031) - Blood vessels damaged from diabetic retinopathy can cause vision loss in two ways. Fragile, abnormal blood vessels can develop and leak blood into the center of the eye, blurring vision. This is proliferative retinopa...
4. `medquad_ans_0000016-10` (score=0.0019) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year. Learn more about a comprehensive dilated eye exam.
5. `medquad_ans_0000016-4__dup2` (score=0.0006) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...

**Answer**:

Research being done for diabetic retinopathy includes studies conducted by the National Eye Institute (NEI) to find better ways to detect, treat, and prevent vision loss in people with diabetes. This research is carried out through laboratory studies and patient-based studies. For example, researchers are investigating drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery. Additionally, research has found that prompt treatment of macular edema with anti-VEGF drugs, with or without laser treatment, results in better vision than laser treatment alone or steroid injections. These drugs reduce fluid leakage and interfere with the growth of new blood vessels in the retina [medquad_ans_0000016-21, medquad_ans_0000016-4__dup2].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000016-21, medquad_ans_0000016-4__dup2`

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.938, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.9100) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000007-3__dup1` (score=0.0093) - Many conditions or factors can directly or indirectly injure the lungs and lead to ARDS. Some common ones are: Sepsis. This is a condition in which bacteria infect the bloodstream. Pneumonia. This is an infection in t...
3. `medquad_ans_0000007-4__dup1` (score=0.0057) - People at risk for ARDS have a condition or illness that can directly or indirectly injure their lungs. Direct Lung Injury Conditions that can directly injure the lungs include: Pneumonia. This is an infection in the ...
4. `medquad_ans_0000007-1__dup1` (score=0.0031) - ARDS, or acute respiratory distress syndrome, is a lung condition that leads to low oxygen levels in the blood. ARDS can be life threatening because your body's organs need oxygen-rich blood to work well. People who d...
5. `medquad_ans_0000076-3__dup2` (score=0.0015) - Repeatedly breathing in foreign substances can cause hypersensitivity pneumonitis (HP). Examples of these substances include molds, dusts, and chemicals. (Mold often is the cause of HP.) These substances also are know...

**Answer**:

Inhalation injuries are injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes, which can damage the eyes and respiratory system and worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. In individuals with asthma, inhalation injuries may exacerbate the condition [medquad_ans_0000516-1__dup1]. Additionally, breathing in harmful fumes or smoke is listed as a direct cause of lung injury that can contribute to conditions like ARDS [medquad_ans_0000007-4__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`, `medquad_ans_0000007-4__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.375, groundedness=0.000, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
4. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.
5. `medquad_ans_0000407-2` (score=0.9993) - Gilbert syndrome is a common condition that is estimated to affect 3 to 7 percent of Americans.

**Answer**:

The provided context does not contain information about the incidence or prevalence of Zollinger-Ellison Syndrome. Therefore, it is not possible to determine how many people are affected by this condition based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.739, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.9949) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
2. `medquad_ans_0000033-2__dup1` (score=0.9923) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
3. `medquad_ans_0001859-5` (score=0.0013) - How might dihydrolipoamide dehydrogenase deficiency be treated? There are currently no consensus recommendations for the management of dihydrolipoamide dehydrogenase (DLD) deficiency. Management can be hard because va...
4. `medquad_ans_0000860-2__dup1` (score=0.0003) - How might Brown-Sequard syndrome be treated?
5. `medquad_ans_0004451-3` (score=0.0001) - How might Noonan syndrome be treated? Management generally focuses on the specific signs and symptoms present in each person. Treatments for the complications of Noonan syndrome (such as cardiovascular abnormalities) ...

**Answer**:

The provided context does not specify the treatments for tetrahydrobiopterin deficiency. It lists diagnostic resources and information about the condition, such as genetic testing and serum phenylalanine screening, but does not detail any therapeutic interventions [medquad_ans_0000974-5]. Therefore, based on the given information, the treatments for tetrahydrobiopterin deficiency cannot be determined.

**Supported citations**: `medquad_ans_0000974-5`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.125, groundedness=0.710, answer_relevance=0.325

**Top docs**:

1. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
2. `medquad_ans_0000989-2` (score=0.9995) - This condition affects an estimated 1 in 50,000 people.
3. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
4. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
5. `medquad_ans_0000708-2` (score=0.9995) - The prevalence of this condition is unknown.

**Answer**:

The incidence of nonsyndromic holoprosencephaly is estimated to range from 1 in 40,000 to 50,000 people, with some estimates suggesting 1 in 50,000 or as low as 1 in 87,000 people. Another estimate indicates it affects 1 to 3 per million people worldwide [medquad_ans_0000195-2][medquad_ans_0000989-2][medquad_ans_0000484-2][medquad_ans_0000770-2]. The exact prevalence remains uncertain [medquad_ans_0000708-2].

**Supported citations**: `medquad_ans_0000195-2`, `medquad_ans_0000989-2`, `medquad_ans_0000484-2`, `medquad_ans_0000770-2`, `medquad_ans_0000708-2`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.636, groundedness=0.047, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000525-3` (score=0.0013) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
2. `medquad_ans_0000525-2` (score=0.0001) - Intrahepatic cholestasis of pregnancy is estimated to affect 1 percent of women of Northern European ancestry. The condition is more common in certain populations, such as women of Araucanian Indian ancestry in Chile ...
3. `medquad_ans_0003240-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile ...
4. `medquad_ans_0000525-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy is a liver disorder that occurs in pregnant women. Cholestasis is a condition that impairs the release of a digestive fluid called bile from liver cells. As a result, bile builds ...
5. `medquad_ans_0003239-1` (score=0.0001) - Intrahepatic cholangiocarcinoma is a cancer that develops in the cells within the bile ducts; both inside and outside the liver. The terms cholangiocarinoma and bile duct cancer are often used to refer to the same con...

**Answer**:

The provided context does not mention progressive familial intrahepatic cholestasis or describe genetic changes related to this condition. Therefore, based on the available information, it is not possible to answer the question. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.9994) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0000108-2__dup1` (score=0.0001) - What are the signs and symptoms of Acute alcohol sensitivity ? The Human Phenotype Ontology provides the following list of signs and symptoms for Acute alcohol sensitivity . If the information is available, the table ...
4. `medquad_ans_0003814-2` (score=0.0001) - What are the signs and symptoms of Marshall-Smith syndrome? Marshall-Smith syndrome is characterized by accelerated skeletal maturation, relative failure to thrive, respiratory difficulties, mental retardation, and un...
5. `medquad_ans_0002480-2` (score=0.0001) - What are the signs and symptoms of Galloway-Mowat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Galloway-Mowat syndrome. If the information is available, the table below ...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=0.714, groundedness=0.066, answer_relevance=0.943

**Top docs**:

1. `medquad_ans_0000324-4` (score=0.9982) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
2. `medquad_ans_0006597-4` (score=0.9924) - Can 48,XXYY syndrome be inherited?
3. `medquad_ans_0000228-4` (score=0.0001) - The various syndromes of which congenital hepatic fibrosis is often a feature can have different inheritance patterns. Most of these disorders are inherited in an autosomal recessive pattern, which means both copies o...
4. `medquad_ans_0005558-3` (score=0.0001) - How is severe combined immunodeficiency (SCID) diagnosed? A diagnosis of severe combined immunodeficiency (SCID) may be suspected if a baby shows any of the following persistent symptoms within the first year of life:...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

The provided context does not contain information about the inheritance of hepatic veno-occlusive disease with immunodeficiency. Therefore, it is not possible to determine based on the given text whether this condition is inherited.

[no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.966, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.9993) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0000057-4__dup3` (score=0.0117) - Symptoms Vary With Disorders There are several types of taste disorders depending on how the sense of taste is affected. People who have taste disorders usually lose their ability to taste or can no longer perceive ta...
3. `medquad_ans_0000013_3-3__dup2` (score=0.0021) - Signs and symptoms of chronic myelomonocytic leukemia include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by CMML or by other conditions. Check with your doctor if you ...
4. `medquad_ans_0000013_3-2__dup4` (score=0.0015) - Signs and symptoms of juvenile myelomonocytic leukemia include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by JMML or by other conditions. Check with your doctor if you...
5. `medquad_ans_0000001_1-2` (score=0.0008) - Signs and symptoms of adult ALL include fever, feeling tired, and easy bruising or bleeding. The early signs and symptoms of ALL may be like the flu or other common diseases. Check with your doctor if you have any of ...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. [medquad_ans_0005902-1]

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.944, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000159-7` (score=0.9732) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
2. `medquad_ans_0000159-8` (score=0.9253) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
3. `medquad_ans_0000159-4__dup2` (score=0.0088) - Lupus nephritis is diagnosed through urine and blood tests and a kidney biopsy: - Urinalysis. Urinalysis is testing of a urine sample. The urine sample is collected in a special container in a health care providers of...
4. `medquad_ans_0000159-6` (score=0.0006) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
5. `medquad_ans_0000186-2__dup1` (score=0.0003) - There is no cure for lupus. Treatment is symptomatic. With a combination of medication, rest, exercise, proper nutrition, and stress management, most individuals with lupus can often achieve remission or reduce their ...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent further damage to the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication such as cyclophosphamide or mycophenolate mofetil, typically used in combination with prednisone [medquad_ans_0000159-8]. If lupus nephritis is causing high blood pressure, medications to lower blood pressure may be necessary, as this can significantly slow the progression of kidney disease [medquad_ans_0000159-8]. Additionally, people with kidney disease may need to consult a health care provider about dietary changes, such as reducing sodium intake, especially if they have high blood pressure [medquad_ans_0000159-7]. Diagnosis is typically confirmed through urine and blood tests,

**Supported citations**: `medquad_ans_0000159-8`, `medquad_ans_0000159-7`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.519, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000233-2__dup1` (score=0.9969) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...
2. `medquad_ans_0000246-2__dup2` (score=0.9927) - For those with mild to moderate symptoms, many physicians suggest certain lifestyle changes and activities to reduce or eliminate symptoms. Decreased use of caffeine, alcohol, and tobacco may provide some relief. Phys...
3. `medquad_ans_0000869-5` (score=0.9811) - These resources address the diagnosis or management of restless legs syndrome: - Agency for Healthcare Research and Quality: Options for Treating Restless Legs Syndrome - Genetic Testing Registry: Restless legs syndro...
4. `medquad_ans_0000114-6` (score=0.9709) - Restless legs syndrome (RLS) has no cure. If a condition or medicine triggers RLS, it may go away or get better if the trigger is relieved or stopped. RLS can be treated. The goals of treatment are to: Prevent or reli...
5. `medquad_ans_0000114-5__dup1` (score=0.0009) - Your doctor will diagnose restless legs syndrome (RLS) based on your signs and symptoms, your medical and family histories, a physical exam, and test results. Your doctor will use this information to rule out other co...

**Answer**:

Treatments for Restless Legs Syndrome (RLS) include lifestyle changes and medications. Lifestyle changes may involve decreasing the use of caffeine, alcohol, and tobacco, which can provide some relief. Other helpful measures include taking a hot bath, massaging the legs, or using a heating pad or ice pack. In some cases, physicians may suggest supplements to correct deficiencies in iron, folate, and magnesium. For mild to moderate symptoms, lifestyle changes and periodic use of medicines are often sufficient. For more severe RLS, daily medications are typically prescribed. Approved medications by the U.S. Food and Drug Administration include dopaminergics (such as ropinirole and pramipexole), benzodiazepines (central nervous system depressants), opioids, and anticonvulsants (such as gabapentin enacarbil and rotigotine). The Relaxis pad, which provides 30 minutes of vibrations at the site of

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=0.750, groundedness=0.993, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002574-1` (score=0.9311) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
2. `medquad_ans_0000127-1__dup3` (score=0.8372) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
3. `medquad_ans_0002573-1` (score=0.6410) - Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue th...
4. `medquad_ans_0002573-5` (score=0.0010) - Is genetic testing available for glioblastoma? Genetic testing is not available for many people with glioblastoma since most of these tumors occur sporadically (by chance) and are not caused by a genetic mutation. How...
5. `medquad_ans_0002575-1` (score=0.0004) - Gliomatosis cerebri is a type of brain cancer. It is a variant form of glioblastoma multiforme. It is characterized by scattered and widespread tumor cells that can cause the cerebrum, cerebellum, or brain stem to enl...

**Answer**:

Glioblastoma is a malignant (cancerous) brain tumor that develops from a specific type of brain cell called an astrocyte. These cells help support and nourish neurons (nerve cells of the brain) and form scar tissue that helps repair brain damage in response to injury. Glioblastomas are often very aggressive and grow into surrounding brain tissue. Signs and symptoms, such as headache, nausea, vomiting, and/or drowsiness, may develop when the tumor begins to put excess pressure on the brain. Affected people may also experience other features depending on the size and location of the tumor. In most cases, the exact underlying cause is unknown; however, they can rarely occur in [medquad_ans_0002573-1].

**Supported citations**: `medquad_ans_0002573-1`
**Unsupported citations**: _none_

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0005496-3` (score=0.9944) - What causes schizencephaly? The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, s...
2. `medquad_ans_0000251-4__dup1` (score=0.0004) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to pre...
3. `medquad_ans_0000189-4__dup1` (score=0.0001) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes of the National Institutes of Health (NIH) conduct research related to megalencephaly in laboratories at the NIH and also suppor...
4. `medquad_ans_0000064-1__dup1` (score=0.0001) - Cephalic disorders are congenital conditions that stem from damage to or abnormal development of the budding nervous system. Most cephalic disorders are caused by a disturbance that occurs very early in the developmen...
5. `medquad_ans_0000251-3__dup1` (score=0.0001) - The prognosis for individuals with schizencephaly varies depending on the size of the clefts and the extent of neurological disabilities.

**Answer**:

The exact cause of schizencephaly is unknown. A small number of people with schizencephaly are found to have changes (mutations) in one of four genes: EMX2, SIX3, SHH, and COL4A1. Rarely, schizencephaly can affect more than one family member, supporting a genetic cause in some cases. Schizencephaly has also been linked to a variety of non-genetic factors, including young maternal age and certain medications and infections that can cause vascular disruptions (disruption of blood flow or blood supply) in a developing baby [medquad_ans_0005496-3].

**Supported citations**: `medquad_ans_0005496-3`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.800, groundedness=0.884, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000309-5__dup1` (score=0.0001) - Is genetic testing available for amyotrophic lateral sclerosis? Yes. Clinical genetic testing is currently available for several genes in which mutations are known to cause ALS. Genetic testing on a research basis is ...
2. `medquad_ans_0000551-5` (score=0.0001) - These resources address the diagnosis or management of juvenile primary lateral sclerosis: - Gene Review: Gene Review: ALS2-Related Disorders - Genetic Testing Registry: Juvenile primary lateral sclerosis These resour...
3. `medquad_ans_0000028-1__dup3` (score=0.0001) - Amyotrophic lateral sclerosis (ALS) is a nervous system disease that attacks nerve cells called neurons in your brain and spinal cord. These neurons transmit messages from your brain and spinal cord to your voluntary ...
4. `medquad_ans_0000316-1__dup1` (score=0.0001) - What are the signs and symptoms of Amyotrophic lateral sclerosis type 6? The Human Phenotype Ontology provides the following list of signs and symptoms for Amyotrophic lateral sclerosis type 6. If the information is a...
5. `medquad_ans_0000309-1__dup1` (score=0.0001) - Amyotrophic lateral sclerosis (ALS), also referred to as "Lou Gehrig's disease," is a progressive motor neuron disease which leads to problems with muscle control and movement. There are various types of ALS, which ar...

**Answer**:

Genetic testing is available for several genes in which mutations are known to cause amyotrophic lateral sclerosis (ALS). Additionally, genetic testing on a research basis is available for select susceptibility genes associated with ALS [medquad_ans_0000309-5__dup1]. However, the specific genetic changes or mutations related to ALS are not detailed in the provided context. For a comprehensive list of genes and their associated mutations, one may refer to the GeneTests website, which provides information on laboratories offering clinical and research genetic testing for ALS [medquad_ans_0000309-5__dup1].

**Supported citations**: `medquad_ans_0000309-5__dup1`
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.920, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000581-5` (score=0.9964) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
2. `medquad_ans_0000033-2__dup1` (score=0.9946) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
3. `medquad_ans_0003473-6` (score=0.9531) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
4. `medquad_ans_0000860-2__dup1` (score=0.1584) - How might Brown-Sequard syndrome be treated?
5. `medquad_ans_0000406-5__dup1` (score=0.0115) - What treatment is available for antisynthetase syndrome? Corticosteroids are typically the first-line of treatment and may be required for several months or years. These medications are often given orally; however, in...

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

**Scores**: context_relevance=0.857, groundedness=0.817, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000116-7` (score=0.9878) - Treatment depends on the severity of symptoms. The gastroenterologist will - review the medications the person is taking - make recommendations to change or stop certain medications - recommend that the person quit sm...
2. `medquad_ans_0000116-8` (score=0.9757) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during ...
3. `medquad_ans_0001421-3` (score=0.9645) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
4. `medquad_ans_0006236-5` (score=0.1558) - How might tubular aggregate myopathy be treated?
5. `medquad_ans_0000163-2__dup1` (score=0.0095) - The chronic inflammatory myopathies cant be cured in most adults but many of the symptoms can be treated. Options include medication, physical therapy, exercise, heat therapy (including microwave and ultrasound), orth...

**Answer**:

The provided context does not contain specific information about the treatments for collagenous colitis. While it details treatments for microscopic colitis, including medications such as antidiarrheals, corticosteroids, anti-inflammatory drugs, cholestyramine, antibiotics, immunomodulators, and anti-TNF therapies, as well as dietary changes and, in rare cases, surgery, there is no direct information on treatments for collagenous colitis. Therefore, based on the given context, the treatments for collagenous colitis cannot be determined.

[medquad_ans_0000116-7] [medquad_ans_0000116-8]

**Supported citations**: `medquad_ans_0000116-7`, `medquad_ans_0000116-8`
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.778, groundedness=0.970, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000418-7` (score=0.8627) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
2. `medquad_ans_0000418-1__dup2` (score=0.0259) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
3. `medquad_ans_0000904-1__dup2` (score=0.0200) - Your teeth are made of a hard, bonelike material. Inside the tooth are nerves and blood vessels. You need your teeth for many activities you may take for granted. These include eating, speaking and even smiling. But t...
4. `medquad_ans_0000396-3` (score=0.0161) - The FH gene provides instructions for making an enzyme called fumarase (also known as fumarate hydratase). Fumarase participates in an important series of reactions known as the citric acid cycle or Krebs cycle, which...
5. `medquad_ans_0000015-8__dup1` (score=0.0112) - Angiostrongylus cantonensis Prevention of A. cantonensis infections involves educating persons residing in or traveling to areas where the parasite is found about not ingesting raw or undercooked snails and slugs, fre...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis), wash your hands with warm water and soap after handling raw meat. Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; therefore, these methods are not reliable. Freeze pork less than 6 inches thick for 20 days at 5°F (-15°C) to kill any worms. However, freezing wild game meats may not effectively kill all worms because some worm species that infect wild game animals are freeze-resistant. Clean meat grinders thoroughly after each use. To help prevent Trichinella infection in animal populations, do not allow pigs or wild animals to eat uncooked meat, scraps, or carcasses of any animals, including rats, which may be infected with Trichinella [medquad_ans_0000418-7]. Trichinellosis is caused by eating

**Supported citations**: `medquad_ans_0000418-7`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=1.000, groundedness=0.944, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000536-3` (score=0.9939) - What causes autoimmune hepatitis? Although the exact cause of autoimmune hepatitis is unknown, evidence suggests that liver injury in a patient with autoimmune hepatitis is the result of a cell-mediated immunologic at...
2. `medquad_ans_0000192-9` (score=0.9926) - Some cases of viral hepatitis cannot be attributed to the hepatitis A, B, C, D, or E viruses, or even the less common viruses that can infect the liver, such as cytomegalovirus, Epstein-Barr virus, herpesvirus, parvov...
3. `medquad_ans_0000079-3__dup2` (score=0.0058) - In most cases, an autoimmune response is thought to cause immune thrombocytopenia (ITP). Normally, your immune system helps your body fight off infections and diseases. In ITP, however, your immune system attacks and ...
4. `medquad_ans_0000193-2__dup3` (score=0.0005) - Causes of cirrhosis include - heavy alcohol use - some drugs, medicines, and harmful chemicals - infections - chronic hepatitis B, C, or Dviral infections that attack the liver - autoimmune hepatitis, which causes the...
5. `medquad_ans_0000530-3__dup1` (score=0.0005) - What causes autoimmune atrophic gastritis? Autoimmune atrophic gastritis is considered an autoimmune disorder. In people who are affected by this condition, the immune system mistakenly attacks the healthy cells of th...

**Answer**:

Autoimmune hepatitis is caused by a cell-mediated immunologic attack on the liver, which may be triggered by genetic factors, viral infections, or chemical agents. The exact cause is unknown, but evidence suggests a genetic component, as autoimmune hepatitis sometimes occurs in relatives of people with autoimmune diseases [medquad_ans_0000536-3].

**Supported citations**: `medquad_ans_0000536-3`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.948, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.9580) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0004883-1` (score=0.0078) - Pilocytic astrocytoma is an often benign, slow-growing tumor of the brain or spinal cord. The tumor may be in the form of a cyst and usually does not spread to nearby tissues. Symptoms vary depending upon the size and...
3. `medquad_ans_0000006_5-1` (score=0.0029) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...
4. `medquad_ans_0000006_3-3` (score=0.0026) - Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk factors doesnt mean that you will not get cancer. Talk with ...
5. `medquad_ans_0000006_2-1` (score=0.0024) - Key Points - A childhood brain or spinal cord tumor is a disease in which abnormal cells form in the tissues of the brain or spinal cord. - The brain controls many important body functions. - The spinal cord connects ...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.857, groundedness=0.975, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000013_3-1__dup3` (score=0.8728) - Key Points - Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. - Signs and symptoms of atypical chronic myelogenous leukemia in...
2. `medquad_ans_0000013_2-2__dup3` (score=0.0018) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.
3. `medquad_ans_0000013_2-1__dup3` (score=0.0004) - Chronic myelogenous leukemia is a disease in which too many white blood cells are made in the bone marrow. See the PDQ summary on Chronic Myelogenous Leukemia Treatment for information on diagnosis, staging, and treat...
4. `medquad_ans_0000013_2-1__dup2` (score=0.0001) - Chronic neutrophilic leukemia is a disease in which too many blood stem cells become a type of white blood cell called neutrophils. Neutrophils are infection -fighting blood cells that surround and destroy dead cells ...
5. `medquad_ans_0000013_3-2__dup3` (score=0.0001) - Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak. These and other signs and symptoms may be caused by atypical CML or by other conditions. Check ...

**Answer**:

Atypical chronic myelogenous leukemia is a disease in which too many granulocytes (immature white blood cells) are made in the bone marrow. In atypical chronic myelogenous leukemia (CML), the body tells too many blood stem cells to become a type of white blood cell called granulocytes. Some of these blood stem cells never become mature white blood cells. These immature white blood cells are called blasts. Over time, the granulocytes and blasts crowd out the red blood cells and platelets in the bone marrow. The leukemia cells in atypical CML and CML look alike under a microscope. However, in atypical CML a certain chromosome change, called the "Philadelphia chromosome" is not present [medquad_ans_0000013_3-1__dup3]. Signs and symptoms of atypical chronic myelogenous leukemia include easy bruising or bleeding and feeling tired and weak [medquad

**Supported citations**: `medquad_ans_0000013_3-1__dup3`
**Unsupported citations**: _none_

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.625, groundedness=0.907, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0000996-3` (score=0.9960) - Most cases of trisomy 13 result from having three copies of chromosome 13 in each cell in the body instead of the usual two copies. The extra genetic material disrupts the normal course of development, causing the cha...
2. `medquad_ans_0000996-4` (score=0.0003) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...
3. `medquad_ans_0000770-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0001083-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000784-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The genetic changes related to trisomy 13 include having three copies of chromosome 13 in each cell instead of the usual two copies, which disrupts normal development [medquad_ans_0000996-3]. This can occur due to nondisjunction during the formation of reproductive cells (eggs and sperm), resulting in a reproductive cell with an extra copy of chromosome 13 [medquad_ans_0000996-4]. In some cases, trisomy 13 results from a translocation, where a part of chromosome 13 becomes attached to another chromosome during the formation of reproductive cells or early fetal development [medquad_ans_0000996-3]. In rare instances, only part of chromosome 13 is present in three copies, leading to potentially different physical signs and symptoms [medquad_ans_0000996-3

**Supported citations**: `medquad_ans_0000996-3`, `medquad_ans_0000996-4`
**Unsupported citations**: _none_

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.714, groundedness=0.911, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.9866) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - G...
2. `medquad_ans_0000218-2__dup1` (score=0.9733) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bo...
3. `medquad_ans_0004415-3` (score=0.8279) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
4. `medquad_ans_0004784-5` (score=0.0503) - How might Pelizaeus-Merzbacher disease be treated?
5. `medquad_ans_0000168-4__dup1` (score=0.0100) - The NINDS supports research on neuromuscular disorders such as KSS. The goals of this research are to increase understanding of these disorders, and to find ways to prevent, treat, and, ultimately, cure them. The most...

**Answer**:

There is currently no cure for Niemann-Pick disease. Treatment is supportive and may include care from specialists such as pulmonologists for respiratory problems, cardiologists for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists for gastrointestinal issues. Learning specialists may also be involved to support cognitive development. Bone marrow transplantation has been attempted in a few individuals with type B, and enzyme replacement and gene therapies might be helpful for those with type B. However, restricting one's diet does not prevent the buildup of lipids in cells and tissues. There is no effective treatment for persons with type A [medquad_ans_0000218-2\_\_dup1][medquad_ans_0004415-3].

**Supported citations**: `medquad_ans_0000218-2__dup1`, `medquad_ans_0004415-3`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.9994) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0002679-2` (score=0.0001) - What are the signs and symptoms of Greig cephalopolysyndactyly syndrome? The symptoms of Greig cephalopolysyndactyly syndrome (GCPS) are highly variable, ranging from mild to severe. People with this condition typical...
3. `medquad_ans_0000270-4` (score=0.0001) - Most cases of Dandy-Walker malformation are sporadic, which means they occur in people with no history of the disorder in their family. A small percentage of cases seem to run in families; however, Dandy-Walker malfor...
4. `medquad_ans_0000270-3` (score=0.0001) - Researchers have found mutations in a few genes that are thought to cause Dandy-Walker malformation, but these mutations account for only a small number of cases. Dandy-Walker malformation has also been associated wit...
5. `medquad_ans_0000094-1__dup2` (score=0.0001) - Dandy-Walker Syndrome is a congenital brain malformation involving the cerebellum (an area of the back of the brain that coordinates movement) and the fluid-filled spaces around it. The key features of this syndrome a...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, and wide nasal bridge [

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003789-3` (score=0.9941) - What causes Mantle cell lymphoma? Most lymphomas are not inherited, but rather an acquired disease in which the DNAwithin the cells has been damaged. Damage to the DNA occurs by a combination of different factors. Man...
2. `medquad_ans_0000983-2__dup1` (score=0.0028) - What causes Castleman disease? The exact underlying cause of Castleman disease (CD) is poorly understood. However, some scientists suspect that an increased production of interleukin-6 (IL-6) by the immune system may ...
3. `medquad_ans_0000088-3` (score=0.0003) - Mutations in the FAS gene cause ALPS in approximately 75 percent of affected individuals. The FAS gene provides instructions for making a protein involved in cell signaling that results in the self-destruction of cell...
4. `medquad_ans_0000040-12` (score=0.0001) - Myeloma and lymphoma are other types of blood cancers. Both are common among older adults and occur more often in men than women. Myeloma affects plasma cells, a type of white blood cells typically found in the bone m...
5. `medquad_ans_0000004_7-1` (score=0.0001) - Key Points - Childhood non-Hodgkin lymphoma is a disease in which malignant (cancer) cells form in the lymph system. - The main types of lymphoma are Hodgkin lymphoma and non-Hodgkin lymphoma. - There are three major ...

**Answer**:

Mantle cell lymphoma is caused by DNA damage within cells, often due to a chromosomal translocation. Most lymphomas, including mantle cell lymphoma, are acquired diseases rather than inherited. Some cases are associated with a chromosomal translocation, and environmental factors such as hair dyes and pesticides, as well as immunodeficiencies, may contribute to the development of non-Hodgkin lymphomas, including mantle cell lymphoma [medquad_ans_0003789-3].

**Supported citations**: `medquad_ans_0003789-3`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=0.750, groundedness=0.020, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000324-4` (score=0.9978) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
2. `medquad_ans_0006597-4` (score=0.9332) - Can 48,XXYY syndrome be inherited?
3. `medquad_ans_0000714-4` (score=0.0026) - Nail-patella syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation f...
4. `medquad_ans_0003718-4` (score=0.0007) - Is Lynch syndrome an inherited condition? Lynch syndrome cancer risk is inherited in an autosomal dominant pattern, which means one inherited copy of the altered gene in each cell is sufficient to increase cancer risk...
5. `medquad_ans_0002382-4` (score=0.0002) - How is Fine-Lubinsky syndrome inherited? Almost all people reported to have FineLubinsky syndrome (FLS) have been the only affected people in their families (these cases were sporadic). There has been one report of an...

**Answer**:

The provided context does not contain information about the inheritance of Legg-Calvé-Perthes disease. Therefore, based on the given context, it is not possible to determine whether Legg-Calvé-Perthes disease is inherited. [no_context]

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

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.9994) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
3. `medquad_ans_0000094-1__dup1` (score=0.0001) - What are the signs and symptoms of Acromesomelic dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acromesomelic dysplasia Maroteaux type. If the information ...
4. `medquad_ans_0005819-2` (score=0.0001) - What are the signs and symptoms of Spondylocostal dysostosis 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocostal dysostosis 1. If the information is available, the tabl...
5. `medquad_ans_0005820-2` (score=0.0001) - What are the signs and symptoms of Spondylocostal dysostosis 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocostal dysostosis 2. If the information is available, the tabl...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, and tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000251-1` (score=0.8085) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
2. `medquad_ans_0000835-1__dup1` (score=0.0224) - Children vary in their development of speech and language skills. Health professionals have milestones for what's normal. These milestones help determine if a child is on track or if he or she may need extra help. For...
3. `medquad_ans_0006007-1` (score=0.0173) - TAR syndrome is characterized by the absence of a bone called the radius in each forearm, short stature, and thrombocytopenia. The thrombocytopenia often appears first in infancy but becomes less severe or returns to ...
4. `medquad_ans_0004775-1` (score=0.0056) - Pearson syndrome is a mitochondrial disorder characterized by transfusion-dependent sideroblastic anemia and pancreatic dysfunction resulting in in malabsorption and chronic diarrhea. The features of this progressive ...
5. `medquad_ans_0000779-1__dup2` (score=0.0032) - Rett syndrome is a rare genetic disease that causes developmental and nervous system problems, mostly in girls. It's related to autism spectrum disorder. Babies with Rett syndrome seem to grow and develop normally at ...

**Answer**:

Cri-du-chat syndrome is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that sounds like that of a cat. The disorder is characterized by intellectual disability and delayed development, small head size (microcephaly), low birth weight, and weak muscle tone (hypotonia) in infancy. Affected individuals also have distinctive facial features, including widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children with cri-du-chat syndrome are born with a heart defect [medquad_ans_0000251-1].

**Supported citations**: `medquad_ans_0000251-1`
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.9992) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0003881-2` (score=0.1891) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...
3. `medquad_ans_0000722-2__dup1` (score=0.0003) - What are the signs and symptoms of Bilateral perisylvian polymicrogyria? The signs and symptoms of bilateral perisylvian polymicrogyria (BPP) vary but may include: Partial paralysis of muscles on both sides of the fac...
4. `medquad_ans_0000718-1__dup1` (score=0.0001) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
5. `medquad_ans_0006296-1` (score=0.0001) - What are the signs and symptoms of Uropathy distal obstructive polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Uropathy distal obstructive polydactyly. If the informatio...

**Answer**:

The symptoms of Bilateral frontoparietal polymicrogyria may include: moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003414-3` (score=0.9965) - What causes Konigsmark Knox Hussels syndrome? Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene. In most cases, this condition is caused by a mutation that replaces the amino acid ar...
2. `medquad_ans_0006110-1` (score=0.0001) - Tietz syndrome is a rare condition that affects the development of melanocytes, the cells in our body that produce and contain melanin (the pigment that gives color to skin, hair, and eyes). Signs and symptoms of this...
3. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
4. `medquad_ans_0003384-1` (score=0.0001) - Keutel syndrome is an inherited condition characterized by cartilage calcification in the ears, nose, larnyx, trachea (voice box), and ribs; pulmonary artery stenoses; brachytelephalangism (short fingers and nails tha...
5. `medquad_ans_0001853-3` (score=0.0001) - What causes diffuse idiopathic skeletal hyperostosis ? The exact underlying cause of diffuse idiopathic skeletal hyperostosis (DISH) is poorly understood. However, several factors have been associated with an increase...

**Answer**:

Konigsmark Knox Hussels syndrome is caused by a particular mutation in the OPA1 gene, specifically a mutation that replaces the amino acid arginine with the amino acid histidine at position 445 in the OPA1 protein, written as Arg445His or R445H [medquad_ans_0003414-3].

**Supported citations**: `medquad_ans_0003414-3`
**Unsupported citations**: _none_

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.507, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004584-2` (score=0.9992) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...
2. `medquad_ans_0004589-2` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...
3. `medquad_ans_0004584-1` (score=0.0001) - Orofaciodigital syndrome 1 (OFD1), also called orofaciodigital syndrome type 1, is a condition that affects the development of the oral cavity (the mouth and teeth), facial features, and digits (fingers and toes). Thi...
4. `medquad_ans_0000943-1__dup1` (score=0.0001) - What are the signs and symptoms of Cardiocranial syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cardiocranial syndrome. If the information is available, the table below in...
5. `medquad_ans_0004102-2` (score=0.0001) - What are the signs and symptoms of Moebius syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Moebius syndrome. If the information is available, the table below includes how o...

**Answer**:

The symptoms of Orofaciodigital syndrome 1 include oral features such as a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other facial abnormalities. Finger and toe abnormalities may include shortness (brachydactyly), webbing or joining (syndactyly), abnormally curved fingers (clinodactyly), or other malformations. Brain abnormalities such as cysts and kidney problems like polycystic kidney disease may also occur. About half of individuals have some degree of learning disability, usually mild. Additionally, symptoms include bifid tongue (90%), broad alveolar ridges (90%), cleft upper lip (90%), frontal bossing (90%), hypert

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=1.000, groundedness=0.906, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000863-1__dup1` (score=0.9972) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
2. `medquad_ans_0000718-1__dup2` (score=0.9967) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
3. `medquad_ans_0000969-1__dup2` (score=0.9965) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
4. `medquad_ans_0000130-1__dup3` (score=0.0026) - Summary : If you need a mastectomy, you have a choice about whether or not to have surgery to rebuild the shape of the breast. Instead of breast reconstruction, you could choose to wear a breast form that replaces the...
5. `medquad_ans_0000745-5` (score=0.0004) - These resources address the diagnosis or management of oculodentodigital dysplasia: - Genetic Testing Registry: Oculodentodigital dysplasia - MedlinePlus Encyclopedia: Webbing of the fingers or toes - UC Davis Childre...

**Answer**:

Yes, there is information about surgery. Surgery can relieve or prevent pain, reduce symptoms, improve body function, or identify problems such as through a biopsy. Some surgeries, like heart surgery, can save a life. Laparoscopic surgery allows for smaller incisions using a thin tube with a camera and small tools. After surgery, there are risks including infection, bleeding, reaction to anesthesia, or accidental injury, and pain is almost always present [medquad_ans_0000863-1__dup1]. Cosmetic surgery can reshape body parts such as breasts, ears, eyes, face, hair, nose, and tummy [medquad_ans_0000718-1__dup2]. Weight loss surgery helps individuals with extreme obesity who cannot lose weight through diet and exercise or have obesity-related health issues; it often limits food intake and affects digestion and nutrient absorption, with risks including infections, hernias, and blood

**Supported citations**: `medquad_ans_0000863-1__dup1`, `medquad_ans_0000718-1__dup2`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.985, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0002935-1` (score=0.9831) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
2. `medquad_ans_0002709-1` (score=0.0019) - Hairy cell leukemia is a rare, slow-growing cancer of the blood in which the bone marrow makes too many B cells (lymphocytes), a type of white blood cell that fights infection. The condition is named after these exces...
3. `medquad_ans_0002934-1` (score=0.0011) - Human T-cell leukemia virus, type 1 (HTLV-1) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, some affected people may later de...
4. `medquad_ans_0000010-1__dup4` (score=0.0005) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, however, the bone marrow produces abnormal white blood cells. Thes...
5. `medquad_ans_0000968-1` (score=0.0003) - T-cell immunodeficiency, congenital alopecia, and nail dystrophy is a type of severe combined immunodeficiency (SCID), which is a group of disorders characterized by an almost total lack of immune protection from fore...

**Answer**:

Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles. It can also be spread from mother to child during birth or breastfeeding. There is no cure or treatment for HTLV-2, and it is considered a lifelong condition; however, most infected people remain asymptomatic (show no symptoms) throughout life [medquad_ans_0002935-1].

**Supported citations**: `medquad_ans_0002935-1`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.716, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000567-1__dup2` (score=0.9791) - When you breathe, your lungs take in oxygen from the air and deliver it to the bloodstream. The cells in your body need oxygen to work and grow. During a normal day, you breathe nearly 25,000 times. People with lung d...
2. `medquad_ans_0000007-1__dup1` (score=0.8598) - ARDS, or acute respiratory distress syndrome, is a lung condition that leads to low oxygen levels in the blood. ARDS can be life threatening because your body's organs need oxygen-rich blood to work well. People who d...
3. `medquad_ans_0000522-1__dup1` (score=0.5592) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
4. `medquad_ans_0000007-4__dup1` (score=0.1474) - People at risk for ARDS have a condition or illness that can directly or indirectly injure their lungs. Direct Lung Injury Conditions that can directly injure the lungs include: Pneumonia. This is an infection in the ...
5. `medquad_ans_0001152-1` (score=0.0115) - Children's interstitial and diffuse lung disease (chILD) is not a single condition, but a group of rare lung diseases found in infants, children and adolescents that can range from mild to severe. All types of chILD d...

**Answer**:

The provided context does not contain information about interstitial lung diseases. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0000522-1__dup1] is incomplete and does not provide a definition or description of interstitial lung disease.

**Supported citations**: `medquad_ans_0000522-1__dup1`
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.065, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0006460-4` (score=0.2627) - Is Wolff-Parkinson-White syndrome inherited?
2. `medquad_ans_0003743-4` (score=0.0021) - Is Madelung disease inherited? Although the exact cause of Madelung disease is unknown, most cases are not thought to be inherited. However, more than one family member can occasionally be affected by this condition w...
3. `medquad_ans_0000983-3__dup1` (score=0.0011) - Is Castleman disease inherited? Although the exact underlying cause of Castleman disease is unknown, it is thought to occur sporadically in people with no family history of the condition.
4. `medquad_ans_0002175-4` (score=0.0011) - Is essential tremor inherited? About half of all cases of essential tremor appear to occur because of a genetic mutation. This is referred to as familial tremor. In these cases, essential tremor appears to be passed t...
5. `medquad_ans_0000139-2__dup2` (score=0.0005) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...

**Answer**:

The provided context does not contain information about whether Melnick-Needles syndrome is inherited. Therefore, based on the given sources, it is not possible to determine the inheritance pattern of Melnick-Needles syndrome. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=1.000, groundedness=0.538, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000908-1__dup2` (score=0.9954) - Your trachea, or windpipe, is one part of your airway system. Airways are pipes that carry oxygen-rich air to your lungs. They also carry carbon dioxide, a waste gas, out of your lungs. When you inhale, air travels fr...
2. `medquad_ans_0006138-1` (score=0.9703) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
3. `medquad_ans_0006140-1` (score=0.0106) - Tracheobronchopathia osteoplastica (TO) is a rare condition of the large airways. It is characterized by the presence of multiple growths (nodules) made of bone and cartilage tissue, in the submucosa of the tracheobro...
4. `medquad_ans_0006137-1` (score=0.0010) - Tracheal agenesis is a rare birth defect in which the trachea (windpipe) is completely absent (agenesis) or significantly underdeveloped (atresia). Signs and symptoms include polyhydramnios during pregnancy and respir...
5. `medquad_ans_0006138-2` (score=0.0001) - What are the signs and symptoms of tracheobronchomalacia? Tracheobronchomalacia (TBM) is a condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway ...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse. There are two forms of TBM: a congenital form (called primary TBM) that typically develops during infancy or early childhood, and an acquired form (called secondary TBM) that is usually seen in adults. Some affected people may initially have no signs or symptoms, but the condition is typically progressive and most people will eventually develop characteristic features such as shortness of breath, cough, sputum retention, and wheezing or stridor with breathing. Most cases of primary TBM are caused by genetic conditions that weaken the walls of the airway, while the secondary form often occurs due to trauma, chronic inflammation, and/or prolonged compression of the airways. Treatment is generally only required in those who have

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
5. `medquad_ans_0000810-1` (score=0.0002) - Popliteal pterygium syndrome is a condition that affects the development of the face, skin, and genitals. Most people with this disorder are born with a cleft lip, a cleft palate (an opening in the roof of the mouth),...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.893, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000045-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000770-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000298-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000122-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000554-4` (score=0.9615) - Kallmann syndrome 1 (caused by ANOS1 gene mutations) has an X-linked recessive pattern of inheritance. The ANOS1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only on...

**Answer**:

Yes, Kallmann syndrome can be inherited, and it can follow different patterns of inheritance. Kallmann syndrome 1, caused by ANOS1 gene mutations, is inherited in an X-linked recessive pattern, where males are more frequently affected due to having only one X chromosome. In females, two mutated copies of the gene are typically required to cause the disorder. Additionally, some forms of Kallmann syndrome can be inherited in an autosomal dominant pattern, where one copy of the altered gene is sufficient to cause the disorder. In other cases, it can follow an autosomal recessive pattern, requiring mutations in both copies of the gene [medquad_ans_0000554-4]. Some cases result from new mutations and occur in individuals with no family history of the disorder.

**Supported citations**: `medquad_ans_0000554-4`
**Unsupported citations**: _none_

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.9994) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005798-1` (score=0.0001) - What are the signs and symptoms of Spinocerebellar ataxia X-linked type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia X-linked type 3. If the information ...
3. `medquad_ans_0005970-1` (score=0.0001) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...
4. `medquad_ans_0005967-1` (score=0.0001) - Syndactyly type 3 (SD3) is a limb abnormality present at birth that is characterized by complete fusion of the 4th and 5th fingers on both hands. In most cases only the soft tissue is fused, but in some cases the bone...
5. `medquad_ans_0000813-1__dup1` (score=0.0001) - What are the signs and symptoms of Brachydactyly type A5? The Human Phenotype Ontology provides the following list of signs and symptoms for Brachydactyly type A5. If the information is available, the table below incl...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.537, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000232-1__dup2` (score=0.8652) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...
2. `medquad_ans_0000133-1__dup2` (score=0.7908) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
3. `medquad_ans_0000197-1__dup3` (score=0.1864) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
4. `medquad_ans_0000066-1__dup3` (score=0.0310) - Asthma is a chronic disease that affects your airways. Your airways are tubes that carry air in and out of your lungs. If you have asthma, the inside walls of your airways become sore and swollen. In the United States...
5. `medquad_ans_0000021-1__dup3` (score=0.0154) - Espaol Bronchitis (bron-KI-tis) is a condition in which the bronchial tubes become inflamed. These tubes carry air to your lungs. (For more information about the bronchial tubes and airways, go to the Diseases and Con...

**Answer**:

Chronic bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus and can also cause shortness of breath, wheezing, a low fever, and chest tightness. It is one type of COPD (chronic obstructive pulmonary disease). The inflamed bronchial tubes produce a lot of mucus, leading to coughing and difficulty breathing. Cigarette smoking is the most common cause, and long-term exposure to air pollution, fumes, or dust may also cause it. Chronic bronchitis is a long-term condition that keeps coming back or never goes away completely. If you smoke, it is important to quit. Treatment often includes medicines to open the airways and help clear mucus, and may also include oxygen therapy and pulmonary rehabilitation [medquad_ans_0000197-1\_\_dup3

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.625, groundedness=0.731, answer_relevance=0.725

**Top docs**:

1. `medquad_ans_0000233-2__dup1` (score=0.9969) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...
2. `medquad_ans_0000936-5` (score=0.9910) - These resources address the diagnosis or management of spinal muscular atrophy with progressive myoclonic epilepsy: - Genetic Testing Registry: Jankovic Rivera syndrome - Muscular Dystrophy Association: Spinal Muscula...
3. `medquad_ans_0000935-5` (score=0.9884) - These resources address the diagnosis or management of spinal muscular atrophy: - Gene Review: Gene Review: Spinal Muscular Atrophy - Gene Review: Gene Review: Spinal Muscular Atrophy, X-Linked Infantile - Genetic Tes...
4. `medquad_ans_0000934-5` (score=0.0224) - These resources address the diagnosis or management of spinal and bulbar muscular atrophy: - Gene Review: Gene Review: Spinal and Bulbar Muscular Atrophy - Genetic Testing Registry: Bulbo-spinal atrophy X-linked - Med...
5. `medquad_ans_0000173-4__dup1` (score=0.0182) - Between 2003 and 2012, the NINDS piloted the Spinal Muscular Atrophy Project to expedite therapeutics development for this hereditary neurodegenerative disease. The Project was designed to accelerate the research proc...

**Answer**:

The provided context does not specify treatments for Spinal Muscular Atrophy. While it mentions resources related to diagnosis and management, such as genetic testing, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, it does not detail specific treatments for Spinal Muscular Atrophy [medquad_ans_0000935-5][medquad_ans_0000934-5][medquad_ans_0000936-5]. Therefore, based on the given information, the exact treatments for Spinal Muscular Atrophy cannot be determined.

**Supported citations**: `medquad_ans_0000935-5`, `medquad_ans_0000934-5`, `medquad_ans_0000936-5`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.898, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000013_2-2__dup3` (score=0.9939) - See the PDQ summary about Chronic Myelogenous Leukemia Treatment for information.
2. `medquad_ans_0000001_1-7` (score=0.9738) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
3. `medquad_ans_0000040-4__dup2` (score=0.9614) - There are many treatment options for people with leukemia. The choice of treatment depends on your age and general health, the type of leukemia you have, whether or not it has spread outside the bone marrow, and other...
4. `medquad_ans_0000040-19` (score=0.4059) - Standard treatments for leukemia include chemotherapy, biological therapy, radiation therapy, and immunotherapy. Some patients receive a combination of treatments. Learn more about treatments for acute myeloid leukemi...
5. `medquad_ans_0000040-6` (score=0.0329) - Researchers are conducting clinical trials in many parts of the country. Clinical trials test an intervention such as a drug, therapy, medical device, or behavior in many people to see if it is safe and effective. Cli...

**Answer**:

The treatments for Adult Acute Lymphoblastic Leukemia (ALL) include chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and biologic therapy. The treatment is typically divided into two phases: remission induction therapy, which aims to kill leukemia cells in the blood and bone marrow to achieve remission, and post-remission therapy, which seeks to eliminate any remaining leukemia cells that could lead to relapse. Central nervous system (CNS) sanctuary therapy is also commonly given during each phase to address leukemia cells that may hide in the brain and spinal cord. This therapy includes systemic chemotherapy, intrathecal chemotherapy, and radiation therapy to the brain. Additionally, patients may consider participating in clinical trials, which are research studies aimed at improving current treatments or testing new ones [medquad_ans_0000001_1-7].

**Supported citations**: `medquad_ans_0000001_1-7`
**Unsupported citations**: _none_

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002439-3` (score=0.9899) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
2. `medquad_ans_0001042-3__dup1` (score=0.2452) - What causes cerebellar degeneration? Cerebellar degeneration can be caused by a variety of different conditions. Neurological diseases that can lead to cerebellar degeneration include: Acute and hemorrhagic stroke can...
3. `medquad_ans_0000133-3__dup1` (score=0.0002) - Generally, within 15 to 20 years after the appearance of the first symptoms, the person is confined to a wheelchair, and in later stages of the disease, individuals may become completely incapacitated. Friedreich's at...
4. `medquad_ans_0000078-2` (score=0.0001) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.
5. `medquad_ans_0000388-3` (score=0.0001) - Mutations in the FXN gene cause Friedreich ataxia. This gene provides instructions for making a protein called frataxin. Although its role is not fully understood, frataxin appears to be important for the normal funct...

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [medquad_ans_0002439-3].

**Supported citations**: `medquad_ans_0002439-3`
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.667, groundedness=0.528, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000013-4__dup2` (score=0.9705) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
2. `medquad_ans_0000341-1` (score=0.8466) - Familial atrial fibrillation is an inherited condition that disrupts the heart's normal rhythm. This condition is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which cau...
3. `medquad_ans_0002230-1` (score=0.8236) - Familial atrial fibrillation is an inherited heart condition that disrupts the heart's rhythm. It is characterized by erratic electrical activity in the heart's upper chambers (the atria), causing an irregular respons...
4. `medquad_ans_0000341-2` (score=0.7779) - Atrial fibrillation is the most common type of sustained abnormal heart rhythm (arrhythmia), affecting more than 3 million people in the United States. The risk of developing this irregular heart rhythm increases with...
5. `medquad_ans_0000070-1__dup4` (score=0.0162) - An arrhythmia is a problem with the speed or rhythm of the heartbeat. Atrial fibrillation (AF) is the most common type of arrhythmia. The cause is a disorder in the heart's electrical system. Often, people who have AF...

**Answer**:

Familial atrial fibrillation is an inherited heart condition that disrupts the heart's normal rhythm, characterized by uncoordinated or erratic electrical activity in the heart's upper chambers (the atria). This leads to a fast and irregular heartbeat (arrhythmia), which can cause symptoms such as dizziness, chest pain, palpitations, shortness of breath, or fainting. It also increases the risk of stroke and sudden death. The condition may be caused by changes (mutations) in various genes, some of which have not been identified, and is most often inherited in an autosomal dominant manner, although autosomal recessive inheritance has been reported. Complications can occur at any age, though some individuals may never experience associated health problems. The incidence of the familial form is unknown, but recent studies suggest that up to 30 percent of all people with atrial fibrillation may have a family history of the condition [medquad*ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.8245) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0004625-1` (score=0.0137) - Osteomesopyknosis is a bone disorder characterized by abnormal hardening of bone (osteosclerosis). It is generally limited to the axial spine, pelvis, and proximal part of the long bones, which is what distinguishes t...
3. `medquad_ans_0000238-1__dup1` (score=0.0024) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper ...
4. `medquad_ans_0000088-2__dup2` (score=0.0008) - Treatment for craniosynostosis generally consists of surgery to improve the symmetry and appearance of the head and to relieve pressure on the brain and the cranial nerves. For some children with less severe problems,...
5. `medquad_ans_0000088-1__dup2` (score=0.0002) - Craniosynostosis is a birth defect of the skull characterized by the premature closure of one or more of the fibrous joints between the bones of the skull (called the cranial sutures) before brain growth is complete. ...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the associated symptoms vary significantly from person to person, even within a single family. Features of the condition are generally present at birth and may include skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected people may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). Osteopathia striata cranial sclerosis is caused by changes (mutations) in the WTX gene and is inherited in an X-linked dominant manner. Treatment is based on the signs and symptoms present in each person [medquad_ans_0004

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.537, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004097-2` (score=0.9992) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
2. `medquad_ans_0000230-1__dup2` (score=0.0001) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their...
3. `medquad_ans_0000141-1` (score=0.0001) - Buschke-Ollendorff syndrome is a hereditary disorder of connective tissues, which are tissues that provide strength and flexibility to structures throughout the body. Specifically, the condition is characterized by sk...
4. `medquad_ans_0004097-1` (score=0.0001) - Mixed connective tissue disease (MCTD) is a rare autoimmune disorder that is characterized by features commonly seen in three different connective tissue disorders: systemic lupus erythematosus, scleroderma, and polym...
5. `medquad_ans_0006289-1` (score=0.0001) - Urachal cancer is a rare type of bladder cancer, making up less than 1% of all bladder cancers. Only about 350 cases have been described in the medical literature to date. The urachus is a primitive structure which be...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints and/or inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), kidney disease, neuro

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=0.857, groundedness=0.327, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000098-2__dup1` (score=0.9580) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow ...
2. `medquad_ans_0000004-23` (score=0.6805) - Currently, no treatment can stop Alzheimer's disease. However, four medications are used to treat its symptoms. These medicines may help maintain thinking, memory, and speaking skills for a limited time. They work by ...
3. `medquad_ans_0000004-32` (score=0.4979) - People with Alzheimer's disease, those with mild cognitive impairment, those with a family history of Alzheimers, and healthy people with no memory problems who want to help scientists test new treatments may be able ...
4. `medquad_ans_0000004-24` (score=0.1477) - Currently, no medicines or treatments are known to prevent Alzheimer's disease, but scientists are studying many possibilities. These possibilities include lifestyle factors such as exercise and physical activity, a h...
5. `medquad_ans_0000003-30` (score=0.1407) - As Alzheimers disease gets worse, you will need more help to care for the person. It's okay to seek help whenever you need it. Several kinds of help are available. - Home health care agencies send a home health aide o...

**Answer**:

The provided context does not address treatments specifically for Alzheimer's caregiving. While it discusses medications for Alzheimer's disease and cognitive exercises that may benefit individuals with dementia, there is no information on treatments aimed at supporting or improving caregiving. Therefore, based on the given context, the question cannot be answered.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.900, groundedness=0.896, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000511-3` (score=0.9972) - Idiopathic inflammatory myopathy is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Researchers have identi...
2. `medquad_ans_0000547-3` (score=0.0003) - Juvenile idiopathic arthritis is thought to arise from a combination of genetic and environmental factors. The term "idiopathic" indicates that the specific cause of the disorder is unknown. Its signs and symptoms res...
3. `medquad_ans_0000293-4` (score=0.0001) - Distal myopathy 2 is inherited in an autosomal dominant pattern, which means one copy of the altered MATR3 gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000511-2` (score=0.0001) - The incidence of idiopathic inflammatory myopathy is approximately 2 to 8 cases per million people each year. For unknown reasons, polymyositis and dermatomyositis are about twice as common in women as in men, while s...
5. `medquad_ans_0003179-4` (score=0.0001) - How is inclusion body myopathy 2 inherited? Inclusion body myopathy 2 is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with ...

**Answer**:

Genetic changes related to idiopathic inflammatory myopathy include variations in several genes, particularly within the human leukocyte antigen (HLA) complex. These HLA gene variations are associated with an increased risk of developing the disorder. The HLA complex plays a role in helping the immune system distinguish the body's own proteins from foreign proteins, and specific variations in these genes may influence susceptibility to idiopathic inflammatory myopathy [medquad_ans_0000511-3]. Researchers are also studying other genes related to immune function to better understand their contribution to the disease risk [medquad_ans_0000511-3].

**Supported citations**: `medquad_ans_0000511-3`
**Unsupported citations**: _none_

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0005225-5` (score=0.9947) - Is genetic testing available for pyruvate kinase deficiency? Yes. GeneTests lists laboratories offering clinical genetic testing for this condition. Clinical genetic tests are ordered to help diagnose an affected pers...
2. `medquad_ans_0005223-5` (score=0.9899) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
3. `medquad_ans_0005926-5` (score=0.0002) - How is succinic semialdehyde dehydrogenase deficiency diagnosed? The diagnosis of succinic semialdehyde dehydrogenase (SSADH) deficiency is based upon a thorough clinical exam, the identification of features consisten...
4. `medquad_ans_0000855-2` (score=0.0001) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
5. `medquad_ans_0000856-2` (score=0.0001) - Pyruvate kinase deficiency is the most common inherited cause of nonspherocytic hemolytic anemia. More than 500 affected families have been identified, and studies suggest that the disorder may be underdiagnosed becau...

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.961, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.9991) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
3. `medquad_ans_0006565-2` (score=0.0001) - What are the signs and symptoms of 1q21.1 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 1q21.1 microdeletion syndrome. If the information is available, the ...
4. `medquad_ans_0006554-2` (score=0.0001) - What are the signs and symptoms of 15q11.2 microdeletion? The signs and symptoms in people with a 15q11.2 microdeletion can vary widely. Some people with the microdeletion don't have any apparent features, while other...
5. `medquad_ans_0006562-3` (score=0.0001) - What causes 17q23.2q23.2 microdeletion syndrome? The syndrome is caused by an interstitial deletion (a deletion that does not involve the ends of a chromosome) encompassing bands 23.1 to 23.2 on the long (q) arm of ch...

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

**Scores**: context_relevance=1.000, groundedness=0.927, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.9952) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0005347-3` (score=0.0079) - What causes Rett syndrome? Rett syndrome is typically caused by changes (mutations) in the MECP2 gene. This gene provides instructions for making a protein (MeCP2) needed for the development of the nervous system and ...
3. `medquad_ans_0000635-3` (score=0.0001) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
4. `medquad_ans_0000814-4` (score=0.0001) - More than 99 percent of PPM-X syndrome cases occur in people with no history of the disorder in their family. Many of these cases result from new mutations in the MECP2 gene. A few families with more than one affected...
5. `medquad_ans_0000635-2` (score=0.0001) - The prevalence of MECP2 duplication syndrome is unknown; approximately 120 affected individuals have been reported in the scientific literature. It is estimated that this condition is responsible for 1 to 2 percent of...

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
4. `medquad_ans_0004633-1` (score=0.0001) - Osteopetrosis is a bone disease that makes bones abnormally dense and prone to breakage (fracture). Researchers have described several major types of osteopetrosis, which are usually distinguished by their pattern of ...
5. `medquad_ans_0000096-1__dup1` (score=0.0001) - What are the signs and symptoms of Acroosteolysis dominant type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acroosteolysis dominant type. If the information is available, the ta...

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

**Scores**: context_relevance=0.667, groundedness=0.716, answer_relevance=0.578

**Top docs**:

1. `medquad_ans_0000033-2__dup1` (score=0.9955) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
2. `medquad_ans_0000200-2__dup2` (score=0.9933) - Currently there is no cure for these disease syndromes.Medical care is directed at treating systemic conditions and improving the person's quality of life. Physical therapy and daily exercise may delay joint problems ...
3. `medquad_ans_0000084-2__dup1` (score=0.9910) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
4. `medquad_ans_0000163-2__dup1` (score=0.9877) - The chronic inflammatory myopathies cant be cured in most adults but many of the symptoms can be treated. Options include medication, physical therapy, exercise, heat therapy (including microwave and ultrasound), orth...
5. `medquad_ans_0003459-4` (score=0.9687) - How might Lambert-Eaton myasthenic syndrome be treated? Medications and therapies used to treat Lambert-Eaton myasthenic syndrome may include anticholinesterase agents (e.g., Pyridostigmine), guanidine hydrochloride, ...

**Answer**:

The provided context does not contain information about the treatments for Lambert-Eaton myasthenic syndrome. [medquad_ans_0003459-4] mentions the question but does not provide an answer or details about treatments. Therefore, based on the given context, the treatments for Lambert-Eaton myasthenic syndrome cannot be determined.

**Supported citations**: `medquad_ans_0003459-4`
**Unsupported citations**: _none_

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.925, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.9993) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
2. `medquad_ans_0000065-3` (score=0.0001) - Mutations in the ARG1 gene cause arginase deficiency. Arginase deficiency belongs to a class of genetic diseases called urea cycle disorders. The urea cycle is a sequence of reactions that occurs in liver cells. This ...
3. `medquad_ans_0000065-1` (score=0.0001) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
4. `medquad_ans_0000445-1__dup1` (score=0.0001) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The symptoms of Arginase deficiency include aminoaciduria, behavioral abnormality, cognitive impairment, neurological speech impairment, EEG abnormality, hemiplegia/hemiparesis, hyperammonemia, seizures, anorexia, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1]. These symptoms are based on the Human Phenotype Ontology and are reported as approximate frequencies in patients with the condition [medquad_ans_0000445-2__dup1]. The underlying cause is the impaired function of arginase due to mutations in the ARG1 gene, leading to disrupted urea cycle and accumulation of ammonia and arginine [medquad_ans_0000065-3].

**Supported citations**: `medquad_ans_0000445-2__dup1`, `medquad_ans_0000065-3`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.889, groundedness=0.943, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000022-2__dup1` (score=0.9864) - Treatment for VHL varies according to the location and size of the tumor. In general, the objective of treatment is to treat the tumors before they grow to a size large enough to cause permanent problems by putting pr...
2. `medquad_ans_0006376-6` (score=0.9676) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
3. `medquad_ans_0004784-5` (score=0.8496) - How might Pelizaeus-Merzbacher disease be treated?
4. `medquad_ans_0000139-5__dup1` (score=0.0074) - Treatment for von Willebrand disease (VWD) is based on the type of VWD you have and how severe it is. Most cases of VWD are mild, and you may need treatment only if you have surgery, tooth extraction, or an accident. ...
5. `medquad_ans_0001020-5` (score=0.0060) - These resources address the diagnosis or management of von Hippel-Lindau syndrome: - Brigham and Women's Hospital - Gene Review: Gene Review: Von Hippel-Lindau Syndrome - Genetic Testing Registry: Von Hippel-Lindau sy...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat tumors when they cause symptoms but are still small enough to avoid permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All individuals with VHL disease should be carefully monitored by a physician or medical team familiar with the disorder [medquad_ans_0006376-6]. Additionally, treatment aims to remove tumors before they grow large enough to cause pressure on the brain or spinal cord, which could impair vision, cause deafness, or block cerebrospinal fluid flow [medquad_ans_0000022-2__dup1].

**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0000022-2__dup1`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.106, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001091-1` (score=0.8518) - 3-M syndrome is a disorder that causes short stature (dwarfism), unusual facial features, and skeletal abnormalities. The name of this condition comes from the initials of three researchers who first identified it: Mi...
2. `medquad_ans_0004451-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
3. `medquad_ans_0004450-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
4. `medquad_ans_0004456-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...
5. `medquad_ans_0004454-1` (score=0.5114) - Noonan syndrome is a genetic disorder that causes abnormal development of multiple parts of the body. Features of Noonan syndrome may include a distinctive facial appearance, short stature, a broad or webbed neck, con...

**Answer**:

Noonan syndrome 3 is not mentioned in the provided context. Therefore, based on the available information, it cannot be identified or described. [doc_id: medquad_ans_0004451-1] does not reference "Noonan syndrome 3" or any such variant.

**Supported citations**: _none_
**Unsupported citations**: `doc_id: medquad_ans_0004451-1`

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.9991) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0000054-11` (score=0.9986) - People who have P.A.D. may have symptoms when walking or climbing stairs. These may include pain, numbness, aching, or heaviness in the leg muscles. Symptoms may also include cramping in the affected leg(s) and in the...
3. `medquad_ans_0002966-1` (score=0.0004) - What are the signs and symptoms of Hydrocephalus, costovertebral dysplasia, and Sprengel anomaly? The Human Phenotype Ontology provides the following list of signs and symptoms for Hydrocephalus, costovertebral dyspla...
4. `medquad_ans_0005835-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia with multiple dislocations? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia with ...
5. `medquad_ans_0001315-2` (score=0.0001) - What are the signs and symptoms of Chronic atypical neutrophilic dermatosis with lipodystrophy and elevated temperature? The Human Phenotype Ontology provides the following list of signs and symptoms for Chronic atypi...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, neck muscle hypoplasia, rib segmentation abnormalities, hemivertebrae, spina bifida occulta, cervical segmentation defects, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.934, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000882-3` (score=0.9988) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...
2. `medquad_ans_0001041-3` (score=0.0002) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
3. `medquad_ans_0000556-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000283-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...
5. `medquad_ans_0000667-3` (score=0.0001) - The causes of Moebius syndrome are unknown, although the condition probably results from a combination of environmental and genetic factors. Researchers are working to identify and describe specific genes related to t...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene, which cause autosomal recessive Robinow syndrome by preventing the production of any functional ROR2 protein, disrupting developmental signaling pathways essential for the formation of the skeleton, heart, and genitals [medquad_ans_0000882-3]. Autosomal dominant Robinow syndrome can result from mutations in the WNT5A or DVL1 gene, with DVL1 mutations leading to the osteosclerotic form of the condition. These mutations alter the production or function of their respective proteins, impairing chemical signaling critical for early development [medquad_ans_0000882-3]. In some cases, no mutation is identified in the ROR2, WNT5A, or DVL1 genes, and the cause remains unknown [medquad_ans_0000882-3].

**Supported citations**: `medquad_ans_0000882-3`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=1.000, groundedness=0.174, answer_relevance=0.855

**Top docs**:

1. `medquad_ans_0000174-11` (score=0.9894) - Eating the right foods can help you feel better while on peritoneal dialysis. Talk with your dialysis centers dietitian to find a meal plan that works for you. Your dietary needs will depend on your treatment and othe...
2. `medquad_ans_0000179-12` (score=0.8733) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should tal...
3. `medquad_ans_0000167-17` (score=0.4150) - A dietitian specializes in helping people who have kidney disease choose the right foods and plan healthy meals. People with any kind of kidney disease, including PKD, should talk with a dietitian about foods that sho...
4. `medquad_ans_0000158-13` (score=0.1524) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A person...
5. `medquad_ans_0000180-8` (score=0.0358) - Eating the right foods can help you feel better when you are on dialysis or have a kidney transplant. Staying healthy with kidney failure requires watching how much of these elements are included in your diet: - Prote...

**Answer**:

The provided context does not contain information about "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, it is not possible to answer the question based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.778, groundedness=0.945, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.9994) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000675-2` (score=0.8117) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
3. `medquad_ans_0000676-2` (score=0.5824) - Mucolipidosis III gamma is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
4. `medquad_ans_0000674-2` (score=0.0122) - Mucolipidosis II alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
5. `medquad_ans_0000677-1` (score=0.0001) - Mucolipidosis type IV is an inherited disorder characterized by delayed development and vision impairment that worsens over time. The severe form of the disorder is called typical mucolipidosis type IV, and the mild f...

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=0.750, groundedness=0.994, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.9368) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000193-1__dup4` (score=0.0106) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes p...
3. `medquad_ans_0000977-1__dup1` (score=0.0020) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from m...
4. `medquad_ans_0000159-6` (score=0.0019) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
5. `medquad_ans_0000063-12` (score=0.0007) - Skin cancer occurs when cancer cells form in the tissues of the skin. The skin is mainly made up of two layers: the inner layer, called the dermis, and the outer layer, called the epidermis. Within the epidermis, ther...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.920, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0001045-3` (score=0.9992) - Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. The ABCD1 gene provides instructions for producing the adrenoleukodystrophy protein (ALDP), which is involved in transporting certain fat molecules call...
2. `medquad_ans_0001051-3` (score=0.0001) - Mutations in and near the TAF1 gene can cause X-linked dystonia-parkinsonism. The TAF1 gene provides instructions for making part of a protein called transcription factor IID (TFIID). This protein is active in cells a...
3. `medquad_ans_0001143-1` (score=0.0001) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...
4. `medquad_ans_0001062-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
5. `medquad_ans_0003225-3` (score=0.0001) - How is intellectual disability-developmental delay-contractures syndrome inherited? Intellectual disability-developmental delay-contractures syndrome syndrome is inherited in an X-linked recessive manner and is caused...

**Answer**:

Mutations in the ABCD1 gene cause X-linked adrenoleukodystrophy. These mutations result in a deficiency of the adrenoleukodystrophy protein (ALDP), which is responsible for transporting very long-chain fatty acids (VLCFAs) into peroxisomes. The lack of this protein disrupts the transport and breakdown of VLCFAs, leading to their accumulation in the body. This accumulation is believed to be toxic to the adrenal cortex and myelin and may trigger an inflammatory response in the brain, contributing to the destruction of myelin and the signs and symptoms of the disease [medquad_ans_0001045-3].

**Supported citations**: `medquad_ans_0001045-3`
**Unsupported citations**: _none_

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.625, groundedness=0.972, answer_relevance=0.425

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0001101-2` (score=0.9995) - The prevalence of this disorder is estimated to be 1 in 7,500 to 20,000 people.
3. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000988-2` (score=0.9981) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
5. `medquad_ans_0000577-2` (score=0.0074) - Lamellar ichthyosis is estimated to affect 1 in 100,000 individuals in the United States. This condition is more common in Norway, where an estimated 1 in 91,000 individuals are affected.

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is thought to be one in 538 people. Among Americans of European descent, it is estimated to affect one in 100,000 people. The cardiac form is more common among people with African ancestry, affecting between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.825, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.9994) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0001795-2` (score=0.0001) - What are the signs and symptoms of Dermatitis herpetiformis ? The Human Phenotype Ontology provides the following list of signs and symptoms for Dermatitis herpetiformis . If the information is available, the table be...
3. `medquad_ans_0000676-2__dup1` (score=0.0001) - What are the symptoms of Bell's palsy?
4. `medquad_ans_0001754-1` (score=0.0001) - What are the signs and symptoms of Deafness, dystonia, and cerebral hypomyelination ? The Human Phenotype Ontology provides the following list of signs and symptoms for Deafness, dystonia, and cerebral hypomyelination...
5. `medquad_ans_0003573-2` (score=0.0001) - What are the signs and symptoms of Leukoencephalopathy with brain stem and spinal cord involvement and lactate elevation ? The Human Phenotype Ontology provides the following list of signs and symptoms for Leukoenceph...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include those listed by the Human Phenotype Ontology, which are derived from data collected through Orphanet, a European rare disease database. The frequency of these symptoms is typically reported as a rough estimate of the percentage of patients who have the feature, often expressed as a fraction (e.g., 25/25 meaning all patients in a study had the symptom). However, the specific list of symptoms for Dens in dente and palatal invaginations is not provided in the given context [medquad_ans_0001780-1]. Therefore, the exact symptoms cannot be identified from the available information.

**Supported citations**: `medquad_ans_0001780-1`
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.500, groundedness=0.931, answer_relevance=0.300

**Top docs**:

1. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
2. `medquad_ans_0000874-2` (score=0.9995) - This condition affects an estimated 1 in 8,500 females.
3. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
4. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.
5. `medquad_ans_0000782-2` (score=0.9993) - Periventricular heterotopia is a rare condition. Its incidence is unknown.

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2].

**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.818, groundedness=0.770, answer_relevance=0.745

**Top docs**:

1. `medquad_ans_0000190-4__dup1` (score=0.9984) - The NINDS supports research on neurological disorders such as Melkersson-Rosenthal syndrome. Much of this research is aimed at increasing knowledge of these disorders and finding ways to treat, prevent, and ultimately...
2. `medquad_ans_0000172-4__dup1` (score=0.9983) - NINDS supports and conducts research on neurobehavioral disorders such as Klver-Bucy syndrome. Much of the research focuses on learning more about these disorders and finding ways to prevent and treat them.
3. `medquad_ans_0000006_3-8` (score=0.0382) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000025_1-8` (score=0.0043) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000223-4__dup1` (score=0.0017) - The National Institute of Neurological Disorders and Stroke (NINDS) conducts PD research in laboratories at the National Institutes of Health (NIH) and also supports additional research through grants to major medical...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Landau-Kleffner Syndrome. [medquad_ans_0000190-4__dup1], [medquad_ans_0000172-4__dup1], and [medquad_ans_0000006_3-8] discuss research on neurological and neurobehavioral disorders such as Melkersson-Rosenthal syndrome, Klver-Bucy syndrome, and cancer treatments in clinical trials, but none reference Landau-Kleffner Syndrome. Therefore, based on the given information, no specific research or clinical trials for Landau-Kleffner Syndrome are described.

**Supported citations**: `medquad_ans_0000190-4__dup1`, `medquad_ans_0000172-4__dup1`, `medquad_ans_0000006_3-8`
**Unsupported citations**: _none_

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.617, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000631-1__dup1` (score=0.8258) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
2. `medquad_ans_0004290-1` (score=0.1678) - Myotonic dystrophy is an inherited condition that affects the muscles and other body systems. It is the most common form of muscular dystrophy that begins in adulthood, usually in a person's 20s or 30s. This condition...
3. `medquad_ans_0000306-1` (score=0.1412) - Muscular dystrophies are a group of genetic conditions characterized by progressive muscle weakness and wasting (atrophy). The Duchenne and Becker types of muscular dystrophy are two related conditions that primarily ...
4. `medquad_ans_0000710-1` (score=0.0819) - Myotonic dystrophy is part of a group of inherited disorders called muscular dystrophies. It is the most common form of muscular dystrophy that begins in adulthood. Myotonic dystrophy is characterized by progressive m...
5. `medquad_ans_0003601-1` (score=0.0265) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...

**Answer**:

Muscular dystrophy (MD) is a group of more than 30 inherited diseases that cause muscle weakness and muscle loss [medquad_ans_0000631-1]. These conditions are characterized by progressive muscle weakness and wasting (atrophy) [medquad_ans_0000306-1]. The different types of MD can vary in terms of whom they affect, which muscles are involved, and the specific symptoms presented [medquad_ans_0000631-1]. Some forms appear in infancy or childhood, while others may not manifest until middle age or later [medquad_ans_0000631-1]. All forms of MD worsen as muscle strength decreases, and most people eventually lose the ability to walk [medquad_ans_0000631-1]. Duchenne and Becker muscular dystrophies are two related conditions that primarily affect skeletal

**Supported citations**: `medquad_ans_0000306-1`
**Unsupported citations**: `medquad_ans_0000631-1`

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.342, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.8062) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0004953-1` (score=0.0003) - Polyembryoma is a type of tumor that develops from the cells of the gonads (testes in men or ovaries in women). Such tumors are called germ cell tumors. Polyembryomas have a distinctive look because they are composed ...
3. `medquad_ans_0005020-1` (score=0.0002) - What are the signs and symptoms of Preaxial polydactyly type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 2. If the information is available, the tabl...
4. `medquad_ans_0000024-1` (score=0.0001) - Adult polyglucosan body disease is a condition that affects the nervous system. People with this condition have problems walking due to reduced sensation in their legs (peripheral neuropathy) and progressive muscle we...
5. `medquad_ans_0005022-1` (score=0.0001) - What are the signs and symptoms of Preaxial polydactyly type 4? The Human Phenotype Ontology provides the following list of signs and symptoms for Preaxial polydactyly type 4. If the information is available, the tabl...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding with no other physical anomalies or intellectual impairment, or it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits, with surgical management depending on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002519-2` (score=0.9994) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0004788-1` (score=0.0001) - What are the signs and symptoms of Pellagra like syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Pellagra like syndrome. If the information is available, the table below in...
3. `medquad_ans_0002519-3` (score=0.0001) - How is hereditary geniospasm inherited? Hereditary geniospasm is inherited in an autosomal dominant manner. This means that having only one mutated copy of the causative gene in each body cell is sufficient to cause s...
4. `medquad_ans_0003320-2` (score=0.0001) - What are the signs and symptoms of Juvenile dermatomyositis? The Human Phenotype Ontology provides the following list of signs and symptoms for Juvenile dermatomyositis. If the information is available, the table belo...
5. `medquad_ans_0004534-2` (score=0.0001) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...

**Answer**:

The symptom of Geniospasm is chin myoclonus. [medquad_ans_0002519-2]

**Supported citations**: `medquad_ans_0002519-2`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.800, groundedness=0.975, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000512-3` (score=0.9928) - The cause of idiopathic pulmonary fibrosis is unknown, although the disease probably results from a combination of genetic and environmental factors. It is likely that genetic changes increase a person's risk of devel...
2. `medquad_ans_0000049-4__dup2` (score=0.0034) - Cystic fibrosis (CF) affects both males and females and people from all racial and ethnic groups. However, the disease is most common among Caucasians of Northern European descent. CF also is common among Latinos and ...
3. `medquad_ans_0000049-3__dup2` (score=0.0029) - A defect in the CFTR gene causes cystic fibrosis (CF). This gene makes a protein that controls the movement of salt and water in and out of your body's cells. In people who have CF, the gene makes a protein that doesn...
4. `medquad_ans_0000089-2__dup2` (score=0.0005) - Marfan syndrome is a genetic disorder. A mutation, or change, in the gene that controls how the body makes fibrillin causes Marfan syndrome. Fibrillin is a protein that plays a major role in your body's connective tis...
5. `medquad_ans_0000109-4__dup2` (score=0.0001) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develo...

**Answer**:

Genetic changes related to idiopathic pulmonary fibrosis include mutations in the TERC and TERT genes, which have been found in about 15 percent of all cases of familial pulmonary fibrosis and a smaller percentage of cases of sporadic idiopathic pulmonary fibrosis. These genes provide instructions for making components of an enzyme called telomerase, which maintains telomeres at the ends of chromosomes. [medquad_ans_0000512-3]

**Supported citations**: `medquad_ans_0000512-3`
**Unsupported citations**: _none_

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.920, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000585-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000362-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000095-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000673-4` (score=0.9988) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000813-4` (score=0.9934) - Potocki-Shaffer syndrome follows an autosomal dominant inheritance pattern, which means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. In some cases, an affected per...

**Answer**:

Potocki-Shaffer syndrome follows an autosomal dominant inheritance pattern, which means a deletion of genetic material from one copy of chromosome 11 is sufficient to cause the disorder. In some cases, an affected person inherits the chromosome with a deleted segment from an affected parent. More commonly, the condition results from a deletion that occurs during the formation of reproductive cells (eggs and sperm) in a parent or in early fetal development. These cases occur in people with no history of the disorder in their family [medquad_ans_0000813-4]. Therefore, Potocki-Shaffer syndrome can be inherited, but in some cases, it is not inherited from a parent and arises de novo.

**Supported citations**: `medquad_ans_0000813-4`
**Unsupported citations**: _none_

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=1.000, groundedness=0.725, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000897-1__dup1` (score=0.9972) - Summary : Most young children get sick. It is hard for parents to know what is serious. You can learn what the common warning signs are. In the end, trust your intuition. If you are worried about your toddler, call yo...
2. `medquad_ans_0000189-1__dup2` (score=0.9970) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
3. `medquad_ans_0000182-1__dup2` (score=0.9965) - Summary : As parents, we want to keep our children safe from harm. Take steps to keep your children safe: - Install the right child safety seat in your car - Teach children how to cross the street safely - Make sure t...
4. `medquad_ans_0000925-1__dup1` (score=0.9907) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
5. `medquad_ans_0000227-5` (score=0.0975) - These resources address the diagnosis or management of CHILD syndrome: - Gene Review: Gene Review: NSDHL-Related Disorders - Genetic Testing Registry: Child syndrome These resources from MedlinePlus offer information ...

**Answer**:

No, there is no information about Child Care in the provided context. The context discusses child health, safety, and medical concerns, but does not address Child Care. [medquad_ans_0000897-1\_\_dup1][medquad_ans_0000189-1__dup2][medquad_ans_0000182-1\_\_dup2][medquad_ans_0000925-1__dup1][medquad_ans_0000227-5]

**Supported citations**: `medquad_ans_0000897-1__dup1`, `medquad_ans_0000189-1__dup2`, `medquad_ans_0000182-1__dup2`, `medquad_ans_0000925-1__dup1`, `medquad_ans_0000227-5`
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.500, groundedness=0.945, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
4. `medquad_ans_0000046-2` (score=0.9995) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
5. `medquad_ans_0000970-2` (score=0.9995) - This condition is very rare; however, the exact prevalence is unknown.

**Answer**:

Alternating hemiplegia of childhood affects approximately 1 in 1 million people [medquad_ans_0000046-2].

**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003687-2` (score=0.9992) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype ...
2. `medquad_ans_0000159-3__dup2` (score=0.9977) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
3. `medquad_ans_0000159-6` (score=0.0062) - In many cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few, if any, further complications. However, even with treatment, 10 to 30 percent of people with lupus nephri...
4. `medquad_ans_0000001_1-2` (score=0.0002) - Signs and symptoms of adult ALL include fever, feeling tired, and easy bruising or bleeding. The early signs and symptoms of ALL may be like the flu or other common diseases. Check with your doctor if you have any of ...
5. `medquad_ans_0000569-1__dup2` (score=0.0002) - If you have lupus, your immune system attacks healthy cells and tissues by mistake. This can damage your joints, skin, blood vessels and organs. There are many kinds of lupus. The most common type, systemic lupus eryt...

**Answer**:

The symptoms of Lupus include abdominal pain, abnormality of temperature regulation, abnormality of the heart valves, abnormality of the pericardium, alopecia, arthralgia, arthritis, autoimmunity, chest pain, cutaneous photosensitivity, skin rash, thrombocytopenia, thrombophlebitis, abnormal pyramidal signs, abnormal tendon morphology, abnormality of the autonomic nervous system, abnormality of the endocardium, abnormality of the pleura, anorexia, arterial thrombosis, aseptic leukocyturia, bone marrow hypocellularity, conjunctival telangiectasia, cranial nerve paralysis, cutis marmorata, dry skin, eczema, edema of the lower limbs, glomerulopathy, hallucinations, hematuria, hepatomegaly, hyperkeratosis, hypoproteinemia, increased antibody level in blood

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.531, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-1` (score=0.2765) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
2. `medquad_ans_0001395-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
3. `medquad_ans_0001398-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001397-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
5. `medquad_ans_0000073-1__dup2` (score=0.0065) - Cerebro-oculo-facio-skeletal syndrome (COFS) is a pediatric, genetic, degenerative disorder that involves the brain and the spinal cord. It is characterized by craniofacial and skeletal abnormalities, severely reduced...

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging (progeria). It is associated with severe photosensitivity, failure to thrive, microcephaly, and impaired development of the nervous system. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain visible on brain scans. The syndrome is divided into three subtypes: classical (type I or type A), which has an onset in early childhood (usually after age 1 year) and is sometimes called "moderate" or "classic" Cockayne syndrome; type II (type B), which is more severe and presents at birth with growth and developmental abnormalities, also known as cerebro-oculo-facio-skeletal (COFS) syndrome or Pena-Shokeir syndrome type II; and type III (type C), which is the mildest form and appears later in childhood. Cockayne

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.667, groundedness=0.874, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000029-9__dup1` (score=0.9863) - When gingivitis is not treated, it can advance to periodontitis (which means "inflammation around the tooth.") In periodontitis, gums pull away from the teeth and form "pockets" that become infected. The body's immune...
2. `medquad_ans_0000413-1__dup1` (score=0.9751) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the t...
3. `medquad_ans_0000029-6__dup1` (score=0.9748) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...
4. `medquad_ans_0000029-8__dup1` (score=0.9484) - Gingivitis is inflammation of the gums. In gingivitis, the gums become red, swollen and can bleed easily. Gingivitis is a mild form of gum disease. It can usually be reversed with daily brushing and flossing, and regu...
5. `medquad_ans_0000029-19` (score=0.9084) - In deep cleaning, the dentist, periodontist, or dental hygienist removes the plaque through a method called scaling and root planing. Scaling means scraping off the tartar from above and below the gum line. Root plani...

**Answer**:

Gum (periodontal) disease is an infection of the tissues that hold your teeth in place. It ranges from mild inflammation of the gums, known as gingivitis, to serious damage involving the destruction of the bone, gums, and connective tissue that support the teeth. In gingivitis, the gums become red, swollen, and may bleed easily, but there is no loss of bone or tissue. If left untreated, gingivitis can progress to periodontitis, where the gums pull away from the teeth and form infected pockets. The body's immune response to bacteria in the plaque leads to the breakdown of bone and tissue that support the teeth, potentially resulting in tooth loss [medquad_ans_0000029-6\_\_dup1][medquad_ans_0000413-1__dup1][medquad_ans_0000029-8__dup1].

**Supported citations**: `medquad_ans_0000029-6__dup1`, `medquad_ans_0000413-1__dup1`, `medquad_ans_0000029-8__dup1`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000505-2` (score=0.9993) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
2. `medquad_ans_0000494-2` (score=0.9992) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
3. `medquad_ans_0000029-3__dup1` (score=0.0695) - Many people with Type I CM are asymptomatic and do not know they have the condition. Many individuals with the more severe types of CM and have surgery see a reduction in their symptoms and/or prolonged periods of rel...
4. `medquad_ans_0003059-2` (score=0.0001) - What are the signs and symptoms of Hypokalemic periodic paralysis? Hypokalemic periodic paralysis involves attacks of muscle weakness or loss of muscle movement (paralysis) that come and go. The weakness or paralysis ...
5. `medquad_ans_0003059-1` (score=0.0001) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=0.800, groundedness=0.982, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.8399) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0004833-1` (score=0.0052) - Peutz-Jeghers syndrome (PJS) is an inherited condition that is associated with an increased risk of growths along the lining of the gastrointestinal tract (called hamartomatous polyps) and certain types of cancer. Mos...
3. `medquad_ans_0000054-7` (score=0.0009) - Peripheral arterial disease (P.A.D.) is a disease in which plaque (plak) builds up in the arteries that carry blood to your head, organs, and limbs. Plaque is made up of fat, cholesterol, calcium, fibrous tissue, and ...
4. `medquad_ans_0000227-1__dup1` (score=0.0002) - Pelizaeus-Merzbacher disease (PMD) is a rare, progressive, degenerative central nervous system disorder in which coordination, motor abilities, and intellectual function deteriorate. The disease is one of a group of g...
5. `medquad_ans_0005016-1` (score=0.0001) - Prader-Willi syndrome (PWS) is a genetic condition that affects many parts of the body. Infants with PWS have severe hypotonia (low muscle tone), feeding difficulties, and slow growth. In later infancy or early childh...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (relative constriction of the diaphysis and flaring of the metaphysis), widening of the ribs and clavicles, platyspondyly (flattening of the spine bones), and cortical thinning. It is inherited in an autosomal recessive manner and only about 30 cases have been reported in the literature. Cranial involvement is minimal, with some individuals showing mild hyperostosis of the skull base and thickening of the frontal and occipital bones [medquad_ans_0005212-1].

**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.800, groundedness=0.928, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000135-3__dup2` (score=0.9952) - Weak or damaged valves in the veins can cause varicose veins. After your arteries and capillaries deliver oxygen-rich blood to your body, your veins return the blood to your heart. The veins in your legs must work aga...
2. `medquad_ans_0000108-3__dup2` (score=0.0331) - Major Causes Pulmonary embolism (PE) usually begins as a blood clot in a deep vein of the leg. This condition is called deep vein thrombosis. The clot can break free, travel through the bloodstream to the lungs, and b...
3. `medquad_ans_0000108-4__dup2` (score=0.0002) - Pulmonary embolism (PE) occurs equally in men and women. The risk increases with age. For every 10 years after age 60, the risk of having PE doubles. Certain inherited conditions, such as factor V Leiden, increase the...
4. `medquad_ans_0000104-1__dup4` (score=0.0001) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
5. `medquad_ans_0000108-8` (score=0.0001) - Preventing pulmonary embolism (PE) begins with preventing deep vein thrombosis (DVT). Knowing whether you're at risk for DVT and taking steps to lower your risk are important. Exercise your lower leg muscles if you're...

**Answer**:

Deep vein thrombosis (DVT) occurs when blood flow in the deep veins of the legs is restricted and slows down, which can lead to blood clot formation [medquad_ans_0000108-3__dup2]. This can happen due to prolonged periods of immobility, such as after some types of surgery, during long trips in a car or airplane, or when staying in bed for an extended time [medquad_ans_0000108-3__dup2]. Blood clots are also more likely to develop in veins that have been damaged from surgery or injury [medquad_ans_0000108-3__dup2].

**Supported citations**: `medquad_ans_0000108-3__dup2`
**Unsupported citations**: _none_

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=0.714, groundedness=0.645, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0004086-1` (score=0.0234) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at...
2. `medquad_ans_0000665-1` (score=0.0088) - Mitochondrial trifunctional protein deficiency is a rare condition that prevents the body from converting certain fats to energy, particularly during periods without food (fasting). Signs and symptoms of mitochondrial...
3. `medquad_ans_0004079-1` (score=0.0011) - Complex II deficiency is a mitochondrial disease. Mitochondria are specialized compartments in cells that create more than 90% of the energy needed by the body. In mitochondrial diseases, the mitochondria don't work c...
4. `medquad_ans_0000661-1` (score=0.0010) - Mitochondrial complex III deficiency is a genetic condition that can affect several parts of the body, including the brain, kidneys, liver, heart, and the muscles used for movement (skeletal muscles). Signs and sympto...
5. `medquad_ans_0000661-2` (score=0.0001) - The prevalence of mitochondrial complex III deficiency is unknown, although the condition is thought to be rare.

**Answer**:

The provided context does not contain information about Mitochondrial complex IV deficiency. Therefore, it is not possible to answer the question based on the given text. [medquad_ans_0004079-1] discusses complex II deficiency, but there is no mention of complex IV deficiency.

**Supported citations**: `medquad_ans_0004079-1`
**Unsupported citations**: _none_

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=0.857, groundedness=0.523, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.4373) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0000026-1__dup1` (score=0.0015) - Apraxia (called "dyspraxia" if mild) is a neurological disorder characterized by loss of the ability to execute or carry out skilled movements and gestures, despite having the desire and the physical ability to perfor...
3. `medquad_ans_0000495-1__dup1` (score=0.0007) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
4. `medquad_ans_0000388-2` (score=0.0004) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
5. `medquad_ans_0000097-1` (score=0.0004) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, which makes it difficult to move their eyes side-to-side, requiring them to turn their head to see peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4, while in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Individuals

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=0.909, groundedness=0.543, answer_relevance=0.909

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.9399) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0000897-1` (score=0.0012) - Sensorineural deafness and male infertility is a condition characterized by hearing loss and an inability to father children. Affected individuals have moderate to severe sensorineural hearing loss, which is caused by...
3. `medquad_ans_0000273-1` (score=0.0008) - Deafness and myopia syndrome is a disorder that causes problems with both hearing and vision. People with this disorder have moderate to profound hearing loss in both ears that may worsen over time. The hearing loss m...
4. `medquad_ans_0001496-1` (score=0.0002) - Congenital deafness with vitiligo and achalasia is a syndrome characterized by deafness present from birth (congenital), associated with short stature, vitiligo, muscle wasting and achalasia (swallowing difficulties)....
5. `medquad_ans_0000623-1` (score=0.0002) - Mandibulofacial dysostosis with microcephaly (MFDM) is a disorder that causes abnormalities of the head and face. People with this disorder often have an unusually small head at birth, and the head does not grow at th...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). Rarely, affected individuals have some underdeveloped inner ear structures in one or both ears. The abnormalities of the inner ear cause a form of hearing loss called sensorineural deafness that is present from birth (congenital). Because the inner ear is important for balance as well as hearing, development of motor skills, such as sitting and crawling, may be delayed in affected infants. In addition, people with LAMM syndrome often have abnormally small outer ears (microtia) with narrow ear canals. They can also have unusually small, widely spaced teeth (microdontia) [medquad_ans_000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000556-4` (score=0.9985) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
2. `medquad_ans_0000283-4` (score=0.9984) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...
3. `medquad_ans_0000883-4` (score=0.9978) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...
4. `medquad_ans_0000324-4` (score=0.9964) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
5. `medquad_ans_0000064-4` (score=0.9898) - Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in ...

**Answer**:

Apert syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Almost all cases of Apert syndrome result from new mutations in the gene, and occur in people with no history of the disorder in their family. Individuals with Apert syndrome, however, can pass along the condition to the next generation [medquad_ans_0000064-4]. Therefore, Apert syndrome is inherited, although most cases arise from new mutations.

**Supported citations**: `medquad_ans_0000064-4`
**Unsupported citations**: _none_

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.1319) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-3__dup2` (score=0.0073) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
3. `medquad_ans_0000213-2__dup1` (score=0.0050) - Generally, intensive care is needed. The neuroleptic or antipsychotic drug is discontinued, and the fever is treated aggressively. A muscle relaxant may be prescribed. Dopaminergic drugs, such as a dopamine agonist, h...
4. `medquad_ans_0000341-1__dup3` (score=0.0047) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
5. `medquad_ans_0000359-1__dup1` (score=0.0039) - A fever is a body temperature that is higher than normal. It is not an illness. It is part of your body's defense against infection. Most bacteria and viruses that cause infections do well at the body's normal tempera...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air that contains airborne barnyard dust contaminated by dried placental material, birth fluids, and excreta of infected animals. Other modes of transmission to humans, including tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission, are rare. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, the affected people with acute Q fever infection develop a chronic Q fever. Treatment of the acute form is

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.913, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0000038-5__dup1` (score=0.9936) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
2. `medquad_ans_0000113-4__dup2` (score=0.0287) - People who have diseases or conditions that affect the muscles, nerves, bones, or tissues that support breathing are at risk for respiratory failure. People who have lung diseases or conditions also are at risk for re...
3. `medquad_ans_0000020-4__dup2` (score=0.0239) - People who have conditions that damage the lungs or increase the risk of lung infections are at risk for bronchiectasis. Such conditions include: Cystic fibrosis. This disease leads to almost half of the cases of bron...
4. `medquad_ans_0000038-4__dup2` (score=0.0159) - Researchers don't yet know all of the causes of childhood interstitial lung disease (chILD). Many times, these diseases have no clear cause. Some conditions and factors that may cause or lead to chILD include: Inherit...
5. `medquad_ans_0000038-9` (score=0.0145) - At this time, most types of childhood interstitial lung disease (chILD) can't be prevented. People who have a family history of inherited (genetic) interstitial lung disease may want to consider genetic counseling. A ...

**Answer**:

Children who have a family history of interstitial lung disease or childhood interstitial lung disease (chILD), inherited surfactant disorders or a family history of such disorders, problems with aspiration, immune system disorders, exposure to environmental lung irritants such as molds and chemicals, systemic or autoimmune diseases (e.g., collagen vascular disease or inflammatory bowel disease), have undergone cancer treatments (e.g., radiation and chemotherapy), or have had a bone marrow or lung transplant are at risk for childhood interstitial lung disease [medquad_ans_0000038-5__dup1]. Additionally, children with certain surfactant problems or pulmonary hypertension, developmental or growth disorders, or those who have undergone bone marrow transplants are at higher risk of death, indicating increased severity [medquad_ans_0000038-5__dup1].

**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.9011) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000391-3` (score=0.0009) - FTDP-17 is caused by mutations in the MAPT gene. This gene is located on chromosome 17, which is how the disease got its name. The MAPT gene provides instructions for making a protein called tau. This protein is found...
3. `medquad_ans_0000533-1__dup1` (score=0.0009) - Juvenile arthritis (JA) is arthritis that happens in children. It causes joint swelling, pain, stiffness, and loss of motion. It can affect any joint, but is more common in the knees, hands, and feet. In some cases it...
4. `medquad_ans_0006001-1` (score=0.0006) - Takayasu arteritis is a condition that causes inflammation of the main blood vessel that carries blood from the heart to the rest of the body (aorta) and its associated branched blood vessels. As a result of the infla...
5. `medquad_ans_0000178-3` (score=0.0003) - CHMP2B-related frontotemporal dementia results from mutations in the CHMP2B gene. This gene provides instructions for making a protein called charged multivesicular body protein 2B. This protein is active in the brain...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels. It is generally diagnosed in late childhood or early adulthood and specifically affects the temporal arteries, which are located at the lower sides of the skull, directly underneath the temple. Affected individuals often have no signs or symptoms aside from a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is generally treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.521, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.9565) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
2. `medquad_ans_0001082-1` (score=0.9285) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0003264-1` (score=0.0049) - Isobutyryl-CoA dehydrogenase deficiency (IBD deficiency) is an inborn error of valine (an amino acid) metabolism. The symptoms, which may not develop until later in infancy or childhood, can include failure to thrive,...
4. `medquad_ans_0001082-2` (score=0.0004) - 2-methylbutyryl-CoA dehydrogenase deficiency is a rare disorder; its actual incidence is unknown. This disorder is more common, however, among Hmong populations in southeast Asia and in Hmong Americans. 2-methylbutyry...
5. `medquad_ans_0000855-2` (score=0.0003) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of the amino acid isoleucine. The inability to process isoleucine correctly leads to the buildup of the amino acid in the body, which can cause a variety of health problems ranging from severe and life-threatening to mild or absent. Signs and symptoms can begin a few days after birth or later in childhood and often include poor feeding, lack of energy (lethargy), vomiting, and irritability. These symptoms may progress to serious medical problems such as difficulty breathing, seizures, and coma. Additional issues can include poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills like standing and walking. Symptoms may be triggered by prolonged fasting, infections, or increased protein intake.

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

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005219-3` (score=0.9912) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...
2. `medquad_ans_0000741-2` (score=0.0002) - Northern epilepsy appears to affect only individuals of Finnish ancestry, particularly those from the Kainuu region of northern Finland. Approximately 1 in 10,000 individuals in this region have the condition.
3. `medquad_ans_0000852-2` (score=0.0001) - Pyridoxal 5'-phosphate-dependent epilepsy is a rare condition; approximately 14 cases have been described in the scientific literature.
4. `medquad_ans_0000853-2` (score=0.0001) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
5. `medquad_ans_0005219-4` (score=0.0001) - How might pyridoxine-dependent epilepsy be treated? Anticonvulsant drugs, which are usually given to control seizures, are ineffective in people with pyridoxine-dependent epilepsy. Instead, people with this type of se...

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The ALDH7A1 gene provides instructions for making an enzyme called α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the protein building block (amino acid) lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. It is unclear how a lack of pyridoxine causes the seizures that are characteristic of this condition. Some individuals with pyridoxine-dependent epilepsy do not have identified mutations in the ALDH7A1 gene. In these cases, the cause of the condition is unknown [medquad_ans_00052

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=1.000, groundedness=0.952, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.9949) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
2. `medquad_ans_0000001_7-2` (score=0.0056) - The risk factors for childhood AML, childhood CML, JMML, and MDS are similar. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer...
3. `medquad_ans_0000001_2-2` (score=0.0033) - Smoking, previous chemotherapy treatment, and exposure to radiation may affect the risk of adult AML. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean ...
4. `medquad_ans_0000004_6-5` (score=0.0027) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...
5. `medquad_ans_0000001_6-6` (score=0.0023) - Key Points Past treatment for cancer and certain genetic conditions affect the risk of having childhood ALL. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does no...

**Answer**:

Children who are infected with the Epstein-Barr virus, those infected with the human immunodeficiency virus (HIV), children with certain diseases of the immune system, children with a personal history of mononucleosis ("mono"), and children who have a parent or sibling with a personal history of Hodgkin lymphoma are at risk for childhood Hodgkin lymphoma. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma in children due to its effect on the immune system [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=0.800, groundedness=0.961, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.8953) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0001795-1` (score=0.7929) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0000088-11` (score=0.0098) - Most people with celiac disease have a significant improvement in symptoms when they follow a gluten-free diet. Health care providers typically refer people to a dietitian who specializes in treating people with the d...
4. `medquad_ans_0000084-2` (score=0.0050) - Atopic dermatitis is a common disorder that affects 10 to 20 percent of children and 5 to 10 percent of adults.
5. `medquad_ans_0002241-1` (score=0.0029) - Familial dermographism is a condition also known as skin writing. When people who have dermatographia lightly scratch their skin, the scratches redden into a raised wheal similar to hives. Signs and symptoms of dermat...

**Answer**:

Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These lesions are more common on the knees, elbows, buttocks, and shoulder blades. The condition is often associated with the inability to digest gluten and may present with symptoms such as fluid-filled sores, red lesions resembling hives, and itchiness, redness, and burning. A burning sensation may precede lesion formation, and the lesions are usually scratched off by the time a patient presents for a physical exam, appearing as erosions and excoriations. It is also associated with celiac disease, and patients may experience dental enamel defects in permanent teeth. Less than 20 percent of people with dermatitis herpetiformis have symptoms of celiac disease [medquad_ans_0001795-1]. [medquad_ans_0000095-1

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: _none_

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.857, groundedness=0.576, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000210-2__dup2` (score=0.9956) - Treatment for myotonia may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help muscle function.
2. `medquad_ans_0000709-5` (score=0.9926) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
3. `medquad_ans_0000084-2__dup1` (score=0.9908) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
4. `medquad_ans_0000040-2__dup1` (score=0.9899) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...
5. `medquad_ans_0000040-3__dup1` (score=0.9090) - Most individuals with myotonia congenita lead long, productive lives. Although muscle stiffness may interfere with walking, grasping, chewing, and swallowing, it is usually relieved with exercise.

**Answer**:

Treatments for myotonia congenita may include medications such as quinine, phenytoin, and mexiletine, which are anticonvulsant drugs. Physical therapy and other rehabilitative measures may also help improve muscle function. In most cases, stiff muscles resolve with exercise or light movement, especially after rest. For individuals with more limiting symptoms, these medications and rehabilitative therapies have shown some success [medquad_ans_0000210-2__dup2, medquad_ans_0000040-2__dup1]. Most people with myotonia congenita do not require special treatments, as symptoms often improve with activity [medquad_ans_0000040-3__dup1].

**Supported citations**: `medquad_ans_0000040-3__dup1`
**Unsupported citations**: `medquad_ans_0000210-2__dup2, medquad_ans_0000040-2__dup1`

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=1.000, groundedness=0.464, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000133-13` (score=0.8860) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
2. `medquad_ans_0000133-11` (score=0.8679) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the i...
3. `medquad_ans_0000133-12` (score=0.0097) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
4. `medquad_ans_0001037-5` (score=0.0008) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Hu...
5. `medquad_ans_0000276-4__dup1` (score=0.0004) - The National Institute of Neurological Disorders and Stroke, the Eunice Kennedy Shriver National Institute of Child Health and Human Development, and other institutes of the National Institutes of Health (NIH) conduct...

**Answer**:

For Wilson disease, treatment involves a lifelong effort to reduce and control copper levels in the body. This may include medications, changes in eating, diet, and nutrition, and in some cases, a liver transplant. People should reduce their dietary copper intake by avoiding high-copper foods such as shellfish, liver, mushrooms, nuts, and chocolate, especially during initial treatment; however, they should consult a health care provider to determine if these foods can be consumed in moderation during maintenance treatment. Additionally, individuals should check the copper levels in their tap water, as water from copper pipes or wells may contain copper residue; running water helps lower copper levels. People should not use copper containers or cookware to store or prepare food or drinks. It is also important to discuss the use of complementary and alternative medical practices, including vitamins and dietary supplements, with a health care provider, as some supplements contain copper. If a supplement is recommended, a pharmacist can suggest copper-free options

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.896, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.9980) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000920-3` (score=0.0001) - Most people with Smith-Magenis syndrome have a deletion of genetic material from a specific region of chromosome 17. Although this region contains multiple genes, researchers believe that the loss of one particular ge...
3. `medquad_ans_0000592-2` (score=0.0001) - Lennox-Gastaut syndrome affects an estimated 1 in 50,000 to 1 in 100,000 children. This condition accounts for about 4 percent of all cases of childhood epilepsy. For unknown reasons, it appears to be more common in m...
4. `medquad_ans_0001041-3` (score=0.0001) - Mutations in the PRKAG2 gene cause Wolff-Parkinson-White syndrome. A small percentage of all cases of Wolff-Parkinson-White syndrome are caused by mutations in the PRKAG2 gene. Some people with these mutations also ha...
5. `medquad_ans_0000883-4` (score=0.0001) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component [medquad_ans_0000592-3]. In about one-third of cases, the cause is unknown and described as cryptogenic, meaning there is no identifiable underlying reason [medquad_ans_0000592-3]. In the remaining two-thirds of cases, the syndrome is symptomatic and associated with existing neurological problems such as brain injuries, blood flow issues, infections, or brain malformations like cortical dysplasia, or it may occur as part of a genetic disorder such as tuberous sclerosis complex [medquad_ans_0000592-3]. However, no specific gene mutations have been identified as directly causing Lennox-Gastaut syndrome.

**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=0.400, groundedness=0.969, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000065-2` (score=0.9995) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.
2. `medquad_ans_0000066-2` (score=0.9994) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
3. `medquad_ans_0000050-2` (score=0.9991) - The prevalence of aminoacylase 1 deficiency is unknown.
4. `medquad_ans_0000711-2` (score=0.9990) - N-acetylglutamate synthase deficiency is a very rare disorder. Only a few cases have been reported worldwide, and the overall incidence is unknown.
5. `medquad_ans_0000420-2` (score=0.2978) - The worldwide incidence of glycine encephalopathy is unknown. Its frequency has been studied in only a few regions: this condition affects about 1 in 55,000 newborns in Finland and about 1 in 63,000 newborns in Britis...

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
5. `medquad_ans_0000379-1` (score=0.0001) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. ...

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths (papillomas) around the nostrils, lips, anus, and female genitalia, and may also be present in the throat, causing problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly), and a deep split in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=1.000, groundedness=0.866, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0000418-1__dup2` (score=0.9876) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
2. `medquad_ans_0000418-2__dup1` (score=0.9752) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
3. `medquad_ans_0000415-1__dup2` (score=0.9549) - A single-celled parasite called Toxoplasma gondii causes a disease known as toxoplasmosis. While the parasite is found throughout the world, more than 60 million people in the United States may be infected with the To...
4. `medquad_ans_0000418-7` (score=0.0132) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
5. `medquad_ans_0000269-2__dup2` (score=0.0017) - There are three different filarial species that can cause lymphatic filariasis in humans. Most of the infections worldwide are caused by Wuchereria bancrofti. In Asia, the disease can also be caused by Brugia malayi a...

**Answer**:

People at risk for trichinellosis (also known as trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating undercooked wild game, especially bear meat, puts individuals at risk. Additionally, individuals who taste very small amounts of undercooked meat during preparation or cooking are also at risk. Pregnant women and those with compromised immune systems are not specifically mentioned as being at risk for trichinellosis in the provided context, though they are noted as being cautious with toxoplasmosis [medquad_ans_0000418-2__dup1].

The primary risk group identified in the context is consumers of raw or undercooked meat, especially from wild animals or domestic pigs [medquad_ans_0000418-1__dup2].

**Supported citations**: `medquad_ans_0000418-2__dup1`, `medquad_ans_0000418-1__dup2`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.940, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.9994) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0000108-3__dup1` (score=0.0019) - As children with Dravet syndrome get older, their decline in cognitive function stabilizes, and in many, it improves slightly. However, most teenagers with Dravet syndrome are dependent on caregivers. The degree of co...
3. `medquad_ans_0003700-2` (score=0.0001) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available...
4. `medquad_ans_0005016-2` (score=0.0001) - What are the signs and symptoms of Prader-Willi syndrome? In infancy, Prader-Willi syndrome (PWS) is characterized by weak muscle tone (hypotonia), feeding difficulties, poor growth, and delayed development. In later ...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The symptoms of Dravet syndrome include absence seizures, ataxia, autosomal dominant inheritance, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2]. Additionally, as children with Dravet syndrome get older, their decline in cognitive function stabilizes and may improve slightly, though most teenagers remain dependent on caregivers, and the degree of cognitive impairment correlates with the frequency of seizures [medquad_ans_0000108-3__dup1].

**Supported citations**: `medquad_ans_0001924-2`, `medquad_ans_0000108-3__dup1`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.954, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000826-3` (score=0.9974) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, whi...
2. `medquad_ans_0000011-13` (score=0.0010) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
3. `medquad_ans_0001030-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene and occur in pe...
4. `medquad_ans_0000575-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000516-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...

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
3. `medquad_ans_0005127-2` (score=0.0001) - What are the signs and symptoms of Pseudoachondroplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudoachondroplasia. If the information is available, the table below includ...
4. `medquad_ans_0003459-2` (score=0.0001) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
5. `medquad_ans_0000314-1` (score=0.0001) - Ellis-van Creveld syndrome is an inherited disorder of bone growth that results in very short stature (dwarfism). People with this condition have particularly short forearms and lower legs and a narrow chest with shor...

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, cleft palate, cleft upper lip, high palate, hydronephrosis, hypoplasia of the radius, hypotelorism, long face, mesomelia, narrow face, oligodactyly (hands), proximal placement of thumb, and proximal radial head dislocation [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.910, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.9994) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0004524-1` (score=0.0001) - What are the signs and symptoms of Okamoto syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Okamoto syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0000764-1__dup1` (score=0.0001) - Summary : Radiation is energy that travels in the form of waves or high-speed particles. It occurs naturally in sunlight. Man-made radiation is used in X-rays, nuclear weapons, nuclear power plants and cancer treatmen...
4. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
5. `medquad_ans_0004384-2` (score=0.0001) - What are the signs and symptoms of Neuronal ceroid lipofuscinosis 10 ? The Human Phenotype Ontology provides the following list of signs and symptoms for Neuronal ceroid lipofuscinosis 10 . If the information is avail...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of the blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is based on data from the Human Phenotype Ontology, which compiles information from sources such as Orphanet, with frequencies often expressed as percentages or fractions representing the proportion of patients exhibiting each symptom [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=0.111, groundedness=0.000, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000484-2` (score=0.9995) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
2. `medquad_ans_0000195-2` (score=0.9995) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
3. `medquad_ans_0000091-2` (score=0.9995) - This condition is rare, affecting fewer than 1 per million people.
4. `medquad_ans_0000770-2` (score=0.9995) - The precise incidence of this condition is unknown. It is estimated to affect 1 to 3 per million people worldwide.
5. `medquad_ans_0001065-2` (score=0.9994) - The estimated incidence of X-linked thrombocytopenia is between 1 and 10 per million males worldwide; this condition is rarer among females.

**Answer**:

The provided context does not contain specific information about the incidence of thrombocytopenia-absent radius syndrome. Therefore, it is not possible to determine how many people are affected by this condition based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.909, groundedness=0.812, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000006_3-8` (score=0.9880) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000030_1-7` (score=0.0021) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000265-3__dup1` (score=0.0007) - Even with the best of care, children with Tay-Sachs disease usually die by age 4, from recurring infection.
4. `medquad_ans_0000031_1-8` (score=0.0004) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000032_1-8` (score=0.0001) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not mention any research or clinical trials being conducted for Tay-Sachs Disease. [medquad_ans_0000006_3-8] and [medquad_ans_0000030_1-7] discuss clinical trials for cancer treatments, including therapies involving lenalidomide and combinations of biologic or chemotherapy agents, but do not reference Tay-Sachs Disease or related research. Therefore, no information is available in the context to answer the question.

**Supported citations**: `medquad_ans_0000006_3-8`, `medquad_ans_0000030_1-7`
**Unsupported citations**: _none_

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.4357) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000010-1__dup2` (score=0.0318) - Agnosia is a rare disorder characterized by an inability to recognize and identify objects or persons. People with agnosia may have difficulty recognizing the geometric features of an object or face or may be able to ...
3. `medquad_ans_0000008-2` (score=0.0004) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
4. `medquad_ans_0000025-1__dup1` (score=0.0001) - An acanthoma is a small, reddish bump that usually develops on the skin of an older adult. There are several types of acanthoma, including "acantholytic", "epidermolytic", "clear cell", and "melanoacanthoma". Though m...
5. `medquad_ans_0000141-2__dup1` (score=0.0001) - What are the signs and symptoms of ADCY5-related dyskinesia ? The Human Phenotype Ontology provides the following list of signs and symptoms for ADCY5-related dyskinesia . If the information is available, the table be...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. Achromatopsia also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals can also have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green. Complete achromatops

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=0.714, groundedness=0.797, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000029-18` (score=0.9947) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
2. `medquad_ans_0000052-2__dup1` (score=0.9872) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
3. `medquad_ans_0000039-9` (score=0.9869) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.
4. `medquad_ans_0000039-11` (score=0.9823) - Treatments your doctor will likely recommend before knee replacement include - exercises to strengthen the muscles around the knee and improve flexibility - weight loss, if needed, to reduce the load the knee must bea...
5. `medquad_ans_0000039-26` (score=0.9592) - You can learn more about knee replacement from the following resources. National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) Information Clearinghouse National Institutes of Health 1 AMS Circl...

**Answer**:

The provided context does not specify treatments for knee replacement surgery itself, but rather outlines treatments recommended before knee replacement. These include:

- Exercises to strengthen the muscles around the knee and improve flexibility
- Weight loss, if needed, to reduce the load the knee must bear
- Walking aids such as canes to reduce stress on the joint
- Shoe inserts to improve knee alignment
- Medicines to relieve pain

The main reason for knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems while preserving movement [medquad_ans_0000039-9]. However, specific treatments for the procedure (e.g., surgical techniques, post-operative care) are not detailed in the context.

Therefore, based on the given information, the treatments recommended prior to knee replacement are listed above. No treatments for the surgery itself are described. [medquad_ans_0000039-11]

**Supported citations**: `medquad_ans_0000039-9`, `medquad_ans_0000039-11`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.872, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000606-1__dup1` (score=0.0006) - Metabolism is the process your body uses to get or make energy from the food you eat. Food is made up of proteins, carbohydrates, and fats. Chemicals in your digestive system break the food parts down into sugars and ...
2. `medquad_ans_0000090-1__dup2` (score=0.0001) - Metabolicsyndrome is the name for a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. The term "metabolic" refers to the biochemical processes involv...
3. `medquad_ans_0000090-5__dup1` (score=0.0001) - Metabolic syndrome is a group of risk factors that raises your risk for heart disease and other health problems, such as diabetes and stroke. These risk factors can increase your risk for health problems even if they'...
4. `medquad_ans_0002288-1` (score=0.0001) - What are the signs and symptoms of Familial partial lipodystrophy associated with PPARG mutations? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial partial lipodystrophy asso...
5. `medquad_ans_0000090-3__dup2` (score=0.0001) - Metabolic syndrome has several causes that act together. You can control some of the causes, such as overweight and obesity, an inactive lifestyle, and insulin resistance. You can't control other factors that may play...

**Answer**:

The provided context does not describe the symptoms of Metabolic Syndrome. It outlines the risk factors associated with Metabolic Syndrome, such as a large waistline, high triglyceride levels, low HDL cholesterol, high blood pressure, and high fasting blood sugar [medquad_ans_0000090-1__dup2]. However, it does not list specific symptoms. Therefore, based on the given information, the symptoms of Metabolic Syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000090-1__dup2`
**Unsupported citations**: _none_

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005211-1` (score=0.9994) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
2. `medquad_ans_0002152-2` (score=0.0001) - What are the signs and symptoms of Erdheim-Chester disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Erdheim-Chester disease. If the information is available, the table below ...
3. `medquad_ans_0000986-2__dup1` (score=0.0001) - What are the signs and symptoms of Catamenial pneumothorax? The Human Phenotype Ontology provides the following list of signs and symptoms for Catamenial pneumothorax. If the information is available, the table below ...
4. `medquad_ans_0002060-1` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome, periodontitis type ? The Human Phenotype Ontology provides the following list of signs and symptoms for Ehlers-Danlos syndrome, periodontitis type . If the in...
5. `medquad_ans_0002259-1` (score=0.0001) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

The symptoms of Pyknoachondrogenesis include abnormal vertebral ossification, abnormality of pelvic girdle bone morphology, abnormality of the mouth, abnormality of the sacrum, depressed nasal ridge, enlarged thorax, increased bone mineral density, low-set, posteriorly rotated ears, macrocephaly, micromelia, palpebral edema, premature birth, short stature, short thorax, thickened nuchal skin fold, and stillbirth. These symptoms are reported in approximately 90% of patients with this condition [medquad_ans_0005211-1].

**Supported citations**: `medquad_ans_0005211-1`
**Unsupported citations**: _none_

### Example 243 (`0003965-2`)

**Question**: What are the symptoms of Metachromatic leukodystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003965-2` (score=0.9993) - What are the signs and symptoms of Metachromatic leukodystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy. If the information is available, the ta...
2. `medquad_ans_0003966-1` (score=0.0163) - What are the signs and symptoms of Metachromatic leukodystrophy due to saposin B deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Metachromatic leukodystrophy due to sapos...
3. `medquad_ans_0003965-4` (score=0.0002) - Who might consider genetic carrier testing for a family history of metachromatic leukodystrophy? If someone has a family history of metachromatic leukodystrophy (MLD) or someone is known to be a carrier for MLD, indiv...
4. `medquad_ans_0003290-1` (score=0.0001) - What are the signs and symptoms of Jansen type metaphyseal chondrodysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Jansen type metaphyseal chondrodysplasia. If the informat...
5. `medquad_ans_0000181-1__dup2` (score=0.0001) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, limitation of joint mobility, muscular hypotonia, nystagmus, optic atrophy, aganglionic megacolon, ataxia, Babinski sign, bulbar palsy, cholecystitis, chorea, delusions, dysarthria, dystonia, EMG: neuropathic changes, emotional lability, gallbladder dysfunction, hallucinations, hyporeflexia, increased CSF protein, intellectual disability, loss of speech, mental deterioration, peripheral demyelination, progressive peripheral neuropathy, spastic tetraplegia, tetraplegia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000362-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000095-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000585-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000673-4` (score=0.9986) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000786-4` (score=0.9985) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not mention chronic granulomatous disease or its inheritance pattern. Therefore, it is not possible to determine from the given information whether chronic granulomatous disease is inherited.

Answer: Insufficient context [medquad_ans_0000362-4, medquad_ans_0000095-4, medquad_ans_0000585-4, medquad_ans_0000673-4, medquad_ans_0000786-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000362-4, medquad_ans_0000095-4, medquad_ans_0000585-4, medquad_ans_0000673-4, medquad_ans_0000786-4`

### Example 245 (`0000061-3__dup1`)

**Question**: What is the outlook for Central Cord Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.941, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000062-3__dup1` (score=0.9964) - Central pain syndrome is not a fatal disorder, but the syndrome causes disabling chronic pain and suffering among the majority of individuals who have it.
2. `medquad_ans_0000061-3__dup1` (score=0.9886) - The prognosis for central cord syndrome varies, but most people whose syndrome is caused by trauma have some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict he likeliho...
3. `medquad_ans_0000266-3__dup1` (score=0.9799) - With treatment, individuals with tethered spinal cord syndrome have a normal life expectancy. However, some neurological and motor impairments may not be fully correctable. Surgery soon after symptoms emerge appears t...
4. `medquad_ans_0000085-2__dup1` (score=0.0086) - Currently, only central core disease has an effective treatment (see above). There are no known cures for any of these disorders. Supportive treatment may involve orthopedic treatments, as well as physical, occupation...
5. `medquad_ans_0000061-2__dup1` (score=0.0046) - There is no cure for central cord syndrome although some people recover near-normal function. There is no standard course of treatment, although drug therapy, surgery, and rest are often part of the program. Magnetic ...

**Answer**:

The outlook for central cord syndrome varies, but most people who have it due to trauma experience some recovery of neurological function. Evaluation of abnormal signals on MRI images can help predict the likelihood of natural neurological recovery. Those who receive medical intervention soon after injury often have good outcomes. Many people recover substantial function after their initial injury, and the ability to walk is recovered in most cases, although some impairment may remain. Improvement typically occurs first in the legs, then the bladder, and may be seen in the arms; hand function recovers last, if at all. Recovery is generally better in younger patients compared to those over the age of 50 [medquad_ans_0000061-3__dup1]. There is no cure for central cord syndrome, but treatment options such as drug therapy, surgery, and rest are often part of the management plan, and early surgery may improve recovery chances [medquad_ans_00000

**Supported citations**: `medquad_ans_0000061-3__dup1`
**Unsupported citations**: _none_

### Example 246 (`0000094-3__dup3`)

**Question**: What causes Cyclic Vomiting Syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.631, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000094-3__dup3` (score=0.9953) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way ...
2. `medquad_ans_0000094-12` (score=0.0025) - During the prodrome and vomiting phases of cyclic vomiting syndrome, a person will generally take in little or no nutrition by mouth. During the recovery phase, the person may be quite hungry as soon as the vomiting s...
3. `medquad_ans_0000261-3` (score=0.0017) - Although the causes of cyclic vomiting syndrome have yet to be determined, researchers have proposed several factors that may contribute to the disorder. These factors include changes in brain function, hormonal abnor...
4. `medquad_ans_0000192-2__dup2` (score=0.0012) - Symptoms include - jaundice, which causes a yellowing of the skin and eyes - fatigue - abdominal pain - loss of appetite - nausea - vomiting - diarrhea - low grade fever - headache However, some people do not have sym...
5. `medquad_ans_0000107-8` (score=0.0006) - The problems of gastroparesis can include - severe dehydration due to persistent vomiting - gastroesophageal reflux disease (GERD), which is GER that occurs more than twice a week for a few weeks; GERD can lead to eso...

**Answer**:

The cause of cyclic vomiting syndrome is unknown. However, some experts believe that possible problems with bodily functions may contribute to the cause, such as gastrointestinal motility, central nervous system function, autonomic nervous system function, hormone imbalances, and in children, an abnormal inherited gene [medquad_ans_0000094-3__dup3]. Researchers have also proposed that the condition may be related to changes in brain function, hormonal abnormalities, and gastrointestinal problems, and many believe it is a migraine-like condition involving changes in signaling between nerve cells in certain areas of the brain [medquad_ans_00000261-3].

**Supported citations**: `medquad_ans_0000094-3__dup3`
**Unsupported citations**: `medquad_ans_00000261-3`

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000189-1__dup2` (score=0.9981) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
2. `medquad_ans_0000180-1__dup2` (score=0.6764) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
3. `medquad_ans_0000177-1__dup2` (score=0.6129) - Summary : Children's healthy development depends on safe and positive experiences when they are very young. If you work or go to school, you want to know that your child is in good hands while you are away. You may ch...
4. `medquad_ans_0000925-1__dup1` (score=0.0046) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
5. `medquad_ans_0000603-1__dup1` (score=0.0020) - Summary : Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel and act as we cope with life. It also helps determine how we handle stress, relate to others, and mak...

**Answer**:

Yes, there is information about child mental health. The context indicates that it is important to recognize and treat mental illnesses in children early on, as they become a regular part of the child's behavior and are more difficult to treat later. Warning signs of a more serious mental health problem include problems in more than one setting, changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to younger behaviors like bedwetting, signs of being upset such as sadness or tearfulness, self-destructive behaviors like head-banging or sudden injuries, and repeated thoughts of death. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history. Treatments include medicines and talk therapy [medquad_ans_0000180-1__dup2].

**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.111, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000088-4__dup2` (score=0.9988) - A person may experience digestive signs and symptoms, or symptoms in other parts of the body. Digestive signs and symptoms are more common in children and can include - abdominal bloating - chronic diarrhea - constipa...
2. `medquad_ans_0001618-2` (score=0.9976) - What are the signs and symptoms of Cowden syndrome? Cowden syndrome is characterized primarily by multiple, noncancerous growths (called hamartomas) on various parts of the body. Approximately 99% of people affected b...
3. `medquad_ans_0004375-1` (score=0.0001) - What are the signs and symptoms of Neurofibromatosis-Noonan syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Neurofibromatosis-Noonan syndrome. If the information is availab...
4. `medquad_ans_0000003-2__dup3` (score=0.0001) - Signs and symptoms of Cushing's syndrome vary, but most people with the disorder have upper body obesity, a rounded face, increased fat around the neck, and relatively slender arms and legs. Children tend to be obese ...
5. `medquad_ans_0004450-2` (score=0.0001) - What are the signs and symptoms of Noonan syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Noonan syndrome. If the information is available, the table below includes how oft...

**Answer**:

The provided context does not mention Cowden syndrome or its symptoms. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.889, groundedness=0.785, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.9929) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000084-2__dup1` (score=0.0662) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
3. `medquad_ans_0000576-5` (score=0.0004) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
4. `medquad_ans_0006267-3` (score=0.0003) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
5. `medquad_ans_0004219-1` (score=0.0001) - Muscular dystrophy (MD) refers to a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...

**Answer**:

The provided context does not specify the treatments for Fukuyama congenital muscular dystrophy. While it lists resources related to the diagnosis or management of Fukuyama congenital muscular dystrophy, such as gene reviews, genetic testing registry entries, and MedlinePlus encyclopedia entries, no specific treatments are mentioned [medquad_ans_0000395-5]. Therefore, based on the available information, the treatments for Fukuyama congenital muscular dystrophy cannot be determined.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.855, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000004_6-4` (score=0.9976) - Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general...
2. `medquad_ans_0003789-4` (score=0.0865) - How is Mantle cell lymphoma diagnosed? Mantle cell lymphoma is diagnosed by a biopsy (surgical removal) of the lymph nodes. If lymph nodes are not easily accessible to be biopsied, a fine needle aspiration may be perf...
3. `medquad_ans_0000004_1-4` (score=0.0148) - Tests that examine the lymph nodes are used to detect (find) and diagnose adult Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general sign...
4. `medquad_ans_0000004_7-2` (score=0.0001) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...
5. `medquad_ans_0000004_6-3` (score=0.0001) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with ...

**Answer**:

Childhood Hodgkin lymphoma is diagnosed through a combination of tests that examine the lymph system. These include a physical exam and history, CT scan (CAT scan), PET scan (positron emission tomography scan), chest x-ray, complete blood count (CBC), blood chemistry studies, sedimentation rate (ESR), and lymph node biopsy. The physical exam and history assess general signs of health and past medical history. Imaging tests such as CT and PET scans help detect and locate malignant tumor cells, with PET scans showing increased glucose uptake by malignant cells. Chest x-rays evaluate the chest area, while blood tests like CBC, blood chemistry studies, and sedimentation rate provide information on blood cell counts and signs of inflammation, which may indicate lymphoma. A lymph node biopsy, which involves removing all or part of a lymph node, is often performed to confirm the diagnosis [medquad_ans_0000004_6-4].

**Supported citations**: `medquad_ans_0000004_6-4`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000362-4` (score=0.9989) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000585-4` (score=0.9989) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000673-4` (score=0.9986) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
4. `medquad_ans_0000786-4` (score=0.9985) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000700-4` (score=0.9985) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not specify whether KBG syndrome is inherited. While several documents describe autosomal dominant inheritance patterns for certain conditions, none explicitly mention KBG syndrome or its inheritance pattern. Therefore, based on the given information, it is not possible to determine if KBG syndrome is inherited.

[Note: No specific document cites KBG syndrome, so no citation can be made from the provided context.]

**Supported citations**: _none_
**Unsupported citations**: `Note: No specific document cites KBG syndrome, so no citation can be made from the provided context.`

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.667, groundedness=0.993, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000195-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
2. `medquad_ans_0000194-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000193-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000197-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
5. `medquad_ans_0000196-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe intellectual and physical disability. Additional symptoms may include epilepsy, painful, itchy skin lesion (chilblains), vision problems, and joint stiffness. Symptoms usually progress over several months before the disease course stabilizes. There are six different types of Aicardi-Goutieres syndrome, which are distinguished by the gene that causes the condition: TREX1, RNASEH2A, RNASEH2B, RNASEH2C, SAMHD1, and ADAR genes. Most cases are inherited in an autosomal recessive pattern, although rare autosomal dominant cases have been reported. Treatment is symptomatic and supportive [medquad_ans_0000195-1__dup1].

**Supported citations**: `medquad_ans_0000195-1__dup1`
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.833, groundedness=0.720, answer_relevance=0.967

**Top docs**:

1. `medquad_ans_0000006_3-8` (score=0.9938) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
2. `medquad_ans_0000025_1-8` (score=0.9928) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000006_2-7` (score=0.9924) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000016_1-8` (score=0.9921) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000019_2-7` (score=0.9905) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...

**Answer**:

The provided context does not specify any research or clinical trials being conducted specifically for Renal Cell Cancer. While the context mentions general information about clinical trials and treatments such as lenalidomide and biologic therapy, it does not reference Renal Cell Cancer or any trials related to it. [medquad_ans_0000006_3-8] and [medquad_ans_0000025_1-8] discuss general cancer treatments and clinical trials but lack specific details about Renal Cell Cancer. Therefore, based on the given information, it is not possible to determine what research or clinical trials are being done for Renal Cell Cancer.

**Supported citations**: `medquad_ans_0000006_3-8`, `medquad_ans_0000025_1-8`
**Unsupported citations**: _none_

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.958, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000190-9` (score=0.9146) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
2. `medquad_ans_0000184-9` (score=0.8577) - - Cirrhosis is a condition in which the liver slowly deteriorates and is unable to function normally due to chronic, or long lasting, injury. Scar tissue replaces healthy liver tissue and partially blocks the flow of ...
3. `medquad_ans_0000193-7` (score=0.5855) - To prevent cirrhosis, - see your doctor for treatment of your liver disease. Many of the causes of cirrhosis are treatable. Early treatment may prevent cirrhosis. - try to keep your weight in the normal range. Being o...
4. `medquad_ans_0000188-2__dup2` (score=0.0037) - NASH is usually a silent disease with few or no symptoms. Patients generally feel well in the early stages and only begin to have symptomssuch as fatigue, weight loss, and weaknessonce the disease is more advanced or ...
5. `medquad_ans_0000184-4__dup2` (score=0.0035) - As the liver fails, complications may develop. In some people, complications may be the first signs of the disease. Complications of cirrhosis may include the following: Portal hypertension. The portal vein carries bl...

**Answer**:

For primary biliary cirrhosis, a healthcare provider may prescribe ursodiol (Actigall, Urso) to treat the condition. Early treatment with this medication reduces the likelihood of needing a liver transplant and improves survival. A liver transplant may be considered when cirrhosis leads to liver failure or when treatment for complications is ineffective [medquad_ans_0000190-9].

**Supported citations**: `medquad_ans_0000190-9`
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=0.750, groundedness=0.932, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000299-1__dup2` (score=0.9778) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
2. `medquad_ans_0000967-1__dup1` (score=0.0016) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoi...
3. `medquad_ans_0000378-4` (score=0.0014) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of Floating-Harbor syndrome result from new mutations...
4. `medquad_ans_0000378-2` (score=0.0004) - Floating-Harbor syndrome is a rare disorder; only about 50 cases have been reported in the medical literature.
5. `medquad_ans_0000137-1__dup1` (score=0.0004) - Acute respiratory distress syndrome (ARDS) is a life-threatening lung condition that prevents enough oxygen from getting to the lungs and into the blood. People who develop ARDS often are very ill with another disease...

**Answer**:

Drowning occurs when people get too much water in their lungs, which can happen even with as little as an inch or two of water. It can be particularly dangerous for babies in a sink or bathtub and for preschoolers in swimming pools. Individuals with seizure disorders are also at increased risk. Drowning can happen quickly and silently. Precautions include installing fences around pools, supervising children near any body of water, avoiding alcohol or sedatives when swimming or boating, wearing life jackets when boating, and learning CPR [medquad_ans_0000299-1__dup2].

**Supported citations**: `medquad_ans_0000299-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=0.857, groundedness=0.990, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.9604) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
2. `medquad_ans_0000313-1` (score=0.2940) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
3. `medquad_ans_0002063-1` (score=0.0005) - Ehlers-Danlos syndrome (EDS), vascular type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. It is generally considered the most severe form of Ehlers-Danlos syndrome....
4. `medquad_ans_0000313-2` (score=0.0003) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...
5. `medquad_ans_0002055-1` (score=0.0001) - Ehlers-Danlos syndrome (EDS), dermatosparaxis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common symptoms include soft, doughy skin that is extremely fragile...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is an inherited connective tissue disorder caused by defects in a protein called collagen. It is characterized by hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera. This type is caused by mutations in the PLOD1 gene and is inherited in an autosomal recessive manner. Treatment is focused on preventing serious complications and relieving associated signs and symptoms [medquad_ans_0002058-1].

**Supported citations**: `medquad_ans_0002058-1`
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.971, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.9994) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0001862-2` (score=0.0008) - What are the signs and symptoms of Dihydropyrimidine dehydrogenase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropyrimidine dehydrogenase deficiency. If the info...
3. `medquad_ans_0000898-2` (score=0.0001) - Sepiapterin reductase deficiency appears to be a rare condition. At least 30 cases have been described in the scientific literature.
4. `medquad_ans_0002595-2` (score=0.0001) - What are the signs and symptoms of glucose-6-phosphate dehydrogenase (G6PD) deficiency? People with G6PD deficiency do not have signs of the disease unless their red blood cells are exposed to certain chemicals in foo...
5. `medquad_ans_0006580-1` (score=0.0001) - What are the signs and symptoms of 3 alpha methylcrotonyl-CoA carboxylase 2 deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for 3 alpha methylcrotonyl-CoA carboxylase 2 defic...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment (90%), microcephaly (90%), cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.143, groundedness=0.000, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000989-2` (score=0.9994) - This condition affects an estimated 1 in 50,000 people.
2. `medquad_ans_0000484-2` (score=0.9994) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000195-2` (score=0.9994) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000790-2` (score=0.9994) - The prevalence of this condition is uncertain; estimates range from 1 in 25,000 to 300,000 individuals.
5. `medquad_ans_0000091-2` (score=0.9993) - This condition is rare, affecting fewer than 1 per million people.

**Answer**:

The provided context does not contain any information about monilethrix or its incidence or prevalence. Therefore, it is not possible to determine how many people are affected by monilethrix based on the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.3190) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0000708-2` (score=0.0024) - The prevalence of this condition is unknown.
3. `medquad_ans_0000439-2` (score=0.0021) - This condition is very rare; its prevalence is unknown.
4. `medquad_ans_0000767-2` (score=0.0021) - This condition is very rare; its prevalence is unknown.
5. `medquad_ans_0002259-1` (score=0.0002) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. Most people with Hartnup disease are asymptomatic, but those who do show symptoms typically experience onset between the ages of 3 and 9, although it may occasionally present in adulthood. Signs and symptoms include skin photosensitivity, neurologic findings, psychiatric symptoms, and ocular findings. Mental development is usually normal, though a few cases with intellectual impairment have been reported. Treatment may include a high-protein diet, protection from sunlight, vitamin supplementation, avoidance of certain drugs/medications, and in some cases, nicotinamide supplements and tryptophan ethyl ester [medquad_ans_0002733-1].

**Supported citations**: `medquad_ans_0002733-1`
**Unsupported citations**: _none_

### Example 260 (`0000428-3`)

**Question**: What are the genetic changes related to glycogen storage disease type VII ?

**Scores**: context_relevance=1.000, groundedness=0.835, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000428-1` (score=0.0001) - Glycogen storage disease type VII (GSDVII) is an inherited disorder caused by an inability to break down a complex sugar called glycogen in muscle cells. A lack of glycogen breakdown interferes with the function of mu...
2. `medquad_ans_0002623-1` (score=0.0001) - Glycogen storage disease type 7 (GSD7) is an inherited condition in which the body is unable to break down glycogen (a complex sugar) in the muscle cells. Because glycogen is an important source of energy, this can in...
3. `medquad_ans_0002615-3` (score=0.0001) - What causes glycogen storage disease type 13? Glycogen storage disease type 13 (GSD13) is caused by changes (mutations) in the ENO3 gene. Glycogen is a substance that is stored in muscle tissue and is used as an impor...
4. `medquad_ans_0002620-1` (score=0.0001) - Glycogen storage disease type 4 (GSD 4) is part of a group of disorders which lead to abnormal accumulation of glycogen (a storage form of glucose) in various parts of the body. Symptoms of GSD 4 usually begin in infa...
5. `medquad_ans_0000424-5` (score=0.0001) - These resources address the diagnosis or management of glycogen storage disease type IV: - Gene Review: Gene Review: Glycogen Storage Disease Type IV - Genetic Testing Registry: Glycogen storage disease, type IV - Med...

**Answer**:

The provided context does not mention specific genetic changes or mutations associated with glycogen storage disease type VII. Therefore, based on the given information, the genetic changes related to GSDVII cannot be determined. [medquad_ans_0000428-1]

**Supported citations**: `medquad_ans_0000428-1`
**Unsupported citations**: _none_

### Example 261 (`0000977-5`)

**Question**: What are the treatments for thiamine-responsive megaloblastic anemia syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.764, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000977-5` (score=0.9933) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
2. `medquad_ans_0000102-7` (score=0.0065) - Doctors treat pernicious anemia by replacing the missing vitamin B12 in the body. People who have pernicious anemia may need lifelong treatment. The goals of treating pernicious anemia include: Preventing or treating ...
3. `medquad_ans_0006056-6` (score=0.0014) - How might thalassemia be treated? The best treatment options depend on the severity of thalassemia. People affected by a mild form often need little to no treatment, while people with intermediate to severe thalassemi...
4. `medquad_ans_0001478-6` (score=0.0001) - How might congenital anosmia be treated? Unfortunately, there is currently no known cure or treatment for congenital anosmia.
5. `medquad_ans_0000258-4__dup1` (score=0.0001) - How might alpha-thalassemia be treated? Treatment of alpha-thalassemia often includes blood transfusions to provide healthy blood cells that have normal hemoglobin. Bone marrow transplant has helped to cure a small nu...

**Answer**:

The provided context does not contain information about the treatments for thiamine-responsive megaloblastic anemia syndrome. Therefore, a specific answer cannot be generated based on the given context. [medquad_ans_0000977-5] mentions resources that address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome, but does not detail the treatments themselves.

**Supported citations**: `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=1.000, groundedness=0.903, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000167-18` (score=0.8975) - - Polycystic kidney disease (PKD) is a genetic disorder that causes numerous cysts to grow in the kidneys. - A gene mutation, or defect, causes polycystic kidney disease. - Autosomal dominant PKD is the most common fo...
2. `medquad_ans_0000155-1__dup4` (score=0.3167) - If you have advanced chronic kidney disease (CKD), you may soon need treatment to do the work your kidneys can no longer do. Learning about your treatment options for kidney failure will help you make the best choice ...
3. `medquad_ans_0000173-5__dup1` (score=0.0240) - People with a solitary kidney do not need to eat a special diet. However, people with reduced kidney function may need to make changes to their diet to slow the progression of kidney disease. More information about re...
4. `medquad_ans_0000038-14` (score=0.0038) - Kidney disease is often called a "silent" disease, because most people have no symptoms in early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NOT wait for sympto...
5. `medquad_ans_0000135-1__dup4` (score=0.0003) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...

**Answer**:

For Polycystic Kidney Disease (PKD), there is currently no cure, but treatment can ease symptoms and prolong life. Management includes controlling complications such as high blood pressure through lifestyle changes, diet, and blood pressure medications. People with PKD should talk with a dietitian about foods to add and those that might be harmful. Treatment also addresses symptoms like pain, hematuria, and urinary tract infections, and may involve imaging tests (ultrasound, CT scans, MRI) and genetic testing for diagnosis and risk assessment. In advanced cases, treatment options for kidney failure may be necessary [medquad_ans_0000167-18].

**Supported citations**: `medquad_ans_0000167-18`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.909, groundedness=0.807, answer_relevance=0.836

**Top docs**:

1. `medquad_ans_0000018-14` (score=0.9978) - For information about the clinical trial on this topic, visit: http://www.clinicaltrials.gov/ct/show/NCT00372320?order=1. If you would like to read an interview with Dr. Bruce Baum, the study's principal investigator,...
2. `medquad_ans_0000090-4__dup1` (score=0.9973) - Much of the on-going research on RMDs is aimed at prevention and rehabilitation. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs.
3. `medquad_ans_0000025_1-8` (score=0.9866) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
4. `medquad_ans_0000006_3-8` (score=0.9800) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
5. `medquad_ans_0000061-5__dup2` (score=0.9730) - Scientists are making rapid progress in understanding the complexities of rheumatoid arthritis. They are learning more about how and why it develops and why some people have more severe symptoms than others. Research ...

**Answer**:

The provided context does not specify any research or clinical trials being conducted specifically for Rheumatoid Arthritis (RMDs). While it mentions that much of the ongoing research on RMDs is aimed at prevention and rehabilitation and that the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs [medquad_ans_0000090-4__dup1], no details about specific clinical trials or treatments being studied for Rheumatoid Arthritis are given. Therefore, based on the available information, the exact nature of research or clinical trials for Rheumatoid Arthritis cannot be determined.

**Supported citations**: `medquad_ans_0000090-4__dup1`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.917, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0002561-4` (score=0.9770) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
2. `medquad_ans_0000860-2__dup1` (score=0.3046) - How might Brown-Sequard syndrome be treated?
3. `medquad_ans_0000883-5__dup1` (score=0.0525) - How might Buschke Ollendorff syndrome be treated? There is currently no cure for BOS. Surgical removal of lesions on or under the skin may be done for cosmetic purposes. In some patients, surgical treatment of deafnes...
4. `medquad_ans_0006423-3` (score=0.0321) - How might Wernicke-Korsakoff syndrome be treated?
5. `medquad_ans_0006133-6` (score=0.0129) - Is there treatment for Townes-Brocks syndrome? Treatment is directed towards the specific symptoms, including immediate surgical intervention for imperforate anus; surgery for severe malformations of the hands; routin...

**Answer**:

Surgery is typically the treatment for Glass-Chapman-Hockley syndrome, aiming to increase the space in the anterior part of the skull. The operation is usually performed between 9 to 12 months of age and may involve other sutures if they are affected [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.852, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000868-5` (score=0.9961) - These resources address the diagnosis or management of Renpenning syndrome: - Genetic Testing Registry: Renpenning syndrome 1 - Greenwood Genetics Center: X-Linked Intellectual Disability - Kennedy Krieger Institute: ...
2. `medquad_ans_0003615-3` (score=0.9334) - How might CREST syndrome be treated? Unfortunately, CREST syndrome has no known cure. The condition carries both physical and psychological consequences, so a holistic approach to management should be taken. Treatment...
3. `medquad_ans_0000247-2__dup2` (score=0.1648) - There is no cure for Rett syndrome. Treatment for the disorder is symptomatic, focusing on the management of symptoms, and supportive. Medication may be needed for breathing irregularities and motor difficulties, and ...
4. `medquad_ans_0005119-5` (score=0.0784) - How might Proud syndrome be treated? The treatment of Proud syndrome is based on the signs and symptoms present in each person. For example, spasticity may be treated with a variety of therapies including medications ...
5. `medquad_ans_0000146-2__dup1` (score=0.0099) - There are no standard treatments for hereditary neuropathies. Treatment is mainly symptomatic and supportive. Medical treatment includes physical therapy and if needed, pain medication. Orthopedic surgery may be neede...

**Answer**:

The provided context does not contain information about the treatments for Renpenning syndrome. While it mentions resources that address the diagnosis or management of Renpenning syndrome, such as genetic testing and genetic counseling, specific treatment options are not detailed [medquad_ans_0000868-5]. Therefore, based on the available information, the treatments for Renpenning syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000868-5`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.988, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001105-1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
2. `medquad_ans_0001104-2` (score=0.0007) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2F? The subtypes of Charcot-Marie-Tooth type 2, including type 2F, have similar signs and symptoms. Affected individuals usually become symptomatic b...
3. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
4. `medquad_ans_0001092-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
5. `medquad_ans_0001094-2__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1E? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1E. If the information is ava...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 2G include onion bulb formation (7.5%), areflexia, autosomal recessive inheritance, axonal degeneration/regeneration, decreased motor nerve conduction velocity, decreased number of peripheral myelinated nerve fibers, distal amyotrophy, distal muscle weakness, distal sensory impairment, flexion contracture, neonatal onset, pes cavus, spinal deformities, split hand, and vocal cord paresis [medquad_ans_0001105-1].

**Supported citations**: `medquad_ans_0001105-1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001393-2` (score=0.9988) - What signs and symptoms are associated with coccygodynia? The classic symptom is pain when pressure is applied to the tailbone, such as when sitting on a hard chair. Symptoms usually improve with relief of pressure wh...
2. `medquad_ans_0000676-2__dup1` (score=0.0005) - What are the symptoms of Bell's palsy?
3. `medquad_ans_0003065-1` (score=0.0001) - What are the signs and symptoms of Hypomandibular faciocranial dysostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypomandibular faciocranial dysostosis. If the information ...
4. `medquad_ans_0001633-1` (score=0.0001) - What are the signs and symptoms of Craniofacial dyssynostosis? The Human Phenotype Ontology provides the following list of signs and symptoms for Craniofacial dyssynostosis. If the information is available, the table ...
5. `medquad_ans_0000943-1__dup1` (score=0.0001) - What are the signs and symptoms of Cardiocranial syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cardiocranial syndrome. If the information is available, the table below in...

**Answer**:

The symptoms of coccygodynia include pain when pressure is applied to the tailbone, such as when sitting on a hard chair, immediate and severe pain when moving from sitting to standing, pain during bowel movements, pain during sex, and a deep ache in the region of the tailbone. Symptoms usually improve with relief of pressure when standing or walking [medquad_ans_0001393-2].

**Supported citations**: `medquad_ans_0001393-2`
**Unsupported citations**: _none_

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.910, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.9994) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0004026-1` (score=0.0001) - What are the signs and symptoms of Microcephaly, holoprosencephaly, and intrauterine growth retardation? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, holoprosencepha...
3. `medquad_ans_0004549-2` (score=0.0001) - What are the signs and symptoms of Opitz G/BBB syndrome? Opitz G/BBB syndrome mainly affects structures along the midline of the body. The most common features of the condition are wide-spaced eyes (hypertelorism); de...
4. `medquad_ans_0005496-2` (score=0.0001) - What are the signs and symptoms of Schizencephaly? Signs and symptoms of schizencephaly may include: Developmental delay Seizures Abnormally small head (microcephaly) Intellectual disability Partial or complete paraly...
5. `medquad_ans_0002040-1` (score=0.0001) - What are the signs and symptoms of Ectrodactyly and ectodermal dysplasia without cleft lip/palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Ectrodactyly and ectodermal dysplas...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. Preaxial hand polydactyly, proptosis, and seizures are also listed as possible symptoms. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which draws from Orphanet and other sources, with frequencies often expressed as percentages or fractions [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003402-5` (score=0.9964) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
2. `medquad_ans_0000562-2` (score=0.0001) - Klippel-Feil syndrome is estimated to occur in 1 in 40,000 to 42,000 newborns worldwide. Females seem to be affected slightly more often than males.
3. `medquad_ans_0000170-3__dup1` (score=0.0001) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.
4. `medquad_ans_0000114-5__dup1` (score=0.0001) - Your doctor will diagnose restless legs syndrome (RLS) based on your signs and symptoms, your medical and family histories, a physical exam, and test results. Your doctor will use this information to rule out other co...
5. `medquad_ans_0000170-2__dup1` (score=0.0001) - Treatment for Klippel-Feil Syndrome is symptomatic and may include surgery to relieve cervical or craniocervical instability and constriction of the spinal cord, and to correct scoliosis. Physical therapy may also be ...

**Answer**:

Klippel Feil syndrome is diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are needed to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to the high incidence of hearing loss, and various lab tests to assess organ function. Additional tests or consultations with specialists may be recommended depending on the features present in each person with KFS [medquad_ans_0003402-5].

**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.957, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001497-1` (score=0.9537) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
2. `medquad_ans_0000464-1__dup1` (score=0.5711) - A hernia happens when part of an internal organ or tissue bulges through a weak area of muscle. Most hernias are in the abdomen. There are several types of hernias, including - Inguinal, in the groin. This is the the ...
3. `medquad_ans_0000222-1` (score=0.0661) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
4. `medquad_ans_0000002-1` (score=0.0366) - An abdominal wall defect is an opening in the abdomen through which various abdominal organs can protrude. This opening varies in size and can usually be diagnosed early in fetal development, typically between the ten...
5. `medquad_ans_0002458-1` (score=0.0068) - Fryns syndrome is a condition that affects the development of many parts of the body. Signs and symptoms vary widely among affected individuals. Many affected individuals have a defect in the diaphragm muscle such as ...

**Answer**:

Congenital diaphragmatic hernia (CDH) is a birth defect characterized by the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest. It ranges in severity from a thinned area in the diaphragm to its complete absence. This defect results in an abnormal opening that allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This crowding can lead to underdevelopment of the lungs (pulmonary hypoplasia), potentially causing life-threatening breathing difficulties from birth. CDH may occur with other malformations and can present with symptoms such as breathing problems or abdominal pain, though in about 1 percent of cases, it is asymptomatic and may be detected incidentally during imaging for other reasons [medquad_ans_0001497-1]. It is often classified by position,

**Supported citations**: `medquad_ans_0001497-1`
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0001088-1__dup1` (score=0.0761) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
2. `medquad_ans_0000173-1` (score=0.0431) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
3. `medquad_ans_0001090-1__dup1` (score=0.0026) - Charcot-Marie-Tooth disease type 1A (CMT1A) is a type of inherited neurological disorder that affects the peripheral nerves. Affected individuals experience weakness and wasting (atrophy) of the muscles of the lower l...
4. `medquad_ans_0001094-1__dup1` (score=0.0020) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...
5. `medquad_ans_0000173-2` (score=0.0014) - Charcot-Marie-Tooth disease is the most common inherited disorder that involves the peripheral nerves, affecting an estimated 150,000 people in the United States. It occurs in populations worldwide with a prevalence o...

**Answer**:

Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves, which connect the brain and spinal cord to muscles and sensory cells that detect sensations such as touch, pain, heat, and sound. Defects in many different genes cause different forms of this disease. Common symptoms may include foot drop, foot deformity, loss of lower leg muscle, numbness in the foot or leg, slapping gait, and weakness of the hips, legs, or feet. The disease usually becomes apparent in adolescence or early adulthood, but onset may occur anytime from early childhood through late adulthood. Symptoms vary in severity, even among family members, and may include balance difficulties, clumsiness, muscle weakness in the feet, foot abnormalities such as high arches or flat feet, and decreased sensitivity to touch. While there is currently no cure, management options include physical therapy, occupational therapy, braces, orthopedic devices, pain medication, and

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000685-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000205-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000784-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000998-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000858-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether monilethrix is inherited. While multiple documents describe autosomal recessive inheritance patterns, none explicitly mention monilethrix or its mode of inheritance. Therefore, based on the given information, it is not possible to determine if monilethrix is inherited.

[Note: Insufficient context to answer the question.]

**Supported citations**: _none_
**Unsupported citations**: `Note: Insufficient context to answer the question.`

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.857, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000215-3__dup2` (score=0.0924) - Most UTIs are caused by bacteria that live in the bowel. The bacterium Escherichia coli (E. coli) causes the vast majority of UTIs. The urinary tract has several systems to prevent infection. The points where the uret...
2. `medquad_ans_0000223-1__dup3` (score=0.0533) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. Bacteria \* are the most common cause of UTIs. Normally, bacteria that enter the urinary...
3. `medquad_ans_0000214-1__dup4` (score=0.0212) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
4. `medquad_ans_0000215-1__dup3` (score=0.0212) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...
5. `medquad_ans_0000223-2__dup2` (score=0.0116) - Most UTIs are caused by bacteria that live in the bowel, the part of the digestive tract where stool is changed from liquid to solid. The bacterium Escherichia coli (E. coli) causes most UTIs. The urinary tract has se...

**Answer**:

A urinary tract infection (UTI) in children is an infection in the urinary tract caused primarily by bacteria, most commonly Escherichia coli (E. coli). These infections occur when bacteria enter the urinary tract and overcome the body's natural defenses, despite the presence of protective mechanisms such as one-way valves at the ureter-bladder junction that prevent urine reflux and regular urination that flushes out microbes. Children who delay urination, have inadequate fluid intake, or suffer from chronic constipation (defined as fewer than two bowel movements per week) are at higher risk because holding in urine or blocked urine flow allows bacteria to grow. Some children are genetically prone to recurrent UTIs, similar to how some children are prone to respiratory or ear infections [medquad_ans_0000215-3__dup2]. UTIs can affect different parts of the urinary tract, including the urethra (urethritis), bladder (cyst

**Supported citations**: `medquad_ans_0000215-3__dup2`
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=0.875, groundedness=0.971, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000135-2__dup2` (score=0.0063) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
2. `medquad_ans_0000135-1__dup4` (score=0.0007) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...
3. `medquad_ans_0000804-5` (score=0.0005) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autoso...
4. `medquad_ans_0000804-2` (score=0.0003) - Polycystic kidney disease is a fairly common genetic disorder. It affects about 500,000 people in the United States. The autosomal dominant form of the disease is much more common than the autosomal recessive form. Au...
5. `medquad_ans_0000152-6` (score=0.0002) - Complications of IgA nephropathy include - high blood pressure - acute kidney failuresudden and temporary loss of kidney function - chronic kidney failurereduced kidney function over a period of time - nephrotic syndr...

**Answer**:

Complications of Polycystic Kidney Disease (PKD) include high blood pressure and problems with blood vessels in the brain and heart [medquad_ans_0000135-2__dup2].

**Supported citations**: `medquad_ans_0000135-2__dup2`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=0.800, groundedness=0.938, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.8358) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000949-1__dup1` (score=0.0440) - Viruses are capsules with genetic material inside. They are very tiny, much smaller than bacteria. Viruses cause familiar infectious diseases such as the common cold, flu and warts. They also cause severe illnesses su...
3. `medquad_ans_0000123-1__dup3` (score=0.0038) - Viral gastroenteritis is inflammation of the lining of the stomach, small intestine, and large intestine. Several different viruses can cause viral gastroenteritis, which is highly contagious and extremely common. Vir...
4. `medquad_ans_0000123-4__dup2` (score=0.0034) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
5. `medquad_ans_0000361-1__dup1` (score=0.0013) - Fifth disease is a viral infection caused by parvovirus B19. The virus only infects humans; it's not the same parvovirus that dogs and cats can get. Fifth disease mostly affects children. Symptoms can include a low fe...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, which are a group of related viruses. Infection with these viruses leads to an inflammation of the stomach and intestines, resulting in symptoms such as diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms include fever, headache, or body aches. Norovirus can spread from person to person, through contaminated food or water, or by touching a contaminated surface. The infection can be serious, especially in young children and older adults. There is no specific medicine to treat norovirus infections; treatment focuses on bed rest and staying hydrated to prevent dehydration. Proper hand washing and safe food preparation can help prevent infections [medquad_ans_0000649-1__dup1]. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in people of all ages [medquad_ans_0

**Supported citations**: `medquad_ans_0000649-1__dup1`
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.800, groundedness=0.947, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000392-3` (score=0.0234) - The cause of Fryns syndrome is unknown. The disorder is thought to be genetic because it tends to run in families and has features similar to those of other genetic disorders. Duplications and deletions in several chr...
2. `medquad_ans_0000222-3` (score=0.0110) - Congenital diaphragmatic hernia has many different causes. In 10 to 15 percent of affected individuals, the condition appears as a feature of a disorder that affects many body systems, called a syndrome. Donnai-Barrow...
3. `medquad_ans_0000235-3` (score=0.0001) - Mutations in many genes can cause congenital myasthenic syndrome. Mutations in the CHRNE gene are responsible for more than half of all cases. A large number of cases are also caused by mutations in the RAPSN, CHAT, C...
4. `medquad_ans_0000222-2` (score=0.0001) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.
5. `medquad_ans_0000464-1__dup1` (score=0.0001) - A hernia happens when part of an internal organ or tissue bulges through a weak area of muscle. Most hernias are in the abdomen. There are several types of hernias, including - Inguinal, in the groin. This is the the ...

**Answer**:

Genetic changes related to congenital diaphragmatic hernia include duplications and deletions in several chromosome regions, which have been associated with the condition and some of its features, such as those seen in Fryns syndrome [medquad_ans_0000392-3]. Some syndromes in which congenital diaphragmatic hernia may occur, like Donnai-Barrow, Fryns, and Pallister-Killian mosaic syndrome, are caused by changes in single genes or chromosomal abnormalities affecting multiple genes [medquad_ans_0000222-3]. However, no specific genetic change has been found to cause all of the signs and symptoms of Fryns syndrome, and in approximately 50 to 60 percent of cases, congenital diaphragmatic hernia is isolated with no other major malformations, and in more than 80 percent of cases, there is no known genetic syndrome

**Supported citations**: `medquad_ans_0000392-3`, `medquad_ans_0000222-3`
**Unsupported citations**: _none_

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.589, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004534-3` (score=0.9950) - What causes Omenn syndrome? Omenn syndrome is a genetically heterogeneous condition (meaning that it may be caused by a number of different genes). While most cases are attributed to mutations in the RAG genes (RAG-1 ...
2. `medquad_ans_0000238-4` (score=0.0027) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0000104-4` (score=0.0027) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000465-4` (score=0.0027) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000011-13` (score=0.0019) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...

**Answer**:

Omenn syndrome is a genetically heterogeneous condition that may be caused by mutations in several genes, including RAG1 and RAG2 genes located on chromosome band 11p13. Recent reports have also described cases caused by mutations in ARTEMIS, ADA, ILRA2, ILRA7, CHD7, and DNA ligase 4. Additionally, some cases have been associated with 22q11 microdeletion syndrome [medquad_ans_0004534-3]. The condition is inherited in an autosomal dominant pattern, meaning one copy of the altered gene in each cell is sufficient to cause the disorder [medquad_ans_0000238-4, medquad_ans_0000104-4, medquad_ans_0000465-4].

**Supported citations**: `medquad_ans_0004534-3`
**Unsupported citations**: `medquad_ans_0000238-4, medquad_ans_0000104-4, medquad_ans_0000465-4`

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=0.875, groundedness=0.977, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.9864) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0000067-1__dup1` (score=0.0163) - Cerebellar hypoplasia is a neurological condition in which the cerebellum is smaller than usual or not completely developed. Cerebellar hypoplasia is a feature of a number of congenital (present at birth) malformation...
3. `medquad_ans_0005761-1` (score=0.0026) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
4. `medquad_ans_0005792-1` (score=0.0016) - Spinocerebellar ataxia autosomal recessive 7, also called SCAR7, is a slowly progressive hereditary form of spinocerebellar ataxia. Symptoms of SCAR7 can include difficulty walking and writing, speech difficulties (dy...
5. `medquad_ans_0000495-1__dup1` (score=0.0014) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and symptoms of the disorder first appear in early to mid-adulthood and include impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. Individuals may also have difficulty with movements that involve judging distance or scale (dysmetria). Other features include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive and often result in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.375, groundedness=0.000, answer_relevance=0.675

**Top docs**:

1. `medquad_ans_0000233-2__dup1` (score=0.9965) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...
2. `medquad_ans_0000267-2__dup1` (score=0.9954) - Treatment begins with exercise programs and physical therapy to strengthen chest muscles, restore normal posture, and relieve compression by increasing the space of the area the nerve passes through. Doctors will ofte...
3. `medquad_ans_0000255-2__dup1` (score=0.9947) - Treatment may include such medications as baclofen, diazepam, tizanidine or clonazepam. Physical therapy regimens may include muscle stretching and range of motion exercises to help prevent shrinkage or shortening of ...
4. `medquad_ans_0000029-18` (score=0.9944) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
5. `medquad_ans_0000090-2__dup1` (score=0.9936) - Treatment for RMDs usually includes reducing or stopping the motions that cause symptoms. Options include taking breaks to give the affected area time to rest, and adopting stretching and relaxation exercises. Applyin...

**Answer**:

The provided context does not contain specific information about treatments for Brachial Plexus Injuries. Therefore, a definitive answer cannot be given based on the available context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.937, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000218-4__dup2` (score=0.9972) - Urine leakage has many possible causes.
2. `medquad_ans_0000216-8` (score=0.9936) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
3. `medquad_ans_0000216-7` (score=0.0299) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
4. `medquad_ans_0000216-2__dup1` (score=0.0293) - Many types of defects in the urinary tract can cause urine blockage: - Vesicoureteral reflux (VUR). Most children with VUR are born with a ureter that did not grow long enough during development in the womb. The valve...
5. `medquad_ans_0001092-2` (score=0.0033) - This condition is detected in an estimated 1 in 36,000 newborns worldwide.

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), ureterocele, and genetic conditions like prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects. Additionally, spina bifida and other birth defects affecting the spinal cord can lead to urine blockage. Urine blockage may also be associated with prenatal conditions detected via ultrasound, amniocentesis, or chorionic villus sampling (CVS), and may not be apparent until symptoms of a urinary tract infection (UTI) develop [medquad_ans_0000216-8]. When a defect blocks urine flow, it causes urine to back up, leading to hydroure

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

**Scores**: context_relevance=0.750, groundedness=0.065, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000570-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
2. `medquad_ans_0001065-4` (score=0.9958) - This condition is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromosomes), a mutation ...
3. `medquad_ans_0000059-4` (score=0.6511) - When EDA-ID is caused by mutations in the IKBKG gene, it is inherited in an X-linked recessive pattern. The IKBKG gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only o...
4. `medquad_ans_0004758-4` (score=0.0003) - Is Partington syndrome inherited? Partington syndrome is inherited in an X-linked recessive manner. A condition is considered X-linked if the mutated gene that causes the condition is located on the X chromosome, one ...
5. `medquad_ans_0000052-31` (score=0.0002) - Genetics is one of the most exciting areas of Parkinson's disease research. Studying the genes responsible for inherited cases can help researchers understand both inherited and sporadic cases of the disease. Sporadic...

**Answer**:

The provided context does not mention X-linked dystonia-parkinsonism or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether X-linked dystonia-parkinsonism is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 284 (`0000120-1`)

**Question**: What is (are) beta-ketothiolase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.993, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000702-1__dup1` (score=0.9065) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ket...
2. `medquad_ans_0000120-1` (score=0.8601) - Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This disorder also impairs the body's ability to process keto...
3. `medquad_ans_0005518-1` (score=0.2969) - SCOT deficiency is a metabolic disease that is caused by reduced or missing levels of 3-ketoacid CoA transferase. This enzyme is necessary for the body to use ketones. Ketones are substances produced when fat cells br...
4. `medquad_ans_0000122-1` (score=0.0012) - Beta-ureidopropionase deficiency is a disorder that causes excessive amounts of molecules called N-carbamyl-beta-aminoisobutyric acid and N-carbamyl-beta-alanine to be released in the urine. Neurological problems rang...
5. `medquad_ans_0000120-2` (score=0.0002) - Beta-ketothiolase deficiency appears to be very rare. It is estimated to affect fewer than 1 in 1 million newborns.

**Answer**:

Beta-ketothiolase deficiency is an inherited disorder in which the body cannot effectively process a protein building block (amino acid) called isoleucine. This condition also impairs the body's ability to process ketones, which are molecules produced during the breakdown of fats. Signs and symptoms typically appear between the ages of 6 months and 24 months. Affected children experience episodes of vomiting, dehydration, difficulty breathing, extreme tiredness (lethargy), and occasionally seizures. These episodes, called ketoacidotic attacks, sometimes lead to coma. Ketoacidotic attacks are frequently triggered by infections, periods without food (fasting), or increased intake of protein-rich foods. This condition is inherited in an autosomal recessive fashion and is caused by mutations in the ACAT1 gene [medquad_ans_0000702-1__dup1].

**Supported citations**: `medquad_ans_0000702-1__dup1`
**Unsupported citations**: _none_

### Example 285 (`0000006_5-7`)

**Question**: What are the stages of Childhood Brain Stem Glioma ?

**Scores**: context_relevance=0.778, groundedness=0.954, answer_relevance=0.511

**Top docs**:

1. `medquad_ans_0000006_5-7` (score=0.9969) - Key Points - The plan for cancer treatment depends on whether the tumor is in one area of the brain or has spread all through the brain. The plan for cancer treatment depends on whether the tumor is in one area of the...
2. `medquad_ans_0000006_3-7` (score=0.9921) - Key Points - The grade of the tumor is used to plan cancer treatment. - Low-grade astrocytomas - High-grade astrocytomas - An MRI is done after surgery. The grade of the tumor is used to plan cancer treatment. Staging...
3. `medquad_ans_0000006_5-6` (score=0.0401) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis and treatment options depend on: - The type of brain stem glioma. - Where the tumor is found in the brain and if it has spread...
4. `medquad_ans_0000006_2-8` (score=0.0031) - Key Points - There are different types of treatment for children with brain and spinal cord tumors. - Children with brain or spinal cord tumors should have their treatment planned by a team of health care providers wh...
5. `medquad_ans_0000006_5-3` (score=0.0012) - Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer; not having risk factors doesnt mean that you will not get cancer. Talk with ...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.533, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000266-1__dup2` (score=0.3112) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
2. `medquad_ans_0000838-1__dup2` (score=0.2233) - Your spinal cord is a bundle of nerves that runs down the middle of your back. It carries signals back and forth between your body and your brain. It is protected by your vertebrae, which are the bone disks that make ...
3. `medquad_ans_0000266-2__dup1` (score=0.0036) - MRI imaging is often used to evaluate individuals with these symptoms, and can be used to diagnose the location of the tethering, lower than normal position of the conus medullaris, or presence of a tumor or fatty mas...
4. `medquad_ans_0000054-4__dup1` (score=0.0036) - The NINDS supports and conducts a wide range of research on spinal cord disorders such as BSS. The goal of this research is to find ways to prevent, treat, and, ultimately, cure these disorders.
5. `medquad_ans_0000258-2__dup2` (score=0.0025) - Improved emergency care for people with spinal cord injuries and aggressive treatment and rehabilitation can minimize damage to the nervous system and even restore limited abilities. Respiratory complications are ofte...

**Answer**:

Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. These attachments may occur congenitally at the base of the spinal cord (conus medullaris) or develop near the site of a spinal cord injury. The attachments cause an abnormal stretching of the spinal cord, leading to a progressive course. In children, symptoms may include lesions, hairy patches, dimples, or fatty tumors on the lower back; foot and spinal deformities; weakness in the legs; low back pain; scoliosis; and incontinence. The condition is often linked to improper growth of the neural tube during fetal development and is closely associated with spina bifida. It may go undiagnosed until adulthood, when pain, sensory and motor problems, and loss of bowel and bladder control emerge. Delayed symptom presentation is related to the degree of strain on the spinal cord over time

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.869, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000058-3__dup1` (score=0.9958) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work,...
2. `medquad_ans_0000136-5__dup2` (score=0.0001) - The symptoms of dialysis-related amyloidosis may include - pain, stiffness, and fluid in the joints. - abnormal, fluid-containing sacs, called cysts, in some bones. - carpal tunnel syndrome, caused by unusual buildup ...
3. `medquad_ans_0000058-1__dup1` (score=0.0001) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
4. `medquad_ans_0000034-2__dup4` (score=0.0001) - The causes are probably different for different types of diabetic neuropathy. Researchers are studying how prolonged exposure to high blood glucose causes nerve damage. Nerve damage is likely due to a combination of f...
5. `medquad_ans_0000419-1__dup1` (score=0.0001) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as it responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Symptoms may include numbness, tingling, weakness, and sometimes pain in the hand and wrist, which can interfere with daily activities such as driving or reading [medquad_ans_0000058-1__dup1]. While treatment can improve symptoms, complete resolution is not guaranteed for all patients. [medquad_ans_0000058-3__dup1]

**Supported citations**: `medquad_ans_0000058-3__dup1`, `medquad_ans_0000058-1__dup1`
**Unsupported citations**: _none_

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.776, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000033-2__dup1` (score=0.9951) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
2. `medquad_ans_0000296-5` (score=0.9939) - These resources address the diagnosis or management of Donnai-Barrow syndrome: - Gene Review: Gene Review: Donnai-Barrow Syndrome - Genetic Testing Registry: Donnai Barrow syndrome - MedlinePlus Encyclopedia: Diaphrag...
3. `medquad_ans_0000188-2__dup1` (score=0.9909) - MJD is incurable, but some symptoms of the disease can be treated. For those individuals who show parkinsonian features, levodopa therapy can help for many years. Treatment with antispasmodic drugs, such as baclofen, ...
4. `medquad_ans_0000200-2__dup2` (score=0.9899) - Currently there is no cure for these disease syndromes.Medical care is directed at treating systemic conditions and improving the person's quality of life. Physical therapy and daily exercise may delay joint problems ...
5. `medquad_ans_0000860-2__dup1` (score=0.9415) - How might Brown-Sequard syndrome be treated?

**Answer**:

The provided context does not specify treatments for Donnai-Barrow syndrome. While it lists resources that address the diagnosis or management of Donnai-Barrow syndrome, such as Gene Review, Genetic Testing Registry, and MedlinePlus Encyclopedia entries on diaphragmatic hernia, hearing loss, and omphalocele, no specific treatments are described [medquad_ans_0000296-5]. Therefore, based on the available information, the treatments for Donnai-Barrow syndrome cannot be determined.

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

**Scores**: context_relevance=0.917, groundedness=0.988, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000167-11` (score=0.9791) - Autosomal recessive PKD is a rare genetic disorder that affects the liver as well as the kidneys. The signs of autosomal recessive PKD frequently appear in the earliest months of life, even in the womb, so health care...
2. `medquad_ans_0000585-1__dup1` (score=0.9521) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
3. `medquad_ans_0000674-1__dup1` (score=0.0385) - Behr syndrome is a disorder mainly characterized by early-onset optic atrophy, ataxia, and spasticity. Other signs and symptoms may be present and vary from person to person. Although the exact cause is unknown, the s...
4. `medquad_ans_0003414-1` (score=0.0250) - Konigsmark Knox Hussels syndrome is an inherited condition that causes both hearing and vision loss. This condition is characterized by late-onset progressive sensorineural deafness and progressive optic atrophy, whic...
5. `medquad_ans_0004797-1` (score=0.0041) - Pendred syndrome is a condition usually characterized by sensorineural hearing loss in both ears (bilateral) and euthyroid goiter (enlargement of the thyroid gland with normal thyroid gland function). The amount of he...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. It is described by Iwashita et al. in 1969 in a Korean brother and sister, and autosomal recessive inheritance has been suggested [medquad_ans_0000585-1__dup1].

**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.700, groundedness=0.971, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000794-3` (score=0.9991) - Phosphoglycerate kinase deficiency is caused by mutations in the PGK1 gene. This gene provides instructions for making an enzyme called phosphoglycerate kinase, which is involved in a critical energy-producing process...
2. `medquad_ans_0000856-3` (score=0.0254) - Pyruvate kinase deficiency is caused by mutations in the PKLR gene. The PKLR gene is active in the liver and in red blood cells, where it provides instructions for making an enzyme called pyruvate kinase. The pyruvate...
3. `medquad_ans_0000167-2__dup3` (score=0.0019) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
4. `medquad_ans_0000750-3` (score=0.0011) - In most people with Ollier disease, the disorder is caused by mutations in the IDH1 or IDH2 gene. These genes provide instructions for making enzymes called isocitrate dehydrogenase 1 and isocitrate dehydrogenase 2, r...
5. `medquad_ans_0000652-3` (score=0.0007) - Mutations in the MVK gene cause mevalonate kinase deficiency. The MVK gene provides instructions for making the mevalonate kinase enzyme. This enzyme is involved in the production of cholesterol, which is later conver...

**Answer**:

The genetic changes related to phosphoglycerate kinase deficiency are mutations in the PGK1 gene [medquad_ans_0000794-3]. These mutations reduce the activity of the phosphoglycerate kinase enzyme, which disrupts energy production during glycolysis and leads to cell damage or death [medquad_ans_0000794-3]. It is unclear why this abnormality preferentially affects certain cell types, such as red blood cells, brain cells, or muscle cells, and researchers speculate that different mutations may have varying effects on enzyme activity in different cell types [medquad_ans_0000794-3].

**Supported citations**: `medquad_ans_0000794-3`
**Unsupported citations**: _none_

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.900, groundedness=0.033, answer_relevance=0.820

**Top docs**:

1. `medquad_ans_0000849-3` (score=0.0008) - Mutations in the BMPR2 gene are the most common genetic cause of pulmonary arterial hypertension. This gene plays a role in regulating the number of cells in certain tissues. Researchers suggest that a mutation in thi...
2. `medquad_ans_0000832-3` (score=0.0001) - In most cases, the genetic cause of progressive supranuclear palsy is unknown. Rarely, the disease results from mutations in the MAPT gene. Certain normal variations (polymorphisms) in the MAPT gene have also been ass...
3. `medquad_ans_0000104-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000238-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000465-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not contain any information about genetic changes related to pulmonary veno-occlusive disease. Therefore, it is not possible to answer the question based on the given context. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.750, groundedness=0.965, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0001006-3` (score=0.9991) - Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. In the liver, enzymes break down tyrosine in a five step process, resulting in molecules that are either excreted by t...
2. `medquad_ans_0000398-1__dup1` (score=0.1683) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...
3. `medquad_ans_0000750-3` (score=0.0004) - In most people with Ollier disease, the disorder is caused by mutations in the IDH1 or IDH2 gene. These genes provide instructions for making enzymes called isocitrate dehydrogenase 1 and isocitrate dehydrogenase 2, r...
4. `medquad_ans_0000133-3__dup2` (score=0.0002) - Wilson disease is caused by an inherited autosomal recessive mutation, or change, in the ATP7B gene. In an autosomal recessive disease, the child has to inherit the gene mutation from both parents to have an increased...
5. `medquad_ans_0000396-1__dup2` (score=0.0002) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...

**Answer**:

Mutations in the FAH, TAT, and HPD genes can cause tyrosinemia types I, II, and III, respectively. These mutations result in a decrease in the activity of specific enzymes involved in the breakdown of tyrosine—fumarylacetoacetate hydrolase (FAH), tyrosine aminotransferase (TAT), and 4-hydroxyphenylpyruvate dioxygenase (HPD). As a result, tyrosine and its byproducts accumulate to toxic levels, leading to cellular damage in the liver, kidneys, nervous system, and other organs [medquad_ans_0001006-3].

**Supported citations**: `medquad_ans_0001006-3`
**Unsupported citations**: _none_

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.857, groundedness=0.471, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000209-2` (score=0.9953) - Treatment involves restoring normal levels of thyroid hormone and may include thyroid drugs, radioactive iodine, and sometimes partial or complete surgical removal of the thyroid.
2. `medquad_ans_0000436-5` (score=0.9887) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
3. `medquad_ans_0000004-6__dup2` (score=0.9344) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease ...
4. `medquad_ans_0000004-7__dup2` (score=0.8965) - Treatment for Graves disease can sometimes affect pregnancy. After treatment with surgery or radioactive iodine, TSI antibodies can still be present in the blood, even when thyroid levels are normal. If a pregnant wom...
5. `medquad_ans_0000208-2__dup1` (score=0.4472) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...

**Answer**:

The treatments for Graves' Disease include radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves' disease in the United States. In this therapy, patients take radioactive iodine-131 by mouth, which is collected by the thyroid gland and gradually destroys thyroid cells that produce hormone, without affecting other body tissues. Doses may be adjusted to either shut down the thyroid completely or bring hormone production into the normal range, and more than one round may be needed. Results take time, and symptom improvement may not be noticed for several weeks or months. Radioiodine therapy is not used in pregnant women or women who are breastfeeding due to potential harm to the fetus or passage into breast milk. Additionally, radioiodine therapy may worsen Graves' ophthalmopathy (GO) in some individuals, and prescription steroids may be used to prevent this complication. Other treatment options include thyroid drugs and surgical removal of the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.497, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.2434) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000039-3__dup1` (score=0.0037) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
3. `medquad_ans_0000039-4__dup1` (score=0.0015) - The biochemical defects that underlie several NCLs have recently been discovered. An enzyme called palmitoyl-protein thioesterase has been shown to be insufficiently active in the infantile form of Batten disease (thi...
4. `medquad_ans_0000708-2` (score=0.0013) - The prevalence of this condition is unknown.
5. `medquad_ans_0000545-2` (score=0.0009) - Juvenile Batten disease is the most common type of NCL, but its exact prevalence is unknown. Collectively, all forms of NCL affect an estimated 1 in 100,000 individuals worldwide. NCLs are more common in Finland, wher...

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling, and vision loss is often an early symptom that may lead to suspicion during an eye exam. Over time, affected children experience cognitive impairment, worsening seizures, and progressive loss of sight and motor skills, eventually becoming blind, bedridden, and demented. The disease is often fatal by the late teens or twenties. Biochemical defects underlie several forms of Batten disease: in the infantile form (CLN1), there is insufficient activity of the enzyme palmitoyl-protein thioesterase; in the late infantile form (CLN2), a deficiency of an acid protease enzyme is found; in juvenile B

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=0.857, groundedness=0.993, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002061-1` (score=0.5247) - Ehlers-Danlos syndrome progeroid type is a genetic disorder of the connective tissue, which is the material between the cells of the body that gives tissues form and strength. The disorder primarily affects the skin, ...
2. `medquad_ans_0000318-1__dup1` (score=0.2245) - Ehlers-Danlos syndrome (EDS) is a group of inherited disorders that weaken connective tissues. Connective tissues are proteins that support skin, bones, blood vessels, and other organs. EDS usually affects your skin, ...
3. `medquad_ans_0000313-1` (score=0.0070) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
4. `medquad_ans_0000673-4` (score=0.0016) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000886-1__dup1` (score=0.0005) - C syndrome, also known as Opitz trigonocephaly syndrome, is characterized by trigonocephaly, severe intellectual disability, hypotonia, variable cardiac defects, redundant (extra folds of) skin, joint and limb abnorma...

**Answer**:

Ehlers-Danlos syndrome, progeroid type is a genetic disorder of the connective tissue that primarily affects the skin, hair, and skeletal system. Symptoms usually appear by childhood or adolescence and include unusually flexible joints, loose elastic skin, and easy scarring. Unique features of this type include sparse scalp hair and eyebrows, and loose elastic skin on the face, which causes affected individuals to look older than their age. Additional symptoms may include bone weakness, weak muscle tone, mild intellectual disability, and delayed growth in children. This type is caused by mutations in the B4GALT7 gene and is inherited in an autosomal recessive pattern [medquad_ans_0002061-1].

**Supported citations**: `medquad_ans_0002061-1`
**Unsupported citations**: _none_

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.800, groundedness=0.984, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.9991) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000442-3` (score=0.0003) - Mutations in the GAMT gene cause guanidinoacetate methyltransferase deficiency. The GAMT gene provides instructions for making the enzyme guanidinoacetate methyltransferase. This enzyme participates in the two-step pr...
3. `medquad_ans_0000416-2` (score=0.0001) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
4. `medquad_ans_0000252-3` (score=0.0001) - Mutations in the UGT1A1 gene cause Crigler-Najjar syndrome. This gene provides instructions for making the bilirubin uridine diphosphate glucuronosyl transferase (bilirubin-UGT) enzyme, which is found primarily in liv...
5. `medquad_ans_0002601-1` (score=0.0001) - Glutamate formiminotransferase deficiency is an inherited metabolic disorder that affects physical and mental development. There are two forms of this condition, a mild form and a sever form. People with the mild form...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the last two steps of the metabolism of the amino acid histidine and in the production of one of the forms of the vitamin folate [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001169-2` (score=0.9994) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
2. `medquad_ans_0001169-1` (score=0.0001) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
3. `medquad_ans_0004472-1` (score=0.0001) - What are the signs and symptoms of Nystagmus 1, congenital, X- linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Nystagmus 1, congenital, X- linked. If the information is avail...
4. `medquad_ans_0001047-1` (score=0.0001) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...
5. `medquad_ans_0001047-3` (score=0.0001) - X-linked chondrodysplasia punctata 1 is caused by genetic changes involving the ARSE gene. This gene provides instructions for making an enzyme called arylsulfatase E. The function of this enzyme is unknown, although ...

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive (CDPX1) include abnormality of the vertebral column, anosmia, cataract, depressed nasal bridge, epiphyseal stippling, hearing impairment, hypogonadism, ichthyosis, microcephaly, short distal phalanx of finger, short nasal septum, short nose, short stature, and X-linked recessive inheritance [medquad_ans_0001169-2]. Additional common features include shortened fingers, a flat nose, breathing abnormalities, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development [medquad_ans_0001169-1]. The condition is caused by changes in the ARSE gene on the X chromosome and is inherited in an X-linked recessive manner, occurring almost exclusively in males [medquad_ans_000

**Supported citations**: `medquad_ans_0001169-2`, `medquad_ans_0001169-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.857, groundedness=0.972, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000132-10` (score=0.9879) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
2. `medquad_ans_0000165-2__dup1` (score=0.9735) - The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfa...
3. `medquad_ans_0000203-3__dup1` (score=0.9658) - The disease tends to advance rapidly over the course of 5 to 10 years, with progressive loss of motor skills, eventual confinement to bed, and death. There is no remission from the disease. There is currently no cure.
4. `medquad_ans_0000132-6` (score=0.0375) - People with Whipple disease may have complications caused by malnutrition, which is due to damaged villi in the small intestine. As a result of delayed diagnosis or treatment, people may experience the following compl...
5. `medquad_ans_0000132-7` (score=0.0006) - A health care provider may use several tests and exams to diagnose Whipple disease, including the following: - medical and family history - physical exam - blood tests - upper GI endoscopy and enteroscopy A patient ma...

**Answer**:

The standard treatment for Whipple's disease is a prolonged course of antibiotics, which may last up to two years, including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfadizine or solfamethoxazole can treat neurological symptoms. Relapsing neurologic Whipple's disease, marked by bouts of worsening symptoms, is sometimes treated with a combination of antibiotics and weekly injections of interferon gamma, a substance made by the body that activates the immune system. [medquad_ans_0000165-2__dup1]

**Supported citations**: `medquad_ans_0000165-2__dup1`
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.857, groundedness=0.925, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000085-2__dup1` (score=0.9929) - Currently, only central core disease has an effective treatment (see above). There are no known cures for any of these disorders. Supportive treatment may involve orthopedic treatments, as well as physical, occupation...
2. `medquad_ans_0000131-2__dup1` (score=0.9914) - Treatment begins with a thorough diagnostic evaluation, usually performed by a neurologist, including an assessment of motor and sensory skills, balance and coordination, mental status, reflexes, and functioning of th...
3. `medquad_ans_0000042-2__dup2` (score=0.9856) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as...
4. `medquad_ans_0000676-4__dup1` (score=0.9747) - How might Bell's palsy be treated?
5. `medquad_ans_0000072-2__dup1` (score=0.2812) - Cerebral palsy cant be cured, but treatment will often improve a child's capabilities. In general, the earlier treatment begins the better chance children have of overcoming developmental disabilities or learning new ...

**Answer**:

Treatments for Bell's palsy include steroids such as prednisone, which reduce inflammation and swelling, and antiviral drugs, which may help shorten the course of the disease. Analgesics like aspirin, acetaminophen, or ibuprofen can relieve pain. It is important to protect the eye, especially at night, by keeping it moist with lubricating eye drops to prevent dryness and injury. Additional therapies such as physical therapy, facial massage, or acupuncture may provide a potential small improvement in facial nerve function and pain relief [medquad_ans_0000042-2__dup2].

**Supported citations**: `medquad_ans_0000042-2__dup2`
**Unsupported citations**: _none_
