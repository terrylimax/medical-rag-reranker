# Generation Evaluation

## Summary

- `avg_answer_chars`: 550.2700
- `avg_answer_relevance`: 0.7395
- `avg_citation_presence_rate`: 0.8367
- `avg_context_relevance`: 0.7986
- `avg_empty_answer`: 0.0000
- `avg_end_to_end_latency_ms`: 8213.4724
- `avg_generation_latency_ms`: 5471.2461
- `avg_groundedness`: 0.6754
- `avg_insufficient_context`: 0.0467
- `avg_num_retrieved_docs`: 5.0000
- `avg_rerank_latency_ms`: 2027.5504
- `avg_retrieval_latency_ms`: 714.4530
- `avg_supported_citation_rate`: 0.6417
- `avg_unsupported_citation_rate`: 0.1950
- `generation_remote_concurrency`: 4.0000
- `num_examples`: 300
- `reranker_enabled_rate`: 1.0000

## Examples

### Example 1 (`0006510-1`)

**Question**: What are the symptoms of X-linked lymphoproliferative syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.987, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0006510-1` (score=0.9994) - What are the signs and symptoms of X-linked lymphoproliferative syndrome 1? The Human Phenotype Ontology provides the following list of signs and symptoms for X-linked lymphoproliferative syndrome 1. If the informatio...
2. `medquad_ans_0000004_3-2` (score=0.9888) - Signs of AIDS-related lymphoma include weight loss, fever, and night sweats. These and other signs and symptoms may be caused by AIDS-related lymphoma or by other conditions. Check with your doctor if you have any of ...
3. `medquad_ans_0000001_1-2` (score=0.0024) - Signs and symptoms of adult ALL include fever, feeling tired, and easy bruising or bleeding. The early signs and symptoms of ALL may be like the flu or other common diseases. Check with your doctor if you have any of ...
4. `medquad_ans_0000001_3-3` (score=0.0022) - Signs and symptoms of chronic lymphocytic leukemia include swollen lymph nodes and tiredness. Usually CLL does not cause any signs or symptoms and is found during a routine blood test. Signs and symptoms may be caused...
5. `medquad_ans_0000004_1-3` (score=0.0008) - Signs of adult Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by adult Hodgkin lymphoma or by other conditions. Check with your doc...

**Answer**:

The symptoms of X-linked lymphoproliferative syndrome 1 include cellular immunodeficiency in 90% of patients, decreased antibody level in blood in 50%, hepatomegaly in 50%, lymphadenopathy in 50%, lymphoma in 50%, splenomegaly in 50%, anemia in 7.5%, encephalitis, fulminant hepatitis, hepatic encephalopathy, IgG deficiency, immunodeficiency, increased IgM level, meningitis, pancytopenia, recurrent pharyngitis, reduced natural killer cell activity, thrombocytopenia, and X-linked recessive inheritance [medquad_ans_0006510-1].

**Supported citations**: `medquad_ans_0006510-1`
**Unsupported citations**: _none_

### Example 2 (`0000312-4`)

**Question**: Is early-onset primary dystonia inherited ?

**Scores**: context_relevance=0.400, groundedness=0.553, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000095-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000585-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000362-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000673-4` (score=0.9989) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000575-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not specify whether early-onset primary dystonia is inherited. While several documents mention an autosomal dominant inheritance pattern for certain conditions, none explicitly state that early-onset primary dystonia follows this pattern or confirm its inheritance status. Therefore, based on the given information, it cannot be determined whether early-onset primary dystonia is inherited.

[medquad_ans_0000095-4] [medquad_ans_0000585-4] [medquad_ans_0000362-4] [medquad_ans_0000673-4] [medquad_ans_0000575-4]

**Supported citations**: `medquad_ans_0000095-4`, `medquad_ans_0000585-4`, `medquad_ans_0000362-4`, `medquad_ans_0000673-4`, `medquad_ans_0000575-4`
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

**Scores**: context_relevance=1.000, groundedness=0.975, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000194-2__dup1` (score=0.9994) - What are the signs and symptoms of Aicardi-Goutieres syndrome type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Aicardi-Goutieres syndrome type 2. If the information is availab...
2. `medquad_ans_0000008-6` (score=0.9992) - Many arrhythmias cause no signs or symptoms. When signs or symptoms are present, the most common ones are: Palpitations (feelings that your heart is skipping a beat, fluttering, or beating too hard or fast) A slow hea...
3. `medquad_ans_0000029-6` (score=0.0450) - Some people who have cardiomyopathy never have signs or symptoms. Others don't have signs or symptoms in the early stages of the disease. As cardiomyopathy worsens and the heart weakens, signs and symptoms of heart fa...
4. `medquad_ans_0000059-6` (score=0.0102) - Signs and symptoms depend on the type of heart block you have. First-degree heart block may not cause any symptoms. Signs and symptoms of second- and third-degree heart block include: Fainting Dizziness or light-heade...
5. `medquad_ans_0000197-1__dup1` (score=0.0001) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...

**Answer**:

The symptoms of Aicardi-Goutieres syndrome type 2 include dystonia (5%), microcephaly (5%), spastic paraplegia (5%), basal ganglia calcification, cerebral atrophy, chronic CSF lymphocytosis, and encephalopathy [medquad_ans_0000194-2__dup1].

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

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000990-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000683-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000043-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000784-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000998-4` (score=0.9979) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention hemophilia or specify how it is inherited. Therefore, based on the given information, it is not possible to determine if hemophilia is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 7 (`0003284-1`)

**Question**: What are the symptoms of IVIC syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0003284-1` (score=0.9994) - What are the signs and symptoms of IVIC syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for IVIC syndrome. If the information is available, the table below includes how often t...
2. `medquad_ans_0000139-5__dup2` (score=0.6980) - The signs and symptoms of childhood nephrotic syndrome may include - edemaswelling, most often in the legs, feet, or ankles and less often in the hands or face - albuminuriawhen a childs urine has high levels of album...
3. `medquad_ans_0003248-2` (score=0.0019) - What are the signs and symptoms of intravenous leiomyomatosis? IVL most often does not cause detectable signs or symptoms. In fact, they may be found by chance during surgery. When symptoms do arise, they can include ...
4. `medquad_ans_0003162-2` (score=0.0005) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition ...
5. `medquad_ans_0003103-2` (score=0.0002) - What are the signs and symptoms of IBIDS syndrome? The most common symptoms of Tay syndrome are brittle hair (trichothiodystrophy); dry, thickened, scaling skin (ichthyosis); photosensitivity (abnormal light sensitivi...

**Answer**:

The symptoms of IVIC syndrome include hearing impairment (90%), limitation of joint mobility (90%), short stature (90%), strabismus (90%), abnormal dermatoglyphics (50%), aplasia/hypoplasia of the thumb (50%), radioulnar synostosis (50%), scoliosis (50%), synostosis of carpal bones (50%), triphalangeal thumb (50%), abnormality of the clavicle (7.5%), arrhythmia (7.5%), leukocytosis (7.5%), preaxial hand polydactyly (7.5%), thrombocytopenia (7.5%), and urogenital fistula (7.5%) [medquad_ans_0003284-1].

**Supported citations**: `medquad_ans_0003284-1`
**Unsupported citations**: _none_

### Example 8 (`0000775-1`)

**Question**: What is (are) paroxysmal nocturnal hemoglobinuria ?

**Scores**: context_relevance=0.833, groundedness=0.544, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0004747-1` (score=0.9758) - Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It can occur at any age, but is usually diagnosed in young adulthood. People ...
2. `medquad_ans_0000775-1` (score=0.6112) - Paroxysmal nocturnal hemoglobinuria is an acquired disorder that leads to the premature death and impaired production of blood cells. The disorder affects red blood cells (erythrocytes), which carry oxygen; white bloo...
3. `medquad_ans_0000201-1__dup3` (score=0.0606) - Hematuria is blood in the urine. Two types of blood in the urine exist. Blood that can be seen in the urine is called gross hematuria. Blood that cannot be seen in the urine, except when examined with a microscope, is...
4. `medquad_ans_0000201-2__dup2` (score=0.0022) - Most people with microscopic hematuria do not have symptoms. People with gross hematuria have urine that is pink, red, or cola-colored due to the presence of red blood cells (RBCs). Even a small amount of blood in the...
5. `medquad_ans_0000775-2` (score=0.0005) - Paroxysmal nocturnal hemoglobinuria is a rare disorder, estimated to affect between 1 and 5 per million people.

**Answer**:

Paroxysmal nocturnal hemoglobinuria (PNH) is an acquired disorder that leads to the premature death and impaired production of blood cells. It affects red blood cells (erythrocytes), which carry oxygen; white blood cells (leukocytes), which protect the body from infection; and platelets (thrombocytes), which are involved in blood clotting. PNH can occur at any age but is usually diagnosed in young adulthood and affects both sexes equally. The disorder is caused by acquired, rather than inherited, mutations in the PIGA gene and is not passed down to children of affected individuals. People with PNH experience sudden, recurring episodes of symptoms (paroxysmal symptoms) that may be triggered by stresses on the body, such as infections or physical exertion. During these episodes, red blood cells are prematurely destroyed (hemolysis), leading to hemolytic anemia and signs and symptoms such as fatigue,

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

**Scores**: context_relevance=0.833, groundedness=0.993, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003317-1` (score=0.9712) - Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz ...
2. `medquad_ans_0002121-1` (score=0.0042) - Epidermolysis bullosa (EB) is a group of genetic skin diseases that cause the skin to blister very easily. Blisters form in response to minor injuries or friction, such as rubbing or scratching. There are four main ty...
3. `medquad_ans_0000544-2` (score=0.0003) - Both types of junctional epidermolysis bullosa are rare, affecting fewer than 1 per million people in the United States.
4. `medquad_ans_0002002-1` (score=0.0001) - Dystrophic epidermolysis bullosa (DEB) is one of the major forms of epidermolysis bullosa. The signs and symptoms can vary widely among affected people. In mild cases, blistering may primarily affect the hands, feet, ...
5. `medquad_ans_0002126-3` (score=0.0001) - How might epidermolysis bullosa simplex be treated? There is no cure for epidermolysis bullosa simplex and there is no known treatment proven to completely control all of the symptoms. However, many complications can ...

**Answer**:

Junctional epidermolysis bullosa (JEB) is a type of Epidermolysis Bullosa, a group of genetic conditions that cause the skin to be very fragile and to blister easily. JEB is separated into two categories: the Herlitz type and the Non-Herlitz type. The Herlitz type is very severe, and individuals with this condition often do not survive infancy. The Non-Herlitz type includes several subtypes that cause mild to severe blistering of the skin present at birth or shortly thereafter. JEB is inherited in an autosomal recessive pattern and is caused by mutations in the LAMB3, COL17A1, or LAMC2, and LAMA3 genes. There is no cure for JEB; treatment is focused on management of blistering and prevention of secondary infections [medquad_ans_0003317-1]. Both types

**Supported citations**: `medquad_ans_0003317-1`
**Unsupported citations**: _none_

### Example 11 (`0000668-3`)

**Question**: What are the genetic changes related to molybdenum cofactor deficiency ?

**Scores**: context_relevance=0.700, groundedness=0.935, answer_relevance=0.860

**Top docs**:

1. `medquad_ans_0000668-3` (score=0.9991) - Molybdenum cofactor deficiency is caused by mutations in the MOCS1, MOCS2, or GPHN gene. There are three forms of the disorder, named types A, B, and C (or complementation groups A, B, and C). The forms have the same ...
2. `medquad_ans_0000671-4` (score=0.0003) - Up to 15 percent of Japanese people with moyamoya disease have one or more family members with the condition, indicating that the condition can be passed through generations in families; however, the inheritance patte...
3. `medquad_ans_0000653-2` (score=0.0002) - MOPDII appears to be a rare condition, although its prevalence is unknown.
4. `medquad_ans_0000486-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. Most often, the parents of an individual with an autosomal recessive condition each carry...
5. `medquad_ans_0000668-2` (score=0.0001) - Molybdenum cofactor deficiency is a rare condition that is estimated to occur in 1 in 100,000 to 200,000 newborns worldwide. More than 100 cases have been reported in the medical literature, although it is thought tha...

**Answer**:

The genetic changes related to molybdenum cofactor deficiency are mutations in the MOCS1, MOCS2, or GPHN gene. Mutations in the MOCS1 gene cause type A, mutations in the MOCS2 gene cause type B, and mutations in the GPHN gene cause type C of the disorder [medquad_ans_0000668-3]. These mutations reduce or eliminate the function of the associated protein, impairing molybdenum cofactor biosynthesis, which is essential for the function of metabolic enzymes that break down toxic substances in the body [medquad_ans_0000668-3].

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
5. `medquad_ans_0000930-1__dup1` (score=0.0009) - Carbamoyl phosphate synthetase I deficiency is type of urea cycle disorder. It causes toxic levels of ammonia to accumulate in the blood. Signs and symptoms in newborns may include a lack of energy, unwillingness to e...

**Answer**:

Ornithine translocase deficiency is an inherited disorder that causes ammonia to accumulate in the blood. Ammonia, which is formed when proteins are broken down in the body, is toxic if the levels become too high. The nervous system is especially sensitive to the effects of excess ammonia. Ornithine translocase deficiency varies widely in its severity and age of onset. An infant with ornithine translocase deficiency may be lacking in energy (lethargic) or refuse to eat, or have poorly controlled breathing or body temperature. Some babies with this disorder may experience seizures or unusual body movements, or go into a coma. Episodes of illness may coincide with the introduction of high-protein formulas or solid foods into the diet. In most affected individuals, signs and symptoms of ornithine translocase deficiency do not appear until later in life. Later-onset forms of ornithine translocase deficiency are usually less severe than the

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 13 (`0000362-1__dup2`)

**Question**: Do you have information about Financial Assistance

**Scores**: context_relevance=0.857, groundedness=0.915, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000586-1__dup2` (score=0.9970) - Summary : Medicaid is government health insurance that helps many low-income people in the United States to pay their medical bills. The Federal government sets up general guidelines for the program, but each state ha...
2. `medquad_ans_0000036-6` (score=0.9969) - People who enroll in Medicare can register with www.MyMedicare.gov, a secure online service, and use the site to access their personal Medicare information at any time. People can view their claims and order history, ...
3. `medquad_ans_0000362-1__dup2` (score=0.9954) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
4. `medquad_ans_0000693-1__dup2` (score=0.9929) - Summary : As a patient, you have certain rights. Some are guaranteed by federal law, such as the right to get a copy of your medical records, and the right to keep them private. Many states have additional laws protec...
5. `medquad_ans_0000945-5` (score=0.0929) - These resources from MedlinePlus offer information about the diagnosis and management of various health conditions: - Diagnostic Tests - Drug Therapy - Surgery and Rehabilitation - Genetic Counseling - Palliative Care

**Answer**:

Yes, there is information about financial assistance. Certain government programs and nonprofit organizations can help with medical costs, especially if you don't have insurance or need help with costs that aren't covered. You can also discuss concerns about paying medical bills with your health care provider, social worker, or the business office of your clinic or hospital [medquad_ans_0000362-1__dup2]. Additionally, Medicaid is a government health insurance program that helps low-income people pay their medical bills, and each state may have specific rules regarding cost-sharing and eligibility requirements [medquad_ans_0000586-1__dup2].

**Supported citations**: `medquad_ans_0000362-1__dup2`, `medquad_ans_0000586-1__dup2`
**Unsupported citations**: _none_

### Example 14 (`0004105-1`)

**Question**: What is (are) Mollaret meningitis ?

**Scores**: context_relevance=0.800, groundedness=0.980, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0004105-1` (score=0.8755) - Mollaret meningitis is a rare type of meningitis that is characterized by repeated episodes of fever, stiff neck (meningismus), muscle aches, and severe headaches separated by weeks or months of no symptoms. About hal...
2. `medquad_ans_0000599-1__dup1` (score=0.0084) - Meningococci are a type of bacteria that cause serious infections. The most common infection is meningitis, which is an inflammation of the thin tissue that surrounds the brain and spinal cord. Meningococci can also c...
3. `medquad_ans_0002669-1` (score=0.0073) - Granulomatous amebic encephalitis is a life-threatening infection of the brain caused by the free-living amoebae Acanthamoeba spp., Balamuthia mandrillaris and Sappinia pedata. Acanthamoeba species, are commonly found...
4. `medquad_ans_0000653-5` (score=0.0031) - These resources address the diagnosis or management of MOPDII: - Genetic Testing Registry: Microcephalic osteodysplastic primordial dwarfism type 2 These resources from MedlinePlus offer information about the diagnosi...
5. `medquad_ans_0003916-1` (score=0.0028) - Meningoencephalocele is a type of encephalocele, which is an abnormal sac of fluid, brain tissue, and meninges (membranes that cover the brain and spinal cord) that extends through a defect in the skull. There are two...

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

[medquad_ans_0000309-3__dup1] mentions that in approximately 90-95% of ALS cases, the cause is unknown (sporadic), and in 5-10% of cases, it is genetic, with mutations in genes such as C9orf72, SOD1, TARDBP, FUS, ANG, ALS2, SETX, and VAPB. However, this information pertains to ALS in general, not specifically juvenile ALS. There is no explicit link or evidence in the context connecting these genetic mutations to

**Supported citations**: `medquad_ans_0000309-3__dup1`
**Unsupported citations**: _none_

### Example 16 (`0000219-10`)

**Question**: What to do for What I need to know about Erectile Dysfunction ?

**Scores**: context_relevance=0.778, groundedness=0.952, answer_relevance=0.689

**Top docs**:

1. `medquad_ans_0000219-9` (score=0.9938) - You can prevent many of the causes of ED by making healthy lifestyle choices. Following a healthy diet may help prevent ED. Quitting smoking and getting physical activity are also important ways to prevent ED. Physica...
2. `medquad_ans_0000219-11` (score=0.9849) - - Erectile dysfunction (ED) is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last...
3. `medquad_ans_0000219-1__dup4` (score=0.9725) - Erectile dysfunction\* is when you cannot get or keep an erection firm enough to have sex. You may have ED if you - can get an erection sometimes, though not every time - can get an erection, yet it does not last long ...
4. `medquad_ans_0000219-7` (score=0.9372) - To find the cause of your ED, your doctor may - take your medical and sexual history - ask you questions about your mental health - give you a physical exam - test your blood - give you a nighttime erection test - per...
5. `medquad_ans_0000061-7__dup2` (score=0.7201) - The nerve damage of diabetes may cause sexual or urologic problems. - Sexual problems in men with diabetes include - erectile dysfunction - retrograde ejaculation - Sexual problems in women with diabetes include - dec...

**Answer**:

To understand erectile dysfunction (ED), know that it is when you cannot get or keep an erection firm enough to have sex. You may have ED if you can get an erection sometimes but not every time, can get an erection that does not last long enough for sex, or are unable to get an erection at all [medquad_ans_0000219-11]. An erection occurs when blood flow into the penis increases, making it larger and firmer, and involves the interaction of hormones, blood vessels, nerves, and muscles [medquad_ans_0000219-11]. ED often has multiple causes, and many diseases can damage the nerves, arteries, and muscles involved [medquad_ans_0000219-11].

To determine the cause of ED, your doctor may take your medical and sexual history, ask about your mental health, perform a physical exam, test your

**Supported citations**: `medquad_ans_0000219-11`
**Unsupported citations**: _none_

### Example 17 (`0006267-3`)

**Question**: What are the treatments for Ullrich congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.923, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0006267-3` (score=0.9702) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
2. `medquad_ans_0000981-5` (score=0.0061) - These resources address the diagnosis or management of tibial muscular dystrophy: - Gene Review: Gene Review: Udd Distal Myopathy - Genetic Testing Registry: Distal myopathy Markesbery-Griggs type These resources from...
3. `medquad_ans_0000202-5` (score=0.0019) - These resources address the diagnosis or management of collagen VI-related myopathy: - Gene Review: Gene Review: Collagen Type VI-Related Disorders - Genetic Testing Registry: Bethlem myopathy - Genetic Testing Regist...
4. `medquad_ans_0000334-5` (score=0.0002) - These resources address the diagnosis or management of facioscapulohumeral muscular dystrophy: - Gene Review: Gene Review: Facioscapulohumeral Muscular Dystrophy - Genetic Testing Registry: Facioscapulohumeral muscula...
5. `medquad_ans_0000576-5` (score=0.0002) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...

**Answer**:

Physical therapy, including early mobilization, regular stretching, and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation often become necessary in the first or second decade of life. Prevention of chest infections may be achieved with the use of antibiotics. Feeding difficulties leading to failure to thrive may be managed by gastrostomy. Surgery may be needed for contractures and scoliosis. Some reports indicate that people with Ullrich congenital muscular dystrophy may benefit from cyclosporin A, although more studies are needed to confirm its benefits [medquad_ans_0006267-3]. Additionally, resources such as the Genetic Testing Registry and Muscular Dystrophy UK provide information on the potential use of cyclosporine A for Ullrich congenital muscular dystrophy [medquad_ans_0000202-5].

**Supported citations**: `medquad_ans_0006267-3`, `medquad_ans_0000202-5`
**Unsupported citations**: _none_

### Example 18 (`0000653-4`)

**Question**: Is microcephalic osteodysplastic primordial dwarfism type II inherited ?

**Scores**: context_relevance=1.000, groundedness=0.749, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0002838-1` (score=0.0001) - Hereditary multiple osteochondromas (HMO) (formerly called hereditary multiple exostoses) is a genetic condition in which people develop multiple benign (noncancerous) bone tumors that are covered by cartilage (called...
2. `medquad_ans_0000653-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is a condition characterized by short stature (dwarfism) with other skeletal abnormalities (osteodysplasia) and an unusually small head size (microcep...
3. `medquad_ans_0004006-1` (score=0.0001) - Microcephalic osteodysplastic primordial dwarfism type 2 (MOPD2) is a condition characterized by short stature (dwarfism), skeletal abnormalities and an unusually small head size (microcephaly). Other signs and sympto...
4. `medquad_ans_0000187-1` (score=0.0001) - CHST3-related skeletal dysplasia is a genetic condition characterized by bone and joint abnormalities that worsen over time. Affected individuals have short stature throughout life, with an adult height under 4 and a ...
5. `medquad_ans_0004550-1` (score=0.0001) - Opsismodysplasia is a rare skeletal dysplasia characterized by congenital short stature and characteristic craniofacial abnormalities. Clinical signs observed at birth include short limbs, small hands and feet, relati...

**Answer**:

The provided context does not specify whether microcephalic osteodysplastic primordial dwarfism type II (MOPDII) is inherited. [medquad_ans_0000653-1] describes the clinical features of MOPDII but does not mention inheritance patterns. Therefore, based on the given information, it is not possible to determine if MOPDII is inherited.

**Supported citations**: `medquad_ans_0000653-1`
**Unsupported citations**: _none_

### Example 19 (`0000742-4`)

**Question**: Is Ochoa syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000249-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0000556-4` (score=0.9985) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
3. `medquad_ans_0000283-4` (score=0.9984) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...
4. `medquad_ans_0000031-4` (score=0.9984) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene. These cases oc...
5. `medquad_ans_0000883-4` (score=0.9977) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...

**Answer**:

The provided context does not mention Ochoa syndrome or its inheritance pattern. Therefore, it is not possible to determine whether Ochoa syndrome is inherited based on the given information. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 20 (`0000531-5`)

**Question**: What are the treatments for isodicentric chromosome 15 syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.813, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000531-5` (score=0.9960) - These resources address the diagnosis or management of isodicentric chromosome 15 syndrome: - Autism Speaks: How is Autism Treated? These resources from MedlinePlus offer information about the diagnosis and management...
2. `medquad_ans_0004132-4` (score=0.0114) - How might mosaic trisomy 14 be treated? Treatment for signs and symptoms of mosaic trisomy 14 focuses on the specific features present in each individual. Infants with congenital heart defects may need surgery or othe...
3. `medquad_ans_0001282-6` (score=0.0071) - How might chromosome 4q deletion be treated? Because chromosome 4q deletion affects many different systems of the body, medical management is often provided by a team of doctors and other healthcare professionals. Tre...
4. `medquad_ans_0000531-2` (score=0.0001) - Isodicentric chromosome 15 syndrome occurs in about 1 in 30,000 newborns.
5. `medquad_ans_0000996-4` (score=0.0001) - Most cases of trisomy 13 are not inherited and result from random events during the formation of eggs and sperm in healthy parents. An error in cell division called nondisjunction results in a reproductive cell with a...

**Answer**:

The provided context does not contain specific information about the treatments for isodicentric chromosome 15 syndrome. While it mentions that resources from MedlinePlus offer information on diagnosis and management of various health conditions, including genetic counseling and palliative care, no detailed treatments are described for isodicentric chromosome 15 syndrome specifically [medquad_ans_0000531-5]. Therefore, based on the available context, the treatments for isodicentric chromosome 15 syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000531-5`
**Unsupported citations**: _none_

### Example 21 (`0006224-3`)

**Question**: What causes Troyer syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0006224-3` (score=0.9931) - What causes Troyer syndrome? Troyer syndrome is caused by mutations in the SPG20 gene. This gene gives the body instructions to make a protein called spartin, which is present in many body tissues, including those of ...
2. `medquad_ans_0000115-2__dup2` (score=0.0118) - Scientists are unsure about what causes Mntriers disease; however, researchers think that most people acquire, rather than inherit, the disease. In extremely rare cases, siblings have developed Mntriers disease as chi...
3. `medquad_ans_0003909-3` (score=0.0074) - What causes Mntrier disease? The exact cause of Mntrier disease is unknown. However, it has been associated with cytomegalovirus (CMV) infection in children and Heliobacter pylori (H. pylori) infection in adults. In a...
4. `medquad_ans_0000115-1__dup4` (score=0.0051) - Mntriers disease causes the ridges along the inside of the stomach wallcalled rugaeto enlarge, forming giant folds in the stomach lining. The rugae enlarge because of an overgrowth of mucous cells in the stomach wall....
5. `medquad_ans_0000730-2` (score=0.0001) - Nicolaides-Baraitser syndrome is likely a rare condition; approximately 75 cases have been reported in the scientific literature.

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
5. `medquad_ans_0000989-4` (score=0.0001) - When Treacher Collins syndrome results from mutations in the TCOF1 or POLR1D gene, it is considered an autosomal dominant condition, which means one copy of the altered gene in each cell is sufficient to cause the dis...

**Answer**:

Mutations in the CACNA1C gene are responsible for all reported cases of Timothy syndrome. These mutations change the structure of CaV1.2 channels, causing them to stay open much longer than usual, which leads to an abnormal influx of calcium ions into cells. This calcium overload alters the way cardiac muscle cells beat and can cause arrhythmia [medquad_ans_0000983-3].

**Supported citations**: `medquad_ans_0000983-3`
**Unsupported citations**: _none_

### Example 23 (`0000061-3__dup3`)

**Question**: What are the symptoms of Rheumatoid Arthritis ?

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000049-13` (score=0.9972) - Warning signs of osteoarthritis include - joint pain - swelling or tenderness in one or more joints - stiffness after getting out of bed or sitting for a long time - a crunching feeling or sound of bone rubbing on bon...
2. `medquad_ans_0000061-3__dup3` (score=0.9930) - Swelling and Pain in the Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Rheumatoid arthritis is characterized by i...
3. `medquad_ans_0000061-11` (score=0.9910) - Rheumatoid arthritis can be difficult to diagnose in its early stages because the full range of symptoms develops over time, and only a few symptoms may be present in the early stages. As part of the diagnosis, your d...
4. `medquad_ans_0000049-3__dup3` (score=0.3534) - Pain and Stiffness in Joints Different types of arthritis have different symptoms. In general, people with most forms of arthritis have pain and stiffness in their joints. Osteoarthritis usually develops slowly and ca...
5. `medquad_ans_0000061-10` (score=0.0139) - Rheumatoid arthritis is characterized by inflammation of the joint lining. This inflammation causes warmth, redness, swelling, and pain around the joints. The pain of rheumatoid arthritis varies greatly from person to...

**Answer**:

Symptoms of rheumatoid arthritis include pain and stiffness in the joints, warmth, redness, swelling, and pain around the joints. People may also feel sick, tired, and sometimes feverish. Rheumatoid arthritis generally occurs in a symmetrical pattern, meaning if one knee or hand is affected, the other is also likely to be affected [medquad_ans_0000061-3__dup3].

**Supported citations**: `medquad_ans_0000061-3__dup3`
**Unsupported citations**: _none_

### Example 24 (`0002796-1`)

**Question**: What is (are) Henoch-Schonlein purpura ?

**Scores**: context_relevance=0.600, groundedness=0.976, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002796-1` (score=0.9718) - Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). While the cause of this condition is no...
2. `medquad_ans_0000149-1__dup3` (score=0.9287) - Henoch-Schnlein purpura is a disease that causes small blood vessels in the body to become inflamed and leak. The primary symptom is a rash that looks like many small raised bruises. HSP can also affect the kidneys, d...
3. `medquad_ans_0005502-1` (score=0.2490) - Schnitzler syndrome is a rare autoinflammatory condition. Signs and symptoms of the condition vary but may include urticaria; recurrent fevers; joint pain and inflammation; organomegaly (abnormally enlarged organs); a...
4. `medquad_ans_0005152-1` (score=0.0200) - Pseudopelade of Brocq (PBB) is a slowly progressive, chronic condition characterized by scarring hair loss (cicatricial alopecia). There exists some controversy as to whether PBB is a distinct condition or the common ...
5. `medquad_ans_0002322-1` (score=0.0029) - Febrile ulceronecrotic Mucha-Habermann disease (FUMHD) is a rare and severe form of pityriasis lichenoides et varioliformis acuta (PLEVA). PLEVA is characterized by skin lesions that ulcerate, breakdown, form open sor...

**Answer**:

Henoch-Schonlein purpura (HSP) is a disease that involves purple spots on the skin (purpura), joint pain, digestive problems, and glomerulonephritis (a type of kidney disorder). It may develop as an immune response to an infection, and while the exact cause is not fully understood, it is usually seen in children but can affect people of any age. Most cases go away on their own without treatment, and the main goal of treatment, when needed, is to relieve symptoms such as joint pain, abdominal pain, or swelling. Over-the-counter medicines can be used in many cases, and in some patients with severe arthritis, prednisone, a steroid medicine, may be prescribed [medquad_ans_0002796-1]. HSP can also cause inflammation of small blood vessels, leading to a rash that appears as many small raised bruises, and it can affect the kidneys

**Supported citations**: `medquad_ans_0002796-1`
**Unsupported citations**: _none_

### Example 25 (`0003166-1`)

**Question**: What are the symptoms of Immunodeficiency with hyper IgM type 5 ?

**Scores**: context_relevance=1.000, groundedness=0.983, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0003166-1` (score=0.9994) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 5? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 5. If the information ...
2. `medquad_ans_0003162-2` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 1? Symptoms and physical findings associated with hyper IgM syndrome usually become apparent in the first or second year of life. This condition ...
3. `medquad_ans_0006606-1` (score=0.0001) - What are the signs and symptoms of 5q- syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q- syndrome. If the information is available, the table below includes how often the...
4. `medquad_ans_0006607-2` (score=0.0001) - What are the signs and symptoms of 5q14.3 microdeletion syndrome ? The Human Phenotype Ontology provides the following list of signs and symptoms for 5q14.3 microdeletion syndrome . If the information is available, th...
5. `medquad_ans_0003163-1` (score=0.0001) - What are the signs and symptoms of Immunodeficiency with hyper IgM type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Immunodeficiency with hyper IgM type 2. If the information ...

**Answer**:

The symptoms of Immunodeficiency with hyper IgM type 5 include autosomal recessive inheritance, epididymitis, IgA deficiency, IgG deficiency, immunodeficiency, impaired Ig class switch recombination, increased IgM level, lymphadenopathy, recurrent bacterial infections, and recurrent upper and lower respiratory tract infections [medquad_ans_0003166-1].

**Supported citations**: `medquad_ans_0003166-1`
**Unsupported citations**: _none_

### Example 26 (`0000896-3`)

**Question**: What are the genetic changes related to Senior-Lken syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.943, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000896-3` (score=0.9991) - Senior-Lken syndrome can be caused by mutations in one of at least five genes. The proteins produced from these genes are known or suspected to play roles in cell structures called cilia. Cilia are microscopic, finger...
2. `medquad_ans_0000748-3` (score=0.0016) - The Maat-Kievit-Brunner type of Ohdo syndrome results from mutations in the MED12 gene. This gene provides instructions for making a protein that helps regulate gene activity; it is thought to play an essential role i...
3. `medquad_ans_0000556-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000283-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...
5. `medquad_ans_0000986-3` (score=0.0002) - Mutations in the SALL1 gene cause Townes-Brocks Syndrome. The SALL1 gene is part of a group of genes called the SALL family. These genes provide instructions for making proteins that are involved in the formation of t...

**Answer**:

Genetic changes related to Senior-Lken syndrome include mutations in one of at least five genes. These mutations likely lead to problems with the structure and function of cilia, which are microscopic projections on cells involved in signaling pathways. The defective cilia disrupt important chemical signaling within cells, though it remains unclear how specifically these defects lead to nephronophthisis and Leber congenital amaurosis. In some individuals, no mutations in these five known genes are identified, and the genetic cause remains unknown [medquad_ans_0000896-3].

**Supported citations**: `medquad_ans_0000896-3`
**Unsupported citations**: _none_

### Example 27 (`0000610-2`)

**Question**: How many people are affected by lymphedema-distichiasis syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.931, answer_relevance=0.350

**Top docs**:

1. `medquad_ans_0000610-2` (score=0.9988) - The prevalence of lymphedema-distichiasis syndrome is unknown. Because the extra eyelashes can be overlooked during a medical examination, researchers believe that some people with this condition may be misdiagnosed a...
2. `medquad_ans_0000113-2` (score=0.9950) - Beckwith-Wiedemann syndrome affects an estimated 1 in 13,700 newborns worldwide. The condition may actually be more common than this estimate because some people with mild symptoms are never diagnosed.
3. `medquad_ans_0000644-2` (score=0.0064) - The prevalence of Meige disease is unknown. Collectively, the many types of primary lymphedema affect an estimated 1 in 100,000 people younger than 20; Meige disease is the most common type of primary lymphedema. For ...
4. `medquad_ans_0000278-2` (score=0.0014) - The prevalence of Denys-Drash syndrome is unknown; at least 150 affected individuals have been reported in the scientific literature.
5. `medquad_ans_0000313-2` (score=0.0005) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...

**Answer**:

The prevalence of lymphedema-distichiasis syndrome is unknown [medquad_ans_0000610-2].

**Supported citations**: `medquad_ans_0000610-2`
**Unsupported citations**: _none_

### Example 28 (`0000399-1__dup1`)

**Question**: Do you have information about Genetic Testing

**Scores**: context_relevance=0.857, groundedness=0.864, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000399-1__dup1` (score=0.9977) - Summary : Genetic tests are tests on blood and other tissue to find genetic disorders. Over 2000 tests are available. Doctors use genetic tests for several reasons. These include - Finding genetic diseases in unborn b...
2. `medquad_ans_0000397-1__dup1` (score=0.1860) - Summary : Genetic counseling provides information and support to people who have, or may be at risk for, genetic disorders. A genetic counselor meets with you to discuss genetic risks. The counseling may be for yourse...
3. `medquad_ans_0000049-5__dup2` (score=0.0061) - - Mutations in single genes can cause rare forms of diabetes. - Genetic testing can identify many forms of monogenic diabetes. - A physician evaluates whether genetic testing is appropriate. - A correct diagnosis aide...
4. `medquad_ans_0000013-25` (score=0.0037) - Genetic testing may be helpful whether the test identifies a mutation or not. Test results can - serve as a relief, eliminating some of the uncertainty about a health condition - help doctors make recommendations for ...
5. `medquad_ans_0000395-1__dup2` (score=0.0012) - Summary : Genes are the building blocks of inheritance. Passed from parent to child, they contain instructions for making proteins. If genes don't produce the right proteins or don't produce them correctly, a child ca...

**Answer**:

Yes, I have information about genetic testing. Genetic tests are conducted on blood and other tissues to detect genetic disorders. Over 2000 such tests are available, and doctors use them for various purposes, including identifying genetic diseases in unborn babies, determining if individuals carry a gene that could be passed to their children, screening embryos for disease, testing adults before symptoms appear, making a diagnosis in someone with symptoms, and determining the best type or dose of medication for a person [medquad_ans_0000399-1__dup1]. Genetic testing may also provide relief by reducing uncertainty about a health condition, assist doctors in making treatment or monitoring recommendations, support individuals in making health and family-related decisions, and help in taking preventive steps such as earlier and more frequent screenings [medquad_ans_0000013-25]. In specific cases, such as monogenic diabetes, genetic testing can identify mutations and lead

**Supported citations**: `medquad_ans_0000399-1__dup1`, `medquad_ans_0000013-25`
**Unsupported citations**: _none_

### Example 29 (`0000149-1`)

**Question**: What is (are) capillary malformation-arteriovenous malformation syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.538, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000149-1` (score=0.8681) - Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system, which is the body's complex network of blood vessels. The vascular system consists of arteries, which carry oxy...
2. `medquad_ans_0000030-1__dup2` (score=0.0268) - Arteriovenous malformations (AVMs) are abnormal, snarled tangles of blood vessels that cause multiple irregular connections between the arteries and veins. These malformations most often occur in the spinal cord and i...
3. `medquad_ans_0003238-1` (score=0.0137) - Intracranial arteriovenous malformations (AVMs) are abnormal connections between the arteries and veins in the brain. Most people with brain or spinal AVMs experience few, if any, major symptoms. About 12 percent of p...
4. `medquad_ans_0000060-1__dup2` (score=0.0070) - Cerebral cavernous malformations (CCMs) are vascular lesions comprised of clusters of tightly packed, abnormally thin-walled small blood vessels (capillaries) that displace normal neurological tissue in the brain or s...
5. `medquad_ans_0000687-1` (score=0.0053) - Multiple cutaneous and mucosal venous malformations (also known as VMCM) are bluish patches (lesions) on the skin (cutaneous) and the mucous membranes, such as the lining of the mouth and nose. These lesions represent...

**Answer**:

Capillary malformation-arteriovenous malformation syndrome (CM-AVM) is a disorder of the vascular system characterized by capillary malformations (CMs), which are composed of enlarged capillaries that increase blood flow near the surface of the skin and appear as multiple small, round, pink or red spots on the skin, typically on the face, arms, and legs. These spots may be visible at birth or develop during childhood. In most cases, capillary malformations do not cause health problems and may be the only sign of the disorder. However, some individuals with CM-AVM also have more serious vascular abnormalities such as arteriovenous malformations (AVMs) and arteriovenous fistulas (AVFs), which are abnormal connections between arteries, veins, and capillaries that can lead to complications including abnormal bleeding, migraine headaches, seizures, and heart failure. These complications may appear in infancy or early

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 30 (`0001992-1`)

**Question**: What are the symptoms of Dystonia 16 ?

**Scores**: context_relevance=1.000, groundedness=0.959, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0001992-1` (score=0.9994) - What are the signs and symptoms of Dystonia 16? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 16. If the information is available, the table below includes how often the s...
2. `medquad_ans_0000112-3__dup1` (score=0.9985) - The initial symptoms can be very mild and may be noticeable only after prolonged exertion, stress, or fatigue. Dystonias often progress through various stages. Initially, dystonic movements are intermittent and appear...
3. `medquad_ans_0001994-1` (score=0.0010) - What are the signs and symptoms of Dystonia 18? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 18. If the information is available, the table below includes how often the s...
4. `medquad_ans_0001995-1` (score=0.0003) - What are the signs and symptoms of Dystonia 19? The Human Phenotype Ontology provides the following list of signs and symptoms for Dystonia 19. If the information is available, the table below includes how often the s...
5. `medquad_ans_0001903-2` (score=0.0003) - What are the signs and symptoms of Dopa-responsive dystonia? The most common form of dopa-responsive dystonia (DRD) is autosomal dominant DRD (caused by a mutation in the GCH1 gene). This form of DRD is usually charac...

**Answer**:

The symptoms of Dystonia 16 include cognitive impairment (5%), abnormal pyramidal signs, bradykinesia, delayed speech and language development, dysarthria, dysphagia, gait disturbance, hyperreflexia, involuntary movements, laryngeal dystonia, limb dystonia, lower limb pain, morphological abnormality of the pyramidal tract, motor delay, parkinsonism, postural tremor, progressive nature of the condition, retrocollis [medquad_ans_0001992-1].

**Supported citations**: `medquad_ans_0001992-1`
**Unsupported citations**: _none_

### Example 31 (`0000096-2__dup1`)

**Question**: What are the treatments for Septo-Optic Dysplasia ?

**Scores**: context_relevance=0.857, groundedness=0.877, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000899-5` (score=0.9950) - These resources address the diagnosis or management of septo-optic dysplasia: - Genetic Testing Registry: Septo-optic dysplasia sequence - MedlinePlus Encyclopedia: Growth Hormone Deficiency - MedlinePlus Encyclopedia...
2. `medquad_ans_0005548-5` (score=0.9652) - Can septo-optic dysplasia be cured? There is no cure for septo-optic dysplasia. Treatment is symptomatic. Hormone deficiencies may be treated with hormone replacement therapy. The optical problems are generally not tr...
3. `medquad_ans_0000016-14` (score=0.2261) - Research found that that prompt treatment of macular edema with anti-VEGF drugs, with or without laser treatment, resulted in better vision than laser treatment alone or steroid injections. When injected into the eye,...
4. `medquad_ans_0004461-4` (score=0.0222) - How might Norrie disease be treated? Because most males with Norrie disease (ND) have complete retinal detachment at the time of birth, surgical intervention after that time is typically not effective for preserving s...
5. `medquad_ans_0003876-4` (score=0.0004) - How might Meesmann corneal dystrophy be treated? Treatment is usually not needed unless a person is experiencing symptoms. Most people only need lubricating eye drops. If symptoms are more severe, therapeutic contact ...

**Answer**:

Treatment for septo-optic dysplasia is symptomatic and does not include a cure. Hormone deficiencies associated with the condition may be managed with hormone replacement therapy. Optical problems are generally not treatable, and vision, physical, and occupational therapies may be required [medquad_ans_0005548-5]. Additional management strategies such as genetic counseling, drug therapy, and palliative care may be considered as part of a comprehensive care plan [medquad_ans_0000899-5].

**Supported citations**: `medquad_ans_0005548-5`, `medquad_ans_0000899-5`
**Unsupported citations**: _none_

### Example 32 (`0000208-7`)

**Question**: What are the complications of Prostate Enlargement: Benign Prostatic Hyperplasia ?

**Scores**: context_relevance=1.000, groundedness=0.994, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000208-7` (score=0.9931) - The complications of benign prostatic hyperplasia may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney da...
2. `medquad_ans_0000208-11` (score=0.9898) - The complications of benign prostatic hyperplasia treatment depend on the type of treatment. Medications Medications used to treat benign prostatic hyperplasia may have side effects that sometimes can be serious. Men ...
3. `medquad_ans_0000058-19` (score=0.9847) - Yes. Any of the symptoms caused by prostate cancer may also be due to enlargement of the prostate, which is not cancer. If you have any of the symptoms mentioned in question #10, see your doctor or a urologist to find...
4. `medquad_ans_0000208-13` (score=0.3245) - Researchers have not found that eating, diet, and nutrition play a role in causing or preventing benign prostatic hyperplasia. However, a health care provider can give information about how changes in eating, diet, or...
5. `medquad_ans_0000058-26` (score=0.0243) - Surgery is a common treatment for early stage prostate cancer. It is used to remove the cancer. The surgeon may remove the entire prostate -- a type of surgery called radical prostatectomy -- or, in some cases, remove...

**Answer**:

The complications of benign prostatic hyperplasia (prostate enlargement) may include - acute urinary retention - chronic, or long lasting, urinary retention - blood in the urine - urinary tract infections (UTIs) - bladder damage - kidney damage - bladder stones. Most men with benign prostatic hyperplasia do not develop these complications. However, kidney damage in particular can be a serious health threat when it occurs [medquad_ans_0000208-7].

Complications related to treatment include side effects from medications such as hives, rash, itching, shortness of breath, rapid or irregular heartbeat, painful erection lasting hours, swelling of the face, eyes, lips, tongue, throat, arms, hands, feet, ankles, or legs, difficulty breathing or swallowing, chest pain, dizziness or fainting when standing up suddenly, sudden decrease or loss of vision, blurred vision, sudden decrease or loss of hearing, chest

**Supported citations**: `medquad_ans_0000208-7`
**Unsupported citations**: _none_

### Example 33 (`0000001-2__dup1`)

**Question**: What are the treatments for Absence of the Septum Pellucidum ?

**Scores**: context_relevance=0.667, groundedness=0.725, answer_relevance=0.756

**Top docs**:

1. `medquad_ans_0000236-2__dup1` (score=0.9962) - Treatment may include physical therapy, medication for seizures, and the placement of a shunt in the brain to remove excess fluid in the brain.
2. `medquad_ans_0000029-18` (score=0.9948) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
3. `medquad_ans_0000085-2__dup1` (score=0.6420) - Currently, only central core disease has an effective treatment (see above). There are no known cures for any of these disorders. Supportive treatment may involve orthopedic treatments, as well as physical, occupation...
4. `medquad_ans_0000072-7` (score=0.2375) - Many holes in the heart don't need treatment, but some do. Those that do often are repaired during infancy or early childhood. Sometimes adults are treated for holes in the heart if problems develop. The treatment you...
5. `medquad_ans_0000033-2__dup1` (score=0.1454) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...

**Answer**:

Answer: The provided context does not contain any information about the treatments for Absence of the Septum Pellucidum. [medquad_ans_0000236-2\_\_dup1][medquad_ans_0000029-18][medquad_ans_0000085-2\_\_dup1][medquad_ans_0000072-7] None of these sources mention Absence of the Septum Pellucidum or its treatment. Therefore, based on the given context, the treatments for this condition cannot be determined.

**Supported citations**: `medquad_ans_0000236-2__dup1`, `medquad_ans_0000029-18`, `medquad_ans_0000085-2__dup1`, `medquad_ans_0000072-7`
**Unsupported citations**: _none_

### Example 34 (`0002322-2`)

**Question**: What are the symptoms of Febrile Ulceronecrotic Mucha-Habermann disease ?

**Scores**: context_relevance=1.000, groundedness=0.517, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002322-2` (score=0.9992) - What are the signs and symptoms of febrile ulceronecrotic Mucha-Habermann disease? Initial symptoms of FUMHD include red scaly skin legions (papules) that ulcerate, breakdown, form open sores, then a red-brown crust (...
2. `medquad_ans_0003540-2` (score=0.0001) - What are the symptoms reported in children who have Lemierre syndrome? In children and adolescents, Lemierre syndrome usually begins with a severe sore throat, persistent fever, and possibly chills. Some cases begin w...
3. `medquad_ans_0000004_7-2` (score=0.0001) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...
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

1. `medquad_ans_0001090-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000959-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000405-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000685-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000045-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

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
3. `medquad_ans_0000390-2` (score=0.0001) - Frontonasal dysplasia is likely a rare condition; at least 100 cases have been reported in the scientific literature.
4. `medquad_ans_0000389-1` (score=0.0001) - Frontometaphyseal dysplasia is a disorder involving abnormalities in skeletal development and other health problems. It is a member of a group of related conditions called otopalatodigital spectrum disorders, which al...
5. `medquad_ans_0002374-2` (score=0.0001) - What are the symptoms of fibrous dysplasia? Fibrous dysplasia may cause no symptoms, mild symptoms, or severe symptoms. The most common symptoms are bone pain, bone deformities, fractures, and skin pigmentation differ...

**Answer**:

The symptoms of Frontometaphyseal dysplasia include abnormality of dental morphology, abnormality of frontal sinus, abnormality of the metaphyses, bowing of the long bones, camptodactyly of finger, craniofacial hyperostosis, hypertelorism, limitation of joint mobility, prominent supraorbital ridges, abnormal form of the vertebral bodies, abnormality of the palate, accelerated skeletal maturation, aplasia/hypoplasia of the thumb, arachnodactyly, conductive hearing impairment, elbow dislocation, scoliosis, sensorineural hearing impairment, skeletal muscle atrophy, synostosis of carpal bones, ulnar deviation of finger, abnormality of the larynx, abnormality of the urethra, complete atrioventricular canal defect, craniosynostosis, tracheal stenosis, ureteral stenosis, ankle

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 37 (`0000128-9`)

**Question**: What are the treatments for What I need to know about Diverticular Disease ?

**Scores**: context_relevance=0.818, groundedness=0.809, answer_relevance=0.673

**Top docs**:

1. `medquad_ans_0000116-7` (score=0.9914) - Treatment depends on the severity of symptoms. The gastroenterologist will - review the medications the person is taking - make recommendations to change or stop certain medications - recommend that the person quit sm...
2. `medquad_ans_0000097-9` (score=0.9854) - Diverticulitis can attack suddenly and cause complications, such as - an abscessa painful, swollen, pus-filled area just outside the colon wallcaused by infection - a perforationa small tear or hole in the diverticula...
3. `medquad_ans_0000128-7` (score=0.9832) - Treatment for diverticular disease depends on how serious the problem is and whether you are suffering from diverticulosis or diverticulitis. Most people get better by changing their diet. If you have rectal bleeding,...
4. `medquad_ans_0000128-9` (score=0.9815) - A doctor may prescribe antibiotics and recommend following a liquid diet. Most people get better with this treatment. Some people may need surgery and other treatments. - Surgery. Serious problems from diverticulitis ...
5. `medquad_ans_0000097-8` (score=0.9577) - A health care provider may treat the symptoms of diverticulosis with a high-fiber diet or fiber supplements, medications, and possibly probiotics. Treatment for diverticular disease varies, depending on whether a pers...

**Answer**:

The provided context does not contain information about the treatments for diverticular disease in general. It specifically discusses complications of diverticulitis (such as abscess, perforation, peritonitis, fistula, and intestinal obstruction) and mentions antibiotic treatment for abscesses and possible drainage, but does not cover comprehensive treatments for diverticular disease. Therefore, based on the available context, it is not possible to provide a complete answer to the question.

[medquad_ans_0000097-9]

**Supported citations**: `medquad_ans_0000097-9`
**Unsupported citations**: _none_

### Example 38 (`0000619-1__dup1`)

**Question**: What are the symptoms of Baraitser Brett Piesowicz syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.542, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000619-1__dup1` (score=0.9995) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...
2. `medquad_ans_0002911-2` (score=0.0003) - What are the signs and symptoms of Holt-Oram syndrome? People with Holt-Oram syndrome have abnormally developed bones in their upper limbs. At least one abnormality in the bones of the wrist (carpal bones) is present....
3. `medquad_ans_0004414-2` (score=0.0002) - What are the signs and symptoms of Nicolaides-Baraitser syndrome? Nicolaides-Baraitser syndrome (NCBRS) is typically characterized by intellectual disability, seizures, short stature, sparse hair, distinctive facial f...
4. `medquad_ans_0002441-2` (score=0.0001) - What are the signs and symptoms of Froelich syndrome? Signs and symptoms of Froelich syndrome include obesity, small testes, delay in the onset of puberty, short stature (compared to other family members of the same s...
5. `medquad_ans_0002652-2` (score=0.0001) - What are the signs and symptoms of Gordon syndrome? Gordon syndrome belongs to a group of conditions known as the distal arthrogryposes, which are characterized by stiffness and impaired mobility of certain joints of ...

**Answer**:

The symptoms of Baraitser Brett Piesowicz syndrome include cerebral calcification, hyperreflexia, hypertonia, microcephaly, seizures, abnormality of movement, cerebral cortical atrophy, cataract, opacification of the corneal stroma, renal insufficiency, anteverted nares, cerebellar hypoplasia, decreased liver function, elevated hepatic transaminases, failure to thrive, hepatomegaly, high palate, increased CSF protein, intellectual disability, profound, jaundice, lissencephaly, long philtrum, low-set ears, microretrognathia, muscular hypotonia of the trunk, nystagmus, pachygyria, petechiae, phenotypic variability, polymicrogyria, sloping forehead, spasticity, splenomegaly, thrombocytopenia, ventriculomeg

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 39 (`0003911-2`)

**Question**: What are the treatments for Meningioma ?

**Scores**: context_relevance=0.667, groundedness=0.050, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000052-2__dup1` (score=0.9958) - The three most commonly used treatments are surgery, radiation, and chemotherapy. Doctors also may prescribe steroids to reduce the tumor-related swelling inside the CNS.
2. `medquad_ans_0000209-2` (score=0.9954) - Treatment involves restoring normal levels of thyroid hormone and may include thyroid drugs, radioactive iodine, and sometimes partial or complete surgical removal of the thyroid.
3. `medquad_ans_0000001-8__dup1` (score=0.9895) - No single treatment is effective for all patients. Treatment should be individualized, and often combined, depending on patient characteristics such as age and tumor size. If the tumor has not yet invaded surrounding ...
4. `medquad_ans_0000205-2__dup1` (score=0.9863) - There is no specific treatment to stop or reverse any form of MD. Treatment may include physical therapy, respiratory therapy, speech therapy, orthopedic appliances used for support, and corrective orthopedic surgery....
5. `medquad_ans_0000258-5__dup1` (score=0.9802) - Aseptic meningitis, encephalitis, or meningoencephalitis requires hospitalization and supportive treatment based on severity. Anti-inflammatory drugs, such as corticosteroids, may be considered under specific circumst...

**Answer**:

The provided context does not contain information about the treatments for meningioma. Therefore, based on the given context, it is not possible to answer the question. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 40 (`0000006_7-8`)

**Question**: What are the treatments for Childhood Central Nervous System Germ Cell Tumors ?

**Scores**: context_relevance=1.000, groundedness=0.935, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0000006_7-8` (score=0.9738) - Key Points - There are different types of treatment for patients with childhood central nervous system (CNS) germ cell tumors. - Children with childhood CNS germ cell tumors should have their treatment planned by a te...
2. `medquad_ans_0000006_2-8` (score=0.6526) - Key Points - There are different types of treatment for children with brain and spinal cord tumors. - Children with brain or spinal cord tumors should have their treatment planned by a team of health care providers wh...
3. `medquad_ans_0000006_6-8` (score=0.0341) - Key Points - There are different types of treatment for children who have central nervous system (CNS) embryonal tumors. - Children who have CNS embryonal tumors should have their treatment planned by a team of health...
4. `medquad_ans_0000006_4-8` (score=0.0284) - Key Points - There are different types of treatment for patients with central nervous system atypical teratoid/rhabdoid tumor. - Children with atypical teratoid/rhabdoid tumor should have their treatment planned by a ...
5. `medquad_ans_0000015_1-6` (score=0.0152) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The type of germ cell tumor. - Where the tumor first beg...

**Answer**:

The treatments for Childhood Central Nervous System (CNS) Germ Cell Tumors include radiation therapy, chemotherapy, surgery, high-dose chemotherapy with stem cell rescue, and participation in clinical trials for new treatments [medquad_ans_0000006_7-8]. Treatment is typically planned by a team of health care providers who are experts in treating cancer in children, including a pediatric oncologist and/or a radiation oncologist, along with other specialists such as a pediatric neurosurgeon, neurologist, endocrinologist, ophthalmologist, pediatric nurse specialist, rehabilitation specialist, psychologist, and social worker [medquad_ans_0000006_7-8]. Some treatments are standard, while others are being tested in clinical trials [medquad_ans_0000006_7-8]. Patients may consider participating in clinical trials before, during, or after starting treatment [

**Supported citations**: `medquad_ans_0000006_7-8`
**Unsupported citations**: _none_

### Example 41 (`0004401-1`)

**Question**: What is (are) Neutral lipid storage disease with myopathy ?

**Scores**: context_relevance=0.889, groundedness=0.984, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0004401-1` (score=0.9127) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. The accumulation of fats in muscle tissue leads to muscle weakness (my...
2. `medquad_ans_0000729-1` (score=0.8874) - Neutral lipid storage disease with myopathy is a condition in which fats (lipids) are stored abnormally in organs and tissues throughout the body. People with this condition have muscle weakness (myopathy) due to the ...
3. `medquad_ans_0000927-1__dup1` (score=0.3344) - Cap myopathy is a disorder that primarily affects skeletal muscles, the muscles that the body uses for movement. People with cap myopathy have muscle weakness (myopathy) and poor muscle tone (hypotonia) throughout the...
4. `medquad_ans_0000002-1__dup1` (score=0.0052) - Acid lipase disease or deficiency occurs when the enzyme needed to break down certain fats that are normally digested by the body is lacking or missing, resulting in the toxic buildup of these fats in the bodys cells ...
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
3. `medquad_ans_0000949-1__dup1` (score=0.0097) - Viruses are capsules with genetic material inside. They are very tiny, much smaller than bacteria. Viruses cause familiar infectious diseases such as the common cold, flu and warts. They also cause severe illnesses su...
4. `medquad_ans_0000512-1__dup1` (score=0.0080) - Infectious diseases kill more people worldwide than any other single cause. Infectious diseases are caused by germs. Germs are tiny living things that are found everywhere - in air, soil and water. You can get infecte...
5. `medquad_ans_0000092-1__dup1` (score=0.0066) - Cytomegalovirus (CMV) is a virus found throughout the world that infects between 50 to 80 percent of all adults in the United States by the age of 40. CMV is in the same family of viruses that causes cold sores (herpe...

**Answer**:

Zika is a virus that is spread by mosquitoes. A pregnant mother can pass it to her baby during pregnancy or around the time of birth. A man can spread it to his partner during sexual contact. There have also been reports that the virus has spread through blood transfusions. Most people who get the virus do not get sick. One in five people do get symptoms, which can include a fever, rash, joint pain, and conjunctivitis (pinkeye). Symptoms are usually mild, and start 2 to 7 days after being bitten by an infected mosquito. A blood test can tell whether you have the infection. There are no vaccines or medicines to treat it. Drinking lots of fluids, resting, and taking acetaminophen might help. Zika can cause microcephaly (a serious birth defect of the brain) and other problems in babies whose mothers were infected while pregnant. The Centers for Disease Control and Prevention recommends that pregnant women

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 43 (`0000019-3__dup2`)

**Question**: What causes Broken Heart Syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000019-3__dup2` (score=0.9850) - The cause of broken heart syndrome isnt fully known. However, extreme emotional or physical stress is believed to play a role in causing the temporary disorder. Although symptoms are similar to those of a heart attack...
2. `medquad_ans_0000033-10` (score=0.9841) - Coronary heart disease, or CHD, is the most common underlying cause of a heart attack. Coronary arteries are the blood vessels that bring blood and oxygen to the heart muscle. Most heart attacks are caused by a blood ...
3. `medquad_ans_0000008-4__dup2` (score=0.7459) - An arrhythmia can occur if the electrical signals that control the heartbeat are delayed or blocked. This can happen if the special nerve cells that produce electrical signals don't work properly. It also can happen i...
4. `medquad_ans_0000034-12` (score=0.0337) - Heart failure is caused by other diseases and conditions that damage the heart muscle. It is most commonly caused by coronary artery disease, including heart attack. Diabetes and high blood pressure also contribute to...
5. `medquad_ans_0000034-2__dup3` (score=0.0293) - Heart failure is caused by other diseases or conditions that damage the heart muscle such as coronary artery disease (including heart attacks), diabetes, and high blood pressure. Treating these problems can prevent or...

**Answer**:

Broken heart syndrome is believed to be caused by extreme emotional or physical stress, which triggers a surge of fight or flight hormones such as adrenaline and noradrenaline. This hormone surge overwhelms and "stuns" the heart muscle, leading to temporary changes in heart muscle cells or coronary blood vessels. Potential causes include a hormone surge, coronary artery spasm, or microvascular dysfunction [medquad_ans_0000019-3__dup2].

**Supported citations**: `medquad_ans_0000019-3__dup2`
**Unsupported citations**: _none_

### Example 44 (`0000448-1`)

**Question**: What is (are) Hashimoto thyroiditis ?

**Scores**: context_relevance=0.800, groundedness=0.048, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000013-1__dup4` (score=0.8969) - Thyroid disease is a disorder that affects the thyroid gland. Sometimes the body produces too much or too little thyroid hormone. Thyroid hormones regulate metabolismthe way the body uses energyand affect nearly every...
2. `medquad_ans_0000892-1__dup1` (score=0.8647) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...
3. `medquad_ans_0000008-1__dup6` (score=0.8143) - Hyperthyroidism is a disorder that occurs when the thyroid gland makes more thyroid hormone than the body needs. Hyperthyroidism is sometimes called thyrotoxicosis, the technical term for too much thyroid hormone in t...
4. `medquad_ans_0000495-1__dup2` (score=0.6415) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...
5. `medquad_ans_0000498-1__dup2` (score=0.5367) - Your thyroid is a butterfly-shaped gland in your neck, just above your collarbone. It is one of your endocrine glands, which make hormones. Thyroid hormones control the rate of many activities in your body. These incl...

**Answer**:

The provided context does not mention Hashimoto thyroiditis. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 45 (`0001043-1`)

**Question**: What is (are) Wolman disease ?

**Scores**: context_relevance=0.800, groundedness=0.953, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001043-1` (score=0.7181) - Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). In affected individuals, harmful amounts of lipids accumulate in the spleen, liver, ...
2. `medquad_ans_0006463-1` (score=0.6708) - Wolman disease is a type of lysosomal storage disorder. It is an inherited condition that causes a buildup of lipids (fats) in body organs and calcium deposits in the adrenal glands. Common symptoms in infants include...
3. `medquad_ans_0006380-1` (score=0.0056) - Waardenburg syndrome (WS) is a group of genetic conditions characterized by varying degrees of hearing loss and differences in the coloring (pigmentation) of the eyes, hair, and skin. Signs and symptoms can vary both ...
4. `medquad_ans_0000276-3__dup1` (score=0.0029) - Early onset of the disease may foretell a worse prognosis than later onset. If the disorder is detected early and treated appropriately, an individual with WD can usually enjoy normal health and a normal lifespan. If ...
5. `medquad_ans_0000188-1__dup1` (score=0.0004) - Machado-Joseph disease (MJD), which is also called spinocerebellar ataxia type 3, is a rare hereditary ataxia (ataxia is a medical term meaning lack of muscle control). The disease is characterized by slowly progressi...

**Answer**:

Wolman disease is a rare inherited condition involving the breakdown and use of fats and cholesterol in the body (lipid metabolism). It is a type of lysosomal storage disorder caused by mutations in the LIPA gene and inherited in an autosomal recessive manner. The condition leads to a buildup of lipids in body organs such as the spleen, liver, bone marrow, small intestine, adrenal glands, and lymph nodes, along with calcium deposits in the adrenal glands. Infants are often healthy at birth but soon develop symptoms including enlarged liver and spleen (hepatosplenomegaly), poor weight gain, low muscle tone, jaundice, vomiting, diarrhea, developmental delay, anemia, and poor nutrient absorption. The disease is severe and life-threatening, though new therapies such as bone marrow transplantation and enzyme replacement therapy are showing promise [medquad_ans_0001043-1]medquad_ans_00

**Supported citations**: `medquad_ans_0001043-1`
**Unsupported citations**: _none_

### Example 46 (`0002753-3`)

**Question**: What causes Hemangiopericytoma ?

**Scores**: context_relevance=1.000, groundedness=0.982, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002753-3` (score=0.9955) - What causes hemangiopericytoma? The cause of the disease is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma a...
2. `medquad_ans_0000201-4__dup2` (score=0.0044) - Hematuria can be caused by menstruation, vigorous exercise, sexual activity, viral illness, trauma, or infection, such as a urinary tract infection (UTI). More serious causes of hematuria include - cancer of the kidne...
3. `medquad_ans_0002753-4` (score=0.0001) - What treatment is available for meningeal hemangiopericytoma? Radical surgical resection with removal of all meningeal attachments is typically the preferred treatment. However this treatment option is generally possi...
4. `medquad_ans_0002760-3` (score=0.0001) - What causes hemifacial microsomia? For most people with hemifacial microsomia, the cause is unknown. It is believed that something occurs in the early stages of development, such as a disturbance of the blood supply t...
5. `medquad_ans_0002753-1` (score=0.0001) - Hemangiopericytoma is a term used to described a group of tumors that are derived from pericytes, the cells normally arranged along specific types of blood vessels called capillaries and venules. These types of tumors...

**Answer**:

The cause of hemangiopericytoma is unknown, and no strong clinical data exist to indicate a convincing link to specific causes. Some reports suggest a relationship between hemangiopericytoma and occupational vinyl chloride exposure, as well as exposure to herbicides [medquad_ans_0002753-3].

**Supported citations**: `medquad_ans_0002753-3`
**Unsupported citations**: _none_

### Example 47 (`0000655-3`)

**Question**: What are the genetic changes related to microphthalmia ?

**Scores**: context_relevance=0.875, groundedness=0.974, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000621-3` (score=0.9991) - Mutations in the MLYCD gene cause malonyl-CoA decarboxylase deficiency. The MLYCD gene provides instructions for making an enzyme called malonyl-CoA decarboxylase. Within cells, this enzyme helps regulate the formatio...
2. `medquad_ans_0000588-3` (score=0.9988) - Mutations in the MT-ND1, MT-ND4, MT-ND4L, or MT-ND6 gene can cause LHON. These genes are found in the DNA of cellular structures called mitochondria, which convert the energy from food into a form that cells can use. ...
3. `medquad_ans_0000655-3` (score=0.9980) - Microphthalmia may be caused by changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or...
4. `medquad_ans_0000398-1__dup1` (score=0.9419) - Genes are the building blocks of heredity. They are passed from parent to child. They hold DNA, the instructions for making proteins. Proteins do most of the work in cells. They move molecules from one place to anothe...
5. `medquad_ans_0000013-12` (score=0.3008) - People can't change the genes they inherit from their parents, but they can change other things to prevent diseases that run in the family. This is good news because many diseases result from a combination of a person...

**Answer**:

The genetic changes related to microphthalmia include changes in many genes involved in the early development of the eye, most of which have not been identified. The condition may also result from a chromosomal abnormality affecting one or more genes. Most genetic changes associated with isolated microphthalmia have been identified only in very small numbers of affected individuals [medquad_ans_0000655-3].

**Supported citations**: `medquad_ans_0000655-3`
**Unsupported citations**: _none_

### Example 48 (`0000028-3__dup3`)

**Question**: What are the symptoms of Gout ?

**Scores**: context_relevance=0.833, groundedness=0.790, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000054-11` (score=0.9940) - People who have P.A.D. may have symptoms when walking or climbing stairs. These may include pain, numbness, aching, or heaviness in the leg muscles. Symptoms may also include cramping in the affected leg(s) and in the...
2. `medquad_ans_0000028-18` (score=0.0379) - The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH has more information about gout. Check out the information here. Also, see "Key Words" related to gout.
3. `medquad_ans_0000028-14` (score=0.0125) - To confirm a diagnosis of gout, the doctor inserts a needle into the inflamed joint and draws a sample of synovial fluid, the substance that lubricates a joint. A laboratory technician places some of the fluid on a sl...
4. `medquad_ans_0000093-5__dup1` (score=0.0043) - The most common signs and symptoms of Crohn's disease are - diarrhea - abdominal cramping and pain - weight loss Other general signs and symptoms include - feeling tired - nausea or loss of appetite - fever - anemiaa ...
5. `medquad_ans_0000028-9` (score=0.0026) - Most people with gout have too much uric acid in their blood, a condition called hyperuricemia. Uric acid is a substance that results from the breakdown of purines, which are part of all human tissue and are found in ...

**Answer**:

The symptoms of gout include pain, swelling, redness, and warmth in the affected joint, typically the big toe, but can occur in other joints as well. A diagnosis is confirmed by detecting uric acid crystals in the synovial fluid of the joint via a needle aspiration and microscopic examination [medquad_ans_0000028-14]. Most people with gout have hyperuricemia, a condition characterized by excess uric acid in the blood, which results from the breakdown of purines found in many foods, especially high-protein foods [medquad_ans_0000028-9]. The buildup of needle-like uric acid crystals in joint spaces or connective tissue leads to gout flare-ups [medquad_ans_0000028-9].

**Supported citations**: `medquad_ans_0000028-14`, `medquad_ans_0000028-9`
**Unsupported citations**: _none_

### Example 49 (`0002001-3`)

**Question**: What causes Dystonia 8 ?

**Scores**: context_relevance=0.333, groundedness=0.000, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000704-2` (score=0.0001) - The prevalence of myoclonus-dystonia is unknown. This condition has been described in people worldwide.
2. `medquad_ans_0000312-2` (score=0.0001) - Early-onset primary dystonia is among the most common forms of childhood dystonia. This disorder occurs most frequently in people of Ashkenazi (central and eastern European) Jewish heritage, affecting 1 in 3,000 to 9,...
3. `medquad_ans_0000309-5` (score=0.0001) - These resources address the diagnosis or management of dystonia 6: - Gene Review: Gene Review: Dystonia Overview - Genetic Testing Registry: Dystonia 6, torsion These resources from MedlinePlus offer information about...
4. `medquad_ans_0000312-5` (score=0.0001) - These resources address the diagnosis or management of early-onset primary dystonia: - Gene Review: Gene Review: DYT1 Early-Onset Primary Dystonia - Genetic Testing Registry: Dystonia 1 - MedlinePlus Encyclopedia: Mov...
5. `medquad_ans_0000309-2` (score=0.0001) - The prevalence of dystonia 6 is unknown. Studies indicate that it likely accounts for between 1 and 3 percent of all cases of dystonia. For reasons that are unclear, the disorder appears to be slightly more prevalent ...

**Answer**:

The provided context does not contain information about the causes of Dystonia 8. Therefore, the cause of Dystonia 8 cannot be determined from the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 50 (`0000612-2`)

**Question**: How many people are affected by lysinuric protein intolerance ?

**Scores**: context_relevance=0.667, groundedness=0.978, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000612-2` (score=0.9993) - Lysinuric protein intolerance is estimated to occur in 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations this condition occurs less frequently, but the exact incidence is unk...
2. `medquad_ans_0000066-2` (score=0.9977) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
3. `medquad_ans_0000974-2` (score=0.9958) - This condition is rare, affecting an estimated 1 in 500,000 to 1 in 1 million newborns. In most parts of the world, tetrahydrobiopterin deficiency accounts for 1 to 3 percent of all cases of elevated phenylalanine lev...
4. `medquad_ans_0000837-2` (score=0.0040) - Mild protein C deficiency affects approximately 1 in 500 individuals. Severe protein C deficiency is rare and occurs in an estimated 1 in 4 million newborns.
5. `medquad_ans_0000077-5__dup1` (score=0.0002) - Hypotension can affect people of all ages. However, people in certain age groups are more likely to have certain types of hypotension. Older adults are more likely to have orthostatic and postprandial hypotension. Chi...

**Answer**:

Lysinuric protein intolerance affects an estimated 1 in 60,000 newborns in Finland and 1 in 57,000 newborns in Japan. Outside these populations, the condition occurs less frequently, but the exact incidence is unknown [medquad_ans_0000612-2].

**Supported citations**: `medquad_ans_0000612-2`
**Unsupported citations**: _none_

### Example 51 (`0000170-5__dup1`)

**Question**: What are the symptoms of Renal Artery Stenosis ?

**Scores**: context_relevance=0.750, groundedness=0.075, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000150-4__dup2` (score=0.9987) - Most people with high blood pressure do not have symptoms. In rare cases, high blood pressure can cause headaches. Kidney disease also does not have symptoms in the early stages. A person may have swelling called edem...
2. `medquad_ans_0000071-5__dup1` (score=0.9951) - Because diagnosis is based on blood pressure readings, this condition can go undetected for years, as symptoms do not usually appear until the body is damaged from chronic high blood pressure. Complications of High Bl...
3. `medquad_ans_0000101-5__dup1` (score=0.6197) - Many people who have peripheral artery disease (P.A.D.) dont have any signs or symptoms. Even if you don't have signs or symptoms, ask your doctor whether you should get checked for P.A.D. if you're: Aged 70 or older ...
4. `medquad_ans_0000012-5__dup1` (score=0.4271) - Atherosclerosis usually doesn't cause signs and symptoms until it severely narrows or totally blocks an artery. Many people don't know they have the disease until they have a medical emergency, such as a heart attack ...
5. `medquad_ans_0000038-14` (score=0.1612) - Kidney disease is often called a "silent" disease, because most people have no symptoms in early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NOT wait for sympto...

**Answer**:

The provided context does not mention Renal Artery Stenosis or its symptoms. Therefore, based on the given information, it is not possible to answer the question. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 52 (`0000712-5`)

**Question**: What are the treatments for Naegeli-Franceschetti-Jadassohn syndrome/dermatopathia pigmentosa reticularis ?

**Scores**: context_relevance=0.800, groundedness=0.823, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000712-5` (score=0.9908) - These resources address the diagnosis or management of NFJS/DPR: - Foundation for Ichthyosis and Related Skin Types (FIRST): Palmoplantar Keratodermas - Genetic Testing Registry: Dermatopathia pigmentosa reticularis -...
2. `medquad_ans_0004303-4` (score=0.9591) - Is there a treatment for Naegeli syndrome? Treatment for Naegeli syndrome is based on an individual's symptoms. Dry skin can be moisturized with creams. To avoid overheating, affected individuals should wear appropria...
3. `medquad_ans_0000618-5` (score=0.9550) - These resources address the diagnosis or management of mal de Meleda: - Foundation for Ichthyosis and Related Skin Types: Palmoplantar Keratodermas - Genetic Testing Registry: Acroerythrokeratoderma These resources fr...
4. `medquad_ans_0000273-2__dup2` (score=0.0673) - There is no cure for TSC, although treatment is available for a number of the symptoms. Rapamycin and related drugs are not yet approved by the U.S. Food and Drug Administration (FDA) for any purpose in individuals wi...
5. `medquad_ans_0003746-3` (score=0.0269) - How might Maffucci syndrome be treated? Management aims at relief of symptoms and early detection of malignancies. Individuals with Maffucci syndrome may benefit from consultations with the following specialists: Radi...

**Answer**:

The provided context does not specify treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis. While it mentions that treatment for Naegeli syndrome is based on an individual's symptoms, including moisturizing dry skin with creams, using wet dressings to avoid overheating, and dental care for cavities and tooth loss, this information is specific to Naegeli syndrome and not directly applicable to Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis [medquad_ans_0004303-4]. Therefore, based on the available context, no specific treatments for Naegeli-Franceschetti-Jadassohn syndrome or dermatopathia pigmentosa reticularis can be identified.

**Supported citations**: `medquad_ans_0004303-4`
**Unsupported citations**: _none_

### Example 53 (`0002259-1`)

**Question**: What are the symptoms of Familial hyperaldosteronism type III ?

**Scores**: context_relevance=1.000, groundedness=0.958, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002259-1` (score=0.9994) - What are the signs and symptoms of Familial hyperaldosteronism type III ? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hyperaldosteronism type III . If the information is...
2. `medquad_ans_0003038-1` (score=0.0001) - What are the signs and symptoms of Hypocalcemia, autosomal dominant? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypocalcemia, autosomal dominant. If the information is available...
3. `medquad_ans_0005138-2` (score=0.0001) - What are the signs and symptoms of Pseudohypoaldosteronism type 2? The most common symptom of pseudohypoaldosteronism type 2 is high blood pressure in adolescents or young adults. In its most severe form, it is associ...
4. `medquad_ans_0002270-1` (score=0.0001) - What are the signs and symptoms of Familial hypocalciuric hypercalcemia type 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Familial hypocalciuric hypercalcemia type 2. If the in...
5. `medquad_ans_0000845-5` (score=0.0001) - These resources address the diagnosis or management of pseudohypoaldosteronism type 2: - Gene Review: Gene Review: Pseudohypoaldosteronism Type II - Genetic Testing Registry: Pseudohypoaldosteronism, type 2 These reso...

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

**Scores**: context_relevance=1.000, groundedness=0.854, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0003040-1` (score=0.9993) - What are the signs and symptoms of Hypochromic microcytic anemia with iron overload? The Human Phenotype Ontology provides the following list of signs and symptoms for Hypochromic microcytic anemia with iron overload....
2. `medquad_ans_0000185-5__dup1` (score=0.9933) - Without treatment, iron may build up in the organs and cause complications, including - cirrhosis, or scarring of liver tissue - diabetes - irregular heart rhythms or weakening of the heart muscle - arthritis - erecti...
3. `medquad_ans_0000057-4__dup2` (score=0.7864) - Major Signs and Symptoms Your doctor may suspect you or your child has Fanconi anemia (FA) if you have signs and symptoms of: Anemia Bone marrow failure Birth defects Developmental or eating problems FA is an inherite...
4. `medquad_ans_0000067-5__dup1` (score=0.0019) - Hemochromatosis can affect many parts of the body and cause various signs and symptoms. Many of the signs and symptoms are similar to those of other diseases. Signs and symptoms of hemochromatosis usually don't occur ...
5. `medquad_ans_0000082-4__dup2` (score=0.0002) - The signs and symptoms of iron-deficiency anemia depend on its severity. Mild to moderate iron-deficiency anemia may have no signs or symptoms. When signs and symptoms do occur, they can range from mild to severe. Man...

**Answer**:

The signs and symptoms of Hypochromic microcytic anemia with iron overload include anemia, abnormality of the liver, and autosomal recessive inheritance. The Human Phenotype Ontology provides a list of these signs and symptoms, with frequencies estimated based on studies, though specific symptom frequencies are not detailed in the provided context [medquad_ans_0003040-1]. Additionally, complications associated with iron overload, such as cirrhosis, diabetes, irregular heart rhythms, arthritis, and erectile dysfunction, may occur, though these are not directly classified as symptoms of the anemia itself but rather as complications of iron overload [medquad_ans_0000185-5__dup1]. However, the provided context does not specify which of these complications are directly linked to Hypochromic microcytic anemia with iron overload. Therefore, the primary symptoms listed are anemia and abnormality of the

**Supported citations**: `medquad_ans_0003040-1`, `medquad_ans_0000185-5__dup1`
**Unsupported citations**: _none_

### Example 56 (`0000115-4`)

**Question**: Is benign chronic pemphigus inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000685-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000349-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000990-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000101-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000451-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention benign chronic pemphigus or its mode of inheritance. Therefore, it is not possible to determine based on the given information whether benign chronic pemphigus is inherited. [No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 57 (`0000145-4`)

**Question**: Is Camurati-Engelmann disease inherited ?

**Scores**: context_relevance=1.000, groundedness=0.947, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000673-4` (score=0.9988) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
2. `medquad_ans_0000786-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
3. `medquad_ans_0000883-4` (score=0.9984) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...
4. `medquad_ans_0000324-4` (score=0.9977) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
5. `medquad_ans_0000916-4__dup1` (score=0.9533) - How is Camurati-Engelmann disease inherited? Camurati-Engelmann disease is inherited in an autosomal dominant manner. This means that having a change (mutation) in only one copy of the responsible gene in each cell is...

**Answer**:

Camurati-Engelmann disease is inherited in an autosomal dominant manner, which means having a mutation in only one copy of the responsible gene in each cell is sufficient to cause the condition. In some cases, an affected person inherits the mutated gene from an affected parent, while in other cases, the mutation occurs for the first time in the individual with no family history of the condition, known as a de novo mutation [medquad_ans_0000916-4__dup1]. Therefore, Camurati-Engelmann disease is inherited in an autosomal dominant pattern.

**Supported citations**: `medquad_ans_0000916-4__dup1`
**Unsupported citations**: _none_

### Example 58 (`0000483-3`)

**Question**: What are the genetic changes related to histiocytosis-lymphadenopathy plus syndrome ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000095-2__dup3` (score=0.0011) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...
2. `medquad_ans_0000746-4` (score=0.0001) - This condition is inherited in an X-linked dominant pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromosomes), a ...
3. `medquad_ans_0000371-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000101-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000300-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain any information about histiocytosis-lymphadenopathy plus syndrome or its genetic changes. Therefore, it is not possible to answer the question based on the given context. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 59 (`0000738-4`)

**Question**: Is Noonan syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000370-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000371-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000676-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000998-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000685-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Noonan syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Noonan syndrome is inherited.

Answer: Insufficient context [medquad_ans_0000370-4, medquad_ans_0000371-4, medquad_ans_0000676-4, medquad_ans_0000998-4, medquad_ans_0000685-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000370-4, medquad_ans_0000371-4, medquad_ans_0000676-4, medquad_ans_0000998-4, medquad_ans_0000685-4`

### Example 60 (`0001093-5`)

**Question**: What are the treatments for 3-methylglutaconyl-CoA hydratase deficiency ?

**Scores**: context_relevance=0.875, groundedness=0.806, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0001093-5` (score=0.9952) - These resources address the diagnosis or management of 3-methylglutaconyl-CoA hydratase deficiency: - Baby's First Test - Genetic Testing Registry: 3-Methylglutaconic aciduria - MedlinePlus Encyclopedia: Metabolic Aci...
2. `medquad_ans_0006372-6` (score=0.9421) - How might VLCAD deficiency be treated? Management of VLCAD deficiency depends on many factors, including the form of the condition and the specific signs and symptoms present. For example, people affected by the sever...
3. `medquad_ans_0006585-5` (score=0.0107) - How might 3-hydroxyisobutyric aciduria be treated? There is no cure for 3-hydroxyisobutyric aciduria. Because it is so rare, there is limited evidence to support the effectiveness of treatment. However, affected peopl...
4. `medquad_ans_0005223-6` (score=0.0010) - How might pyruvate dehydrogenase deficiency be treated? Treatment of pyruvate dehydrogenase (PDH) deficiency rarely influences the course of the disease, but goals include stimulating the pyruvate dehydrogenase comple...
5. `medquad_ans_0001859-5` (score=0.0001) - How might dihydrolipoamide dehydrogenase deficiency be treated? There are currently no consensus recommendations for the management of dihydrolipoamide dehydrogenase (DLD) deficiency. Management can be hard because va...

**Answer**:

The provided context does not contain information about the treatments for 3-methylglutaconyl-CoA hydratase deficiency. [medquad_ans_0001093-5] mentions resources related to the diagnosis or management of this condition, but does not specify any treatments. Therefore, based on the available context, the treatments for 3-methylglutaconyl-CoA hydratase deficiency cannot be determined.

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
5. `medquad_ans_0002007-1` (score=0.0157) - Ohtahara syndrome is a neurological disorder characterized by seizures. The disorder affects newborns, usually within the first three months of life (most often within the first 10 days) in the form of epileptic seizu...

**Answer**:

Hyperkalemic periodic paralysis is a genetic condition that causes episodes of extreme muscle weakness, usually beginning in infancy or early childhood. Most often, these episodes involve a temporary inability to move muscles in the arms and legs. Episodes tend to increase in frequency until about age 25, after which they may occur less frequently. Factors that can trigger attacks include rest after exercise, potassium-rich foods, stress, fatigue, and long periods without food. Muscle strength improves between attacks, although many affected people continue to experience mild stiffness, particularly in muscles of the face and hands. This condition is caused by mutations in the SCN4A gene and is inherited in an autosomal dominant fashion [medquad_ans_0003000-1].

**Supported citations**: `medquad_ans_0003000-1`
**Unsupported citations**: _none_

### Example 62 (`0000225-5`)

**Question**: What are the treatments for congenital fibrosis of the extraocular muscles ?

**Scores**: context_relevance=0.800, groundedness=0.817, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000225-5` (score=0.9944) - These resources address the diagnosis or management of congenital fibrosis of the extraocular muscles: - Gene Review: Gene Review: Congenital Fibrosis of the Extraocular Muscles - Genetic Testing Registry: Fibrosis of...
2. `medquad_ans_0002381-3` (score=0.0099) - How might Filippi syndrome be treated? The treatment of Filippi syndrome is directed toward the specific symptoms that are apparent in each individual. Treatment may require the coordinated efforts of a team of medica...
3. `medquad_ans_0002374-4` (score=0.0046) - How might fibrous dysplasia be treated? Unfortunately, there is no cure for fibrous dysplasia. Treatment depends on the symptoms that develop. Fractures often require surgery, but can sometimes be treated with casting...
4. `medquad_ans_0000375-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases of fibrodysplasia ossificans progressiva result from ...
5. `medquad_ans_0000225-2` (score=0.0001) - CFEOM1 is the most common form of congenital fibrosis of the extraocular muscles, affecting at least 1 in 230,000 people. CFEOM1 and CFEOM3 have been reported worldwide, whereas CFEOM2 has been seen in only a few fami...

**Answer**:

The provided context does not contain specific information about the treatments for congenital fibrosis of the extraocular muscles. While it lists resources related to the diagnosis and management of this condition, such as genetic testing and information on extraocular muscle function testing and strabismus, no details on treatment approaches are given [medquad_ans_0000225-5]. Therefore, based on the available context, the treatments for congenital fibrosis of the extraocular muscles cannot be determined.

**Supported citations**: `medquad_ans_0000225-5`
**Unsupported citations**: _none_

### Example 63 (`0000152-1__dup1`)

**Question**: What is (are) Huntington's Disease ?

**Scores**: context_relevance=0.800, groundedness=0.533, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000491-1__dup2` (score=0.6731) - Huntington's disease (HD) is an inherited disease that causes certain nerve cells in the brain to waste away. People are born with the defective gene, but symptoms usually don't appear until middle age. Early symptoms...
2. `medquad_ans_0002944-1` (score=0.4746) - Huntington disease (HD) is an inherited condition that causes progressive degeneration of neurons in the brain. Signs and symptoms usually develop between ages 35 to 44 years and may include uncontrolled movements, lo...
3. `medquad_ans_0000490-1` (score=0.4568) - As its name suggests, a Huntington disease-like (HDL) syndrome is a condition that resembles Huntington disease. Researchers have described four HDL syndromes, designated Huntington disease-like 1 (HDL1) through Hunti...
4. `medquad_ans_0000152-1__dup1` (score=0.4376) - Huntington's disease (HD) is an inherited disorder that causes degeneration of brain cells, called neurons, in motor control regions of the brain, as well as other areas. Symptoms of the disease, which gets progressiv...
5. `medquad_ans_0000489-1` (score=0.1633) - Huntington disease is a progressive brain disorder that causes uncontrolled movements, emotional problems, and loss of thinking ability (cognition). Adult-onset Huntington disease, the most common form of this disorde...

**Answer**:

Huntington's disease (HD) is an inherited condition that causes progressive degeneration of neurons in the brain. It is caused by changes (mutations) in the HTT gene and is inherited in an autosomal dominant manner. Signs and symptoms usually develop between ages 35 to 44 years and may include uncontrolled movements, loss of intellectual abilities, and various emotional and psychiatric problems. Early symptoms may include uncontrolled movements, clumsiness, and balance problems, while later symptoms can involve loss of the ability to walk, talk, and swallow, and some people may stop recognizing family members. People are born with the defective gene, but symptoms typically do not appear until middle age. There is no cure; medicines can help manage symptoms but cannot slow down or stop the disease. If one of your parents has Huntington's disease, you have a 50 percent chance of inheriting it. A blood test can determine if a person has the HD

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 64 (`0000416-2`)

**Question**: How many people are affected by glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.889, groundedness=0.950, answer_relevance=0.533

**Top docs**:

1. `medquad_ans_0000419-2` (score=0.9993) - Glutathione synthetase deficiency is very rare. This disorder has been described in more than 70 people worldwide.
2. `medquad_ans_0000416-2` (score=0.9992) - Glutamate formiminotransferase deficiency is a rare disorder; approximately 20 affected individuals have been identified. Of these, about one-quarter have the severe form of the disorder. Everyone reported with the se...
3. `medquad_ans_0000150-2` (score=0.9990) - Carbamoyl phosphate synthetase I deficiency is a rare disorder; its overall incidence is unknown. Researchers in Japan have estimated that it occurs in 1 in 800,000 newborns in that country.
4. `medquad_ans_0000415-2` (score=0.9988) - GLUT1 deficiency syndrome is a rare disorder. Approximately 500 cases have been reported worldwide since the disorder was first identified in 1991. In Australia, the prevalence of the disorder has been estimated at 1 ...
5. `medquad_ans_0000442-2` (score=0.9986) - Guanidinoacetate methyltransferase deficiency is a very rare disorder. About 80 affected individuals have been described in the medical literature. Of these, approximately one-third are of Portuguese origin.

**Answer**:

Approximately 20 affected individuals have been identified with glutamate formiminotransferase deficiency [medquad_ans_0000416-2].

**Supported citations**: `medquad_ans_0000416-2`
**Unsupported citations**: _none_

### Example 65 (`0000738-2`)

**Question**: How many people are affected by Noonan syndrome ?

**Scores**: context_relevance=0.500, groundedness=0.044, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000115-3__dup3` (score=0.9993) - Mntriers disease is rare. The disease is more common in men than in women. The average age at diagnosis is 55.2
2. `medquad_ans_0000089-3__dup2` (score=0.0068) - People at highest risk for Marfan syndrome are those who have a family history of the condition. If you have Marfan syndrome, you have a 50 percent chance of passing the altered gene to each of your children. Marfan s...
3. `medquad_ans_0000018-13` (score=0.0039) - Sjgren's Syndrome Clinic National Institute of Dental and Craniofacial Research Building 10, Room 1N113 10 Center Drive MSC 1190 Bethesda, MD 20892-1190 301-435-8528 http://www.nidcr.nih.gov/Research/NIDCRLaboratories...
4. `medquad_ans_0000152-4__dup2` (score=0.0012) - IgA nephropathy is one of the most common kidney diseases, other than those caused by diabetes or high blood pressure.1 IgA nephropathy can occur at any age, although the first evidence of kidney disease most frequent...
5. `medquad_ans_0000134-3__dup2` (score=0.0001) - Experts do not know the exact cause of Zollinger-Ellison syndrome. About 25 to 30 percent of gastrinomas are caused by an inherited genetic disorder called multiple endocrine neoplasia type 1 (MEN1).1 MEN1 causes horm...

**Answer**:

The provided context does not contain any information about Noonan syndrome or the number of people affected by it. Therefore, the question cannot be answered based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 66 (`0000339-5__dup1`)

**Question**: How to diagnose Parasites - Lice - Pubic "Crab" Lice ?

**Scores**: context_relevance=0.714, groundedness=0.989, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000053-5__dup2` (score=0.9966) - Body lice infestation is diagnosed by finding eggs and crawling lice in the seams of clothing. Sometimes a body louse can be seen crawling or feeding on the skin. Although body lice and nits can be large enough to be ...
2. `medquad_ans_0000339-5__dup1` (score=0.9943) - Pubic lice are short and crab-like and appear very different from head and body lice. Pubic lice infestation is diagnosed by finding a “crab” louse or eggs on hair in the pubic region or, less commonly, elsewhere on t...
3. `medquad_ans_0000344-5__dup1` (score=0.9895) - If you suspect you have been infected, consult your health care provider immediately. Be sure to tell your health care provider if you have recently been exposed to raccoons or their feces. Diagnosis is difficult beca...
4. `medquad_ans_0000265-8` (score=0.9512) - In people who have been bitten by the flies that carry Loa loa in areas where Loa loa is known to exist, the diagnosis can be made in the following ways: - Identification of the adult worm by a microbiologist or patho...
5. `medquad_ans_0000214-5__dup1` (score=0.0870) - Misdiagnosis of head lice infestation is common. The diagnosis of head lice infestation is best made by finding a live nymph or adult louse on the scalp or hair of a person. Because adult and nymph lice are very small...

**Answer**:

Pubic "crab" lice infestation is diagnosed by finding a "crab" louse or eggs on hair in the pubic region or, less commonly, elsewhere on the body (eyebrows, eyelashes, beard, mustache, armpit, perianal area, groin, trunk, scalp). Although pubic lice and nits can be large enough to be seen with the naked eye, a magnifying lens may be necessary to find lice or eggs [medquad_ans_0000339-5__dup1].

**Supported citations**: `medquad_ans_0000339-5__dup1`
**Unsupported citations**: _none_

### Example 67 (`0005571-3`)

**Question**: What are the treatments for Shapiro syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005571-3` (score=0.9688) - What treatment options have been attempted for Shapiro syndrome? Evaluating effective treatment options for Shapiro syndrome can be difficult because of the limited number of diagnosed cases, the periodic nature of th...
2. `medquad_ans_0000860-2__dup1` (score=0.9474) - How might Brown-Sequard syndrome be treated?
3. `medquad_ans_0003593-6` (score=0.0196) - How might Liddle syndrome be treated? Treatment for Liddle syndrome includes following a low sodium diet as well as taking potassium-sparing diuretics, which reduce blood pressure and correct hypokalemia and metabolic...
4. `medquad_ans_0000275-2` (score=0.0175) - There is no cure for Williams syndrome, nor is there a standard course of treatment. Because WS is an uncommon and complex disorder, multidisciplinary clinics have been established at several centers in the United Sta...
5. `medquad_ans_0000155-3__dup1` (score=0.0069) - How might adenylosuccinase deficiency be treated? At the current time, there are no effective therapies for the treatment of adenylosuccinase deficiency. Treatment is supportive based on the specific features.

**Answer**:

Treatment options for Shapiro syndrome that have been attempted include anticonvulsants, clonidine, cyproheptadine, glycopyrrolate, bromocriptine, chlorpromazine, or sympathectomy. Responses to these treatments have varied among individuals. It is recommended that treatment options be discussed with a healthcare provider, as only a patient's health care provider can determine the appropriate course of treatment [medquad_ans_0005571-3].

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
5. `medquad_ans_0000627-4` (score=0.0014) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. At least 25 percent of Marfan syndrome cases result from a new m...

**Answer**:

FG syndrome is a genetic condition that affects many parts of the body and occurs almost exclusively in males. "FG" represents the surname initials of the first family diagnosed with the disorder. It is characterized by intellectual disability ranging from mild to severe, hypotonia, constipation and/or anal anomalies, a distinctive facial appearance, broad thumbs and great toes, relative macrocephaly, and abnormalities of the corpus callosum. Medical problems such as heart defects, seizures, undescended testicles, and inguinal hernia have also been reported. The condition is inherited in an X-linked recessive pattern, and mutations in genes such as MED12 (FGS1), FLNA (FGS2), CASK (FGS4), UPF3B (FGS6), and BRWD3 (FGS7) are associated with the disorder [medquad_ans_0002359-1]. Affected individuals often

**Supported citations**: `medquad_ans_0002359-1`
**Unsupported citations**: _none_

### Example 69 (`0006242-1`)

**Question**: What are the symptoms of Tukel syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006242-1` (score=0.9994) - What are the signs and symptoms of Tukel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Tukel syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0004003-2` (score=0.0029) - What are the signs and symptoms of Michels syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Michels syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0003459-2` (score=0.0001) - What are the symptoms of Lambert-Eaton myasthenic syndrome? Signs and symptoms of Lambert-Eaton myasthenic syndrome may include: Weakness or loss of movement that varies in severity: Difficulty climbing stairs Difficu...
4. `medquad_ans_0000335-2__dup1` (score=0.0001) - What are the signs and symptoms of Andersen-Tawil syndrome? Anderson-Tawil syndrome causes episodes of muscle weakness (periodic paralysis), changes in heart rhythm (arrhythmia), and developmental abnormalities. The m...
5. `medquad_ans_0003814-2` (score=0.0001) - What are the signs and symptoms of Marshall-Smith syndrome? Marshall-Smith syndrome is characterized by accelerated skeletal maturation, relative failure to thrive, respiratory difficulties, mental retardation, and un...

**Answer**:

The symptoms of Tukel syndrome include carpal bone aplasia, carpal synostosis, compensatory chin elevation, congenital fibrosis of extraocular muscles, nonprogressive restrictive external ophthalmoplegia, postaxial oligodactyly, ptosis, restrictive external ophthalmoplegia, syndactyly [medquad_ans_0006242-1].

**Supported citations**: `medquad_ans_0006242-1`
**Unsupported citations**: _none_

### Example 70 (`0000199-2__dup1`)

**Question**: What are the treatments for Mucolipidoses ?

**Scores**: context_relevance=0.333, groundedness=0.694, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000674-5` (score=0.9859) - These resources address the diagnosis or management of mucolipidosis II alpha/beta: - Gene Review: Gene Review: Mucolipidosis II - Genetic Testing Registry: I cell disease - MedlinePlus Encyclopedia: Clubfoot - Medlin...
2. `medquad_ans_0000675-5` (score=0.8807) - These resources address the diagnosis or management of mucolipidosis III alpha/beta: - Gene Review: Gene Review: Mucolipidosis III Alpha/Beta - Genetic Testing Registry: Pseudo-Hurler polydystrophy - MedlinePlus Encyc...
3. `medquad_ans_0000676-5` (score=0.0573) - These resources address the diagnosis or management of mucolipidosis III gamma: - Gene Review: Gene Review: Mucolipidosis III Gamma - Genetic Testing Registry: Mucolipidosis III Gamma - MedlinePlus Encyclopedia: Cloud...
4. `medquad_ans_0000677-5` (score=0.0223) - These resources address the diagnosis or management of mucolipidosis type IV: - Gene Review: Gene Review: Mucolipidosis IV - Genetic Testing Registry: Ganglioside sialidase deficiency - MedlinePlus Encyclopedia: Gastr...
5. `medquad_ans_0000215-2__dup1` (score=0.0207) - There is no agreed upon standard of treatment for neurosarcoidosis. Doctors generally recommend corticosteroid therapy as first-line therapy for individuals with the condition. Additional treatment with immunomodulato...

**Answer**:

The provided context does not specify any treatments for mucolipidoses. While it lists resources related to the diagnosis and management of various mucolipidosis types (e.g., II alpha/beta, III alpha/beta, III gamma, IV), it does not detail specific treatments such as drug therapies, surgical interventions, or rehabilitation strategies for these conditions. Therefore, based on the given information, the treatments for mucolipidoses cannot be determined.

[medquad_ans_0000674-5] [medquad_ans_0000675-5] [medquad_ans_0000676-5] [medquad_ans_0000677-5]

**Supported citations**: `medquad_ans_0000674-5`, `medquad_ans_0000675-5`, `medquad_ans_0000676-5`, `medquad_ans_0000677-5`
**Unsupported citations**: _none_

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

**Scores**: context_relevance=0.700, groundedness=0.707, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000211-6` (score=0.9968) - Urinary incontinence occurs in 11 to 34 percent of older men. Two to 11 percent of older men report daily UI.1 Although more women than men develop UI, the chances of a man developing UI increase with age because he i...
2. `medquad_ans_0000213-4__dup2` (score=0.0061) - Urinary retention in men becomes more common with age. - In men 40 to 83 years old, the overall incidence of urinary retention is 4.5 to 6.8 per 1,000 men.2 - For men in their 70s, the overall incidence increases to 1...
3. `medquad_ans_0000517-2` (score=0.0038) - Incontinentia pigmenti is an uncommon disorder. Between 900 and 1,200 affected individuals have been reported in the scientific literature. Most of these individuals are female, but several dozen males with incontinen...
4. `medquad_ans_0000214-4__dup2` (score=0.0005) - Urinary tract infections are the second most common type of infection in the body, accounting for about 8.1 million visits to health care providers each year.1 Women are especially prone to UTIs for anatomical reasons...
5. `medquad_ans_0000099-2__dup2` (score=0.0005) - Nearly 18 million U.S. adultsabout one in 12have fecal incontinence.1 People of any age can have a bowel control problem, though fecal incontinence is more common in older adults. Fecal incontinence is slightly more c...

**Answer**:

The provided context does not specify the exact number of people affected by urinary incontinence (UI) in men. It states that urinary incontinence occurs in 11 to 34 percent of older men and that two to 11 percent report daily UI [medquad_ans_0000211-6]. However, no absolute number of affected individuals is given. Therefore, based on the available information, it is not possible to determine the total number of people affected by urinary incontinence in men.

Answer: Insufficient information to determine the exact number of people affected by urinary incontinence in men. [medquad_ans_0000211-6]

**Supported citations**: `medquad_ans_0000211-6`
**Unsupported citations**: _none_

### Example 73 (`0005415-1`)

**Question**: What is (are) Robinow syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.550, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005415-1` (score=0.3207) - Robinow syndrome is a rare disorder that affects the bones as well as other parts of the body. Two forms of Robinow syndrome have been described: autosomal recessive Robinow syndrome, and the milder autosomal dominant...
2. `medquad_ans_0000882-1` (score=0.2402) - Robinow syndrome is a rare disorder that affects the development of many parts of the body, particularly the bones. Researchers have identified two major types of Robinow syndrome. The types are distinguished by the s...
3. `medquad_ans_0000420-1__dup1` (score=0.0009) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...
4. `medquad_ans_0000387-2` (score=0.0005) - Freeman-Sheldon syndrome is a rare disorder; its exact prevalence is unknown.
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
3. `medquad_ans_0005763-1` (score=0.2458) - Spinocerebellar ataxia 13 (SCA13) is a rare sub-type of spinocerebellar ataxias, a group of neurological conditions characterized by degeneration of the brain and spinal cord. Signs and symptoms of SCA13 appear to var...
4. `medquad_ans_0000238-1__dup1` (score=0.1857) - Primary lateral sclerosis (PLS) is a rare neuromuscular disease with slowly progressive weakness in voluntary muscle movement. PLS belongs to a group of disorders known as motor neuron diseases. PLS affects the upper ...
5. `medquad_ans_0005765-1` (score=0.0288) - Spinocerebellar ataxia 15 (SCA15) is a neurological condition characterized by slowly progressive gait and limb ataxia, often in combination with eye movement abnormalities and balance, speech and swallowing difficult...

**Answer**:

Juvenile primary lateral sclerosis is a rare disorder characterized by progressive weakness and tightness (spasticity) of muscles in the arms, legs, and face. It is caused by damage to motor neurons, which are specialized nerve cells in the brain and spinal cord that control muscle movement. Symptoms begin in early childhood and progress slowly over many years. Early symptoms include clumsiness, muscle weakness, and spasticity in the legs, and difficulty with balance. As the condition progresses, spasticity spreads to the arms and hands, and individuals may develop slurred speech, drooling, difficulty swallowing, and an inability to walk. Juvenile primary lateral sclerosis is caused by mutations in the ALS2 gene and is inherited in an autosomal recessive pattern [medquad_ans_0003331-1].

**Supported citations**: `medquad_ans_0003331-1`
**Unsupported citations**: _none_

### Example 75 (`0000554-1__dup2`)

**Question**: What is (are) Leukemia ?

**Scores**: context_relevance=0.750, groundedness=0.986, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000040-11` (score=0.9829) - There are four common types of leukemia. They are chronic lymphocytic leukemia, chronic myeloid leukemia, acute myeloid leukemia, and acute lymphocytic leukemia. Chronic lymphocytic leukemia, chronic myeloid leukemia,...
2. `medquad_ans_0000040-9` (score=0.9663) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
3. `medquad_ans_0000201-1__dup2` (score=0.9485) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
4. `medquad_ans_0000200-1__dup3` (score=0.8027) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...
5. `medquad_ans_0000554-1__dup2` (score=0.7928) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...

**Answer**:

Leukemia is a cancer of the white blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white blood cells builds up in the blood, they crowd out healthy blood cells, making it difficult for the blood to carry out its normal functions [medquad_ans_0000040-9].

**Supported citations**: `medquad_ans_0000040-9`
**Unsupported citations**: _none_

### Example 76 (`0002485-1`)

**Question**: What is (are) Gangliocytoma ?

**Scores**: context_relevance=0.750, groundedness=0.979, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0002485-1` (score=0.7954) - Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. Gangliocytomas may occur in all age groups but most often occur in people between the ages of 10 and 30. The most common si...
2. `medquad_ans_0000087-1__dup2` (score=0.6743) - Tumors are abnormal growths in your body. They are made up of extra cells. Normally, cells grow and divide to form new cells as your body needs them. When cells grow old, they die, and new cells take their place. Some...
3. `medquad_ans_0000063-16` (score=0.0618) - When the cancer spreads from its original tumor location in the skin to another part of the body such as the brain, it is called metastatic skin cancer. It is not the same as a cancer that started in the brain (brain ...
4. `medquad_ans_0000127-1__dup3` (score=0.0233) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
5. `medquad_ans_0000076-3__dup1` (score=0.0078) - Neuroacanthocytosis is a progressive disease, and in some cases may be complicated by poor nutritional status, cardiac abnormalities, and pneumonia.

**Answer**:

Gangliocytoma is a rare type of central nervous system (CNS) tumor made up of mature neurons. It may occur in all age groups but most commonly occurs in people between the ages of 10 and 30. The most common site is the temporal lobe of the brain, although it can arise anywhere in the CNS, including the cerebellum, brainstem, floor of the third ventricle, and spinal cord. Gangliocytomas are among the most frequent tumors associated with epilepsy. Signs and symptoms depend on the tumor's location and may include seizures (most commonly), increased brain pressure, endocrine disorders, and focal symptoms. These tumors are generally slow-growing and usually do not become malignant. Treatment involves surgical removal of the tumor [medquad_ans_0002485-1].

**Supported citations**: `medquad_ans_0002485-1`
**Unsupported citations**: _none_

### Example 77 (`0000130-1__dup1`)

**Question**: What is (are) Miller Fisher Syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.327, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000130-1__dup1` (score=0.7352) - Miller Fisher syndrome is a rare, acquired nerve disease that is considered to be a variant of Guillain-Barr syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of ...
2. `medquad_ans_0004068-1` (score=0.0810) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. Characteristic features include underdeveloped cheek bones, a very small lower jaw, cleft lip and/or palate, abnormalities...
3. `medquad_ans_0000658-1` (score=0.0291) - Miller syndrome is a rare condition that mainly affects the development of the face and limbs. The severity of this disorder varies among affected individuals. Children with Miller syndrome are born with underdevelope...
4. `medquad_ans_0000658-2` (score=0.0004) - Miller syndrome is a rare disorder; it is estimated to affect fewer than 1 in 1 million newborns. At least 30 cases have been reported in the medical literature.
5. `medquad_ans_0000659-1` (score=0.0003) - Miller-Dieker syndrome is a condition characterized by a pattern of abnormal brain development known as lissencephaly. Normally the exterior of the brain (cerebral cortex) is multi-layered with folds and grooves. Peop...

**Answer**:

Miller Fisher syndrome is a rare, acquired nerve disease considered to be a variant of Guillain-Barré syndrome. It is characterized by abnormal muscle coordination, paralysis of the eye muscles, and absence of tendon reflexes. Symptoms may be preceded by a viral illness and include generalized muscle weakness and respiratory failure. The majority of individuals with Miller Fisher syndrome have a unique antibody that characterizes the disorder [medquad_ans_0000130-1].

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
5. `medquad_ans_0001086-2` (score=0.0002) - At least 500 cases of 22q13.3 deletion syndrome are known.

**Answer**:

22q11.2 deletion syndrome is caused by a missing piece (deletion) of part of chromosome 22 in each cell. The deletion occurs near the middle of the chromosome at a location designated q11.2. Most people with 22q11.2 deletion syndrome are missing a piece of the chromosome that contains about 30 to 40 genes, many of which have not been well characterized. Some affected people have smaller deletions. Researchers are working to learn more about all of the genes that contribute to the features of 22q11.2 deletion syndrome. The deletion of a particular gene, TBX1, is probably responsible for many of the syndrome's characteristic signs (such as heart defects, a cleft palate, distinctive facial features, hearing loss, and low calcium levels). Loss of this gene may also contribute to behavioral problems. The loss of another gene, COMT, may

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

**Scores**: context_relevance=1.000, groundedness=0.954, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002456-1` (score=0.9994) - What are the signs and symptoms of Fryns Hofkens Fabry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns Hofkens Fabry syndrome. If the information is available, the ta...
2. `medquad_ans_0002458-2` (score=0.9970) - What are the signs and symptoms of Fryns syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fryns syndrome. If the information is available, the table below includes how often...
3. `medquad_ans_0000123-3__dup1` (score=0.0043) - Individuals with Fabry disease often die prematurely of complications from strokes, heart disease, or kidney failure.
4. `medquad_ans_0003512-1` (score=0.0001) - What are the signs and symptoms of Leber congenital amaurosis 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Leber congenital amaurosis 2. If the information is available, the ta...
5. `medquad_ans_0000619-1__dup1` (score=0.0001) - What are the signs and symptoms of Baraitser Brett Piesowicz syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Baraitser Brett Piesowicz syndrome. If the information is avail...

**Answer**:

The symptoms of Fryns Hofkens Fabry syndrome include ulnar deviation of the finger (90%), distal ulnar hypoplasia, dysplastic radii, hypoplasia of the radius, mesomelic arm shortening, radial bowing, and ulnar deviation of the hand [medquad_ans_0002456-1].

**Supported citations**: `medquad_ans_0002456-1`
**Unsupported citations**: _none_

### Example 81 (`0000050-8`)

**Question**: what research (or clinical trials) is being done for Osteoporosis ?

**Scores**: context_relevance=0.800, groundedness=0.886, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000018-14` (score=0.9978) - For information about the clinical trial on this topic, visit: http://www.clinicaltrials.gov/ct/show/NCT00372320?order=1. If you would like to read an interview with Dr. Bruce Baum, the study's principal investigator,...
2. `medquad_ans_0000008-30` (score=0.9929) - Clinical trials are research studies on people to find out whether a new drug or treatment is both safe and effective. New therapies are tested on people only after laboratory and animal studies show promising results...
3. `medquad_ans_0000050-8` (score=0.9897) - Scientists are pursuing a wide range of basic and clinical studies on osteoporosis. Significant advances in preventing and treating osteoporosis continue to be made. Such advances are the direct result of research foc...
4. `medquad_ans_0000050-28` (score=0.9646) - Consider talking to your doctor about being evaluated for osteoporosis if - you are a man or woman over age 50 or a postmenopausal woman and you break a bone - you are a woman age 65 or older - you are a woman younger...
5. `medquad_ans_0000113-1__dup3` (score=0.6649) - Summary : Strong bones are important for your health. A bone mineral density (BMD) test is the best way to measure your bone health. It compares your bone density, or mass, to that of a healthy person who is the same ...

**Answer**:

Research on osteoporosis is focused on determining the causes and consequences of bone loss at the cellular and tissue levels, assessing risk factors, developing new strategies to maintain and enhance bone density, and reducing fracture risk. This research also explores the roles of genetics, hormones, calcium, vitamin D, drugs, and exercise on bone mass [medquad_ans_0000050-8].

For more information about ongoing research, individuals can visit the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) at NIH [medquad_ans_0000050-8]. Additionally, clinical trials related to osteoporosis are being conducted, though specific trial details are not provided in the context. Information about clinical trials can be found at http://www.cancer.gov/clinicaltrials, which includes trials for breast cancer prevention, though this may not directly apply to osteoporosis [medquad_ans_0

**Supported citations**: `medquad_ans_0000050-8`
**Unsupported citations**: _none_

### Example 82 (`0000430-1__dup1`)

**Question**: Do you have information about Health Fraud

**Scores**: context_relevance=0.857, groundedness=0.923, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000338-1__dup1` (score=0.9973) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...
2. `medquad_ans_0000036-6` (score=0.9972) - People who enroll in Medicare can register with www.MyMedicare.gov, a secure online service, and use the site to access their personal Medicare information at any time. People can view their claims and order history, ...
3. `medquad_ans_0000430-1__dup1` (score=0.9966) - Summary : Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. Keep in mind - if it sounds too good to be true, it's probably a scam. At best, these scams don't work. ...
4. `medquad_ans_0000693-1__dup2` (score=0.9966) - Summary : As a patient, you have certain rights. Some are guaranteed by federal law, such as the right to get a copy of your medical records, and the right to keep them private. Many states have additional laws protec...
5. `medquad_ans_0000694-1__dup1` (score=0.9945) - Summary : You can help prevent medical errors by being an active member of your health care team. Research shows that patients who are more involved with their care tend to get better results. To reduce the risk of me...

**Answer**:

Yes, there is information about health fraud. Health fraud involves selling drugs, devices, foods, or cosmetics that have not been proven effective. It often includes red flags such as "miracle cure," "quick fix," "ancient remedy," "secret ingredient," or "scientific breakthrough." These scams can be dangerous, waste money, and may prevent individuals from receiving necessary treatment. The Food and Drug Administration warns that such scams target people with serious conditions like cancer, diabetes, heart disease, HIV/AIDS, arthritis, and Alzheimer's. Before taking unproven treatments, it is advised to consult a doctor or healthcare professional, especially when considering prescription drugs [medquad_ans_0000430-1__dup1].

**Supported citations**: `medquad_ans_0000430-1__dup1`
**Unsupported citations**: _none_

### Example 83 (`0000055-3__dup2`)

**Question**: What are the symptoms of Prevent diabetes problems: Keep your eyes healthy ?

**Scores**: context_relevance=0.750, groundedness=0.051, answer_relevance=0.483

**Top docs**:

1. `medquad_ans_0000016-10` (score=0.9985) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year. Learn more about a comprehensive dilated eye exam.
2. `medquad_ans_0000055-3__dup2` (score=0.9946) - Often, no symptoms appear during the early stages of diabetes retina problems. As retina problems worsen, your symptoms might include - blurry or double vision - rings, flashing lights, or blank spots in your vision -...
3. `medquad_ans_0000016-24` (score=0.9885) - If you have diabetes, get a comprehensive dilated eye exam at least once a year. Proliferative retinopathy can develop without symptoms. If it gets to this advanced stage, you are at high risk for vision loss or even ...
4. `medquad_ans_0000015-15` (score=0.9867) - Many people with diabetes experience one or more symptoms, including extreme thirst or hunger, a frequent need to urinate and/or fatigue. Some lose weight without trying. Additional signs include sores that heal slowl...
5. `medquad_ans_0000043-2__dup4` (score=0.9731) - Many people have no visible signs or symptoms of diabetes. Symptoms can also be so mild that you might not notice them. More than 5 million people in the United States have type 2 diabetes and do not know it. - increa...

**Answer**:

The provided context does not contain information about symptoms of preventing diabetes problems or specifically about keeping eyes healthy in the context of prevention. While the context discusses symptoms of diabetic retinopathy and diabetes-related eye issues, such as blurry vision, double vision, flashing lights, or dark spots, these are symptoms of existing problems rather than preventive measures.

Therefore, based on the given context, there is no information to answer the question about symptoms of preventing diabetes problems or keeping eyes healthy.

Answer: Insufficient context. [doc_id] not available.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 84 (`0000001-4`)

**Question**: Is Aarskog-Scott syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.742, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000362-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
2. `medquad_ans_0000585-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000095-4` (score=0.9990) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000673-4` (score=0.9988) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
5. `medquad_ans_0000786-4` (score=0.9987) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

Aarskog-Scott syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condition. However, some cases result from new mutations and occur in people with no family history of the disorder [medquad_ans_0000673-4].

Note: While the provided context references autosomal dominant inheritance for several conditions, it does not explicitly state that Aarskog-Scott syndrome is inherited in this manner. Therefore, the answer is based on the pattern of inheritance described in the context for similar conditions, which may not directly confirm the inheritance pattern for Aarskog-Scott syndrome.

Due to the lack of explicit confirmation in the provided context that Aarskog-Scott syndrome specifically follows autosomal dominant inheritance, the answer cannot be definitively established.

Answer: Ins

**Supported citations**: `medquad_ans_0000673-4`
**Unsupported citations**: _none_

### Example 85 (`0001062-3`)

**Question**: What are the genetic changes related to X-linked sideroblastic anemia ?

**Scores**: context_relevance=0.800, groundedness=0.930, answer_relevance=0.560

**Top docs**:

1. `medquad_ans_0001062-3` (score=0.9992) - Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. The ALAS2 gene provides instructions for making an enzyme called erythroid ALA-synthase, which plays a critical role in the production of heme (a compon...
2. `medquad_ans_0000337-3` (score=0.0011) - The inherited form of factor X deficiency, known as congenital factor X deficiency, is caused by mutations in the F10 gene, which provides instructions for making a protein called coagulation factor X. This protein pl...
3. `medquad_ans_0000040-3` (score=0.0004) - Alpha thalassemia X-linked intellectual disability syndrome results from mutations in the ATRX gene. This gene provides instructions for making a protein that plays an essential role in normal development. Although th...
4. `medquad_ans_0000068-4__dup2` (score=0.0001) - The immediate cause of hemolytic anemia is the early destruction of red blood cells. This means that red blood cells are destroyed and removed from the bloodstream before their normal lifespan is over. Many diseases, ...
5. `medquad_ans_0001062-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...

**Answer**:

Mutations in the ALAS2 gene cause X-linked sideroblastic anemia. These mutations impair the activity of erythroid ALA-synthase, disrupting normal heme production and leading to reduced hemoglobin synthesis in erythroblasts. This results in excess iron accumulation in erythroblasts and organ damage due to iron overload. In some cases, mutations in both the ALAS2 and HFE genes can lead to a more severe form of the disorder [medquad_ans_0001062-3].

**Supported citations**: `medquad_ans_0001062-3`
**Unsupported citations**: _none_

### Example 86 (`0000602-1`)

**Question**: What is (are) limb-girdle muscular dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.951, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0000602-1` (score=0.3290) - Limb-girdle muscular dystrophy is a term for a group of diseases that cause weakness and wasting of the muscles in the arms and legs. The muscles most affected are those closest to the body (proximal muscles), specifi...
2. `medquad_ans_0003601-1` (score=0.1475) - Limb-girdle muscular dystrophy is a group of disorders which affect the voluntary muscles around the hips and shoulders. The conditions are progressive, leading to a loss of muscle strength and bulk over a number of y...
3. `medquad_ans_0000631-1__dup1` (score=0.0793) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
4. `medquad_ans_0003610-1` (score=0.0108) - Limb-girdle muscular dystrophy type 2C (LGMD2C) is a condition that affects the muscles and is caused by mutations in the gamma-sarcoglycan gene. This condition belongs to a group of muscle disorders called limb-girdl...
5. `medquad_ans_0001537-1` (score=0.0101) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...

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

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001503-2` (score=0.9990) - What are the signs and symptoms of Congenital dyserythropoietic anemia type 2? The signs and symptoms of CDA II include jaundice, gallstones and an enlarged liver and spleen. This condition also causes the body to abs...
2. `medquad_ans_0000057-4__dup2` (score=0.9948) - Major Signs and Symptoms Your doctor may suspect you or your child has Fanconi anemia (FA) if you have signs and symptoms of: Anemia Bone marrow failure Birth defects Developmental or eating problems FA is an inherite...
3. `medquad_ans_0000148-5__dup1` (score=0.0185) - A child with hemolytic uremic syndrome may develop signs and symptoms similar to those seen with gastroenteritisan inflammation of the lining of the stomach, small intestine, and large intestine such as - vomiting - b...
4. `medquad_ans_0002772-2` (score=0.0004) - What are the signs and symptoms of hemoglobin E disease? Affected individuals can develop mild thalassemia in the first few months of life. While mild splenomegaly and/or anemia can occur, it is generally considered a...
5. `medquad_ans_0003748-2` (score=0.0001) - What are the signs and symptoms of Majeed syndrome? Majeed syndrome is characterized by recurrent episodes of fever and inflammation in the bones and skin. There are two main features of Majeed syndrome: Chronic recur...

**Answer**:

The symptoms of Congenital dyserythropoietic anemia type 2 include jaundice, gallstones, an enlarged liver and spleen (splenomegaly), anemia of inadequate production, endopolyploidy on chromosome studies of bone marrow, reduced activity of N-acetylglucosaminyltransferase II, reticulocytosis, and iron overload leading to complications such as arrhythmia, congestive heart failure, diabetes, and chronic liver disease (cirrhosis). Rarely, mediastinal tumors may occur [medquad_ans_0001503-2].

**Supported citations**: `medquad_ans_0001503-2`
**Unsupported citations**: _none_

### Example 90 (`0000038-16`)

**Question**: What is (are) Kidney Disease ?

**Scores**: context_relevance=0.800, groundedness=0.895, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000538-1__dup1` (score=0.9443) - Your kidneys are two bean-shaped organs, each about the size of your fists. They are located near the middle of your back, just below the rib cage. Inside each kidney about a million tiny structures called nephrons fi...
2. `medquad_ans_0000140-2__dup2` (score=0.9286) - Chronic kidney disease is kidney damage that occurs slowly over many years, often due to diabetes or high blood pressure. Once damaged, the kidneys cant filter blood as they should. This damage can cause wastes to bui...
3. `medquad_ans_0000038-11` (score=0.9072) - Kidney disease -- also known as chronic kidney disease (CKD) -- occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should. According to the Centers for Di...
4. `medquad_ans_0000144-10` (score=0.1711) - Renal failure is any acute or chronic loss of kidney function and is the term used when some kidney function remains. Total kidney failure, sometimes called end-stage renal disease (ESRD), indicates permanent loss of ...
5. `medquad_ans_0000038-14` (score=0.1635) - Kidney disease is often called a "silent" disease, because most people have no symptoms in early kidney disease. In fact, you might feel just fine until your kidneys have almost stopped working. Do NOT wait for sympto...

**Answer**:

Kidney disease, also known as chronic kidney disease (CKD), occurs when kidneys can no longer remove wastes and extra water from the blood or perform other functions as they should [medquad_ans_0000038-11]. It is characterized by damage to the nephrons, which are the tiny structures within the kidneys responsible for filtering blood [medquad_ans_0000538-1__dup1]. This damage may result from genetic problems, injuries, medicines, or conditions such as diabetes and high blood pressure [medquad_ans_0000538-1\_\_dup1][medquad_ans_0000140-2__dup2]. Chronic kidney disease develops slowly over several years, leading to a buildup of wastes in the body and potential complications like mineral and bone disorder [medquad_ans_0000140-2__dup2]. Other

**Supported citations**: `medquad_ans_0000038-11`, `medquad_ans_0000538-1__dup1`, `medquad_ans_0000140-2__dup2`
**Unsupported citations**: _none_

### Example 91 (`0000058-24`)

**Question**: What are the treatments for Prostate Cancer ?

**Scores**: context_relevance=0.857, groundedness=0.232, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000058-24` (score=0.9795) - Surgery, radiation therapy, and hormonal therapy all have the potential to disrupt sexual desire or performance for a short while or permanently. Discuss your concerns with your health care provider. Several options a...
2. `medquad_ans_0000058-28` (score=0.9740) - Hormonal therapy deprives cancer cells of the male hormones they need to grow and survive. This treatment is often used for prostate cancer that has spread to other parts of the body. Sometimes doctors use hormonal th...
3. `medquad_ans_0000058-4__dup3` (score=0.9707) - If tests show that you have cancer, you should talk with your doctor in order to make treatment decisions. Working With a Team of Specialists A team of specialists often treats people with cancer. The team will keep t...
4. `medquad_ans_0000058-26` (score=0.9697) - Surgery is a common treatment for early stage prostate cancer. It is used to remove the cancer. The surgeon may remove the entire prostate -- a type of surgery called radical prostatectomy -- or, in some cases, remove...
5. `medquad_ans_0000058-30` (score=0.8510) - Through research, doctors are trying to find new, more effective ways to treat prostate cancer. Cryosurgery -- destroying cancer by freezing it -- is under study as an alternative to surgery and radiation therapy. To ...

**Answer**:

The treatments for prostate cancer include surgery, radiation therapy, hormonal therapy, and participation in clinical trials. Surgery, such as radical prostatectomy, is commonly used for early-stage prostate cancer to remove the cancer. Radiation therapy is another treatment option. Hormonal therapy deprives cancer cells of male hormones they need to grow and survive and is often used for prostate cancer that has spread or to prevent recurrence after surgery or radiation. Additionally, some men may participate in clinical trials to access new treatments [medquad_ans_0000058-24, medquad_ans_0000058-26, medquad_ans_0000058-28, medquad_ans_0000058-4__dup3].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000058-24, medquad_ans_0000058-26, medquad_ans_0000058-28, medquad_ans_0000058-4__dup3`

### Example 92 (`0000431-1__dup1`)

**Question**: Do you have information about Health Insurance

**Scores**: context_relevance=0.857, groundedness=0.838, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000431-1__dup1` (score=0.9972) - Summary : Health insurance helps protect you from high medical care costs. It is a contract between you and your insurance company. You buy a plan or policy, and the company agrees to pay part of your expenses when yo...
2. `medquad_ans_0000362-1__dup2` (score=0.9971) - Summary : Health care can be costly. If you have health insurance, it usually pays at least part of your medical costs. If you don't have insurance or need help with costs that aren't covered, financial assistance mig...
3. `medquad_ans_0000589-1__dup2` (score=0.9958) - Summary : Medicare is the U.S. government's health insurance program for people age 65 or older. Some people under age 65 can qualify for Medicare, too. They include those with disabilities, permanent kidney failure, ...
4. `medquad_ans_0000036-6` (score=0.9954) - People who enroll in Medicare can register with www.MyMedicare.gov, a secure online service, and use the site to access their personal Medicare information at any time. People can view their claims and order history, ...
5. `medquad_ans_0000338-1__dup1` (score=0.9707) - Summary : Millions of consumers get health information from magazines, TV or the Internet. Some of the information is reliable and up to date; some is not. How can you tell the good from the bad? First, consider the s...

**Answer**:

Yes, there is information about health insurance in the provided context. Health insurance is described as a contract between an individual and an insurance company that helps protect against high medical care costs by covering part of the expenses when medical care is needed [medquad_ans_0000431-1__dup1]. It can be obtained through employers, where the employer often helps pay for the insurance, or purchased independently [medquad_ans_0000431-1__dup1]. Government health insurance programs such as Medicare and Medicaid are also available to eligible individuals [medquad_ans_0000431-1__dup1]. Medicare specifically serves people age 65 or older and certain individuals under 65 with disabilities or chronic conditions [medquad_ans_0000589-1__dup2]. The program includes four parts: Part A (hospital insurance), Part B (medical services

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
5. `medquad_ans_0001085-1` (score=0.0001) - 22q11.2 duplication is a condition caused by an extra copy of a small piece of chromosome 22. The duplication occurs near the middle of the chromosome at a location designated q11.2. The features of this condition var...

**Answer**:

MECP2 duplication syndrome is a genetic condition that occurs almost exclusively in males and is characterized by moderate to severe intellectual disability. It is caused by an extra copy (duplication) of the MECP2 gene in each cell, which results from a duplication of genetic material on the long (q) arm of the X chromosome. The size of the duplication varies from 100,000 to 900,000 base pairs (100 to 900 kilobases), and the MECP2 gene is always included in this duplication, with other genes possibly involved. Other signs and symptoms include infantile hypotonia, delayed motor milestones (such as sitting up or crawling), recurrent infections, poor or absent speech, seizures, and/or spasticity. Some individuals experience developmental regression, and approximately one third cannot walk without assistance. Recurrent respiratory tract infections are a major cause of death, with almost half

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 94 (`0000028_2-5`)

**Question**: What is the outlook for Melanoma ?

**Scores**: context_relevance=0.667, groundedness=0.923, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000017_1-5` (score=0.9940) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - How the melanoma cells look under a microscope. - The si...
2. `medquad_ans_0000063-30` (score=0.0762) - The National Cancer Institute has developed a comprehensive online cancer database called the Physician Data Query (PDQ) to present evidence from the most recent research on melanoma and other skin cancers. Click here...
3. `medquad_ans_0000063-15` (score=0.0225) - Melanoma is the fastest growing and most invasive type of skin cancer. This cancer arises from overgrowth of melanocytes. Melanocytes are specialized skin cells that produce a pigment called melanin.
4. `medquad_ans_0000063-6__dup1` (score=0.0050) - Many Treatment Options There are many treatment options for people with melanoma and non-melanoma skin cancer. The choice of treatment depends on your age and general health, the site of the cancer, the stage of the c...
5. `medquad_ans_0000063-22` (score=0.0031) - Skin cancer is usually visible. Changes in your skin, such as lumps, scabs, red spots, rough patches, or new or irregular moles should be reported to your doctor. You should also see a doctor if you have a mole, birth...

**Answer**:

The outlook for melanoma depends on several factors, including how the melanoma cells look under a microscope, the size and thickness of the tumor, the part of the eye the tumor is in (such as the iris, ciliary body, or choroid), whether the tumor has spread within the eye or to other parts of the body, genetic changes linked to intraocular melanoma, the patient's age and general health, and whether the tumor has recurred after treatment [medquad_ans_0000017_1-5]. Early treatment is associated with better chances for successful outcomes [medquad_ans_0000063-15].

**Supported citations**: `medquad_ans_0000017_1-5`, `medquad_ans_0000063-15`
**Unsupported citations**: _none_

### Example 95 (`0000132-4`)

**Question**: Is boomerang dysplasia inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000283-4` (score=0.9988) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...
2. `medquad_ans_0006597-4` (score=0.9839) - Can 48,XXYY syndrome be inherited?
3. `medquad_ans_0004514-4` (score=0.0315) - How is oculopharyngeal muscular dystrophy inherited?
4. `medquad_ans_0001784-4` (score=0.0205) - How is dentatorubral-pallidoluysian atrophy (DRPLA) inherited?
5. `medquad_ans_0000139-2__dup2` (score=0.0053) - Von Willebrand disease (VWD) is almost always inherited. "Inherited" means that the disorder is passed from parents to children though genes. You can inherit type 1 or type 2 VWD if only one of your parents passes the...

**Answer**:

The provided context does not contain information about the inheritance of boomerang dysplasia. Therefore, it is not possible to determine based on the given text whether boomerang dysplasia is inherited. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 96 (`0000466-3`)

**Question**: What are the genetic changes related to hereditary multiple osteochondromas ?

**Scores**: context_relevance=0.800, groundedness=0.479, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000466-3` (score=0.9991) - Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene and the EXT2 gene provide instructions for producing the proteins exostosin-1 and exostosin-2, respectively. The two exosto...
2. `medquad_ans_0000763-3` (score=0.9988) - Mutations in the COL11A2 gene cause OSMED. The COL11A2 gene is one of several genes that provide instructions for the production of type XI collagen. This type of collagen is important for the normal development of bo...
3. `medquad_ans_0000362-3` (score=0.0007) - Mutation of the ACAN gene can cause familial osteochondritis dissecans. The ACAN gene provides instructions for making the aggrecan protein, which is a component of cartilage. Aggrecan attaches to the other components...
4. `medquad_ans_0000502-3` (score=0.0002) - About 70 percent of all cases of hypochondroplasia are caused by mutations in the FGFR3 gene. This gene provides instructions for making a protein that is involved in the development and maintenance of bone and brain ...
5. `medquad_ans_0000579-3` (score=0.0002) - Langer-Giedion syndrome is caused by the deletion or mutation of at least two genes on chromosome 8. Researchers have determined that the loss of a functional EXT1 gene is responsible for the multiple osteochondromas ...

**Answer**:

Mutations in the EXT1 and EXT2 genes cause hereditary multiple osteochondromas. The EXT1 gene provides instructions for producing the exostosin-1 protein, and the EXT2 gene provides instructions for producing the exostosin-2 protein. These two proteins bind together to form a complex in the Golgi apparatus, where they modify heparan sulfate. When there is a mutation in exostosin-1 or exostosin-2, heparan sulfate cannot be processed correctly and becomes nonfunctional. A mutation in the EXT1 gene results in hereditary multiple osteochondromas type 1, while a mutation in the EXT2 gene causes hereditary multiple osteochondromas type 2. Mutations in the EXT1 gene are likely responsible for 55 to 75 percent of all cases, and symptoms tend to be more severe in type 1. Approximately 15 percent of individuals

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 97 (`0000746-2`)

**Question**: How many people are affected by oculofaciocardiodental syndrome ?

**Scores**: context_relevance=0.375, groundedness=0.000, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000624-2` (score=0.9992) - Manitoba oculotrichoanal syndrome is estimated to occur in 2 to 6 in 1,000 people in a small isolated Ojibway-Cree community in northern Manitoba, Canada. Although this region has the highest incidence of the conditio...
2. `medquad_ans_0000135-2` (score=0.9991) - Branchio-oculo-facial syndrome is a rare condition, although the prevalence is unknown.
3. `medquad_ans_0000151-2` (score=0.9990) - Cardiofaciocutaneous syndrome is a very rare condition whose incidence is unknown. Researchers estimate that 200 to 300 people worldwide have this condition.
4. `medquad_ans_0000745-2` (score=0.6417) - The exact incidence of oculodentodigital dysplasia is unknown. It has been diagnosed in fewer than 1,000 people worldwide. More cases are likely undiagnosed.
5. `medquad_ans_0000344-2` (score=0.3264) - It is estimated that 750,000 people in the United States have dilated cardiomyopathy; roughly half of these cases are familial.

**Answer**:

The provided context does not contain information about oculofaciocardiodental syndrome. Therefore, it is not possible to determine how many people are affected by this condition based on the given text. [No relevant context]

**Supported citations**: _none_
**Unsupported citations**: `No relevant context`

### Example 98 (`0000189-1`)

**Question**: What is (are) citrullinemia ?

**Scores**: context_relevance=0.750, groundedness=0.538, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000189-1` (score=0.0631) - Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described; they have different signs and symptoms and are caused b...
2. `medquad_ans_0000107-1__dup3` (score=0.0056) - Primary ciliary (SIL-e-ar-e) dyskinesia (dis-kih-NE-ze-ah), or PCD, is a rare disease that affects tiny, hair-like structures that line the airways. These structures are called cilia (SIL-e-ah). Cilia move together in...
3. `medquad_ans_0001350-1` (score=0.0046) - Citrullinemia type I is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. This condition, also known as classic citrullinemia, belongs to a class of genetic diseases call...
4. `medquad_ans_0000966-1__dup1` (score=0.0027) - Carnitine palmitoyltransferase 2 (CPT2) deficiency is a condition that prevents the body from using certain fats for energy, particularly during periods without food (fasting). There are three main types of CPT2 defic...
5. `medquad_ans_0004334-1` (score=0.0025) - Neonatal intrahepatic cholestasis caused by citrin deficiency (NICCD) is a liver condition is also known as neonatal-onset type II citrullinemia. NICCD blocks the flow of bile (a digestive fluid produced by the liver)...

**Answer**:

Citrullinemia is an inherited disorder that causes ammonia and other toxic substances to accumulate in the blood. Two forms of citrullinemia have been described: Type I citrullinemia (also known as classic citrullinemia) usually becomes evident in the first few days of life, with symptoms including lethargy, poor feeding, vomiting, seizures, and loss of consciousness. A milder, later-onset form of Type I citrullinemia can occur in childhood or adulthood, associated with intense headaches, partial loss of vision, ataxia, and lethargy. Type II citrullinemia chiefly affects the nervous system, causing confusion, restlessness, memory loss, abnormal behaviors, seizures, and coma; in adult-onset cases, these symptoms can be triggered by medications, infections, surgery, or alcohol intake. Additionally, some individuals with neonatal intrahepatic cholestasis caused by

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

**Scores**: context_relevance=0.857, groundedness=0.738, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000943-5` (score=0.9942) - These resources address the diagnosis or management of spondylocarpotarsal synostosis syndrome: - Gene Review: Gene Review: FLNB-Related Disorders - Genetic Testing Registry: Spondylocarpotarsal synostosis syndrome Th...
2. `medquad_ans_0000949-5` (score=0.9918) - These resources address the diagnosis or management of spondylothoracic dysostosis: - Cleveland Clinic: Spine X-ray - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive These resources from Medli...
3. `medquad_ans_0000944-5` (score=0.9918) - These resources address the diagnosis or management of spondylocostal dysostosis: - Gene Review: Gene Review: Spondylocostal Dysostosis, Autosomal Recessive - Genetic Testing Registry: Jarcho-Levin syndrome - Genetic ...
4. `medquad_ans_0000948-5` (score=0.3362) - These resources address the diagnosis or management of spondyloperipheral dysplasia: - Genetic Testing Registry: Spondyloperipheral dysplasia - MedlinePlus Encyclopedia: Nearsightedness These resources from MedlinePlu...
5. `medquad_ans_0005860-4` (score=0.0968) - What treatment is available for spondylothoracic dysostosis? Many infants born with spondylothoracic dysostosis have difficulty breathing due to their small, malformed chests, and therefore are prone to repeated respi...

**Answer**:

The provided context does not contain specific information about the treatments for spondylocostal dysostosis. While it mentions resources related to diagnosis and management, such as genetic counseling, diagnostic tests, surgery and rehabilitation, and palliative care, it does not explicitly detail treatments for spondylocostal dysostosis [medquad_ans_0000944-5]. Therefore, based on the available context, the specific treatments for spondylocostal dysostosis cannot be determined.

**Supported citations**: `medquad_ans_0000944-5`
**Unsupported citations**: _none_

### Example 101 (`0004825-1`)

**Question**: What are the symptoms of Perry syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.946, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004825-1` (score=0.9994) - What are the signs and symptoms of Perry syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Perry syndrome. If the information is available, the table below includes how often...
2. `medquad_ans_0004102-2` (score=0.0001) - What are the signs and symptoms of Moebius syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Moebius syndrome. If the information is available, the table below includes how o...
3. `medquad_ans_0004143-2` (score=0.0001) - What are the signs and symptoms of Mowat-Wilson syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Mowat-Wilson syndrome. If the information is available, the table below incl...
4. `medquad_ans_0005008-2` (score=0.0001) - What are the signs and symptoms of Potocki-Shaffer syndrome? The signs and symptoms can vary depending on the area and amount deleted. Some individuals with the syndrome have few issues and lead a normal life while ot...
5. `medquad_ans_0002480-2` (score=0.0001) - What are the signs and symptoms of Galloway-Mowat syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Galloway-Mowat syndrome. If the information is available, the table below ...

**Answer**:

The symptoms of Perry syndrome include abnormality of extrapyramidal motor function, respiratory insufficiency, sleep disturbance, tremor, weight loss, developmental regression, hallucinations, hypotension, apathy, autosomal dominant inheritance, bradykinesia, central hypoventilation, dysarthria, hypoventilation, inappropriate behavior, insomnia, mask-like facies, parkinsonism, rapidly progressive course, rigidity, short stepped shuffling gait, vertical supranuclear gaze palsy, and weak voice. These symptoms occur in approximately 90% of patients, except for developmental regression, hallucinations, hypotension, apathy, and others, which occur in about 7.5% of patients [medquad_ans_0004825-1].

**Supported citations**: `medquad_ans_0004825-1`
**Unsupported citations**: _none_

### Example 102 (`0006398-2`)

**Question**: What are the symptoms of Warfarin sensitivity ?

**Scores**: context_relevance=1.000, groundedness=0.916, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0006398-2` (score=0.9992) - What are the signs and symptoms of Warfarin sensitivity? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin sensitivity. If the information is available, the table below includ...
2. `medquad_ans_0006399-1` (score=0.9876) - What are the signs and symptoms of Warfarin syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Warfarin syndrome. If the information is available, the table below includes how...
3. `medquad_ans_0000012-5__dup1` (score=0.0040) - Atherosclerosis usually doesn't cause signs and symptoms until it severely narrows or totally blocks an artery. Many people don't know they have the disease until they have a medical emergency, such as a heart attack ...
4. `medquad_ans_0000108-5__dup1` (score=0.0013) - Major Signs and Symptoms Signs and symptoms of pulmonary embolism (PE) include unexplained shortness of breath, problems breathing, chest pain, coughing, or coughing up blood. An arrhythmia (irregular heartbeat) also ...
5. `medquad_ans_0000031-4__dup3` (score=0.0007) - Carotid artery disease may not cause signs or symptoms until it severely narrows or blocks a carotid artery. Signs and symptoms may include a bruit, a transient ischemic attack(TIA), or a stroke. Bruit During a physic...

**Answer**:

The signs and symptoms of Warfarin sensitivity include abnormality of blood and blood-forming tissues, abnormality of metabolism/homeostasis, and autosomal dominant inheritance. The frequency of these symptoms is based on data from the Human Phenotype Ontology (HPO), which collects information from sources such as Orphanet. However, specific symptom frequencies are not detailed in the provided context [medquad_ans_0006398-2].

**Supported citations**: `medquad_ans_0006398-2`
**Unsupported citations**: _none_

### Example 103 (`0004262-1`)

**Question**: What is (are) Myhre syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.947, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0004262-1` (score=0.7196) - Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, a...
2. `medquad_ans_0000701-1` (score=0.6933) - Myhre syndrome is a condition with features affecting many systems and functions of the body. People with Myhre syndrome usually have delayed development of language and motor skills such as crawling and walking. Most...
3. `medquad_ans_0004272-1` (score=0.0197) - Myofibrillar myopathies (MFM) are a group of neuromuscular disorders characterized by slowly progressive weakness that can involve both proximal muscles (such as hips and shoulders) and distal muscles (those farther a...
4. `medquad_ans_0001030-1__dup1` (score=0.0020) - Central core disease (CCD) is an inherited condition that involves muscle weakness, skeletal abnormalities, and an increased chance of having a severe reaction to some anesthesia medications. Muscle weakness ranges fr...
5. `medquad_ans_0000208-1__dup2` (score=0.0012) - The myopathies are neuromuscular disorders in which the primary symptom is muscle weakness due to dysfunction of muscle fiber. Other symptoms of myopathy can include include muscle cramps, stiffness, and spasm. Myopat...

**Answer**:

Myhre syndrome is a rare inherited disorder characterized by intellectual disability, short stature, unusual facial features, and various bone (skeletal) abnormalities. Other findings may include hearing impairment, abnormal enlargement of the muscles (muscle hypertrophy), and/or joint stiffness. It is caused by mutations in the SMAD4 gene and inherited in an autosomal dominant pattern, with most cases resulting from a new mutation [medquad_ans_0004262-1]. The condition affects multiple systems and includes delayed development of language and motor skills, hearing loss (sensorineural, conductive, or mixed), growth restriction from before birth through adolescence, skin stiffness, and muscular appearance. Skeletal abnormalities include thickened skull bones, flattened spine bones (platyspondyly), broad ribs, underdeveloped pelvic wings (hypoplastic iliac wings), and short fingers and toes (brachydactyly). Facial features commonly include short

**Supported citations**: `medquad_ans_0004262-1`
**Unsupported citations**: _none_

### Example 104 (`0006057-2`)

**Question**: What are the symptoms of Thanatophoric dysplasia ?

**Scores**: context_relevance=0.571, groundedness=0.087, answer_relevance=0.571

**Top docs**:

1. `medquad_ans_0000088-4__dup2` (score=0.9989) - A person may experience digestive signs and symptoms, or symptoms in other parts of the body. Digestive signs and symptoms are more common in children and can include - abdominal bloating - chronic diarrhea - constipa...
2. `medquad_ans_0000149-3__dup2` (score=0.9987) - The symptoms of HSP include the following: - Rash. Leaking blood vessels in the skin cause a rash that looks like bruises or small red dots on the legs, arms, and buttocks. The rash may first look like hives and then ...
3. `medquad_ans_0000014-6` (score=0.0738) - Most people with primary hyperparathyroidism have no symptoms. When symptoms appear, they are often mild and nonspecific, such as - muscle weakness - fatigue and an increased need for sleep - feelings of depression - ...
4. `medquad_ans_0000185-4__dup2` (score=0.0254) - A person with hemochromatosis may notice one or more of the following symptoms: - joint pain - fatigue, or feeling tired - unexplained weight loss - abnormal bronze or gray skin color - abdominal pain - loss of sex dr...
5. `medquad_ans_0000008-4__dup5` (score=0.0043) - Hyperthyroidism has many symptoms that can vary from person to person. Some common symptoms of hyperthyroidism are - nervousness or irritability - fatigue or muscle weakness - heat intolerance - trouble sleeping - han...

**Answer**:

The provided context does not mention Thanatophoric dysplasia or its symptoms. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 105 (`0002417-2`)

**Question**: What are the symptoms of Fragile X syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.979, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0002417-2` (score=0.9990) - What are the signs and symptoms of Fragile X syndrome? Fragile X syndrome is characterized by developmental problems including intellectual disability and delayed speech and language development. Males are usually mor...
2. `medquad_ans_0000383-1` (score=0.0001) - Fragile X-associated tremor/ataxia syndrome (FXTAS) is characterized by problems with movement and thinking ability (cognition). FXTAS is a late-onset disorder, usually occurring after age 50, and its signs and sympto...
3. `medquad_ans_0000383-5` (score=0.0001) - These resources address the diagnosis or management of FXTAS: - Fragile X Research Foundation of Canada: FXTAS - Gene Review: Gene Review: FMR1-Related Disorders - Genetic Testing Registry: Fragile X tremor/ataxia syn...
4. `medquad_ans_0000384-1` (score=0.0001) - Fragile XE syndrome is a genetic disorder that impairs thinking ability and cognitive functioning. Most affected individuals have mild intellectual disability. In some people with this condition, cognitive function is...
5. `medquad_ans_0006593-2` (score=0.0001) - What are the signs and symptoms of 47 XXX syndrome? Many women with 47 XXX syndrome have no symptoms or only mild symptoms. In other cases, symptoms may be more pronounced. Females with 47 XXX syndrome may be taller t...

**Answer**:

The symptoms of Fragile X syndrome include cognitive impairment, joint hypermobility, macroorchidism, neurological speech impairment, otitis media, pes planus, abnormality of the pinna, attention deficit hyperactivity disorder, frontal bossing, intellectual disability, moderate, long face, macrocephaly, mandibular prognathia, muscular hypotonia, narrow face, sinusitis, abnormality of the mitral valve, autism, cerebral cortical atrophy, dilatation of the ascending aorta, seizures, self-injurious behavior, and strabismus [medquad_ans_0002417-2]. Additionally, males may have enlarged testicles (macroorchidism) after puberty, and physical features such as a long and narrow face, large ears, prominent jaw and forehead, unusually flexible fingers, and flat feet may become more apparent with age [medquad_ans_000241

**Supported citations**: `medquad_ans_0002417-2`
**Unsupported citations**: _none_

### Example 106 (`0000056-1__dup1`)

**Question**: What is (are) CADASIL ?

**Scores**: context_relevance=0.750, groundedness=0.512, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000166-1` (score=0.1349) - Cerebral autosomal dominant arteriopathy with subcortical infarcts and leukoencephalopathy, usually called CADASIL, is an inherited condition that causes stroke and other impairments. This condition affects blood flow...
2. `medquad_ans_0001051-1__dup1` (score=0.0563) - Cerebral autosomal recessive arteriopathy with subcortical infarcts and leukoencephalopathy, commonly known as CARASIL, is an inherited condition that causes stroke and other impairments. This progressive condition is...
3. `medquad_ans_0000220-2__dup1` (score=0.0034) - There is no specific treatmentfor OPCA. Physicians may try different medications to treat the ataxia, tremor, and rigidity that are associated with the disorder. Other treatments are directed at specific symptoms. Sti...
4. `medquad_ans_0000238-1__dup2` (score=0.0025) - Coronary artery disease (CAD) is the most common type of heart disease. It is the leading cause of death in the United States in both men and women. CAD happens when the arteries that supply blood to heart muscle beco...
5. `medquad_ans_0000237-1__dup2` (score=0.0017) - Summary : In coronary artery disease (CAD), the arteries that supply blood and oxygen to your heart muscle grow hardened and narrowed. You may try treatments such as lifestyle changes, medicines, and angioplasty, a pr...

**Answer**:

CADASIL is an inherited condition that causes stroke and other impairments by affecting blood flow in small blood vessels, particularly cerebral vessels within the brain. It is characterized by abnormal and gradually dying muscle cells surrounding these blood vessels (vascular smooth muscle cells), leading to arteriopathy. This can result in migraines, often with visual sensations or auras, or recurrent seizures (epilepsy). Damaged blood vessels reduce blood flow and can cause areas of tissue death (infarcts), which may lead to stroke. Strokes in the subcortical region of the brain can cause progressive loss of intellectual function (dementia) and changes in mood and personality. Many individuals with CADASIL also develop leukoencephalopathy, a change in brain white matter visible on MRI. The age of onset and severity of symptoms vary widely among individuals. CADASIL is not associated with common stroke and heart attack risk factors like high blood pressure and high

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 107 (`0000688-1`)

**Question**: What is (are) multiple endocrine neoplasia ?

**Scores**: context_relevance=0.833, groundedness=0.980, answer_relevance=0.833

**Top docs**:

1. `medquad_ans_0000014-1__dup3` (score=0.9826) - Your adrenal, or suprarenal, glands are located on the top of each kidney. These glands produce hormones that you can't live without, including sex hormones and cortisol, which helps you respond to stress and has many...
2. `medquad_ans_0004185-1` (score=0.9647) - Multiple endocrine neoplasia, type 1 (MEN1) is an inherited condition that causes tumors of the endocrine system (the body's network of hormone-producing glands). People affected by MEN1 typically develop tumors of th...
3. `medquad_ans_0000010-1__dup6` (score=0.9398) - MEN1 is an inherited disorder that causes tumors in the endocrine glands and the duodenum, the first part of the small intestine. MEN1 is sometimes called multiple endocrine adenomatosis or Wermer's syndrome, after on...
4. `medquad_ans_0000358-2` (score=0.7778) - Pituitary adenomas, including sporadic tumors, are relatively common; they are identified in an estimated 1 in 1,000 people. FIPA, though, is quite rare, accounting for approximately 2 percent of pituitary adenomas. M...
5. `medquad_ans_0000688-1` (score=0.6657) - Multiple endocrine neoplasia is a group of disorders that affect the body's network of hormone-producing glands (the endocrine system). Hormones are chemical messengers that travel through the bloodstream and regulate...

**Answer**:

Multiple endocrine neoplasia (MEN1) is an inherited condition that causes tumors of the endocrine system, which includes the body's hormone-producing glands. People affected by MEN1 typically develop tumors in the parathyroid gland, pituitary gland, and pancreas, although other glands may also be involved. These tumors are often "functional" and secrete excess hormones, leading to various health problems. The most common signs and symptoms are due to hyperparathyroidism (overactive parathyroid gland) and may include kidney stones, thinning of bones, nausea and vomiting, high blood pressure (hypertension), weakness, and fatigue. MEN1 is caused by mutations in the MEN1 gene and is inherited in an autosomal dominant manner [medquad_ans_0004185-1]. It is rare, occurring in about one in 30,000 people, and affects both sexes

**Supported citations**: `medquad_ans_0004185-1`
**Unsupported citations**: _none_

### Example 108 (`0004155-3`)

**Question**: What causes Mucopolysaccharidosis type I ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0004155-3` (score=0.9959) - What causes mucopolysaccharidosis I (MPS I)? Mutations in the IDUA gene cause MPS I. The IDUA gene provides instructions for producing an enzyme that is involved in the breakdown of large sugar molecules called glycos...
2. `medquad_ans_0000680-2` (score=0.0001) - MPS III is the most common type of mucopolysaccharidosis; the estimated incidence of all four types combined is 1 in 70,000 newborns. MPS IIIA and MPS IIIB are much more common than MPS IIIC and MPS IIID.
3. `medquad_ans_0000200-4__dup1` (score=0.0001) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
4. `medquad_ans_0000200-3__dup1` (score=0.0001) - The mucopolysaccharidoses syndromes share many clinical features but have varying degrees of severity. Most individuals with a mucopolysaccharidosis syndrome generally experience a period of normal development followe...
5. `medquad_ans_0000682-5` (score=0.0001) - These resources address the diagnosis or management of mucopolysaccharidosis type VI: - Emory University Lysosomal Storage Disease Center - Genetic Testing Registry: Mucopolysaccharidosis type VI - MedlinePlus Encyclo...

**Answer**:

Mutations in the IDUA gene cause mucopolysaccharidosis type I (MPS I). The IDUA gene provides instructions for producing an enzyme involved in the breakdown of large sugar molecules called glycosaminoglycans (GAGs). Mutations in the IDUA gene reduce or completely eliminate the function of the IDUA enzyme, leading to the accumulation of GAGs within cells, specifically inside the lysosomes. This accumulation results in lysosomal storage disorders [medquad_ans_0004155-3].

**Supported citations**: `medquad_ans_0004155-3`
**Unsupported citations**: _none_

### Example 109 (`0000825-3`)

**Question**: What are the genetic changes related to primary spontaneous pneumothorax ?

**Scores**: context_relevance=0.500, groundedness=0.064, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000047-3` (score=0.9985) - ACD/MPV can be caused by mutations in the FOXF1 gene. The protein produced from the FOXF1 gene is a transcription factor, which means that it attaches (binds) to specific regions of DNA and helps control the activity ...
2. `medquad_ans_0000109-4__dup2` (score=0.0170) - Pulmonary hypertension (PH) begins with inflammation and changes in the cells that line your pulmonary arteries. Other factors also can affect the pulmonary arteries and cause PH. For example, the condition may develo...
3. `medquad_ans_0000135-2__dup2` (score=0.0050) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
4. `medquad_ans_0000469-3` (score=0.0015) - Mutations in the PMP22 gene cause hereditary neuropathy with liability to pressure palsies. Hereditary neuropathy with liability to pressure palsies is caused by the loss of one copy of the PMP22 gene or alterations w...
5. `medquad_ans_0000022-5__dup1` (score=0.0004) - Many babies who develop bronchopulmonary dysplasia (BPD) are born with serious respiratory distress syndrome (RDS). The signs and symptoms of RDS at birth are: Rapid, shallow breathing Sharp pulling in of the chest be...

**Answer**:

The provided context does not contain any information about genetic changes related to primary spontaneous pneumothorax. Therefore, the answer cannot be determined from the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 110 (`0000066-31`)

**Question**: What is (are) Stroke ?

**Scores**: context_relevance=0.750, groundedness=0.992, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000066-32` (score=0.9961) - For more information on stroke, including research sponsored by the National Institute of Neurological Disorders and Stroke, call 1-800-352-9424 or visit the Web site at www.ninds.nih.gov.
2. `medquad_ans_0000066-23` (score=0.9929) - One test that helps doctors judge the severity of a stroke is the standardized NIH Stroke Scale, developed by the National Institute of Neurological Disorders and Stroke at the National Institutes of Health, or NIH. H...
3. `medquad_ans_0000066-18` (score=0.9921) - Transient ischemic attacks, or TIAs, occur when the warning signs of stroke last only a few moments and then disappear. These brief episodes are also sometimes called "mini-strokes." Although brief, they identify an u...
4. `medquad_ans_0000032-13` (score=0.9905) - At the first sign of a stroke, you should get medical care right away. If blood vessels to your brain are blocked by blood clots, the doctor can give you a "clot-busting" drug. The drug must be given soon after a stro...
5. `medquad_ans_0000057-9` (score=0.9882) - A stroke happens when part of your brain is not getting enough blood and stops working. Depending on the part of the brain that is damaged, a stroke can cause - sudden weakness or numbness of your face, arm, or leg on...

**Answer**:

A stroke happens when part of your brain is not getting enough blood and stops working. Depending on the part of the brain that is damaged, a stroke can cause - sudden weakness or numbness of your face, arm, or leg on one side of your body - sudden confusion, trouble talking, or trouble understanding - sudden dizziness, loss of balance, or trouble walking - sudden trouble seeing in one or both eyes or sudden double vision - sudden severe headache. Sometimes, one or more of these warning signs may happen and then disappear. You might be having a "mini-stroke," also called a TIA or a transient ischemic attack [medquad_ans_0000057-9].

**Supported citations**: `medquad_ans_0000057-9`
**Unsupported citations**: _none_

### Example 111 (`0006276-3`)

**Question**: What causes Uncombable hair syndrome ?

**Scores**: context_relevance=0.600, groundedness=0.000, answer_relevance=0.760

**Top docs**:

1. `medquad_ans_0000145-2__dup3` (score=0.0019) - The causes of Goodpasture syndrome are not fully understood. People who smoke or use hair dyes appear to be at increased risk for this condition. Exposure to hydrocarbon fumes, metallic dust, and certain drugs, such a...
2. `medquad_ans_0000898-4` (score=0.0016) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000101-4` (score=0.0016) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000731-4` (score=0.0016) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000683-4` (score=0.0016) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain information about the causes of Uncombable hair syndrome. Therefore, the question cannot be answered based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 112 (`0000887-4`)

**Question**: Is Russell-Silver syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000770-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000858-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000801-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000898-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000043-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Russell-Silver syndrome or its mode of inheritance. Therefore, it is not possible to determine from the given information whether Russell-Silver syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 113 (`0000969-1__dup2`)

**Question**: Do you have information about Weight Loss Surgery

**Scores**: context_relevance=0.750, groundedness=0.924, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000969-1__dup2` (score=0.9971) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
2. `medquad_ans_0000863-1__dup1` (score=0.0131) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
3. `medquad_ans_0000718-1__dup2` (score=0.0126) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
4. `medquad_ans_0000111-1__dup2` (score=0.0073) - Summary : Do you know if your current weight is healthy? "Underweight", "normal", "overweight", and "obese" are all labels for ranges of weight. Obese and overweight mean that your weight is greater than it should be ...
5. `medquad_ans_0000018-1__dup2` (score=0.0003) - After any operation, you'll have some side effects. There is usually some pain with surgery. There may also be swelling and soreness around the area that the surgeon cut. Your surgeon can tell you which side effects t...

**Answer**:

Yes, there is information about weight loss surgery. Weight loss surgery helps people with extreme obesity to lose weight and may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. It involves different types of surgery that often limit the amount of food you can consume and may affect how you digest and absorb nutrients. All types of weight loss surgery carry risks and complications, such as infections, hernias, and blood clots. Many people lose weight quickly initially, but some regain weight over time. Following diet and exercise recommendations can help keep most of the weight off, and lifelong medical follow-up is required [medquad_ans_0000969-1__dup2].

**Supported citations**: `medquad_ans_0000969-1__dup2`
**Unsupported citations**: _none_

### Example 114 (`0000889-5`)

**Question**: What are the treatments for Saethre-Chotzen syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.749, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000955-5` (score=0.9974) - These resources address the diagnosis or management of SAVI: - Beth Israel Deaconess Medical Center: Autoinflammatory Disease Center - Eurofever Project - Genetic Testing Registry: Sting-associated vasculopathy, infan...
2. `medquad_ans_0000889-5` (score=0.9947) - These resources address the diagnosis or management of Saethre-Chotzen syndrome: - Gene Review: Gene Review: Saethre-Chotzen Syndrome - Genetic Testing Registry: Robinow Sorauf syndrome - Genetic Testing Registry: Sae...
3. `medquad_ans_0000915-5` (score=0.0030) - These resources address the diagnosis or management of Sjgren syndrome: - Genetic Testing Registry: Sjgren's syndrome - MedlinePlus Encyclopedia: Schirmer's Test - National Institute of Dental and Craniofacial Researc...
4. `medquad_ans_0000607-5` (score=0.0023) - These resources address the diagnosis or management of Lujan syndrome: - Gene Review: Gene Review: MED12-Related Disorders - Genetic Testing Registry: X-linked mental retardation with marfanoid habitus syndrome These ...
5. `medquad_ans_0000152-2__dup1` (score=0.0018) - There is no treatment that can stop or reverse the course of HD. Tetrabenazine is prescribed for treating Huntingtons-associated chorea. It is the only drug approved by the U.S. Food and Drug Administration specifical...

**Answer**:

The provided context does not specify treatments for Saethre-Chotzen syndrome. It lists resources such as the Gene Review, Genetic Testing Registry, MedlinePlus Encyclopedia entries on craniosynostosis and skull of a newborn, and mentions diagnostic and management-related categories like genetic counseling, but does not detail specific treatments [medquad_ans_0000889-5]. Therefore, based on the given information, the treatments for Saethre-Chotzen syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000889-5`
**Unsupported citations**: _none_

### Example 115 (`0000512-1`)

**Question**: What is (are) idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=1.000, groundedness=0.497, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000078-1__dup2` (score=0.9214) - Pulmonary fibrosis (PULL-mun-ary fi-BRO-sis) is a disease in which tissue deep in your lungs becomes thick and stiff, or scarred, over time. The formation of scar tissue is called fibrosis. As the lung tissue thickens...
2. `medquad_ans_0000512-1` (score=0.2314) - Idiopathic pulmonary fibrosis is a chronic, progressive lung disease. This condition causes scar tissue (fibrosis) to build up in the lungs, which makes the lungs unable to transport oxygen into the bloodstream effect...
3. `medquad_ans_0000758-1__dup2` (score=0.1210) - Pulmonary fibrosis is a condition in which the tissue deep in your lungs becomes scarred over time. This tissue gets thick and stiff. That makes it hard for you to catch your breath, and your blood may not get enough ...
4. `medquad_ans_0000049-1__dup2` (score=0.0393) - Cystic fibrosis (SIS-tik fi-BRO-sis), or CF, is an inherited disease of the secretory (see-KREH-tor-ee) glands. Secretory glands include glands that make mucus and sweat. "Inherited" means the disease is passed from p...
5. `medquad_ans_0000253-1__dup2` (score=0.0162) - Cystic fibrosis (CF) is an inherited disease of the mucus and sweat glands. It affects mostly your lungs, pancreas, liver, intestines, sinuses, and sex organs. CF causes your mucus to be thick and sticky. The mucus cl...

**Answer**:

Idiopathic pulmonary fibrosis (IPF) is a chronic, progressive lung disease in which scar tissue (fibrosis) builds up in the lungs, making them unable to transport oxygen into the bloodstream effectively. It usually affects people between the ages of 50 and 70 and is characterized by symptoms such as shortness of breath, a persistent dry, hacking cough, loss of appetite, gradual weight loss, and in some cases, widened and rounded tips of the fingers and toes (clubbing) due to oxygen shortage. IPF has no known cause in most cases and is referred to as "idiopathic." When more than one family member has the disease, it is called familial IPF. The disease progresses over time, leading to reduced oxygen supply to the body's organs and tissues, and it is associated with serious complications such as respiratory failure, pulmonary hypertension, heart failure, pulmonary embolism, pneumonia, and lung cancer. Currently,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 116 (`0000213-4__dup1`)

**Question**: what research (or clinical trials) is being done for Neuroleptic Malignant Syndrome ?

**Scores**: context_relevance=0.583, groundedness=0.977, answer_relevance=0.383

**Top docs**:

1. `medquad_ans_0000045-4__dup1` (score=0.9984) - The NINDS conducts and supports research on disorders of the brain and nervous system, including pseudotumor cerebri. This research focuses primarily on increasing scientific understanding of these disorders and findi...
2. `medquad_ans_0000269-4__dup1` (score=0.9983) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes of the National Institutes of Health (NIH) conduct research in laboratories at the NIH and support additional research through g...
3. `medquad_ans_0000021-4__dup1` (score=0.9980) - The NINDS supports and conducts research on neurogenetic disorders such as Angelman syndrome, to develop techniques to diagnose, treat, prevent, and ultimately cure them.
4. `medquad_ans_0000213-4__dup1` (score=0.9974) - The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome. Much of this research focuses on finding ways to prevent and treat the disorder.
5. `medquad_ans_0000155-4__dup1` (score=0.9974) - The NINDS conducts and supports research on neurological disorders, including normal pressure hydrocephalus. Research on disorders such as normal pressure hydrocephalus focuses on increasing knowledge and understandin...

**Answer**:

The NINDS supports research on neurological disorders such as neuroleptic malignant syndrome, and much of this research focuses on finding ways to prevent and treat the disorder [medquad_ans_0000213-4__dup1].

**Supported citations**: `medquad_ans_0000213-4__dup1`
**Unsupported citations**: _none_

### Example 117 (`0000203-8`)

**Question**: How to diagnose Kidney Stones in Adults ?

**Scores**: context_relevance=0.857, groundedness=0.919, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000203-8` (score=0.9959) - To diagnose kidney stones, the health care provider will perform a physical exam and take a medical history. The medical history may include questions about family history of kidney stones, diet, GI problems, and othe...
2. `medquad_ans_0000204-7` (score=0.9953) - The process of diagnosing any illness begins with consideration of the symptoms. Pain or bloody urine may be the first symptom. Urine, blood, and imaging tests will help determine whether symptoms are caused by a ston...
3. `medquad_ans_0000221-8` (score=0.9934) - To diagnose kidney stones, your doctor will do a physical exam and ask about your medical history. The doctor may ask if you have a family history of kidney stones and about your diet, digestive problems, and other he...
4. `medquad_ans_0000221-6` (score=0.0924) - You may have a kidney stone if you - have pain while urinating - see blood in your urine - feel a sharp pain in your back or lower abdomenthe area between your chest and hips The pain may last for a short or long time...
5. `medquad_ans_0000221-10` (score=0.0012) - To prevent kidney stones, you need to know what caused your kidney stone. Your doctor may ask you to try to catch the kidney stone as it passes in your urine. The kidney stone can then be sent to a lab to find out wha...

**Answer**:

To diagnose kidney stones in adults, a healthcare provider will perform a physical exam and take a medical history, which may include questions about family history, diet, gastrointestinal problems, and other diseases. Diagnostic tests include urinalysis, blood tests, abdominal x-ray, and CT scans. Urinalysis tests a urine sample to detect infection or substances that form stones. Blood tests identify biochemical abnormalities that may lead to kidney stones. An abdominal x-ray uses minimal radiation to show the location of stones in the kidney or urinary tract. CT scans use x-rays and computer technology to create three-dimensional images and can show stone locations and conditions that may have caused the stone to form, possibly with the use of contrast medium [medquad_ans_0000203-8].

**Supported citations**: `medquad_ans_0000203-8`
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

Urethral cancer is a disease in which malignant (cancer) cells form in the tissues of the urethra. It is a rare cancer that occurs more often in men than in women. The urethra is the tube that carries urine from the bladder to outside the body; in women, it is about 1 inch long and is just above the vagina, while in men, it is about 8 inches long and goes through the prostate gland and the penis to the outside of the body. In men, the urethra also carries semen. There are different types of urethral cancer that begin in cells that line the urethra, including squamous cell carcinoma (the most common type), transitional cell carcinoma, and adenocarcinoma. Risk factors include a history of bladder cancer and conditions that cause chronic inflammation in the urethra, such as sexually transmitted diseases (including human papillomavirus type 16

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 119 (`0000771-4`)

**Question**: Is paramyotonia congenita inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000178-4` (score=0.9991) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0000947-4` (score=0.9991) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0000104-4` (score=0.9991) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000601-4` (score=0.9991) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000467-4` (score=0.9991) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not mention paramyotonia congenita or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether paramyotonia congenita is inherited.

[No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 120 (`0000735-2__dup1`)

**Question**: What are the symptoms of Biotinidase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0000735-2__dup1` (score=0.9987) - What are the signs and symptoms of Biotinidase deficiency? The signs and symptoms of biotinidase deficiency typically appear within the first few months of life, but the age of onset varies. Children with profound bio...
2. `medquad_ans_0000138-5__dup1` (score=0.4355) - The signs and symptoms of anemia in someone with CKD may include - weakness - fatigue, or feeling tired - headaches - problems with concentration - paleness - dizziness - difficulty breathing or shortness of breath - ...
3. `medquad_ans_0000125-2` (score=0.0001) - Profound or partial biotinidase deficiency occurs in approximately 1 in 60,000 newborns
4. `medquad_ans_0004581-2` (score=0.0001) - What are the signs and symptoms of Ornithine transcarbamylase deficiency? Ornithine transcarbamylase (OTC) deficiency often becomes evident in the first few days of life. An infant with OTC deficiency may be lacking i...
5. `medquad_ans_0000125-4` (score=0.0001) - This condition is inherited in an autosomal recessive pattern, which means both copies of the BTD gene in each cell have mutations. The parents of an individual with biotinidase deficiency each carry one copy of the m...

**Answer**:

The symptoms of Biotinidase deficiency include muscular hypotonia (90%), seizures (90%), alopecia (50%), dry skin (50%), hearing impairment (50%), incoordination (50%), inflammatory abnormality of the eye (50%), optic atrophy (50%), skin rash (50%), abnormality of retinal pigmentation (7.5%), aplasia/hypoplasia of the cerebellum (7.5%), hypertonia (7.5%), muscle weakness (7.5%), myopia (7.5%), reduced consciousness/confusion (7.5%), respiratory insufficiency (7.5%), skin ulcer (7.5%), and visual field defect (7.5%). Other associated symptoms may include feeding difficulties in infancy, recurrent skin infections, seborrheic dermatitis, sensorineural hearing impairment, hepatomegaly, hyperammonemia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 121 (`0000966-1__dup2`)

**Question**: Do you have information about Water Pollution

**Scores**: context_relevance=0.714, groundedness=0.886, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000966-1__dup2` (score=0.9971) - Summary : We all need clean water. People need it to grow crops and to operate factories, and for drinking and recreation. Fish and wildlife depend on it to survive. Many different pollutants can harm our rivers, stre...
2. `medquad_ans_0000298-1__dup1` (score=0.0371) - Summary : We all need to drink water. How much you need depends on your size, activity level, and the weather where you live. The water you drink is a combination of surface water and groundwater. Surface water includ...
3. `medquad_ans_0000331-1__dup1` (score=0.0012) - Summary : Our environment affects our health. If parts of the environment, like the air, water, or soil become polluted, it can lead to health problems. For example, asthma pollutants and chemicals in the air or in th...
4. `medquad_ans_0000505-1__dup1` (score=0.0002) - Summary : We usually think of air pollution as being outdoors, but the air in your house or office could also be polluted. Sources of indoor pollution include - Mold and pollen - Tobacco smoke - Household products and...
5. `medquad_ans_0000677-1__dup1` (score=0.0001) - Summary : Ozone is a gas. It can be good or bad, depending on where it is. "Good" ozone occurs naturally about 10 to 30 miles above the Earth's surface. It shields us from the sun's ultraviolet rays. Part of the good ...

**Answer**:

Yes, there is information about water pollution. The three most common pollutants affecting rivers, streams, lakes, and oceans are soil, nutrients, and bacteria. Rain washes soil into water bodies, which can kill fish eggs, clog fish gills, and block light, causing plant death. Nutrients from fertilizers cause algae growth, turning water green, especially in lakes, ponds, and reservoirs. Bacteria from sewage spills can pollute both fresh and salt water. To help protect water supplies, one should not pour household products like cleansers, medicines, or grease down the drain, and instead dispose of them at hazardous waste sites. Excess grease, diapers, condoms, and pet waste should be thrown in the garbage. Pet waste contains nutrients and germs that can contribute to water pollution [medquad_ans_0000966-1__dup2].

**Supported citations**: `medquad_ans_0000966-1__dup2`
**Unsupported citations**: _none_

### Example 122 (`0000132-3__dup1`)

**Question**: What is the outlook for Foot Drop ?

**Scores**: context_relevance=0.714, groundedness=0.973, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000132-3__dup1` (score=0.9937) - The prognosis for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is...
2. `medquad_ans_0000276-1__dup2` (score=0.9882) - If you have diabetes, your blood glucose, or blood sugar, levels are too high. Over time, this can damage your nerves or blood vessels. Nerve damage from diabetes can cause you to lose feeling in your feet. You may no...
3. `medquad_ans_0000377-1__dup1` (score=0.0135) - Summary : Each step you take involves a complex network of bones, muscles, tendons, and ligaments. This, combined with all of the weight they carry, explains why feet can have problems. To keep your feet healthy - Exa...
4. `medquad_ans_0000378-1__dup2` (score=0.0080) - Each of your feet has 26 bones, 33 joints, and more than 100 tendons, muscles, and ligaments. No wonder a lot of things can go wrong. Here are a few common problems: - Bunions - hard, painful bumps on the big toe join...
5. `medquad_ans_0000050-18` (score=0.0072) - Preventing falls is a special concern for men and women with osteoporosis. Falls can increase the likelihood of fracturing a bone in the hip, wrist, spine, or other part of the skeleton. In addition to the environment...

**Answer**:

The outlook for foot drop depends on the cause. Foot drop caused by trauma or nerve damage usually shows partial or even complete recovery. For progressive neurological disorders, foot drop will be a symptom that is likely to continue as a lifelong disability, but it will not shorten life expectancy [medquad_ans_0000132-3__dup1].

**Supported citations**: `medquad_ans_0000132-3__dup1`
**Unsupported citations**: _none_

### Example 123 (`0000145-1`)

**Question**: What is (are) Camurati-Engelmann disease ?

**Scores**: context_relevance=0.800, groundedness=0.479, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000145-1` (score=0.7351) - Camurati-Engelmann disease is a condition that mainly affects the bones. People with this disease have increased bone density, particularly affecting the long bones of the arms and legs. In some cases, the skull and h...
2. `medquad_ans_0000767-2` (score=0.0014) - This condition is very rare; its prevalence is unknown.
3. `medquad_ans_0000439-2` (score=0.0014) - This condition is very rare; its prevalence is unknown.
4. `medquad_ans_0000302-2` (score=0.0005) - Dowling-Degos disease appears to be a rare condition, although its prevalence is unknown.
5. `medquad_ans_0000137-1__dup2` (score=0.0004) - Gerstmann-Straussler-Scheinker disease (GSS) is an extremely rare, neurodegenerative brain disorder. It is almost always inherited and is found in only a few families around the world. Onset of the disease usually occ...

**Answer**:

Camurati-Engelmann disease is a condition that mainly affects the bones, characterized by increased bone density, particularly in the long bones of the arms and legs. It can also affect the skull and hip bones. Thickened bones may lead to pain in the arms and legs, a waddling walk, muscle weakness, and extreme tiredness. Increased skull density can cause elevated pressure on the brain, resulting in neurological problems such as headaches, hearing loss, vision problems, dizziness (vertigo), ringing in the ears (tinnitus), and facial paralysis. The condition may also cause abnormal curvature of the spine (scoliosis), joint deformities (contractures), knock knees, and flat feet (pes planus). Other features include abnormally long limbs in proportion to height, decreased muscle mass and body fat, and delayed puberty. Symptoms typically begin during adolescence, though the age of onset varies widely. Some individuals may carry the gene mutation without

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 124 (`0002080-2`)

**Question**: What are the symptoms of Emery-Dreifuss muscular dystrophy, X-linked ?

**Scores**: context_relevance=1.000, groundedness=0.989, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0002080-2` (score=0.9994) - What are the signs and symptoms of Emery-Dreifuss muscular dystrophy, X-linked? The Human Phenotype Ontology provides the following list of signs and symptoms for Emery-Dreifuss muscular dystrophy, X-linked. If the in...
2. `medquad_ans_0000631-1__dup1` (score=0.0001) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
3. `medquad_ans_0002213-2` (score=0.0001) - What are the signs and symptoms of Facioscapulohumeral muscular dystrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Facioscapulohumeral muscular dystrophy. If the information ...
4. `medquad_ans_0000316-1` (score=0.0001) - Emery-Dreifuss muscular dystrophy is a condition that chiefly affects muscles used for movement (skeletal muscles) and heart (cardiac) muscle. Among the earliest features of this disorder are joint deformities called ...
5. `medquad_ans_0001537-1` (score=0.0001) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...

**Answer**:

The symptoms of Emery-Dreifuss muscular dystrophy, X-linked include Achilles tendon contracture, elbow flexion contracture, decreased cervical spine flexion due to contractures of posterior cervical muscles, pectus excavatum, elevated serum creatine phosphokinase, primary atrial arrhythmia, atrioventricular block, slow progression, sudden cardiac death, type 1 muscle fiber atrophy, waddling gait, childhood or juvenile onset, and X-linked recessive inheritance [medquad_ans_0002080-2].

**Supported citations**: `medquad_ans_0002080-2`
**Unsupported citations**: _none_

### Example 125 (`0000072-5`)

**Question**: What are the treatments for arterial tortuosity syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.813, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000267-2__dup1` (score=0.9953) - Treatment begins with exercise programs and physical therapy to strengthen chest muscles, restore normal posture, and relieve compression by increasing the space of the area the nerve passes through. Doctors will ofte...
2. `medquad_ans_0000072-5` (score=0.9945) - These resources address the diagnosis or management of arterial tortuosity syndrome: - Gene Review: Gene Review: Arterial Tortuosity Syndrome - Genetic Testing Registry: Arterial tortuosity syndrome - Johns Hopkins Mc...
3. `medquad_ans_0003473-6` (score=0.9101) - How might Laron syndrome be treated? There is currently no cure for Laron syndrome. Treatment is primarily focused on improving growth. The only specific treatment available for this condition is subcutaneous injectio...
4. `medquad_ans_0000273-2__dup2` (score=0.8527) - There is no cure for TSC, although treatment is available for a number of the symptoms. Rapamycin and related drugs are not yet approved by the U.S. Food and Drug Administration (FDA) for any purpose in individuals wi...
5. `medquad_ans_0003593-6` (score=0.5969) - How might Liddle syndrome be treated? Treatment for Liddle syndrome includes following a low sodium diet as well as taking potassium-sparing diuretics, which reduce blood pressure and correct hypokalemia and metabolic...

**Answer**:

The provided context does not contain specific information about the treatments for arterial tortuosity syndrome. While it lists resources that address the diagnosis or management of arterial tortuosity syndrome, such as gene reviews, genetic testing, and information from the National Heart, Lung, and Blood Institute, it does not detail any specific treatments [medquad_ans_0000072-5]. Therefore, based on the given context, the treatments for arterial tortuosity syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000072-5`
**Unsupported citations**: _none_

### Example 126 (`0001092-1__dup1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 1C ?

**Scores**: context_relevance=1.000, groundedness=0.968, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001092-1__dup1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...
2. `medquad_ans_0001090-2__dup1` (score=0.0009) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1A? CMT1 is generally slowly progressive over many years. However, affected individuals often experience long periods without any obvious deteriorati...
3. `medquad_ans_0001088-2__dup1` (score=0.0005) - What are the signs and symptoms of Charcot-Marie-Tooth disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease. If the information is available, the tabl...
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
2. `medquad_ans_0000071-5__dup1` (score=0.9984) - Because diagnosis is based on blood pressure readings, this condition can go undetected for years, as symptoms do not usually appear until the body is damaged from chronic high blood pressure. Complications of High Bl...
3. `medquad_ans_0000016-11` (score=0.9981) - At first, you will see a few specks of blood, or spots, "floating" in your vision. If spots occur, see your eye care professional as soon as possible. You may need treatment before more serious bleeding or hemorrhagin...
4. `medquad_ans_0000012-5__dup1` (score=0.2946) - Atherosclerosis usually doesn't cause signs and symptoms until it severely narrows or totally blocks an artery. Many people don't know they have the disease until they have a medical emergency, such as a heart attack ...
5. `medquad_ans_0000016-3__dup2` (score=0.0007) - Diabetic retinopathy often has no early warning signs. Don't wait for symptoms. Be sure to have a comprehensive dilated eye exam at least once a year to detect the disease before it causes damage to your vision. Visio...

**Answer**:

The symptoms of Retinal arterial macroaneurysm with supravalvular pulmonic stenosis include exudative retinal detachment and pulmonic stenosis [medquad_ans_0005321-1].

**Supported citations**: `medquad_ans_0005321-1`
**Unsupported citations**: _none_

### Example 128 (`0001838-1`)

**Question**: What is (are) Dicarboxylic aminoaciduria ?

**Scores**: context_relevance=0.400, groundedness=0.039, answer_relevance=0.560

**Top docs**:

1. `medquad_ans_0001702-1` (score=0.0060) - D-2-alpha hydroxyglutaric aciduria is an inherited metabolic condition that is associated with progressive brain damage. Signs and symptoms of this condition include developmental delay, seizures, hypotonia, and abnor...
2. `medquad_ans_0000168-1__dup3` (score=0.0026) - Proteinuriaalso called albuminuria or urine albuminis a condition in which urine contains an abnormal amount of protein. Albumin is the main protein in the blood. Proteins are the building blocks for all body parts, i...
3. `medquad_ans_0000016-3` (score=0.0012) - Mutations in the APRT gene cause APRT deficiency. This gene provides instructions for making APRT, an enzyme that helps to convert a DNA building block (nucleotide) called adenine to a molecule called adenosine monoph...
4. `medquad_ans_0001860-1` (score=0.0003) - Dihydropteridine reductase deficiency (DHPR) is a severe form of hyperphenylalaninemia (high levels of the amino acid phenylalanine in the blood) due to impaired renewal of a substance known as tetrahydrobiopterin (BH...
5. `medquad_ans_0000792-5` (score=0.0001) - These resources address the diagnosis or management of phenylketonuria: - Baby's First Test - Gene Review: Gene Review: Phenylalanine Hydroxylase Deficiency - Genetic Testing Registry: Phenylketonuria - MedlinePlus En...

**Answer**:

The provided context does not contain any information about dicarboxylic aminoaciduria. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 129 (`0000949-4`)

**Question**: Is spondylothoracic dysostosis inherited ?

**Scores**: context_relevance=1.000, groundedness=0.708, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000943-4` (score=0.6467) - Spondylocarpotarsal synostosis syndrome caused by FLNB gene mutations is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with ...
2. `medquad_ans_0004514-4` (score=0.0081) - How is oculopharyngeal muscular dystrophy inherited?
3. `medquad_ans_0005859-1` (score=0.0001) - Spondylospinal thoracic dysostosis is an extremely rare skeletal disorder characterized by a short, curved spine and fusion of the spinous processes, short thorax with 'crab-like' configuration of the ribs, underdevel...
4. `medquad_ans_0001034-2` (score=0.0001) - Weyers acrofacial dysostosis appears to be a rare disorder. Only a few affected families have been identified worldwide.
5. `medquad_ans_0000949-2` (score=0.0001) - Spondylothoracic dysostosis affects about one in 200,000 people worldwide. However, it is much more common in people of Puerto Rican ancestry, affecting approximately one in 12,000 people.

**Answer**:

The provided context does not specify how spondylothoracic dysostosis is inherited. While related conditions like spondylocarpotarsal synostosis syndrome and spondylospinal thoracic dysostosis are mentioned as being inherited in an autosomal recessive manner, there is no direct information on the inheritance pattern of spondylothoracic dysostosis [medquad_ans_0000949-2]. Therefore, based on the available information, the inheritance pattern of spondylothoracic dysostosis cannot be determined.

Answer: Insufficient context to determine the inheritance pattern of spondylothoracic dysostosis.

**Supported citations**: `medquad_ans_0000949-2`
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

**Scores**: context_relevance=0.917, groundedness=0.766, answer_relevance=0.783

**Top docs**:

1. `medquad_ans_0000507-2` (score=0.9991) - The prevalence of hypomyelination and congenital cataract is unknown.
2. `medquad_ans_0000218-1` (score=0.0001) - Congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is a rare disorder that affects several parts of the body. It is characterized by a clouding of the lens of the eyes at birth (congenital cataracts) and...
3. `medquad_ans_0001075-1__dup1` (score=0.0001) - Cerulean cataracts are opaque areas that develop in the lens of the eye that often have a bluish or whitish color. They may be present at birth or develop in very early childhood, but may not be diagnosed until adulth...
4. `medquad_ans_0000507-1` (score=0.0001) - Hypomyelination and congenital cataract is an inherited condition that affects the nervous system and the eyes. This disease is one of a group of genetic disorders called leukoencephalopathies. Leukoencephalopathies i...
5. `medquad_ans_0000009-12` (score=0.0001) - Yes. Although most cataracts are related to aging, there are other types of cataract. These include - secondary cataract - traumatic cataract - congenital cataract - radiation cataract. secondary cataract traumatic ca...

**Answer**:

The number of people affected by congenital cataracts, facial dysmorphism, and neuropathy (CCFDN) is unknown based on the provided context. [medquad_ans_0000218-1] states that CCFDN is a rare disorder, but it does not provide specific prevalence data. [medquad_ans_0000507-2] notes that the prevalence of hypomyelination and congenital cataract is unknown, but this does not directly address the prevalence of CCFDN. Therefore, a specific number of affected individuals cannot be determined from the given information.

**Supported citations**: `medquad_ans_0000218-1`, `medquad_ans_0000507-2`
**Unsupported citations**: _none_

### Example 132 (`0000040-3`)

**Question**: What are the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome ?

**Scores**: context_relevance=0.846, groundedness=0.769, answer_relevance=0.846

**Top docs**:

1. `medquad_ans_0000396-1__dup2` (score=0.0077) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...
2. `medquad_ans_0000127-8` (score=0.0002) - You cant prevent thalassemias because theyre inherited (passed from parents to children through genes). However, prenatal tests can detect these blood disorders before birth. Family genetic studies may help find out w...
3. `medquad_ans_0000127-5__dup1` (score=0.0001) - A lack of oxygen in the bloodstream causes the signs and symptoms of thalassemias. The lack of oxygen occurs because the body doesn't make enough healthy red blood cells and hemoglobin. The severity of symptoms depend...
4. `medquad_ans_0000259-3__dup1` (score=0.0001) - How is alpha-thalassemia x-linked intellectual disability syndrome inherited? Alpha-thalassemia x-linked intellectual disability (ATRX) syndrome is caused by a mutation in the ATRX gene and is inherited in an x-linked...
5. `medquad_ans_0000127-4__dup2` (score=0.0001) - Family history and ancestry are the two risk factors for thalassemias. Family History Thalassemias are inheritedthat is, the genes for the disorders are passed from parents to their children. If your parents have miss...

**Answer**:

The provided context does not contain information about the genetic changes related to alpha thalassemia X-linked intellectual disability syndrome. Therefore, a specific answer cannot be given based on the available information. [medquad_ans_0000396-1__dup2] discusses genetic brain disorders such as Tay-Sachs disease and Wilson disease, and [medquad_ans_0000127-5__dup1] describes thalassemias but does not mention X-linked intellectual disability or alpha thalassemia in that context. [medquad_ans_0000127-8] discusses inherited thalassemias and prenatal testing but does not address genetic changes specific to alpha thalassemia X-linked intellectual disability syndrome.

**Supported citations**: `medquad_ans_0000396-1__dup2`, `medquad_ans_0000127-5__dup1`, `medquad_ans_0000127-8`
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

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000071-2__dup3` (score=0.9952) - Type 1 diabetes is caused by a lack of insulin due to the destruction of insulin-producing beta cells in the pancreas. In type 1 diabetesan autoimmune diseasethe bodys immune system attacks and destroys the beta cells...
2. `medquad_ans_0000037-3__dup3` (score=0.9900) - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 di...
3. `medquad_ans_0000027-3__dup3` (score=0.9900) - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 di...
4. `medquad_ans_0000071-3__dup4` (score=0.9900) - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 di...
5. `medquad_ans_0000070-3__dup4` (score=0.9900) - Type 2 diabetesthe most common form of diabetesis caused by a combination of factors, including insulin resistance, a condition in which the bodys muscle, fat, and liver cells do not use insulin effectively. Type 2 di...

**Answer**:

Type 1 diabetes is caused by a lack of insulin due to the destruction of insulin-producing beta cells in the pancreas, which occurs as a result of an autoimmune process where the body's immune system attacks and destroys these beta cells [medquad_ans_0000071-2__dup3]. This autoimmune response is what defines type 1 diabetes, and it typically occurs in children and young adults, though it can appear at any age [medquad_ans_0000071-2__dup3]. Genetic susceptibility also plays a significant role, with certain gene variants—particularly those affecting human leukocyte antigens (HLAs)—increasing the risk of developing the disease [medquad_ans_0000071-2__dup3]. However, the provided context does not cover the causes of other types of diabetes (e.g., type 2 diabetes or gestational diabetes). Therefore, the

**Supported citations**: `medquad_ans_0000071-2__dup3`
**Unsupported citations**: _none_

### Example 135 (`0000016-21`)

**Question**: what research (or clinical trials) is being done for Diabetic Retinopathy ?

**Scores**: context_relevance=0.636, groundedness=0.988, answer_relevance=0.418

**Top docs**:

1. `medquad_ans_0000016-26` (score=0.9974) - National Eye Institute National Institutes of Health 2020 Vision Place Bethesda, MD 20892-3655 301-496-5248 E-mail: 2020@nei.nih.gov www.nei.nih.gov Find eye health organizations that address diabetic eye disease.
2. `medquad_ans_0000016-21` (score=0.9970) - The National Eye Institute, or NEI, is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the lab...
3. `medquad_ans_0000001-29` (score=0.9966) - The National Eye Institute scientists are - studying the possibility of transplanting healthy cells into a diseased retina - evaluating families with a history of AMD to understand genetic and hereditary factors that ...
4. `medquad_ans_0000027-22` (score=0.9942) - Through studies in the laboratory and with patients, the National Eye Institute is seeking better ways to detect, treat, and prevent vision loss in people with glaucoma. For example, researchers have discovered genes ...
5. `medquad_ans_0000106-4__dup1` (score=0.6799) - The NINDS conducts and supports research on diabetic neuropathy to increase understanding of the disorder and find ways to prevent and cure it. New medications are currently being examined to assess improvement or sta...

**Answer**:

The National Eye Institute (NEI) is conducting and supporting research that seeks better ways to detect, treat, and prevent vision loss in people with diabetes. This research is conducted through studies in the laboratory and with patients. For example, researchers are studying drugs for the treatment of proliferative retinopathy that may reduce the need for laser surgery [medquad_ans_0000016-21].

**Supported citations**: `medquad_ans_0000016-21`
**Unsupported citations**: _none_

### Example 136 (`0000516-1__dup1`)

**Question**: What is (are) Inhalation Injuries ?

**Scores**: context_relevance=0.800, groundedness=0.925, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000516-1__dup1` (score=0.9100) - There are a variety of substances you can inhale that can cause acute internal injuries. Particles in the air from fires and toxic fumes can damage your eyes and respiratory system. They also can make chronic heart an...
2. `medquad_ans_0000379-1__dup1` (score=0.8206) - If you've ever gotten a splinter or had sand in your eye, you've had experience with a foreign body. A foreign body is something that is stuck inside you but isn't supposed to be there. You may inhale or swallow a for...
3. `medquad_ans_0000007-3__dup1` (score=0.0093) - Many conditions or factors can directly or indirectly injure the lungs and lead to ARDS. Some common ones are: Sepsis. This is a condition in which bacteria infect the bloodstream. Pneumonia. This is an infection in t...
4. `medquad_ans_0000252-2__dup1` (score=0.0084) - Emergency treatment for a baby who has been shaken usually includes life-sustaining measures such as respiratory support and surgery to stop internal bleeding and bleeding in the brain. Doctors may use brain scans, su...
5. `medquad_ans_0000007-4__dup1` (score=0.0057) - People at risk for ARDS have a condition or illness that can directly or indirectly injure their lungs. Direct Lung Injury Conditions that can directly injure the lungs include: Pneumonia. This is an infection in the ...

**Answer**:

Inhalation injuries are acute internal injuries caused by inhaling harmful substances such as particles in the air from fires or toxic fumes. These substances can damage the eyes and respiratory system and may worsen chronic heart and lung diseases. Symptoms may include coughing and phlegm, a scratchy throat, irritated sinuses, shortness of breath, chest pain or tightness, headaches, stinging eyes, and a runny nose. If a person already has asthma, the condition may worsen [medquad_ans_0000516-1__dup1]. Breathing in harmful fumes or smoke is also listed as a cause of ARDS, indicating its potential to cause significant lung injury [medquad_ans_0000007-4__dup1].

**Supported citations**: `medquad_ans_0000516-1__dup1`, `medquad_ans_0000007-4__dup1`
**Unsupported citations**: _none_

### Example 137 (`0000134-4__dup2`)

**Question**: How many people are affected by Zollinger-Ellison Syndrome ?

**Scores**: context_relevance=0.500, groundedness=0.945, answer_relevance=0.400

**Top docs**:

1. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.
2. `medquad_ans_0000315-2` (score=0.9993) - Emanuel syndrome is a rare disorder; its prevalence is unknown. More than 100 individuals with this condition have been reported.
3. `medquad_ans_0000407-2` (score=0.9993) - Gilbert syndrome is a common condition that is estimated to affect 3 to 7 percent of Americans.
4. `medquad_ans_0000387-2` (score=0.9984) - Freeman-Sheldon syndrome is a rare disorder; its exact prevalence is unknown.
5. `medquad_ans_0000134-4__dup2` (score=0.9982) - Zollinger-Ellison syndrome is rare and only occurs in about one in every 1 million people.1 Although anyone can get Zollinger-Ellison syndrome, the disease is more common among men 30 to 50 years old. A child who has ...

**Answer**:

Zollinger-Ellison syndrome occurs in about one in every 1 million people [medquad_ans_0000134-4__dup2].

**Supported citations**: `medquad_ans_0000134-4__dup2`
**Unsupported citations**: _none_

### Example 138 (`0000974-5`)

**Question**: What are the treatments for tetrahydrobiopterin deficiency ?

**Scores**: context_relevance=0.857, groundedness=0.768, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000974-5` (score=0.9949) - These resources address the diagnosis or management of tetrahydrobiopterin deficiency: - Baby's First Test: Biopterin Defect in Cofactor Biosynthesis - Baby's First Test: Biopterin Defect in Cofactor Regeneration - Ge...
2. `medquad_ans_0000033-2__dup1` (score=0.9923) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
3. `medquad_ans_0006372-6` (score=0.0612) - How might VLCAD deficiency be treated? Management of VLCAD deficiency depends on many factors, including the form of the condition and the specific signs and symptoms present. For example, people affected by the sever...
4. `medquad_ans_0001859-5` (score=0.0013) - How might dihydrolipoamide dehydrogenase deficiency be treated? There are currently no consensus recommendations for the management of dihydrolipoamide dehydrogenase (DLD) deficiency. Management can be hard because va...
5. `medquad_ans_0000898-5` (score=0.0009) - These resources address the diagnosis or management of sepiapterin reductase deficiency: - Gene Review: Gene Review: Sepiapterin Reductase Deficiency - Genetic Testing Registry: Sepiapterin reductase deficiency These ...

**Answer**:

The provided context does not specify the treatments for tetrahydrobiopterin deficiency. It lists diagnostic resources and information about the condition, such as genetic testing and serum phenylalanine screening, but does not detail any therapeutic interventions [medquad_ans_0000974-5]. Therefore, based on the available information, the treatments for tetrahydrobiopterin deficiency cannot be determined.

**Supported citations**: `medquad_ans_0000974-5`
**Unsupported citations**: _none_

### Example 139 (`0000736-2`)

**Question**: How many people are affected by nonsyndromic holoprosencephaly ?

**Scores**: context_relevance=0.375, groundedness=0.822, answer_relevance=0.575

**Top docs**:

1. `medquad_ans_0000708-2` (score=0.9995) - The prevalence of this condition is unknown.
2. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.
3. `medquad_ans_0000366-2` (score=0.9993) - Familial porencephaly is a rare condition, although the exact prevalence is unknown. At least eight affected families have been described in the scientific literature.
4. `medquad_ans_0000736-2` (score=0.9991) - Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns.
5. `medquad_ans_0000331-2` (score=0.9957) - About 30 individuals with this condition have been identified worldwide, mostly in Mediterranean and Arab populations. Although ethylmalonic encephalopathy appears to be very rare, researchers suggest that some cases ...

**Answer**:

Nonsyndromic holoprosencephaly accounts for approximately 25 to 50 percent of all cases of holoprosencephaly, which affects an estimated 1 in 10,000 newborns [medquad_ans_0000736-2]. Therefore, the number of people affected by nonsyndromic holoprosencephaly is estimated to be about 25 to 50 percent of 1 in 10,000 newborns, which translates to approximately 1 in 20,000 to 1 in 40,000 newborns. However, the exact number of individuals affected is not specified in the provided context.

**Supported citations**: `medquad_ans_0000736-2`
**Unsupported citations**: _none_

### Example 140 (`0000829-3`)

**Question**: What are the genetic changes related to progressive familial intrahepatic cholestasis ?

**Scores**: context_relevance=0.636, groundedness=0.122, answer_relevance=0.782

**Top docs**:

1. `medquad_ans_0000525-3` (score=0.0013) - Genetic changes in the ABCB11 or the ABCB4 gene can increase a woman's likelihood of developing intrahepatic cholestasis of pregnancy. The ABCB11 gene provides instructions for making a protein called the bile salt ex...
2. `medquad_ans_0000525-2` (score=0.0001) - Intrahepatic cholestasis of pregnancy is estimated to affect 1 percent of women of Northern European ancestry. The condition is more common in certain populations, such as women of Araucanian Indian ancestry in Chile ...
3. `medquad_ans_0003240-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy (ICP) is a disorder of the liver that occurs in women during pregnancy. Cholestasis is a condition that impairs the release of bile (a digestive juice) from liver cells. The bile ...
4. `medquad_ans_0000525-1` (score=0.0001) - Intrahepatic cholestasis of pregnancy is a liver disorder that occurs in pregnant women. Cholestasis is a condition that impairs the release of a digestive fluid called bile from liver cells. As a result, bile builds ...
5. `medquad_ans_0000183-4__dup3` (score=0.0001) - Biliary atresia likely has multiple causes, though none are yet proven. Biliary atresia is not an inherited disease, meaning it does not pass from parent to child. Therefore, survivors of biliary atresia are not at ri...

**Answer**:

The provided context does not mention progressive familial intrahepatic cholestasis or describe genetic changes related to this condition. Therefore, based on the available information, it is not possible to answer the question. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 141 (`0005678-1`)

**Question**: What are the symptoms of Smith McCort dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.980, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005678-1` (score=0.9994) - What are the signs and symptoms of Smith McCort dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Smith McCort dysplasia. If the information is available, the table below in...
2. `medquad_ans_0005519-1` (score=0.0001) - What are the signs and symptoms of Scott Bryant Graham syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Scott Bryant Graham syndrome. If the information is available, the ta...
3. `medquad_ans_0003814-2` (score=0.0001) - What are the signs and symptoms of Marshall-Smith syndrome? Marshall-Smith syndrome is characterized by accelerated skeletal maturation, relative failure to thrive, respiratory difficulties, mental retardation, and un...
4. `medquad_ans_0004833-2` (score=0.0001) - What are the signs and symptoms of Peutz-Jeghers syndrome? Peutz-Jeghers syndrome (PJS) is characterized primarily by an increased risk of growths along the lining of the gastrointestinal tract (called hamartomatous p...
5. `medquad_ans_0002658-1` (score=0.0001) - What are the signs and symptoms of Gracile bone dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Gracile bone dysplasia. If the information is available, the table below in...

**Answer**:

The symptoms of Smith McCort dysplasia include atlantoaxial instability, barrel-shaped chest, beaking of vertebral bodies, deformed sella turcica, delayed femoral head ossification, disproportionate short-trunk short stature, dolichocephaly, genu valgum, genu varum, hypoplasia of the odontoid process, hypoplastic acetabulae, hypoplastic facial bones, hypoplastic scapulae, irregular epiphyses, kyphosis, metaphyseal irregularity, microcephaly, multicentric femoral head ossification, platyspondyly, prominent sternum, scoliosis, short metacarpal, short phalanx of finger, and waddling gait [medquad_ans_0005678-1].

**Supported citations**: `medquad_ans_0005678-1`
**Unsupported citations**: _none_

### Example 142 (`0000453-4`)

**Question**: Is hepatic veno-occlusive disease with immunodeficiency inherited ?

**Scores**: context_relevance=0.571, groundedness=0.012, answer_relevance=0.914

**Top docs**:

1. `medquad_ans_0000439-2` (score=0.0001) - This condition is very rare; its prevalence is unknown.
2. `medquad_ans_0000767-2` (score=0.0001) - This condition is very rare; its prevalence is unknown.
3. `medquad_ans_0000708-2` (score=0.0001) - The prevalence of this condition is unknown.
4. `medquad_ans_0001005-2` (score=0.0001) - The prevalence of TH deficiency is unknown.
5. `medquad_ans_0004534-2` (score=0.0001) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...

**Answer**:

The provided context does not contain information about whether hepatic veno-occlusive disease with immunodeficiency is inherited. Therefore, based on the given context, this question cannot be answered. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 143 (`0005902-1`)

**Question**: What are the symptoms of Stomatocytosis I ?

**Scores**: context_relevance=1.000, groundedness=0.914, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005902-1` (score=0.9993) - What are the signs and symptoms of Stomatocytosis I? The Human Phenotype Ontology provides the following list of signs and symptoms for Stomatocytosis I. If the information is available, the table below includes how o...
2. `medquad_ans_0000013_3-2__dup1` (score=0.4409) - Signs and symptoms of myelodysplastic/myeloproliferative neoplasm, unclassifiable, include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by MDS/MPN-UC or by other conditi...
3. `medquad_ans_0000003_2-3` (score=0.0097) - Signs of gastrointestinal stromal tumors include blood in the stool or vomit. These and other signs and symptoms may be caused by a GIST or by other conditions. Check with your doctor if you have any of the following:...
4. `medquad_ans_0000013_3-3__dup2` (score=0.0021) - Signs and symptoms of chronic myelomonocytic leukemia include fever, weight loss, and feeling very tired. These and other signs and symptoms may be caused by CMML or by other conditions. Check with your doctor if you ...
5. `medquad_ans_0000001_4-2` (score=0.0021) - Signs and symptoms of chronic myelogenous leukemia include fever, night sweats, and tiredness. These and other signs and symptoms may be caused by CML or by other conditions. Check with your doctor if you have any of ...

**Answer**:

The symptoms of Stomatocytosis I include hemolytic anemia, increased intracellular sodium, increased red cell osmotic fragility, and stomatocytosis. These signs and symptoms are based on the Human Phenotype Ontology, which compiles data from sources such as Orphanet [medquad_ans_0005902-1].

**Supported citations**: `medquad_ans_0005902-1`
**Unsupported citations**: _none_

### Example 144 (`0000159-7`)

**Question**: What to do for Lupus Nephritis ?

**Scores**: context_relevance=0.667, groundedness=0.941, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000159-7` (score=0.9732) - Eating, diet, and nutrition have not been shown to play a role in causing or preventing lupus nephritis. People with kidney disease that progresses may need to talk with a health care provider about changes they may n...
2. `medquad_ans_0000159-8` (score=0.9253) - - Lupus nephritis is kidney inflammation caused by systemic lupus erythematosus (SLE or lupus). - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edema. - Lupus nephritis is diagnosed...
3. `medquad_ans_0000149-8` (score=0.0163) - - Henoch-Schnlein purpura (HSP) is a disease that causes small blood vessels in the body to become inflamed and leak. - HSP is caused by an abnormal immune system response in which the bodys immune system attacks the ...
4. `medquad_ans_0000161-6` (score=0.0112) - Treating nephrotic syndrome includes addressing the underlying cause as well as taking steps to reduce high blood pressure, edema, high cholesterol, and the risks of infection. Treatment usually includes medications a...
5. `medquad_ans_0000159-4__dup2` (score=0.0088) - Lupus nephritis is diagnosed through urine and blood tests and a kidney biopsy: - Urinalysis. Urinalysis is testing of a urine sample. The urine sample is collected in a special container in a health care providers of...

**Answer**:

For lupus nephritis, treatment involves medications that suppress the immune system to prevent further damage to the kidneys. Standard treatment includes a corticosteroid, usually prednisone, to reduce inflammation in the kidneys, and an immunosuppressive medication such as cyclophosphamide or mycophenolate mofetil, typically used in combination with prednisone [medquad_ans_0000159-8]. If lupus nephritis is causing high blood pressure, medications to lower blood pressure may be needed, as this can significantly slow the progression of kidney disease [medquad_ans_0000159-8]. In some cases, treatment is effective in completely or partially controlling lupus nephritis, resulting in few or no further complications, although 10 to 30 percent of people may still develop kidney failure despite treatment [medquad_ans_0000159-

**Supported citations**: `medquad_ans_0000159-8`
**Unsupported citations**: _none_

### Example 145 (`0000114-6`)

**Question**: What are the treatments for Restless Legs Syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.544, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0000233-2__dup1` (score=0.9969) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...
2. `medquad_ans_0000246-2__dup2` (score=0.9927) - For those with mild to moderate symptoms, many physicians suggest certain lifestyle changes and activities to reduce or eliminate symptoms. Decreased use of caffeine, alcohol, and tobacco may provide some relief. Phys...
3. `medquad_ans_0000242-2__dup1` (score=0.9899) - There is currently no effective treatment for PSP, although scientists are searching for better ways to manage the disease. In some patients the slowness, stiffness, and balance problems of PSP may respond to antipark...
4. `medquad_ans_0000188-2__dup1` (score=0.9891) - MJD is incurable, but some symptoms of the disease can be treated. For those individuals who show parkinsonian features, levodopa therapy can help for many years. Treatment with antispasmodic drugs, such as baclofen, ...
5. `medquad_ans_0000114-6` (score=0.9709) - Restless legs syndrome (RLS) has no cure. If a condition or medicine triggers RLS, it may go away or get better if the trigger is relieved or stopped. RLS can be treated. The goals of treatment are to: Prevent or reli...

**Answer**:

For those with mild to moderate symptoms, lifestyle changes and activities such as decreased use of caffeine, alcohol, and tobacco may provide some relief. Physicians may suggest supplements to correct deficiencies in iron, folate, and magnesium. Taking a hot bath, massaging the legs, or using a heating pad or ice pack can help relieve symptoms in some patients. Medications to treat RLS include dopaminergics, benzodiazepines (central nervous system depressants), opioids, and anticonvulsants. The drugs ropinirole, pramipexole, gabapentin enacarbil, and rotigotine have been approved by the U.S. Food and Drug Administration for treating moderate to severe RLS. The Relaxis pad, which provides 30 minutes of vibrations (counterstimulation) that ramp off after 30 minutes, has also been approved by the FDA [medquad_ans_000024

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 146 (`0002573-1`)

**Question**: What is (are) Glioblastoma ?

**Scores**: context_relevance=0.500, groundedness=0.010, answer_relevance=0.500

**Top docs**:

1. `medquad_ans_0000087-1__dup2` (score=0.9838) - Tumors are abnormal growths in your body. They are made up of extra cells. Normally, cells grow and divide to form new cells as your body needs them. When cells grow old, they die, and new cells take their place. Some...
2. `medquad_ans_0000063-16` (score=0.9503) - When the cancer spreads from its original tumor location in the skin to another part of the body such as the brain, it is called metastatic skin cancer. It is not the same as a cancer that started in the brain (brain ...
3. `medquad_ans_0002574-1` (score=0.9311) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
4. `medquad_ans_0000127-1__dup3` (score=0.8372) - A brain tumor is a growth of abnormal cells in the tissues of the brain. Brain tumors can be benign, with no cancer cells, or malignant, with cancer cells that grow quickly. Some are primary brain tumors, which start ...
5. `medquad_ans_0000008-12` (score=0.2350) - When cancer spreads from its original location in the breast to another part of the body such as the brain, it is called metastatic breast cancer, not brain cancer. Doctors sometimes call this "distant" disease.

**Answer**:

The provided context does not contain information about glioblastoma. Therefore, it is not possible to answer the question based on the given text. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 147 (`0005496-3`)

**Question**: What causes Schizencephaly ?

**Scores**: context_relevance=1.000, groundedness=0.790, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000251-4__dup1` (score=0.0004) - The NINDS conducts and supports a wide range of studies that explore the mechanisms of normal brain development. The knowledge gained from these fundamental studies provides the foundation for understanding how to pre...
2. `medquad_ans_0004039-3` (score=0.0004) - What causes microhydranencephaly? In many cases, the exact, underlying cause of microhydranencephaly is unknown. There are reports of families in which the condition is caused by changes (mutations) in the NDE1 gene. ...
3. `medquad_ans_0000126-1__dup3` (score=0.0002) - Most brain malformations begin long before a baby is born. Something damages the developing nervous system or causes it to develop abnormally. Sometimes it's a genetic problem. In other cases, exposure to certain medi...
4. `medquad_ans_0000189-4__dup1` (score=0.0001) - The National Institute of Neurological Disorders and Stroke (NINDS) and other institutes of the National Institutes of Health (NIH) conduct research related to megalencephaly in laboratories at the NIH and also suppor...
5. `medquad_ans_0000064-1__dup1` (score=0.0001) - Cephalic disorders are congenital conditions that stem from damage to or abnormal development of the budding nervous system. Most cephalic disorders are caused by a disturbance that occurs very early in the developmen...

**Answer**:

The provided context does not specify the exact causes of schizencephaly. It mentions that the NINDS conducts studies on normal brain development to understand how to prevent or treat developmental brain defects such as schizencephaly, but it does not identify specific causes [medquad_ans_0000251-4__dup1]. Therefore, based on the given information, the cause of schizencephaly cannot be determined.

**Supported citations**: `medquad_ans_0000251-4__dup1`
**Unsupported citations**: _none_

### Example 148 (`0000052-3`)

**Question**: What are the genetic changes related to amyotrophic lateral sclerosis ?

**Scores**: context_relevance=0.800, groundedness=0.501, answer_relevance=0.880

**Top docs**:

1. `medquad_ans_0000025-3` (score=0.9993) - ALSP is caused by mutations in the CSF1R gene. This gene provides instructions for making a protein called colony stimulating factor 1 receptor (CSF-1 receptor), which is found in the outer membrane of certain types o...
2. `medquad_ans_0000052-3` (score=0.9983) - Mutations in several genes can cause familial ALS and contribute to the development of sporadic ALS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe. Worldwid...
3. `medquad_ans_0000551-3` (score=0.0068) - Mutations in the ALS2 gene cause most cases of juvenile primary lateral sclerosis. This gene provides instructions for making a protein called alsin. Alsin is abundant in motor neurons, but its function is not fully u...
4. `medquad_ans_0000396-1__dup2` (score=0.0002) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...
5. `medquad_ans_0000309-5__dup1` (score=0.0001) - Is genetic testing available for amyotrophic lateral sclerosis? Yes. Clinical genetic testing is currently available for several genes in which mutations are known to cause ALS. Genetic testing on a research basis is ...

**Answer**:

Genetic changes related to amyotrophic lateral sclerosis (ALS) include mutations in several genes, such as C9orf72, SOD1, TARDBP, and FUS. Mutations in the C9orf72 gene account for 30 to 40 percent of familial ALS in the United States and Europe, while SOD1 gene mutations cause 15 to 20 percent of familial ALS worldwide, and TARDBP and FUS gene mutations each account for about 5 percent of cases. The other genes associated with familial ALS contribute to a small proportion of cases. It is estimated that 60 percent of individuals with familial ALS have an identified genetic mutation. The exact mechanisms by which these mutations lead to motor neuron death are unclear, but they are thought to disrupt axon development, impair axonal transport, or prevent the breakdown of toxic substances, leading to motor neuron damage and death [medquad*ans*

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 149 (`0000581-5`)

**Question**: What are the treatments for Laron syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.785, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000581-5` (score=0.9964) - These resources address the diagnosis or management of Laron syndrome: - Children's Hospital of Pittsburgh: Growth Hormone Treatment - Cinncinati Children's Hospital Medical Center: Growth Hormone Therapy - Genetic Te...
2. `medquad_ans_0000004-2__dup1` (score=0.9963) - Treatment for LKS usually consists of medications, such as anticonvulsants and corticosteroids, and speech therapy, which should be started early. A controversial treatment option involves a surgical technique called ...
3. `medquad_ans_0000054-2__dup1` (score=0.9963) - Generally treatment for individuals with BSS focuses on the underlying cause of the disorder. Early treatment with high-dose steroids may be beneficial in many cases. Other treatment is symptomatic and supportive.
4. `medquad_ans_0000033-2__dup1` (score=0.9946) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
5. `medquad_ans_0000208-2__dup1` (score=0.9899) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...

**Answer**:

The provided context does not contain specific information about the treatments for Laron syndrome. While resources related to growth hormone treatment and genetic testing for Laron-type isolated somatotropin defect are mentioned, no details on actual treatments for Laron syndrome are given [medquad_ans_0000581-5]. Therefore, based on the available context, the treatments for Laron syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000581-5`
**Unsupported citations**: _none_

### Example 150 (`0000016-4__dup2`)

**Question**: What are the treatments for Diabetic Retinopathy ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000106-2__dup1` (score=0.9888) - The goal of treating diabetic neuropathy is to prevent further tissue damage and relieve discomfort. The first step is to bring blood sugar levels under control by diet and medication. Another important part of treatm...
2. `medquad_ans_0000055-7` (score=0.9745) - You can help your diabetes retina problems by controlling your - blood glucose - blood pressure - cholesterol and triglycerides, or types of blood fat If your retinopathy still does not improve, then you may need othe...
3. `medquad_ans_0000016-19` (score=0.9650) - Yes. Both treatments are very effective in reducing vision loss. People with proliferative retinopathy can reduce their risk of blindness by 95 percent with timely treatment and appropriate follow-up care. Although bo...
4. `medquad_ans_0000016-15` (score=0.9647) - During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should control their levels of blood s...
5. `medquad_ans_0000016-4__dup2` (score=0.8793) - Preventing Disease Progression During the first three stages of diabetic retinopathy, no treatment is needed, unless you have macular edema. To prevent progression of diabetic retinopathy, people with diabetes should ...

**Answer**:

Treatments for diabetic retinopathy include medicines, laser treatment, and vitrectomy. Medicines, such as injections that block a protein causing abnormal blood vessel growth and fluid leakage, are used to treat macular edema and reduce fluid leakage, allowing the retina to return to normal thickness [medquad_ans_0000055-7]. Laser treatment includes focal laser treatment, which places small laser burns near the macula to slow fluid leakage, and scatter laser treatment, which places thousands of laser burns around new, weak blood vessels to cause them to shrink, thereby reducing the risk of blindness [medquad_ans_0000055-7]. In cases of severe bleeding in the eye, a vitrectomy surgery may be required, where the clouded vitreous gel is removed and replaced with a salt solution [medquad_ans_0000055-7]. Timely treatment for

**Supported citations**: `medquad_ans_0000055-7`
**Unsupported citations**: _none_

### Example 151 (`0001421-3`)

**Question**: What are the treatments for Collagenous colitis ?

**Scores**: context_relevance=0.857, groundedness=0.827, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000116-7` (score=0.9878) - Treatment depends on the severity of symptoms. The gastroenterologist will - review the medications the person is taking - make recommendations to change or stop certain medications - recommend that the person quit sm...
2. `medquad_ans_0000116-8` (score=0.9757) - To help reduce symptoms, a health care provider may recommend the following dietary changes: - avoid foods and drinks that contain caffeine or artificial sugars - drink plenty of liquids to prevent dehydration during ...
3. `medquad_ans_0001421-3` (score=0.9645) - How might collagenous colitis be treated? Treatment for collagenous colitis varies depending on the symptoms and severity in each affected individual. In some cases the condition may resolve on its own (spontaneous re...
4. `medquad_ans_0000163-2__dup1` (score=0.0095) - The chronic inflammatory myopathies cant be cured in most adults but many of the symptoms can be treated. Options include medication, physical therapy, exercise, heat therapy (including microwave and ultrasound), orth...
5. `medquad_ans_0000270-2__dup1` (score=0.0057) - No effective cure currently exists for people with transverse myelitis. Physicians often prescribe corticosteroid therapy during the first few weeks of illness to decrease inflammation. Following initial therapy, the ...

**Answer**:

The provided context does not contain specific information about the treatments for collagenous colitis. While it details treatments for microscopic colitis, including medications such as antidiarrheals, corticosteroids, anti-inflammatory drugs, cholestyramine, antibiotics, immunomodulators, and anti-TNF therapies, as well as dietary changes and, in rare cases, surgery, there is no direct information on treatments for collagenous colitis. Therefore, based on the given context, the treatments for collagenous colitis cannot be determined.

[medquad_ans_0000116-7] [medquad_ans_0000116-8]

**Supported citations**: `medquad_ans_0000116-7`, `medquad_ans_0000116-8`
**Unsupported citations**: _none_

### Example 152 (`0000418-7`)

**Question**: How to prevent Parasites - Trichinellosis (also known as Trichinosis) ?

**Scores**: context_relevance=0.778, groundedness=0.960, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000418-7` (score=0.8627) - - Wash your hands with warm water and soap after handling raw meat. - Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms; homemade jerky and sausage were the cause ...
2. `medquad_ans_0000418-1__dup2` (score=0.0259) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
3. `medquad_ans_0000415-1__dup2` (score=0.0122) - A single-celled parasite called Toxoplasma gondii causes a disease known as toxoplasmosis. While the parasite is found throughout the world, more than 60 million people in the United States may be infected with the To...
4. `medquad_ans_0000015-8__dup1` (score=0.0112) - Angiostrongylus cantonensis Prevention of A. cantonensis infections involves educating persons residing in or traveling to areas where the parasite is found about not ingesting raw or undercooked snails and slugs, fre...
5. `medquad_ans_0000418-2__dup1` (score=0.0075) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...

**Answer**:

To prevent Trichinellosis (also known as Trichinosis), wash your hands with warm water and soap after handling raw meat [medquad_ans_0000418-7]. Curing (salting), drying, smoking, or microwaving meat alone does not consistently kill infective worms, so these methods are not reliable [medquad_ans_0000418-7]. Freeze pork less than 6 inches thick for 20 days at 5°F (-15°C) to kill any worms [medquad_ans_0000418-7]. However, freezing wild game meats may not effectively kill all worms because some worm species that infect wild game animals are freeze-resistant [medquad_ans_0000418-7]. Clean meat grinders thoroughly after each use [medquad_ans_0000418-7]. To

**Supported citations**: `medquad_ans_0000418-7`
**Unsupported citations**: _none_

### Example 153 (`0000536-3`)

**Question**: What causes Autoimmune hepatitis ?

**Scores**: context_relevance=0.750, groundedness=0.961, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000182-3__dup1` (score=0.9962) - A combination of autoimmunity, environmental triggers, and a genetic predisposition can lead to autoimmune hepatitis.
2. `medquad_ans_0000192-9` (score=0.9926) - Some cases of viral hepatitis cannot be attributed to the hepatitis A, B, C, D, or E viruses, or even the less common viruses that can infect the liver, such as cytomegalovirus, Epstein-Barr virus, herpesvirus, parvov...
3. `medquad_ans_0000195-1__dup4` (score=0.6716) - Hepatitis\* B is a virus, or infection, that causes liver disease and inflammation of the liver. Viruses can cause sickness. For example, the flu is caused by a virus. People can pass viruses to each other. Inflammatio...
4. `medquad_ans_0000194-1__dup4` (score=0.2015) - Hepatitis\* A is a virus, or infection, that causes liver disease and inflammation of the liver. Viruses can cause sickness. For example, the flu is caused by a virus. People can pass viruses to each other. Inflammatio...
5. `medquad_ans_0000196-1__dup4` (score=0.0956) - Hepatitis\* C is a virus, or infection, that causes liver disease and inflammation of the liver. Viruses can cause sickness. For example, the flu is caused by a virus. People can pass viruses to each other. Inflammatio...

**Answer**:

Autoimmune hepatitis is caused by a combination of autoimmunity, environmental triggers, and a genetic predisposition [medquad_ans_0000182-3__dup1].

**Supported citations**: `medquad_ans_0000182-3__dup1`
**Unsupported citations**: _none_

### Example 154 (`0000006_3-1`)

**Question**: What is (are) Childhood Astrocytomas ?

**Scores**: context_relevance=0.800, groundedness=0.948, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000006_3-1` (score=0.9580) - Key Points - Childhood astrocytoma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. - Astrocytomas may be benign (not cancer) or malignant (cancer). - The central ...
2. `medquad_ans_0002574-1` (score=0.8300) - Glioma refers to a type of brain tumor that develops from the glial cells, which are specialized cells that surround and support neurons (nerve cells) in the brain. It is generally classified based on which type of gl...
3. `medquad_ans_0000186-1__dup2` (score=0.0463) - Brain tumors are abnormal growths inside the skull. They are among the most common types of childhood cancers. Some are benign tumors, which aren't cancer. They can still be serious. Malignant tumors are cancerous. Ch...
4. `medquad_ans_0004883-1` (score=0.0078) - Pilocytic astrocytoma is an often benign, slow-growing tumor of the brain or spinal cord. The tumor may be in the form of a cyst and usually does not spread to nearby tissues. Symptoms vary depending upon the size and...
5. `medquad_ans_0000006_5-1` (score=0.0029) - Key Points - Childhood brain stem glioma is a disease in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain stem. - Brain tumors may be benign (not cancer) or malignant (cancer). - T...

**Answer**:

Childhood astrocytomas are diseases in which benign (noncancer) or malignant (cancer) cells form in the tissues of the brain. They are tumors that start in star-shaped brain cells called astrocytes, which are a type of glial cell that holds nerve cells in place, brings food and oxygen to them, and helps protect them from disease. Astrocytomas are the most common type of glioma diagnosed in children and can form anywhere in the central nervous system (brain and spinal cord) [medquad_ans_0000006_3-1].

**Supported citations**: `medquad_ans_0000006_3-1`
**Unsupported citations**: _none_

### Example 155 (`0000013_3-1__dup3`)

**Question**: What is (are) Atypical Chronic Myelogenous Leukemia ?

**Scores**: context_relevance=0.571, groundedness=0.006, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000634-1__dup2` (score=0.9725) - Your bone marrow is the spongy tissue inside some of your bones, such as your hip and thigh bones. It contains immature cells, called stem cells. The stem cells can develop into the red blood cells that carry oxygen t...
2. `medquad_ans_0000040-9` (score=0.1082) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
3. `medquad_ans_0000040-10` (score=0.0081) - Acute leukemia gets worse quickly. In chronic leukemia, symptoms develop gradually and are generally not as severe as in acute leukemia.
4. `medquad_ans_0000040-15` (score=0.0053) - There are no standard or over-the-counter tests for leukemia. Your doctor can request lab analyses for leukemia that include blood tests that check the levels and types of blood cells and look for changes in the shape...
5. `medquad_ans_0000201-1__dup2` (score=0.0019) - Leukemia is cancer of the white blood cells. White blood cells help your body fight infection. Your blood cells form in your bone marrow. In leukemia, the bone marrow produces abnormal white blood cells. These cells c...

**Answer**:

The provided context does not contain information about atypical chronic myelogenous leukemia. Therefore, it is not possible to answer the question based on the given text. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 156 (`0000996-3`)

**Question**: What are the genetic changes related to trisomy 13 ?

**Scores**: context_relevance=0.125, groundedness=0.000, answer_relevance=0.625

**Top docs**:

1. `medquad_ans_0000770-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0001083-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000784-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000349-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000101-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain any information about genetic changes related to trisomy 13. Therefore, the question cannot be answered based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 157 (`0000731-5`)

**Question**: What are the treatments for Niemann-Pick disease ?

**Scores**: context_relevance=0.714, groundedness=0.910, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000731-5` (score=0.9866) - These resources address the diagnosis or management of Niemann-Pick disease: - Baby's First Test - Gene Review: Gene Review: Acid Sphingomyelinase Deficiency - Gene Review: Gene Review: Niemann-Pick Disease Type C - G...
2. `medquad_ans_0000218-2__dup1` (score=0.9733) - There is currently no cure for Niemann-Pick disease. Treatment is supportive. Children usually die from infection or progressive neurological loss. There is currently no effective treatment for persons with type A. Bo...
3. `medquad_ans_0004415-3` (score=0.8279) - How might Niemann-Pick disease type A be treated? There is no specific treatment for this disease. Supportive care from the following specialists may be helpful for managing the symptoms: A pulmonologist for respirato...
4. `medquad_ans_0000130-4__dup1` (score=0.0446) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.
5. `medquad_ans_0000168-4__dup1` (score=0.0100) - The NINDS supports research on neuromuscular disorders such as KSS. The goals of this research are to increase understanding of these disorders, and to find ways to prevent, treat, and, ultimately, cure them. The most...

**Answer**:

There is currently no cure for Niemann-Pick disease. Treatment is supportive and may include care from specialists such as pulmonologists for respiratory problems, cardiologists for heart problems, liver and spleen specialists, nutritionists, physical therapists, and gastroenterologists for gastrointestinal issues. Learning specialists may also be involved to address developmental concerns [medquad_ans_0000218-2__dup1]. Bone marrow transplantation has been attempted in a few individuals with type B, and enzyme replacement and gene therapies might be helpful for those with type B [medquad_ans_0000218-2__dup1]. However, restricting one's diet does not prevent the buildup of lipids in cells and tissues [medquad_ans_0000218-2__dup1]. There is no specific treatment for Niemann-Pick disease type A [medquad_ans_0004415-

**Supported citations**: `medquad_ans_0000218-2__dup1`
**Unsupported citations**: _none_

### Example 158 (`0001714-1`)

**Question**: What are the symptoms of Dandy-Walker malformation with postaxial polydactyly ?

**Scores**: context_relevance=1.000, groundedness=0.541, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001714-1` (score=0.9994) - What are the signs and symptoms of Dandy-Walker malformation with postaxial polydactyly? The Human Phenotype Ontology provides the following list of signs and symptoms for Dandy-Walker malformation with postaxial poly...
2. `medquad_ans_0000270-2` (score=0.0001) - Dandy-Walker malformation is estimated to affect 1 in 10,000 to 30,000 newborns.
3. `medquad_ans_0000270-4` (score=0.0001) - Most cases of Dandy-Walker malformation are sporadic, which means they occur in people with no history of the disorder in their family. A small percentage of cases seem to run in families; however, Dandy-Walker malfor...
4. `medquad_ans_0001959-2` (score=0.0001) - What are the signs and symptoms of Dyggve-Melchior-Clausen syndrome? Affected newborns may be small at birth, but otherwise appear normal. Skeletal findings are often recognized first between 1 and 18 months. With age...
5. `medquad_ans_0004948-1` (score=0.0001) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...

**Answer**:

The symptoms of Dandy-Walker malformation with postaxial polydactyly include Dandy-Walker malformation (90%), postaxial hand polydactyly (90%), agenesis of cerebellar vermis, aortic valve stenosis, autosomal recessive inheritance, chorioretinal atrophy, cranial nerve paralysis, depressed nasal bridge, dilated fourth ventricle, dolichocephaly, elevated imprint of the transverse sinuses, frontal bossing, hydrocephalus, low-set ears, macrocephaly, microretrognathia, nystagmus, partial absence of cerebellar vermis, patent ductus arteriosus, posterior embryotoxon, posterior fossa cyst at the fourth ventricle, small palpebral fissure, thinning and bulging of the posterior fossa bones, truncal ataxia, wide nasal bridge [med

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 159 (`0003789-3`)

**Question**: What causes Mantle cell lymphoma ?

**Scores**: context_relevance=0.800, groundedness=0.702, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000130-2__dup2` (score=0.7762) - Many factors can cause thrombocytopenia (a low platelet count). The condition can be inherited or acquired. "Inherited" means your parents pass the gene for the condition to you. "Acquired" means you aren't born with ...
2. `medquad_ans_0000136-4__dup2` (score=0.0278) - Vasculitis occurs if your immune system attacks your blood vessels by mistake. What causes this to happen isn't fully known. A recent or chronic (ongoing) infection may prompt the attack. Your body also may attack its...
3. `medquad_ans_0000104-3__dup2` (score=0.0163) - Pleurisy Many conditions can cause pleurisy. Viral infections are likely the most common cause. Other causes of pleurisy include: Bacterial infections, such as pneumonia (nu-MO-ne-ah) and tuberculosis, and infections ...
4. `medquad_ans_0000129-3__dup2` (score=0.0002) - Primary Thrombocythemia In this condition, faulty stem cells in the bone marrow make too many platelets. What causes this to happen usually isn't known. When this process occurs without other blood cell disorders, it'...
5. `medquad_ans_0000040-12` (score=0.0001) - Myeloma and lymphoma are other types of blood cancers. Both are common among older adults and occur more often in men than women. Myeloma affects plasma cells, a type of white blood cells typically found in the bone m...

**Answer**:

The provided context does not mention Mantle cell lymphoma or its causes. Therefore, based on the given information, it is not possible to answer the question. [medquad_ans_0000130-2__dup2]

**Supported citations**: `medquad_ans_0000130-2__dup2`
**Unsupported citations**: _none_

### Example 160 (`0000589-4`)

**Question**: Is Legg-Calv-Perthes disease inherited ?

**Scores**: context_relevance=0.750, groundedness=0.694, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0006597-4` (score=0.9332) - Can 48,XXYY syndrome be inherited?
2. `medquad_ans_0002200-3` (score=0.0400) - How is Fabry disease inherited? Fabry disease is inherited in an X-linked pattern, which means that the gene that causes the condition is located on the X chromosome. In males (who have only one X chromosome), one mut...
3. `medquad_ans_0000714-4` (score=0.0026) - Nail-patella syndrome is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation f...
4. `medquad_ans_0000044-4` (score=0.0015) - Alport syndrome can have different inheritance patterns. About 80 percent of cases are caused by mutations in the COL4A5 gene and are inherited in an X-linked pattern. This gene is located on the X chromosome, which i...
5. `medquad_ans_0002382-4` (score=0.0002) - How is Fine-Lubinsky syndrome inherited? Almost all people reported to have FineLubinsky syndrome (FLS) have been the only affected people in their families (these cases were sporadic). There has been one report of an...

**Answer**:

The provided context does not contain information about the inheritance of Legg-Calv-Perthes disease. Therefore, it is not possible to determine based on the given text whether Legg-Calv-Perthes disease is inherited.

Answer: Insufficient context [medquad_ans_0006597-4].

**Supported citations**: `medquad_ans_0006597-4`
**Unsupported citations**: _none_

### Example 161 (`0000192-1__dup1`)

**Question**: What is (are) Metachromatic Leukodystrophy ?

**Scores**: context_relevance=0.800, groundedness=0.523, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000192-1__dup1` (score=0.9699) - Metachromatic leukodystrophy (MLD) is one of a group of genetic disorders called the leukodystrophies, which are characterized by the toxic buildup of lipids (fatty materials such as oils and waxes) and other storage ...
2. `medquad_ans_0000647-1` (score=0.2649) - Metachromatic leukodystrophy is an inherited disorder characterized by the accumulation of fats called sulfatides in cells. This accumulation especially affects cells in the nervous system that produce myelin, the sub...
3. `medquad_ans_0003566-1` (score=0.1022) - A leukodystrophy is a type of rare genetic disorder that affects the brain, spinal cord, and other nerves in the body. It is caused by destruction of the white matter of the brain. The white matter degrades due to def...
4. `medquad_ans_0000181-1__dup2` (score=0.0983) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...
5. `medquad_ans_0000555-1__dup1` (score=0.0017) - The leukodystrophies are rare diseases that affect the cells of the brain. Specifically, the diseases affect the myelin sheath, the material that surrounds and protects nerve cells. Damage to this sheath slows down or...

**Answer**:

Metachromatic leukodystrophy (MLD) is an inherited disorder characterized by the accumulation of fats called sulfatides in cells, particularly in the nervous system cells that produce myelin. This accumulation leads to progressive destruction of white matter (leukodystrophy) throughout the nervous system, including in the brain and spinal cord (the central nervous system). MLD is one of a group of genetic disorders called leukodystrophies, which are marked by the toxic buildup of lipids and other storage materials in cells of the white matter of the central nervous system and peripheral nerves. The buildup impairs the growth or development of the myelin sheath, the fatty covering that insulates nerve fibers. MLD is caused by a deficiency of the enzyme arylsulfatase A, which is necessary to break down (metabolize) lipids. It affects both males and females and has three characteristic forms: late infantile, juvenile

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 162 (`0005135-1`)

**Question**: What are the symptoms of Pseudodiastrophic dysplasia ?

**Scores**: context_relevance=1.000, groundedness=0.969, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005135-1` (score=0.9994) - What are the signs and symptoms of Pseudodiastrophic dysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Pseudodiastrophic dysplasia. If the information is available, the tabl...
2. `medquad_ans_0000094-1__dup1` (score=0.0001) - What are the signs and symptoms of Acromesomelic dysplasia Maroteaux type? The Human Phenotype Ontology provides the following list of signs and symptoms for Acromesomelic dysplasia Maroteaux type. If the information ...
3. `medquad_ans_0005819-2` (score=0.0001) - What are the signs and symptoms of Spondylocostal dysostosis 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocostal dysostosis 1. If the information is available, the tabl...
4. `medquad_ans_0005831-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia Shohat type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia Shohat type. If the ...
5. `medquad_ans_0005820-2` (score=0.0001) - What are the signs and symptoms of Spondylocostal dysostosis 2? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondylocostal dysostosis 2. If the information is available, the tabl...

**Answer**:

The symptoms of Pseudodiastrophic dysplasia include abnormal form of the vertebral bodies (90%), elbow dislocation (90%), hypoplasia of the zygomatic bone (90%), scoliosis (90%), omphalocele (7.5%), hypoplasia of midface, hypoplasia of the odontoid process, lumbar hyperlordosis, malar flattening, phalangeal dislocation, platyspondyly, rhizomelia, severe short stature, talipes equinovarus, tongue-like lumbar vertebral deformities [medquad_ans_0005135-1].

**Supported citations**: `medquad_ans_0005135-1`
**Unsupported citations**: _none_

### Example 163 (`0000251-1`)

**Question**: What is (are) cri-du-chat syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0001664-1` (score=0.8285) - Cri du chat syndrome, also known as 5p- (5p minus) syndrome or cat cry syndrome, is a genetic condition that is caused by the deletion of genetic material on the small arm (the p arm) of chromosome 5. Infants with thi...
2. `medquad_ans_0001383-1` (score=0.8260) - Cluttering is a disorder that affects the way a person speaks. It is characterized by a rapid speaking rate and inability to maintain normally expected sound, syllable, phrase, and pausing patterns while speaking. Oth...
3. `medquad_ans_0000251-1` (score=0.8085) - Cri-du-chat (cat's cry) syndrome, also known as 5p- (5p minus) syndrome, is a chromosomal condition that results when a piece of chromosome 5 is missing. Infants with this condition often have a high-pitched cry that ...
4. `medquad_ans_0000247-1__dup2` (score=0.0306) - Rett syndrome is a childhood neurodevelopmental disorder that affects females almost exclusively. The child generally appears to grow and develop normally, before symptoms begin. Loss of muscle tone is usually the fir...
5. `medquad_ans_0006355-1` (score=0.0072) - Vici syndrome is a multisystem disorder characterized by agenesis (failure to develop) of the corpus callosum, cataracts , hypopigmentation of the eyes and hair, cardiomyopathy, and combined immunodeficiency. Hearing ...

**Answer**:

Cri-du-chat syndrome, also known as 5p- (5p minus) syndrome or cat cry syndrome, is a genetic condition caused by the deletion of genetic material on the small arm (the p arm) of chromosome 5. Infants with this condition often have a high-pitched cry that sounds like that of a cat. The disorder is characterized by intellectual disability and delayed development, small head size (microcephaly), low birth weight, weak muscle tone (hypotonia) in infancy, and distinctive facial features such as widely set eyes (hypertelorism), low-set ears, a small jaw, and a rounded face. Some children with cri-du-chat syndrome are born with a heart defect. Most cases are not inherited [medquad_ans_0001664-1][medquad_ans_0000251-1].

**Supported citations**: `medquad_ans_0001664-1`, `medquad_ans_0000251-1`
**Unsupported citations**: _none_

### Example 164 (`0000719-2__dup1`)

**Question**: What are the symptoms of Bilateral frontoparietal polymicrogyria ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000719-2__dup1` (score=0.9992) - What are the signs and symptoms of Bilateral frontoparietal polymicrogyria? The signs and symptoms of bilateral frontoparietal polymicrogyria vary but may include: Moderate to severe intellectual disability Developmen...
2. `medquad_ans_0003881-2` (score=0.1891) - What are the symptoms of polymicrogyria? A wide variety of symptoms may be observed in people with polymicrogyria, including: Cognitive deficits Epilepsy Paralysis of the face, throat, and tongue Difficulty with speec...
3. `medquad_ans_0000722-2__dup1` (score=0.0003) - What are the signs and symptoms of Bilateral perisylvian polymicrogyria? The signs and symptoms of bilateral perisylvian polymicrogyria (BPP) vary but may include: Partial paralysis of muscles on both sides of the fac...
4. `medquad_ans_0000718-1__dup1` (score=0.0001) - Bilateral frontal polymicrogyria is one of the rarest subtypes of polymicrogyria. It is a symmetric and bilateral form (in both brain hemispheres) that only involves the frontal lobes without including the area locate...
5. `medquad_ans_0000722-1__dup1` (score=0.0001) - Bilateral perisylvian polymicrogyria (BPP) is a rare neurological disorder that affects the cerebral cortex (the outer surface of the brain). Signs and symptoms include partial paralysis of muscles on both sides of th...

**Answer**:

The symptoms of Bilateral frontoparietal polymicrogyria may include: moderate to severe intellectual disability, developmental delay, seizures, dysconjugate gaze, ataxia, strabismus, increased muscle tone, finger dysmetria, ankle clonus, Babinski sign, broad-based gait, hyperreflexia, hypertonia, hypoplasia of the brainstem, nystagmus, and truncal ataxia [medquad_ans_0000719-2__dup1].

**Supported citations**: `medquad_ans_0000719-2__dup1`
**Unsupported citations**: _none_

### Example 165 (`0003414-3`)

**Question**: What causes Konigsmark Knox Hussels syndrome ?

**Scores**: context_relevance=0.167, groundedness=0.044, answer_relevance=0.367

**Top docs**:

1. `medquad_ans_0000149-2__dup2` (score=0.0047) - Henoch-Schnlein purpura is caused by an abnormal immune system response in which the bodys immune system attacks the bodys own cells and organs. Usually, the immune system makes antibodies, or proteins, to protect the...
2. `medquad_ans_0000542-1__dup2` (score=0.0001) - Klinefelter syndrome (KS) is a condition that occurs in men who have an extra X chromosome. The syndrome can affect different stages of physical, language, and social development. The most common symptom is infertilit...
3. `medquad_ans_0006110-1` (score=0.0001) - Tietz syndrome is a rare condition that affects the development of melanocytes, the cells in our body that produce and contain melanin (the pigment that gives color to skin, hair, and eyes). Signs and symptoms of this...
4. `medquad_ans_0003384-1` (score=0.0001) - Keutel syndrome is an inherited condition characterized by cartilage calcification in the ears, nose, larnyx, trachea (voice box), and ribs; pulmonary artery stenoses; brachytelephalangism (short fingers and nails tha...
5. `medquad_ans_0001143-1` (score=0.0001) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...

**Answer**:

The provided context does not mention "Konigsmark Knox Hussels syndrome" or any similar condition. Therefore, the cause cannot be determined from the given information.

Note: It is possible that "Konigsmark Knox Hussels syndrome" is a misspelling or misstatement of a known condition such as Henoch-Schönlein purpura (HSP), which is discussed in the context. However, the query specifically asks about "Konigsmark Knox Hussels syndrome," and no such condition is referenced in the provided documents.

Answer: Insufficient context to determine the cause of Konigsmark Knox Hussels syndrome. [no relevant doc_id]

**Supported citations**: _none_
**Unsupported citations**: `no relevant doc_id`

### Example 166 (`0004584-2`)

**Question**: What are the symptoms of Orofaciodigital syndrome 1 ?

**Scores**: context_relevance=1.000, groundedness=0.500, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004584-2` (score=0.9992) - What are the signs and symptoms of Orofaciodigital syndrome 1? Oral features of OFD1 may include a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormali...
2. `medquad_ans_0002679-2` (score=0.0001) - What are the signs and symptoms of Greig cephalopolysyndactyly syndrome? The symptoms of Greig cephalopolysyndactyly syndrome (GCPS) are highly variable, ranging from mild to severe. People with this condition typical...
3. `medquad_ans_0004589-2` (score=0.0001) - What are the signs and symptoms of Orofaciodigital syndrome 2? Although the signs and symptoms that occur in people with orofaciodigital syndrome type 2 may vary, the following findings may be present:Facial findings ...
4. `medquad_ans_0000943-1__dup1` (score=0.0001) - What are the signs and symptoms of Cardiocranial syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Cardiocranial syndrome. If the information is available, the table below in...
5. `medquad_ans_0004102-2` (score=0.0001) - What are the signs and symptoms of Moebius syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Moebius syndrome. If the information is available, the table below includes how o...

**Answer**:

The symptoms of Orofaciodigital syndrome 1 include oral features such as a split (lobed) tongue, benign tumors of the tongue, cleft palate, hypodontia (missing teeth), or other dental abnormalities. Facial features may include hypertelorism (increased width between the eyes), a small nose, micrognathia (small jaw), and other facial abnormalities. Finger and toe abnormalities may include shortness (brachydactyly), webbing or joining (syndactyly), abnormally curved fingers (clinodactyly), or other malformations. Brain abnormalities such as cysts and kidney problems like polycystic kidney disease may also occur. About half of individuals have some degree of learning disability, usually mild. Additionally, symptoms include bifid tongue (90%), broad alveolar ridges (90%), cleft upper lip (90%), frontal bossing (90%), hypert

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 167 (`0000863-1__dup1`)

**Question**: Do you have information about Surgery

**Scores**: context_relevance=0.667, groundedness=0.895, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000863-1__dup1` (score=0.9972) - Summary : There are many reasons to have surgery. Some operations can relieve or prevent pain. Others can reduce a symptom of a problem or improve some body function. Some surgeries are done to find a problem. For exa...
2. `medquad_ans_0000718-1__dup2` (score=0.9967) - Summary : Surgeons can reshape the appearance of body parts through cosmetic surgery. Some of the most common body parts people want to improve through surgery include - Breasts: Increase or reduce the size of breasts...
3. `medquad_ans_0000969-1__dup2` (score=0.9965) - Summary : Weight loss surgery helps people with extreme obesity to lose weight. It may be an option if you cannot lose weight through diet and exercise or have serious health problems caused by obesity. There are diff...
4. `medquad_ans_0000670-1__dup1` (score=0.9964) - Summary : An ostomy is surgery to create an opening (stoma) from an area inside the body to the outside. It treats certain diseases of the digestive or urinary systems. It can be permanent, when an organ must be remov...
5. `medquad_ans_0000584-1__dup2` (score=0.5846) - Summary : A mastectomy is surgery to remove a breast or part of a breast. It is usually done to treat breast cancer. Types of breast surgery include - Total (simple) mastectomy - removal of breast tissue and nipple - ...

**Answer**:

Yes, there is information about surgery. Surgery can relieve or prevent pain, reduce symptoms, improve body function, or identify problems such as through a biopsy. Some surgeries, like heart surgery, can save a life. Laparoscopic surgery allows for smaller incisions using a thin tube with a camera and small tools. After surgery, there are risks including infection, bleeding, reaction to anesthesia, or accidental injury, and pain is almost always present [medquad_ans_0000863-1__dup1]. Cosmetic surgery can reshape body parts such as breasts, ears, eyes, face, hair, nose, and tummy [medquad_ans_0000718-1__dup2]. Weight loss surgery helps individuals with extreme obesity who cannot lose weight through diet and exercise or have obesity-related health issues; it often limits food intake and affects digestion and nutrient absorption, with risks including infections, hernias, and blood

**Supported citations**: `medquad_ans_0000863-1__dup1`, `medquad_ans_0000718-1__dup2`
**Unsupported citations**: _none_

### Example 168 (`0002935-1`)

**Question**: What is (are) Human T-cell leukemia virus type 2 ?

**Scores**: context_relevance=0.875, groundedness=0.992, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0002935-1` (score=0.9831) - Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affect the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affe...
2. `medquad_ans_0000040-9` (score=0.9651) - Leukemia is a cancer of the blood cells. It usually begins in the bone marrow where blood cells are formed. In leukemia, the bone marrow produces abnormal white blood cells. Over time, as the number of abnormal white ...
3. `medquad_ans_0001053-1` (score=0.0204) - X-linked immunodeficiency with magnesium defect, Epstein-Barr virus infection, and neoplasia (typically known by the acronym XMEN) is a disorder that affects the immune system in males. In XMEN, certain types of immun...
4. `medquad_ans_0000244-1__dup3` (score=0.0047) - Content on this page was developed during the 2009-2010 H1N1 pandemic and has not been updated. - The H1N1 virus that caused that pandemic is now a regular human flu virus and continues to circulate seasonally worldwi...
5. `medquad_ans_0000040-24` (score=0.0037) - Researchers are studying various drugs, immunotherapies, and other types of treatments. Because leukemia is a complicated disease, researchers are also studying the effectiveness of using combinations of treatments. F...

**Answer**:

Human T-cell leukemia virus, type 2 (HTLV-2) is a retroviral infection that affects the T cells (a type of white blood cell). Although this virus generally causes no signs or symptoms, scientists suspect that some affected people may later develop neurological problems and/or chronic lung infections. HTLV-2 is spread by blood transfusions, sexual contact, and sharing needles. It can also be spread from mother to child during birth or breast-feeding. There is no cure or treatment for HTLV-2 and it is considered a lifelong condition; however, most infected people remain asymptomatic (show no symptoms) throughout life [medquad_ans_0002935-1].

**Supported citations**: `medquad_ans_0002935-1`
**Unsupported citations**: _none_

### Example 169 (`0000522-1__dup1`)

**Question**: What is (are) Interstitial Lung Diseases ?

**Scores**: context_relevance=0.833, groundedness=0.543, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000567-1__dup2` (score=0.9791) - When you breathe, your lungs take in oxygen from the air and deliver it to the bloodstream. The cells in your body need oxygen to work and grow. During a normal day, you breathe nearly 25,000 times. People with lung d...
2. `medquad_ans_0000043-10` (score=0.8341) - Lung cancer occurs when malignant tumors form in the tissue of the lung. The lungs are a pair of sponge-like organs. The right lung has three sections, called lobes, and is larger than the left lung, which has two lobes.
3. `medquad_ans_0000323-1__dup2` (score=0.6319) - Emphysema is a type of COPD involving damage to the air sacs (alveoli) in the lungs. As a result, your body does not get the oxygen it needs. Emphysema makes it hard to catch your breath. You may also have a chronic c...
4. `medquad_ans_0000522-1__dup1` (score=0.5592) - Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing...
5. `medquad_ans_0000232-1__dup2` (score=0.1192) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...

**Answer**:

Interstitial lung disease is the name for a large group of diseases that inflame or scar the lungs. The inflammation and scarring make it hard to get enough oxygen. The scarring is called pulmonary fibrosis. Breathing in dust or other particles in the air is responsible for some types of interstitial lung diseases. Specific types include - Black lung disease among coal miners, from inhaling coal dust - Farmer's lung, from inhaling farm dust - Asbestosis, from inhaling asbestos fibers - Siderosis, from inhaling iron from mines or welding fumes - Silicosis, from inhaling silica dust. Other causes include autoimmune diseases or occupational exposures to molds, gases, or fumes. Some types of interstitial lung disease have no known cause. Treatment depends on the type of exposure and the stage of the disease. It may involve medicines, oxygen therapy, or a lung transplant in severe cases [medquad_ans_0000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 170 (`0000645-4`)

**Question**: Is Melnick-Needles syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000045-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000298-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000998-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000349-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000898-4` (score=0.9989) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention Melnick-Needles syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether Melnick-Needles syndrome is inherited.

[No supporting source]

**Supported citations**: _none_
**Unsupported citations**: `No supporting source`

### Example 171 (`0006138-1`)

**Question**: What is (are) Tracheobronchomalacia ?

**Scores**: context_relevance=0.750, groundedness=0.538, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000908-1__dup2` (score=0.9954) - Your trachea, or windpipe, is one part of your airway system. Airways are pipes that carry oxygen-rich air to your lungs. They also carry carbon dioxide, a waste gas, out of your lungs. When you inhale, air travels fr...
2. `medquad_ans_0000133-1__dup2` (score=0.9724) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
3. `medquad_ans_0006138-1` (score=0.9703) - Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak. This can cause the airway to become narrow or collapse. There are two forms of ...
4. `medquad_ans_0006140-1` (score=0.0106) - Tracheobronchopathia osteoplastica (TO) is a rare condition of the large airways. It is characterized by the presence of multiple growths (nodules) made of bone and cartilage tissue, in the submucosa of the tracheobro...
5. `medquad_ans_0006137-1` (score=0.0010) - Tracheal agenesis is a rare birth defect in which the trachea (windpipe) is completely absent (agenesis) or significantly underdeveloped (atresia). Signs and symptoms include polyhydramnios during pregnancy and respir...

**Answer**:

Tracheobronchomalacia (TBM) is a rare condition that occurs when the walls of the airway (specifically the trachea and bronchi) are weak, causing the airway to become narrow or collapse. There are two forms of TBM: a congenital form (called primary TBM) that typically develops during infancy or early childhood and an acquired form (called secondary TBM) that is usually seen in adults. Some affected people may initially have no signs or symptoms, but the condition is typically progressive and most people will eventually develop characteristic features such as shortness of breath, cough, sputum retention (inability to clear mucus from the respiratory tract), and wheezing or stridor with breathing. Most cases of primary TBM are caused by genetic conditions that weaken the walls of the airway, while the secondary form often occurs due to trauma, chronic inflammation, and/or prolonged compression of the airways

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 172 (`0006030-1`)

**Question**: What is (are) Temporomandibular ankylosis ?

**Scores**: context_relevance=0.800, groundedness=0.993, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000880-1__dup2` (score=0.9909) - The temporomandibular joint (TMJ) connects your jaw to the side of your head. When it works well, it enables you to talk, chew, and yawn. For people with TMJ dysfunction, problems with the joint and muscles around it ...
2. `medquad_ans_0006030-1` (score=0.4411) - Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people m...
3. `medquad_ans_0000922-1` (score=0.0003) - SOST-related sclerosing bone dysplasia is a disorder of bone development characterized by excessive bone formation (hyperostosis). As a result of hyperostosis, bones throughout the body are denser and wider than norma...
4. `medquad_ans_0000582-1` (score=0.0003) - Larsen syndrome is a disorder that affects the development of bones throughout the body. The signs and symptoms of Larsen syndrome vary widely even within the same family. Affected individuals are usually born with in...
5. `medquad_ans_0000763-1` (score=0.0002) - Otospondylomegaepiphyseal dysplasia (OSMED) is a skeletal disorder characterized by skeletal abnormalities, distinctive facial features, and severe hearing loss. The condition involves the ears (oto-), affects the bon...

**Answer**:

Temporomandibular ankylosis is a condition that occurs when the temporomandibular joint (the joint that connects the jaw to the side of the head) becomes fused by bony or fibrous tissue. As a result, affected people may experience pain, speech impairment, and difficulty chewing and swallowing. It can interfere with nutrition, oral hygiene, and the normal growth of the face and/or jaw. Although the condition can be diagnosed in people of all ages, it generally occurs during the first and second decades of life. Temporomandibular ankylosis is most commonly caused by trauma or infection; it may also be associated with certain conditions such as ankylosing spondylitis, rheumatoid arthritis, or psoriasis. The condition is typically treated surgically [medquad_ans_0006030-1].

**Supported citations**: `medquad_ans_0006030-1`
**Unsupported citations**: _none_

### Example 173 (`0000554-4`)

**Question**: Is Kallmann syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.000, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000205-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000371-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000770-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000663-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000883-4` (score=0.9979) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...

**Answer**:

The provided context does not contain information about the inheritance pattern of Kallmann syndrome. Therefore, it is not possible to determine based on the given text whether Kallmann syndrome is inherited [doc_id].

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 174 (`0005967-2`)

**Question**: What are the symptoms of Syndactyly type 3 ?

**Scores**: context_relevance=1.000, groundedness=0.895, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005967-2` (score=0.9994) - What are the signs and symptoms of Syndactyly type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 3. If the information is available, the table below includes how...
2. `medquad_ans_0005798-1` (score=0.0001) - What are the signs and symptoms of Spinocerebellar ataxia X-linked type 3? The Human Phenotype Ontology provides the following list of signs and symptoms for Spinocerebellar ataxia X-linked type 3. If the information ...
3. `medquad_ans_0003957-1` (score=0.0001) - What are the signs and symptoms of Mesomelia-synostoses syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Mesomelia-synostoses syndrome. If the information is available, the ...
4. `medquad_ans_0000810-1__dup1` (score=0.0001) - What are the signs and symptoms of Brachydactyly type A2? The Human Phenotype Ontology provides the following list of signs and symptoms for Brachydactyly type A2. If the information is available, the table below incl...
5. `medquad_ans_0005970-1` (score=0.0001) - What are the signs and symptoms of Syndactyly type 9? The Human Phenotype Ontology provides the following list of signs and symptoms for Syndactyly type 9. If the information is available, the table below includes how...

**Answer**:

The symptoms of Syndactyly type 3 include camptodactyly of the finger in approximately 50% of patients, short toe in 7.5% of patients, 4-5 finger syndactyly, absence of the middle phalanx of the 5th finger, short 5th finger, and autosomal dominant inheritance [medquad_ans_0005967-2].

**Supported citations**: `medquad_ans_0005967-2`
**Unsupported citations**: _none_

### Example 175 (`0000197-1__dup3`)

**Question**: What is (are) Chronic Bronchitis ?

**Scores**: context_relevance=0.800, groundedness=0.931, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000232-1__dup2` (score=0.8652) - COPD (chronic obstructive pulmonary disease) makes it hard for you to breathe. The two main types are chronic bronchitis and emphysema. The main cause of COPD is long-term exposure to substances that irritate and dama...
2. `medquad_ans_0000133-1__dup2` (score=0.7908) - The bronchi are two tubes that branch off the trachea, or windpipe. The bronchi carry air to your lungs. The most common problem with the bronchi is bronchitis, an inflammation of the tubes. Bronchitis can be acute or...
3. `medquad_ans_0000012-1__dup5` (score=0.4235) - Chronic obstructive pulmonary disease, or COPD, is a progressive lung disease in which the airways of the lungs become damaged, making it hard to breathe. You may also have heard COPD called other names, like emphysem...
4. `medquad_ans_0000197-1__dup3` (score=0.1864) - Bronchitis is an inflammation of the bronchial tubes, the airways that carry air to your lungs. It causes a cough that often brings up mucus. It can also cause shortness of breath, wheezing, a low fever, and chest tig...
5. `medquad_ans_0000066-1__dup3` (score=0.0310) - Asthma is a chronic disease that affects your airways. Your airways are tubes that carry air in and out of your lungs. If you have asthma, the inside walls of your airways become sore and swollen. In the United States...

**Answer**:

Chronic bronchitis is one of the two main types of chronic obstructive pulmonary disease (COPD). It is characterized by a persistent cough that produces a lot of mucus. The main cause of COPD, including chronic bronchitis, is long-term exposure to irritants such as cigarette smoke, air pollution, chemical fumes, or dust. Chronic bronchitis involves inflammation of the bronchi, the tubes that carry air to the lungs, and over time, leads to damage and narrowing of the airways, making it difficult to breathe. [medquad_ans_0000232-1__dup2] [medquad_ans_0000012-1__dup5]

**Supported citations**: `medquad_ans_0000232-1__dup2`, `medquad_ans_0000012-1__dup5`
**Unsupported citations**: _none_

### Example 176 (`0000173-2__dup2`)

**Question**: What are the treatments for Spinal Muscular Atrophy ?

**Scores**: context_relevance=0.750, groundedness=0.791, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000233-2__dup1` (score=0.9969) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...
2. `medquad_ans_0000937-5` (score=0.9966) - These resources address the diagnosis or management of SMARD1: - Genetic Testing Registry: Spinal muscular atrophy with respiratory distress 1 These resources from MedlinePlus offer information about the diagnosis and...
3. `medquad_ans_0000255-2__dup1` (score=0.9952) - Treatment may include such medications as baclofen, diazepam, tizanidine or clonazepam. Physical therapy regimens may include muscle stretching and range of motion exercises to help prevent shrinkage or shortening of ...
4. `medquad_ans_0000208-2__dup1` (score=0.9945) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
5. `medquad_ans_0000112-2__dup1` (score=0.9944) - No one treatment has been found to be universally effective. Instead, doctors use a variety of therapies (medications, surgery, and other treatments such as physical therapy, splinting, stress management, and biofeedb...

**Answer**:

The provided context does not contain specific information about the treatments for Spinal Muscular Atrophy (SMARD1). While it mentions resources related to the diagnosis and management of SMARD1, including genetic testing, diagnostic tests, drug therapy, surgery and rehabilitation, genetic counseling, and palliative care, it does not detail the specific treatments available for this condition [medquad_ans_0000937-5]. Therefore, based on the given context, a definitive answer about treatments for Spinal Muscular Atrophy cannot be provided.

**Supported citations**: `medquad_ans_0000937-5`
**Unsupported citations**: _none_

### Example 177 (`0000001_1-7`)

**Question**: What are the treatments for Adult Acute Lymphoblastic Leukemia ?

**Scores**: context_relevance=0.889, groundedness=0.235, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000040-18` (score=0.9944) - Treatment depends on a number of factors, including the type of leukemia, the patient's age and general health, where leukemia cells have collected in the body, and whether the leukemia has been treated before. Certai...
2. `medquad_ans_0000040-20` (score=0.9763) - Chemotherapy is a cancer treatment that uses drugs to kill cancer cells. This is the most common treatment for most types of leukemia. Chemotherapy may be taken by mouth in pill form, by injection directly into a vein...
3. `medquad_ans_0000001_1-7` (score=0.9738) - Key Points - There are different types of treatment for patients with adult ALL. - The treatment of adult ALL usually has two phases. - Four types of standard treatment are used: - Chemotherapy - Radiation therapy - C...
4. `medquad_ans_0000040-4__dup2` (score=0.9614) - There are many treatment options for people with leukemia. The choice of treatment depends on your age and general health, the type of leukemia you have, whether or not it has spread outside the bone marrow, and other...
5. `medquad_ans_0000040-21` (score=0.9463) - Biological therapy is a treatment that uses a person's own immune system to fight leukemia. This therapy uses special substances to stimulate the immune system's ability to fight cancer. Some patients with chronic lym...

**Answer**:

The treatments for Adult Acute Lymphoblastic Leukemia (ALL) include chemotherapy, radiation therapy, chemotherapy with stem cell transplant, targeted therapy, and biologic therapy. The treatment typically occurs in two phases: remission induction therapy, which aims to kill leukemia cells in the blood and bone marrow to achieve remission, and post-remission therapy, which targets any remaining leukemia cells to prevent relapse [medquad_ans_0000040-18, medquad_ans_0000040-20, medquad_ans_0000040-1]. Additionally, patients may consider participating in clinical trials, which are research studies to improve current treatments or test new ones [medquad_ans_0000040-1]. Some treatments are standard, while others are being tested in clinical trials [medquad_ans_0000040-1

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000040-18, medquad_ans_0000040-20, medquad_ans_0000040-1`, `medquad_ans_0000040-1`

### Example 178 (`0002439-3`)

**Question**: What causes Friedreich ataxia ?

**Scores**: context_relevance=1.000, groundedness=0.991, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0002439-3` (score=0.9899) - What causes Friedreich ataxia? Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known...
2. `medquad_ans_0000133-3__dup1` (score=0.0002) - Generally, within 15 to 20 years after the appearance of the first symptoms, the person is confined to a wheelchair, and in later stages of the disease, individuals may become completely incapacitated. Friedreich's at...
3. `medquad_ans_0000323-2` (score=0.0001) - Episodic ataxia is uncommon, affecting less than 1 in 100,000 people. Only types 1 and 2 have been identified in more than one family, and type 2 is by far the most common form of the condition.
4. `medquad_ans_0000077-2` (score=0.0001) - The prevalence of ataxia neuropathy spectrum is unknown.
5. `medquad_ans_0000078-2` (score=0.0001) - Ataxia with oculomotor apraxia is a rare condition. Type 1 is a common form of ataxia in Portugal and Japan. Type 2 is estimated to occur in 1 in 900,000 individuals worldwide.

**Answer**:

Friedreich ataxia is caused by mutations in the FXN gene. This gene provides instructions for making a protein called frataxin. One region of the FXN gene contains a segment of DNA known as a GAA trinucleotide repeat. Normally, this segment is repeated 5 to 33 times within the FXN gene. In people with Friedreich ataxia, the GAA segment is repeated 66 to more than 1,000 times. The abnormally long GAA trinucleotide repeat disrupts the production of frataxin, which severely reduces the amount of this protein in cells. Certain nerve and muscle cells cannot function properly with a shortage of frataxin, leading to the characteristic signs and symptoms of Friedreich ataxia [medquad_ans_0002439-3].

**Supported citations**: `medquad_ans_0002439-3`
**Unsupported citations**: _none_

### Example 179 (`0002230-1`)

**Question**: What is (are) Familial atrial fibrillation ?

**Scores**: context_relevance=0.833, groundedness=0.712, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000057-5__dup1` (score=0.9813) - People who have Fanconi anemia (FA) are born with the disorder. They may or may not show signs or symptoms of it at birth. For this reason, FA isn't always diagnosed when a person is born. In fact, most people who hav...
2. `medquad_ans_0000013-4__dup2` (score=0.9705) - Atrial fibrillation (AF) occurs if the heart's electrical signals don't travel through the heart in a normal way. Instead, they become very rapid and disorganized. Damage to the heart's electrical system causes AF. Th...
3. `medquad_ans_0000072-4__dup2` (score=0.0386) - Mothers of children who are born with atrial septal defects (ASDs), ventricular septal defects (VSDs), or other heart defects may think they did something wrong during their pregnancies. However, most of the time, doc...
4. `medquad_ans_0000070-1__dup4` (score=0.0162) - An arrhythmia is a problem with the speed or rhythm of the heartbeat. Atrial fibrillation (AF) is the most common type of arrhythmia. The cause is a disorder in the heart's electrical system. Often, people who have AF...
5. `medquad_ans_0000057-2__dup2` (score=0.0159) - Fanconi anemia (FA) is an inherited disease. The term inherited means that the disease is passed from parents to children through genes. At least 13 faulty genes are associated with FA. FA occurs when both parents pas...

**Answer**:

The provided context does not mention familial atrial fibrillation. Therefore, it is not possible to answer the question based on the given information. [medquad_ans_0000057-5__dup1]

**Supported citations**: `medquad_ans_0000057-5__dup1`
**Unsupported citations**: _none_

### Example 180 (`0004627-1`)

**Question**: What is (are) Osteopathia striata cranial sclerosis ?

**Scores**: context_relevance=1.000, groundedness=0.962, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0004627-1` (score=0.8245) - Osteopathia striata cranial sclerosis is a type of skeletal dysplasia, which refers to a group of genetic conditions that affect the bones and hinder growth and development. The severity of the condition and the assoc...
2. `medquad_ans_0005681-1` (score=0.0316) - Sneddon syndrome is a progressive condition characterized by livedo reticularis (bluish net-like patterns of discoloration on the skin) and neurological abnormalities. Symptoms may include headache, dizziness, high bl...
3. `medquad_ans_0004625-1` (score=0.0137) - Osteomesopyknosis is a bone disorder characterized by abnormal hardening of bone (osteosclerosis). It is generally limited to the axial spine, pelvis, and proximal part of the long bones, which is what distinguishes t...
4. `medquad_ans_0000922-1` (score=0.0092) - SOST-related sclerosing bone dysplasia is a disorder of bone development characterized by excessive bone formation (hyperostosis). As a result of hyperostosis, bones throughout the body are denser and wider than norma...
5. `medquad_ans_0000760-1` (score=0.0026) - Osteoporosis-pseudoglioma syndrome is a rare condition characterized by severe thinning of the bones (osteoporosis) and eye abnormalities that lead to vision loss. In people with this condition, osteoporosis is usuall...

**Answer**:

Osteopathia striata cranial sclerosis is a type of skeletal dysplasia characterized by genetic changes (mutations) in the WTX gene, inherited in an X-linked dominant manner. It affects bone development and growth, leading to features such as skeletal abnormalities (particularly at the ends of long bones), sclerosis (hardening) of the craniofacial bones, macrocephaly (unusually large head size), and characteristic facial features. Some affected individuals may also have developmental delay, hearing loss, heart defects, and/or ophthalmoplegia (paralysis of the muscles surrounding the eyes). The severity and associated symptoms vary significantly among individuals, even within the same family. Treatment is based on the specific signs and symptoms present in each person [medquad_ans_0004627-1].

**Supported citations**: `medquad_ans_0004627-1`
**Unsupported citations**: _none_

### Example 181 (`0004097-2`)

**Question**: What are the symptoms of Mixed connective tissue disease ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0004097-2` (score=0.9992) - What are the signs and symptoms of Mixed connective tissue disease? People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus eryt...
2. `medquad_ans_0000093-5__dup1` (score=0.0013) - The most common signs and symptoms of Crohn's disease are - diarrhea - abdominal cramping and pain - weight loss Other general signs and symptoms include - feeling tired - nausea or loss of appetite - fever - anemiaa ...
3. `medquad_ans_0002374-2` (score=0.0004) - What are the symptoms of fibrous dysplasia? Fibrous dysplasia may cause no symptoms, mild symptoms, or severe symptoms. The most common symptoms are bone pain, bone deformities, fractures, and skin pigmentation differ...
4. `medquad_ans_0000230-1__dup2` (score=0.0001) - Connective tissue is the material inside your body that supports many of its parts. It is the "cellular glue" that gives your tissues their shape and helps keep them strong. It also helps some of your tissues do their...
5. `medquad_ans_0004049-1` (score=0.0001) - What are the signs and symptoms of Microphthalmia syndromic 7? The Human Phenotype Ontology provides the following list of signs and symptoms for Microphthalmia syndromic 7. If the information is available, the table ...

**Answer**:

People with mixed connective tissue disease (MCTD) have symptoms that overlap with several connective tissue disorders, including systemic lupus erythematosus, polymyositis, scleroderma, and rheumatoid arthritis. Common symptoms include pain in multiple joints, inflammation of joints (arthritis), muscle weakness, fevers, and fatigue. Other signs and symptoms may include accumulation of fluid in the tissue of the hands (edema), skin findings such as lupus-like rashes, reddish patches over the knuckles, violet coloring of the eyelids, loss of hair (alopecia), and dilation of small blood vessels around the fingernails (periungual telangiectasia). Dysfunction of the esophagus (hypomotility), abnormalities in lung function leading to breathing difficulties or pulmonary hypertension, heart involvement (such as pericarditis, myocarditis, or aortic insufficiency), neurologic abnormalities (e

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 182 (`0000003-14`)

**Question**: What are the treatments for Alzheimer's Caregiving ?

**Scores**: context_relevance=0.857, groundedness=0.327, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000098-2__dup1` (score=0.9580) - Drugs to specifically treat Alzheimers disease and some other progressive dementias are now available. Although these drugs do not halt the disease or reverse existing brain damage, they can improve symptoms and slow ...
2. `medquad_ans_0000004-23` (score=0.6805) - Currently, no treatment can stop Alzheimer's disease. However, four medications are used to treat its symptoms. These medicines may help maintain thinking, memory, and speaking skills for a limited time. They work by ...
3. `medquad_ans_0000004-32` (score=0.4979) - People with Alzheimer's disease, those with mild cognitive impairment, those with a family history of Alzheimers, and healthy people with no memory problems who want to help scientists test new treatments may be able ...
4. `medquad_ans_0000003-30` (score=0.1407) - As Alzheimers disease gets worse, you will need more help to care for the person. It's okay to seek help whenever you need it. Several kinds of help are available. - Home health care agencies send a home health aide o...
5. `medquad_ans_0000004-21` (score=0.1315) - An early, accurate diagnosis of Alzheimer's disease helps people and their families plan for the future. It gives them time to discuss care options, find support, and make legal and financial arrangements while the pe...

**Answer**:

The provided context does not address treatments specifically for Alzheimer's caregiving. While it discusses medications for Alzheimer's disease and cognitive exercises that may benefit individuals with dementia, there is no information on treatments aimed at supporting or improving caregiving. Therefore, based on the given context, the question cannot be answered.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 183 (`0000511-3`)

**Question**: What are the genetic changes related to idiopathic inflammatory myopathy ?

**Scores**: context_relevance=0.200, groundedness=0.000, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000786-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
2. `medquad_ans_0000575-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
3. `medquad_ans_0000516-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000700-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000149-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutation from one...

**Answer**:

The provided context does not contain information about the genetic changes related to idiopathic inflammatory myopathy. Therefore, the question cannot be answered based on the given context. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 184 (`0005223-5`)

**Question**: How to diagnose Pyruvate dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.550, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0005223-5` (score=0.9899) - Is genetic testing available for pyruvate dehydrogenase deficiency? Genetic testing is available for pyruvate dehydrogenase deficiency. GeneTests lists the names of laboratories that are performing genetic testing for...
2. `medquad_ans_0000855-2` (score=0.0001) - Pyruvate dehydrogenase deficiency is believed to be a rare condition; however, its prevalence is unknown.
3. `medquad_ans_0000856-2` (score=0.0001) - Pyruvate kinase deficiency is the most common inherited cause of nonspherocytic hemolytic anemia. More than 500 affected families have been identified, and studies suggest that the disorder may be underdiagnosed becau...
4. `medquad_ans_0000854-2` (score=0.0001) - Pyruvate carboxylase deficiency is a rare condition, with an estimated incidence of 1 in 250,000 births worldwide. This disorder appears to be much more common in some Algonkian Indian tribes in eastern Canada.
5. `medquad_ans_0003451-1` (score=0.0001) - Lactate dehydrogenase deficiency is a condition that affects how the body breaks down sugar to use as energy in cells, primarily muscle cells. There are two types of lactate dehydrogenase deficiency: lactate dehydroge...

**Answer**:

The diagnosis of pyruvate dehydrogenase (PDH) deficiency may be considered in any individual with early-onset neurological disease, especially if it appears to be associated with structural abnormalities in the brain and unexplained lactic acidosis. When lactic acid (also called lactate) and pyruvate in the blood do not seem to be significantly high, an important clue to the diagnosis may be high concentrations of lactate and/or pyruvate in the cerebrospinal fluid (the fluid that surrounds the brain and spinal cord). Additionally, magnetic resonance spectroscopy (MRS) of the brain may show concentrations of lactate in the central nervous system. Analysis of serum and urine amino acids usually shows hyperalaninemia (high levels of the amino acid alanine). When lactic acidosis is present, other disorders involving pyruvate abnormalities are part of the differential diagnosis. However, in all of these conditions,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 185 (`0006562-2`)

**Question**: What are the symptoms of 17q23.1q23.2 microdeletion syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.938, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0006562-2` (score=0.9991) - What are the signs and symptoms of 17q23.1q23.2 microdeletion syndrome? 17q23.1q23.2 microdeletion syndrome is characterized by developmental delay, microcephaly, short stature, heart defects and hand, foot and limb a...
2. `medquad_ans_0001240-1` (score=0.0024) - What are the signs and symptoms of Chromosome 17q deletion? The Human Phenotype Ontology provides the following list of signs and symptoms for Chromosome 17q deletion. If the information is available, the table below ...
3. `medquad_ans_0006565-2` (score=0.0001) - What are the signs and symptoms of 1q21.1 microdeletion syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for 1q21.1 microdeletion syndrome. If the information is available, the ...
4. `medquad_ans_0006554-2` (score=0.0001) - What are the signs and symptoms of 15q11.2 microdeletion? The signs and symptoms in people with a 15q11.2 microdeletion can vary widely. Some people with the microdeletion don't have any apparent features, while other...
5. `medquad_ans_0006562-3` (score=0.0001) - What causes 17q23.2q23.2 microdeletion syndrome? The syndrome is caused by an interstitial deletion (a deletion that does not involve the ends of a chromosome) encompassing bands 23.1 to 23.2 on the long (q) arm of ch...

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

**Scores**: context_relevance=1.000, groundedness=0.927, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0003860-3` (score=0.9952) - What causes MECP2 duplication syndrome? MECP2 duplication syndrome occurs when there is an extra copy (duplication) of the MECP2 gene in each cell. This is generally caused by a duplication of genetic material located...
2. `medquad_ans_0005347-3` (score=0.0079) - What causes Rett syndrome? Rett syndrome is typically caused by changes (mutations) in the MECP2 gene. This gene provides instructions for making a protein (MeCP2) needed for the development of the nervous system and ...
3. `medquad_ans_0000635-3` (score=0.0001) - MECP2 duplication syndrome is caused by a genetic change in which there is an extra copy of the MECP2 gene in each cell. This extra copy of the MECP2 gene is caused by a duplication of genetic material on the long (q)...
4. `medquad_ans_0000814-4` (score=0.0001) - More than 99 percent of PPM-X syndrome cases occur in people with no history of the disorder in their family. Many of these cases result from new mutations in the MECP2 gene. A few families with more than one affected...
5. `medquad_ans_0000636-2` (score=0.0001) - MECP2-related severe neonatal encephalopathy is likely a rare condition. Twenty to 30 affected males have been reported in the scientific literature.

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

**Scores**: context_relevance=0.444, groundedness=0.010, answer_relevance=0.711

**Top docs**:

1. `medquad_ans_0000033-2__dup1` (score=0.9955) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
2. `medquad_ans_0000208-2__dup1` (score=0.9938) - Treatments for the myopathies depend on the disease or condition and specific causes. Supportive and symptomatic treatment may be the only treatment available or necessary for some disorders. Treatment for other disor...
3. `medquad_ans_0000200-2__dup2` (score=0.9933) - Currently there is no cure for these disease syndromes.Medical care is directed at treating systemic conditions and improving the person's quality of life. Physical therapy and daily exercise may delay joint problems ...
4. `medquad_ans_0000206-2__dup1` (score=0.9931) - Myasthenia gravis can be controlled. Some medications improve neuromuscular transmission and increase muscle strength, and some suppress the production of abnormal antibodies. These medications must be used with caref...
5. `medquad_ans_0000084-2__dup1` (score=0.9910) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...

**Answer**:

The provided context does not contain information about the treatments for Lambert Eaton myasthenic syndrome. [medquad_ans_0000033-2__dup1, medquad_ans_0000208-2__dup1, medquad_ans_0000200-2__dup2, medquad_ans_0000206-2__dup1, medquad_ans_0000084-2__dup1] discuss treatments for conditions such as ataxia-telangiectasia, myopathies, various disease syndromes, myasthenia gravis, and congenital myasthenia, but none mention Lambert Eaton myasthenic syndrome. Therefore, based on the given context, the treatments for Lambert Eaton myasthenic syndrome cannot be determined.

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000033-2__dup1, medquad_ans_0000208-2__dup1, medquad_ans_0000200-2__dup2, medquad_ans_0000206-2__dup1, medquad_ans_0000084-2__dup1`

### Example 189 (`0000445-2__dup1`)

**Question**: What are the symptoms of Arginase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.972, answer_relevance=0.886

**Top docs**:

1. `medquad_ans_0000445-2__dup1` (score=0.9993) - What are the signs and symptoms of Arginase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Arginase deficiency. If the information is available, the table below includes...
2. `medquad_ans_0000451-2__dup1` (score=0.0001) - What are the signs and symptoms of Aromatic L-amino acid decarboxylase deficiency? Symptoms, which typically present during the first year of life, include severe developmental delay, weak muscle tone (hypotonia), mus...
3. `medquad_ans_0000065-1` (score=0.0001) - Arginase deficiency is an inherited disorder that causes the amino acid arginine (a building block of proteins) and ammonia to accumulate gradually in the blood. Ammonia, which is formed when proteins are broken down ...
4. `medquad_ans_0000445-1__dup1` (score=0.0001) - Arginase deficiency is an inherited metabolic condition in which the body is unable to process the amino acid (a building block of protein), arginine. Consequently, people affected by the condition have high levels of...
5. `medquad_ans_0004301-2` (score=0.0001) - What are the signs and symptoms of N-acetylglutamate synthetase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for N-acetylglutamate synthetase deficiency. If the informatio...

**Answer**:

The symptoms of Arginase deficiency include aminoaciduria (90%), behavioral abnormality (90%), cognitive impairment (90%), neurological speech impairment (90%), EEG abnormality (50%), hemiplegia/hemiparesis (50%), hyperammonemia (50%), and seizures (50%). Other reported symptoms are anorexia, autosomal recessive inheritance, diaminoaciduria, hyperactivity, intellectual disability, irritability, oroticaciduria, postnatal growth retardation, progressive spastic quadriplegia, and vomiting [medquad_ans_0000445-2__dup1].

**Supported citations**: `medquad_ans_0000445-2__dup1`
**Unsupported citations**: _none_

### Example 190 (`0000022-2__dup1`)

**Question**: What are the treatments for Von Hippel-Lindau Disease (VHL) ?

**Scores**: context_relevance=0.889, groundedness=0.947, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000022-2__dup1` (score=0.9864) - Treatment for VHL varies according to the location and size of the tumor. In general, the objective of treatment is to treat the tumors before they grow to a size large enough to cause permanent problems by putting pr...
2. `medquad_ans_0000085-2__dup1` (score=0.9834) - Currently, only central core disease has an effective treatment (see above). There are no known cures for any of these disorders. Supportive treatment may involve orthopedic treatments, as well as physical, occupation...
3. `medquad_ans_0006376-6` (score=0.9676) - How might von Hippel-Lindau (VHL) disease be treated? Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of tumors. In general, the goal is to treat growths when they cause symptoms, but ar...
4. `medquad_ans_0000022-3__dup1` (score=0.0160) - The prognosis for individuals with VHL depends on then number, location, and complications of the tumors. Untreated, VHL may result in blindness and/or permanent brain damage. With early detection and treatment the pr...
5. `medquad_ans_0000139-5__dup1` (score=0.0074) - Treatment for von Willebrand disease (VWD) is based on the type of VWD you have and how severe it is. Most cases of VWD are mild, and you may need treatment only if you have surgery, tooth extraction, or an accident. ...

**Answer**:

Treatment for Von Hippel-Lindau (VHL) disease depends on the location and size of the tumors. In general, the goal is to treat tumors when they cause symptoms but are still small enough to avoid permanent damage. Treatment usually involves surgical removal of tumors. Radiation therapy may be used in some cases. All individuals with VHL disease should be carefully monitored by a physician or medical team familiar with the disorder [medquad_ans_0006376-6]. Additionally, the objective of treatment is to remove tumors before they grow large enough to cause pressure on the brain or spinal cord, which could impair vision, cause deafness, or block cerebrospinal fluid flow [medquad_ans_0000022-2__dup1]. Early detection and treatment significantly improve the prognosis [medquad_ans_0000022-3__dup1].

**Supported citations**: `medquad_ans_0006376-6`, `medquad_ans_0000022-2__dup1`, `medquad_ans_0000022-3__dup1`
**Unsupported citations**: _none_

### Example 191 (`0004453-1`)

**Question**: What is (are) Noonan syndrome 3 ?

**Scores**: context_relevance=0.800, groundedness=0.063, answer_relevance=0.480

**Top docs**:

1. `medquad_ans_0001091-1` (score=0.8518) - 3-M syndrome is a disorder that causes short stature (dwarfism), unusual facial features, and skeletal abnormalities. The name of this condition comes from the initials of three researchers who first identified it: Mi...
2. `medquad_ans_0001276-1` (score=0.7563) - Chromosome 3p- syndrome is a rare chromosome abnormality that occurs when there is a missing copy of the genetic material located towards the end of the short arm (p) of chromosome 3. The severity of the condition and...
3. `medquad_ans_0000738-1` (score=0.4222) - Noonan syndrome is a condition that affects many areas of the body. It is characterized by mildly unusual facial characteristics, short stature, heart defects, bleeding problems, skeletal malformations, and many other...
4. `medquad_ans_0001094-1` (score=0.1166) - 3MC syndrome is a disorder characterized by unusual facial features and problems affecting other tissues and organs of the body. The distinctive facial features of people with 3MC syndrome include widely spaced eyes (...
5. `medquad_ans_0000420-1__dup1` (score=0.0088) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...

**Answer**:

The provided context does not mention Noonan syndrome 3 or any information related to it. Therefore, based on the given context, the answer cannot be determined. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 192 (`0005865-2`)

**Question**: What are the symptoms of Sprengel deformity ?

**Scores**: context_relevance=1.000, groundedness=0.963, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0005865-2` (score=0.9991) - What are the signs and symptoms of Sprengel deformity? Signs and symptoms of Sprengel deformity can vary depending on the severity and whether additional skeletal or muscular abnormalities are present. Some people may...
2. `medquad_ans_0000054-11` (score=0.9986) - People who have P.A.D. may have symptoms when walking or climbing stairs. These may include pain, numbness, aching, or heaviness in the leg muscles. Symptoms may also include cramping in the affected leg(s) and in the...
3. `medquad_ans_0000034-3__dup4` (score=0.0340) - Symptoms depend on the type of neuropathy and which nerves are affected. Some people with nerve damage have no symptoms at all. For others, the first symptom is often numbness, tingling, or pain in the feet. Symptoms ...
4. `medquad_ans_0000002-3__dup3` (score=0.0001) - Adrenal Insufficiency The most common symptoms of adrenal insufficiency are - chronic, or long lasting, fatigue - muscle weakness - loss of appetite - weight loss - abdominal pain Other symptoms of adrenal insufficien...
5. `medquad_ans_0005827-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia joint laxity? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia joint laxity. If th...

**Answer**:

The symptoms of Sprengel deformity include shoulder asymmetry, an elevated shoulder blade that may cause a lump in the back of the base of the neck, underdeveloped or incomplete muscles in the surrounding area, and limited movement of the shoulder and arm on the affected side. Additional features that may be present include abnormality of the shoulder (seen in approximately 90% of patients), cleft palate (in about 7.5% of patients), scoliosis, neck muscle hypoplasia, rib segmentation abnormalities, hemivertebrae, spina bifida occulta, cervical segmentation defects, and abnormalities of the collarbone or chest organs [medquad_ans_0005865-2].

**Supported citations**: `medquad_ans_0005865-2`
**Unsupported citations**: _none_

### Example 193 (`0000882-3`)

**Question**: What are the genetic changes related to Robinow syndrome ?

**Scores**: context_relevance=0.778, groundedness=0.931, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000882-3` (score=0.9988) - Autosomal recessive Robinow syndrome results from mutations in the ROR2 gene. This gene provides instructions for making a protein whose function is not well understood, although it is involved in chemical signaling p...
2. `medquad_ans_0000538-3` (score=0.0003) - Changes in the DNA near the SOX9 gene are the most common genetic cause of isolated Pierre Robin sequence. It is likely that changes in other genes, some of which have not been identified, also cause isolated Pierre R...
3. `medquad_ans_0000564-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000556-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
5. `medquad_ans_0000283-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...

**Answer**:

Genetic changes related to Robinow syndrome include mutations in the ROR2 gene for autosomal recessive Robinow syndrome, and mutations in the WNT5A or DVL1 gene for autosomal dominant Robinow syndrome. The ROR2 gene mutations prevent the production of any functional ROR2 protein, disrupting chemical signaling pathways essential for early development, particularly in the formation of the skeleton, heart, and genitals. Mutations in WNT5A or DVL1 alter the production or function of their respective proteins, impairing chemical signaling important for early development. In some cases, no mutation is identified in the ROR2, WNT5A, or DVL1 genes, and the cause remains unknown [medquad_ans_0000882-3].

**Supported citations**: `medquad_ans_0000882-3`
**Unsupported citations**: _none_

### Example 194 (`0000157-8`)

**Question**: What is (are) Kidney Failure: Eat Right to Feel Right on Hemodialysis ?

**Scores**: context_relevance=0.909, groundedness=0.089, answer_relevance=0.564

**Top docs**:

1. `medquad_ans_0000157-11` (score=0.9904) - Before you were on dialysis, your doctor may have told you to follow a low-protein diet. Being on dialysis changes this. Most people on dialysis are encouraged to eat as much high-quality protein as they can. Protein ...
2. `medquad_ans_0000179-12` (score=0.8733) - Eating healthy foods can help you keep up your energy and strength. All dialysis and transplant centers have a dietitian. The dietitian helps people with kidney failure learn about healthy food choices. You should tal...
3. `medquad_ans_0000539-1__dup1` (score=0.8510) - Healthy kidneys clean your blood by removing excess fluid, minerals, and wastes. They also make hormones that keep your bones strong and your blood healthy. But if the kidneys are damaged, they don't work properly. Ha...
4. `medquad_ans_0000038-27` (score=0.3250) - When your kidneys fail, they are no longer able to filter blood and remove waste from your body well enough to maintain health. Kidney failure causes harmful waste and excess fluid to build up in your body. Your blood...
5. `medquad_ans_0000158-13` (score=0.1524) - For people who are on dialysis or approaching total kidney failure, adequate nutrition is important for maintaining energy, strength, healthy sleep patterns, bone health, heart health, and good mental health. A person...

**Answer**:

The provided context does not contain a direct reference or summary of "Kidney Failure: Eat Right to Feel Right on Hemodialysis." While it includes information about dietary recommendations for individuals on hemodialysis—such as limiting fluid, sodium, and potassium, and emphasizing high-quality protein—it does not explicitly describe or summarize the specific program or resource titled "Kidney Failure: Eat Right to Feel Right on Hemodialysis." Therefore, based on the given context, this specific resource or title cannot be identified or described.

Answer: Insufficient context to identify or describe "Kidney Failure: Eat Right to Feel Right on Hemodialysis." [medquad_ans_0000157-11, medquad_ans_0000179-12]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000157-11, medquad_ans_0000179-12`

### Example 195 (`0000677-2`)

**Question**: How many people are affected by mucolipidosis type IV ?

**Scores**: context_relevance=0.556, groundedness=0.945, answer_relevance=0.467

**Top docs**:

1. `medquad_ans_0000677-2` (score=0.9994) - Mucolipidosis type IV is estimated to occur in 1 in 40,000 people. About 70 percent of affected individuals have Ashkenazi Jewish ancestry.
2. `medquad_ans_0000675-2` (score=0.8117) - Mucolipidosis III alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
3. `medquad_ans_0000676-2` (score=0.5824) - Mucolipidosis III gamma is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
4. `medquad_ans_0000674-2` (score=0.0122) - Mucolipidosis II alpha/beta is a rare disorder, although its exact prevalence is unknown. It is estimated to occur in about 1 in 100,000 to 400,000 individuals worldwide.
5. `medquad_ans_0000394-2` (score=0.0061) - Fucosidosis is a rare condition; approximately 100 cases have been reported worldwide. This condition appears to be most prevalent in Italy, Cuba, and the southwestern United States.

**Answer**:

Mucolipidosis type IV is estimated to occur in 1 in 40,000 people [medquad_ans_0000677-2].

**Supported citations**: `medquad_ans_0000677-2`
**Unsupported citations**: _none_

### Example 196 (`0000795-1__dup1`)

**Question**: What is (are) Scars ?

**Scores**: context_relevance=1.000, groundedness=0.994, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000795-1__dup1` (score=0.9368) - A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chi...
2. `medquad_ans_0000193-1__dup4` (score=0.0106) - Cirrhosis\* is scarring of the liver. Scar tissue forms because of injury or long-term disease. Scar tissue replaces healthy liver tissue and blocks the normal flow of blood through the liver. A healthy liver - makes p...
3. `medquad_ans_0000193-6` (score=0.0084) - If too much scar tissue forms, your liver could fail. Then you will need a liver transplant. A liver transplant can return you to good health. For information about liver transplantation, see the booklet What I need t...
4. `medquad_ans_0000712-1__dup1` (score=0.0038) - Piercings and tattoos are body decorations that go back to ancient times. Body piercing involves making a hole in the skin so that you can insert jewelry. This is often in the earlobe, but can be in other parts of the...
5. `medquad_ans_0000977-1__dup1` (score=0.0020) - An injury is damage to your body. It is a general term that refers to harm caused by accidents, falls, hits, weapons, and more. In the U.S., millions of people injure themselves every year. These injuries range from m...

**Answer**:

A scar is a permanent patch of skin that grows over a wound. It forms when your body heals itself after a cut, scrape, burn, or sore. You can also get scars from surgery that cuts through the skin, infections like chickenpox, or skin conditions like acne. Scars are often thicker, as well as pinker, redder, or shinier, than the rest of your skin. How your scar looks depends on - How big and deep your wound is - Where it is - How long it takes to heal - Your age - Your inherited tendency to scar Scars usually fade over time but never go away completely. If the way a scar looks bothers you, various treatments might minimize it. These include surgical revision, dermabrasion, laser treatments, injections, chemical peels, and creams. [medquad_ans_0000795-1__dup1]

**Supported citations**: `medquad_ans_0000795-1__dup1`
**Unsupported citations**: _none_

### Example 197 (`0001045-3`)

**Question**: What are the genetic changes related to X-linked adrenoleukodystrophy ?

**Scores**: context_relevance=0.556, groundedness=0.029, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000015-1__dup2` (score=0.0005) - The adrenal glands are small glands located on top of each kidney. They produce hormones that you can't live without, including sex hormones and cortisol. Cortisol helps you respond to stress and has many other import...
2. `medquad_ans_0000002-1__dup4` (score=0.0001) - Adrenal insufficiency is an endocrine, or hormonal, disorder that occurs when the adrenal glands do not produce enough of certain hormones. The adrenal glands are located just above the kidneys. Adrenal insufficiency ...
3. `medquad_ans_0001143-1` (score=0.0001) - CHILD syndrome, also known as congenital hemidysplasia with ichthyosiform erythroderma and limb defects, is a genetic condition that is typically characterized by large patches of skin that are red and inflamed (eryth...
4. `medquad_ans_0003225-3` (score=0.0001) - How is intellectual disability-developmental delay-contractures syndrome inherited? Intellectual disability-developmental delay-contractures syndrome syndrome is inherited in an X-linked recessive manner and is caused...
5. `medquad_ans_0003442-1` (score=0.0001) - L1 syndrome is a mild to severe congenital disorder with hydrocephalus of varying degrees of severity, intellectual disability, spasticity of the legs, and adducted thumbs. It includes several conditions, some more se...

**Answer**:

The provided context does not contain information about genetic changes related to X-linked adrenoleukodystrophy. Therefore, a definitive answer cannot be given based on the available information. [doc_id] not available

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 198 (`0000988-2`)

**Question**: How many people are affected by transthyretin amyloidosis ?

**Scores**: context_relevance=0.750, groundedness=0.972, answer_relevance=0.450

**Top docs**:

1. `medquad_ans_0000988-2` (score=0.9981) - The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence of this condition is thought to be one in 538 people. Transthyretin amyloidosis is less common among Americans of Europe...
2. `medquad_ans_0000457-2` (score=0.9945) - The prevalence of hereditary cerebral amyloid angiopathy is unknown. The Dutch type is the most common, with over 200 affected individuals reported in the scientific literature.
3. `medquad_ans_0000577-2` (score=0.0074) - Lamellar ichthyosis is estimated to affect 1 in 100,000 individuals in the United States. This condition is more common in Norway, where an estimated 1 in 91,000 individuals are affected.
4. `medquad_ans_0000988-1` (score=0.0001) - Transthyretin amyloidosis is a slowly progressive condition characterized by the buildup of abnormal deposits of a protein called amyloid (amyloidosis) in the body's organs and tissues. These protein deposits most fre...
5. `medquad_ans_0000027-1__dup3` (score=0.0001) - Amyloidosis occurs when abnormal proteins called amyloids build up and form deposits. The deposits can collect in organs such as the kidney and heart. This can cause the organs to become stiff and unable to work the w...

**Answer**:

The exact incidence of transthyretin amyloidosis is unknown. In northern Portugal, the incidence is estimated to be one in 538 people. Among Americans of European descent, it affects one in 100,000 people. The cardiac form is more common among people with African ancestry, affecting between 3 percent and 3.9 percent of African Americans and approximately 5 percent of people in some areas of West Africa [medquad_ans_0000988-2].

**Supported citations**: `medquad_ans_0000988-2`
**Unsupported citations**: _none_

### Example 199 (`0001780-1`)

**Question**: What are the symptoms of Dens in dente and palatal invaginations ?

**Scores**: context_relevance=1.000, groundedness=0.348, answer_relevance=0.927

**Top docs**:

1. `medquad_ans_0001780-1` (score=0.9994) - What are the signs and symptoms of Dens in dente and palatal invaginations? The Human Phenotype Ontology provides the following list of signs and symptoms for Dens in dente and palatal invaginations. If the informatio...
2. `medquad_ans_0003361-2` (score=0.0003) - What are the signs and symptoms of KBG syndrome? KBG syndrome is often characterized by distinctive facial features, skeletal abnormalities, short stature, large upper teeth (macrodontia), and developmental delay or i...
3. `medquad_ans_0000024_3-3` (score=0.0001) - Signs of lip and oral cavity cancer include a sore or lump on the lips or in the mouth. These and other signs and symptoms may be caused by lip and oral cavity cancer or by other conditions. Check with your doctor if ...
4. `medquad_ans_0000263-1__dup2` (score=0.0001) - Dentures are false teeth made to replace teeth you have lost. Dentures can be complete or partial. Complete dentures cover your entire upper or lower jaw. Partials replace one or a few teeth. Advances in dentistry hav...
5. `medquad_ans_0001133-2` (score=0.0001) - What are the signs and symptoms of Cherubism? Cherubism is characterized by abnormal bone tissue in the lower part of the face. Beginning in early childhood, both the lower jaw (the mandible) and the upper jaw (the ma...

**Answer**:

The signs and symptoms of Dens in dente and palatal invaginations include features listed by the Human Phenotype Ontology (HPO), which compile information from sources such as Orphanet. These symptoms are reported with approximate frequencies, often as percentages or fractions indicating how commonly they occur in patients with the condition. Specific symptoms may include dental abnormalities such as dens in dente (a type of dental malformation where a small, conical tooth is embedded within a larger tooth) and palatal invaginations (a developmental defect where a portion of the palate invaginates into the dental tissue). The frequency of these symptoms is based on studies, with some frequencies listed as exact numbers (e.g., 25/25 meaning all patients in a study had the symptom), while others may lack specific data due to limited study samples. For precise symptom details and frequencies, the HPO list is recommended [medquad_ans_00

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 200 (`0000782-2`)

**Question**: How many people are affected by periventricular heterotopia ?

**Scores**: context_relevance=0.750, groundedness=0.797, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000782-2` (score=0.9993) - Periventricular heterotopia is a rare condition. Its incidence is unknown.
2. `medquad_ans_0000479-2` (score=0.0004) - The prevalence of heterotaxy syndrome is estimated to be 1 in 10,000 people worldwide. However, researchers suspect that the condition is underdiagnosed, and so it may actually be more common than this. Heterotaxy syn...
3. `medquad_ans_0004818-2` (score=0.0001) - What are the signs and symptoms of periventricular nodular heterotopia? The condition is first noticed when seizures appear, often during the teenage years. The nodules around the ventricles are then typically discove...
4. `medquad_ans_0000782-1` (score=0.0001) - Periventricular heterotopia is a condition in which nerve cells (neurons) do not migrate properly during the early development of the fetal brain, from about the 6th week to the 24th week of pregnancy. Heterotopia mea...
5. `medquad_ans_0004818-3` (score=0.0001) - What are the recommended evaluations for patients diagnosed with periventricular nodular heterotopia? The following evaluations are recommended:[1823] Imaging exams of the brain to establish the diagnosis Evaluation b...

**Answer**:

The incidence of periventricular heterotopia is unknown [medquad_ans_0000782-2]. Therefore, the number of people affected by periventricular heterotopia cannot be determined from the provided context.

**Supported citations**: `medquad_ans_0000782-2`
**Unsupported citations**: _none_

### Example 201 (`0000004-4__dup1`)

**Question**: what research (or clinical trials) is being done for Landau-Kleffner Syndrome ?

**Scores**: context_relevance=0.455, groundedness=0.000, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000190-4__dup1` (score=0.9984) - The NINDS supports research on neurological disorders such as Melkersson-Rosenthal syndrome. Much of this research is aimed at increasing knowledge of these disorders and finding ways to treat, prevent, and ultimately...
2. `medquad_ans_0000172-4__dup1` (score=0.9983) - NINDS supports and conducts research on neurobehavioral disorders such as Klver-Bucy syndrome. Much of the research focuses on learning more about these disorders and finding ways to prevent and treat them.
3. `medquad_ans_0000168-4__dup1` (score=0.9980) - The NINDS supports research on neuromuscular disorders such as KSS. The goals of this research are to increase understanding of these disorders, and to find ways to prevent, treat, and, ultimately, cure them. The most...
4. `medquad_ans_0000162-4__dup1` (score=0.9977) - The NINDS supports broad and varied programs of research on epilepsy and other seizure disorders. This research is aimed at discovering new ways to prevent, diagnose, and treat these disorders and, ultimately, to find...
5. `medquad_ans_0000124-4__dup1` (score=0.9947) - The NINDS supports and conducts research on neurogenetic disorders such as Fahr's Syndrome. The goals of this research are to locate and understand the actions of the genes involved in this disorder. Finding these gen...

**Answer**:

The provided context does not mention Landau-Kleffner Syndrome or any research or clinical trials specifically related to it. Therefore, based on the given information, no details about research for Landau-Kleffner Syndrome can be provided. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 202 (`0000205-1__dup2`)

**Question**: What is (are) Muscular Dystrophy ?

**Scores**: context_relevance=1.000, groundedness=0.967, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0004219-1` (score=0.9721) - Muscular dystrophy (MD) refers to a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...
2. `medquad_ans_0000205-1__dup2` (score=0.9314) - The muscular dystrophies (MD) are a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or ...
3. `medquad_ans_0001537-1` (score=0.9001) - Congenital muscular dystrophy (CMD) refers to a group of inherited conditions that affect the muscles and are present at birth or in early infancy. The severity of the condition, the associated signs and symptoms and ...
4. `medquad_ans_0000631-1__dup1` (score=0.8258) - Muscular dystrophy (MD) is a group of more than 30 inherited diseases. They all cause muscle weakness and muscle loss. Some forms of MD appear in infancy or childhood. Others may not appear until middle age or later. ...
5. `medquad_ans_0004290-1` (score=0.1678) - Myotonic dystrophy is an inherited condition that affects the muscles and other body systems. It is the most common form of muscular dystrophy that begins in adulthood, usually in a person's 20s or 30s. This condition...

**Answer**:

Muscular dystrophy (MD) refers to a group of more than 30 genetic diseases characterized by progressive weakness and degeneration of the skeletal muscles that control movement. Some forms of MD are seen in infancy or childhood, while others may not appear until middle age or later. The disorders differ in terms of the distribution and extent of muscle weakness (some forms of MD also affect cardiac muscle), age of onset, rate of progression, and pattern of inheritance [medquad_ans_0004219-1]. Duchenne MD is the most common form of MD and primarily affects boys, caused by the absence of dystrophin, a protein involved in maintaining the integrity of muscle, with onset between 3 and 5 years and rapid progression, leading to loss of walking ability by age 12 and later need for a respirator [medquad_ans_0000205-1__dup2]. Facioscap

**Supported citations**: `medquad_ans_0004219-1`, `medquad_ans_0000205-1__dup2`
**Unsupported citations**: _none_

### Example 203 (`0004948-1`)

**Question**: What is (are) Polydactyly ?

**Scores**: context_relevance=0.750, groundedness=0.350, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0004948-1` (score=0.8062) - Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that t...
2. `medquad_ans_0000218-1__dup2` (score=0.0028) - A polyp is an extra piece of tissue that grows inside your body. Colonic polyps grow in the large intestine, or colon. Most polyps are not dangerous. However, some polyps may turn into cancer or already be cancer. To ...
3. `medquad_ans_0004953-1` (score=0.0003) - Polyembryoma is a type of tumor that develops from the cells of the gonads (testes in men or ovaries in women). Such tumors are called germ cell tumors. Polyembryomas have a distinctive look because they are composed ...
4. `medquad_ans_0000108-5` (score=0.0003) - These resources address the diagnosis or management of Bardet-Biedl syndrome: - Gene Review: Gene Review: Bardet-Biedl Syndrome - Genetic Testing Registry: Bardet-Biedl syndrome - MedlinePlus Encyclopedia: Obesity - M...
5. `medquad_ans_0000767-5` (score=0.0001) - These resources address the diagnosis or management of Pallister-Hall syndrome: - Gene Review: Gene Review: Pallister-Hall Syndrome - Genetic Testing Registry: Pallister-Hall syndrome - MedlinePlus Encyclopedia: Epigl...

**Answer**:

Polydactyly is a condition in which a person has more than five fingers per hand or five toes per foot. It is the most common birth defect of the hand and foot. Polydactyly can occur as an isolated finding such that the person has no other physical anomalies or intellectual impairment. However, it can occur in association with other birth defects and cognitive abnormalities as part of a genetic syndrome. In some cases, the extra digits may be well-formed and functional. Surgery may be considered especially for poorly formed digits or very large extra digits. Surgical management depends greatly on the complexity of the deformity [1] [2].

**Supported citations**: _none_
**Unsupported citations**: `1`, `2`

### Example 204 (`0002519-2`)

**Question**: What are the symptoms of Geniospasm ?

**Scores**: context_relevance=1.000, groundedness=0.908, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0002519-2` (score=0.9994) - What are the signs and symptoms of Geniospasm? The Human Phenotype Ontology provides the following list of signs and symptoms for Geniospasm. If the information is available, the table below includes how often the sym...
2. `medquad_ans_0000088-4__dup2` (score=0.9987) - A person may experience digestive signs and symptoms, or symptoms in other parts of the body. Digestive signs and symptoms are more common in children and can include - abdominal bloating - chronic diarrhea - constipa...
3. `medquad_ans_0000145-3__dup3` (score=0.0057) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breat...
4. `medquad_ans_0000005-3__dup4` (score=0.0001) - Many people with Hashimotos disease have no symptoms at first. As the disease slowly progresses, the thyroid usually enlarges and may cause the front of the neck to look swollen. The enlarged thyroid, called a goiter,...
5. `medquad_ans_0000016_1-3` (score=0.0001) - Signs and symptoms of extragonadal germ cell tumors include breathing problems and chest pain. Malignant extragonadal germ cell tumors may cause signs and symptoms as they grow into nearby areas. Other conditions may ...

**Answer**:

The signs and symptoms of Geniospasm include chin myoclonus. This information is derived from the Human Phenotype Ontology, which lists chin myoclonus as a feature associated with Geniospasm [medquad_ans_0002519-2].

**Supported citations**: `medquad_ans_0002519-2`
**Unsupported citations**: _none_

### Example 205 (`0000512-3`)

**Question**: What are the genetic changes related to idiopathic pulmonary fibrosis ?

**Scores**: context_relevance=0.500, groundedness=0.024, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000049-4__dup2` (score=0.0034) - Cystic fibrosis (CF) affects both males and females and people from all racial and ethnic groups. However, the disease is most common among Caucasians of Northern European descent. CF also is common among Latinos and ...
2. `medquad_ans_0000049-3__dup2` (score=0.0029) - A defect in the CFTR gene causes cystic fibrosis (CF). This gene makes a protein that controls the movement of salt and water in and out of your body's cells. In people who have CF, the gene makes a protein that doesn...
3. `medquad_ans_0000089-2__dup2` (score=0.0005) - Marfan syndrome is a genetic disorder. A mutation, or change, in the gene that controls how the body makes fibrillin causes Marfan syndrome. Fibrillin is a protein that plays a major role in your body's connective tis...
4. `medquad_ans_0000181-3__dup2` (score=0.0002) - Alagille syndrome is caused by a gene mutation, or defect. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the DNA sequence that makes up a gene. DNA, or deoxyribon...
5. `medquad_ans_0000349-4` (score=0.0002) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not contain any information about the genetic changes related to idiopathic pulmonary fibrosis. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 206 (`0000813-4`)

**Question**: Is Potocki-Shaffer syndrome inherited ?

**Scores**: context_relevance=0.750, groundedness=0.024, answer_relevance=0.950

**Top docs**:

1. `medquad_ans_0000883-4` (score=0.9980) - This condition is typically inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person inherits the mutatio...
2. `medquad_ans_0000324-4` (score=0.9977) - This condition is not inherited. It arises from a somatic mutation in histiocytes or their precursor cells during an individual's lifetime.
3. `medquad_ans_0006597-4` (score=0.0018) - Can 48,XXYY syndrome be inherited?
4. `medquad_ans_0000772-4` (score=0.0004) - Most cases of Parkes Weber syndrome occur in people with no history of the condition in their family. These cases are described as sporadic. When Parkes Weber syndrome is caused by mutations in the RASA1 gene, it is s...
5. `medquad_ans_0000313-4` (score=0.0001) - The inheritance pattern of Ehlers-Danlos syndrome varies by type. The arthrochalasia, classical, hypermobility, and vascular forms of the disorder have an autosomal dominant pattern of inheritance. Autosomal dominant ...

**Answer**:

The provided context does not contain information about the inheritance pattern of Potocki-Shaffer syndrome. Therefore, based on the given documents, it is not possible to determine whether Potocki-Shaffer syndrome is inherited. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 207 (`0000177-1__dup2`)

**Question**: Do you have information about Child Care

**Scores**: context_relevance=1.000, groundedness=0.800, answer_relevance=0.657

**Top docs**:

1. `medquad_ans_0000076-1__dup3` (score=0.9978) - Summary : There are many new responsibilities when you have a baby. One of them is to make sure they get the checkups that they need. Well-baby exams are important in making sure that your baby is growing and developi...
2. `medquad_ans_0000897-1__dup1` (score=0.9972) - Summary : Most young children get sick. It is hard for parents to know what is serious. You can learn what the common warning signs are. In the end, trust your intuition. If you are worried about your toddler, call yo...
3. `medquad_ans_0000189-1__dup2` (score=0.9970) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
4. `medquad_ans_0000925-1__dup1` (score=0.9907) - Summary : It can be scary when your baby is sick, especially when it is not an everyday problem like a cold or a fever. You may not know whether the problem is serious or how to treat it. If you have concerns about yo...
5. `medquad_ans_0000227-5` (score=0.0975) - These resources address the diagnosis or management of CHILD syndrome: - Gene Review: Gene Review: NSDHL-Related Disorders - Genetic Testing Registry: Child syndrome These resources from MedlinePlus offer information ...

**Answer**:

No, there is no information about Child Care in the provided context. The context discusses well-child visits, well-baby exams, immunizations, screenings, and health concerns in children, but does not address "Child Care" as a topic. [medquad_ans_0000076-1\_\_dup3][medquad_ans_0000897-1__dup1][medquad_ans_0000189-1\_\_dup2][medquad_ans_0000925-1__dup1][medquad_ans_0000227-5]

**Supported citations**: `medquad_ans_0000076-1__dup3`, `medquad_ans_0000897-1__dup1`, `medquad_ans_0000189-1__dup2`, `medquad_ans_0000925-1__dup1`, `medquad_ans_0000227-5`
**Unsupported citations**: _none_

### Example 208 (`0000046-2`)

**Question**: How many people are affected by alternating hemiplegia of childhood ?

**Scores**: context_relevance=0.700, groundedness=0.908, answer_relevance=0.780

**Top docs**:

1. `medquad_ans_0000046-2` (score=0.9995) - Alternating hemiplegia of childhood is a rare condition that affects approximately 1 in 1 million people.
2. `medquad_ans_0000955-2` (score=0.9994) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.
3. `medquad_ans_0000950-2` (score=0.9927) - The worldwide prevalence of sporadic hemiplegic migraine is unknown. Studies suggest that in Denmark about 1 in 10,000 people have hemiplegic migraine and that the condition occurs equally in families with multiple af...
4. `medquad_ans_0000351-2` (score=0.0563) - The worldwide prevalence of familial hemiplegic migraine is unknown. Studies suggest that in Denmark about 1 in 10,000 people have hemiplegic migraine and that the condition occurs equally in families with multiple af...
5. `medquad_ans_0000039-3__dup1` (score=0.0006) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...

**Answer**:

Approximately 1 in 1 million people are affected by alternating hemiplegia of childhood [medquad_ans_0000046-2].

**Supported citations**: `medquad_ans_0000046-2`
**Unsupported citations**: _none_

### Example 209 (`0003687-2`)

**Question**: What are the symptoms of Lupus ?

**Scores**: context_relevance=1.000, groundedness=0.550, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003687-2` (score=0.9992) - What are the signs and symptoms of Lupus? You can read about the signs and symptoms of lupus from MedlinePlus and the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS). The Human Phenotype ...
2. `medquad_ans_0000159-3__dup2` (score=0.9977) - The symptoms of lupus nephritis may include high blood pressure, foamy urine, and edemaswelling, usually in the legs, feet, or ankles and less often in the hands or face. Kidney problems often develop at the same time...
3. `medquad_ans_0000144-5__dup2` (score=0.0114) - The signs and symptoms of glomerular disease include - albuminuria: large amounts of protein in the urine - hematuria: blood in the urine - reduced glomerular filtration rate: inefficient filtering of wastes from the ...
4. `medquad_ans_0000182-6` (score=0.0043) - The most common symptoms of autoimmune hepatitis are - fatigue - joint pain - nausea - loss of appetite - pain or discomfort over the liver - skin rashes - dark yellow urine - light-colored stools - jaundice, or yello...
5. `medquad_ans_0000136-6` (score=0.0011) - The signs and symptoms of vasculitis vary. They depend on the type of vasculitis you have, the organs involved, and the severity of the condition. Some people may have few signs and symptoms. Other people may become v...

**Answer**:

The symptoms of Lupus include abdominal pain, abnormality of temperature regulation, abnormality of the heart valves, abnormality of the pericardium, alopecia, arthralgia, arthritis, autoimmunity, chest pain, cutaneous photosensitivity, skin rash, thrombocytopenia, thrombophlebitis, abnormal pyramidal signs, abnormal tendon morphology, abnormality of the autonomic nervous system, abnormality of the endocardium, abnormality of the pleura, anorexia, arterial thrombosis, aseptic leukocyturia, bone marrow hypocellularity, conjunctival telangiectasia, cranial nerve paralysis, cutis marmorata, dry skin, eczema, edema of the lower limbs, glomerulopathy, hallucinations, hematuria, hepatomegaly, hyperkeratosis, hypoproteinemia, increased antibody level in blood

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 210 (`0001395-1`)

**Question**: What is (are) Cockayne syndrome ?

**Scores**: context_relevance=0.800, groundedness=0.524, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000194-1` (score=0.2765) - Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging. Features of this disorder include a failure to gain weight and grow at the expected rate (failure to thrive), a...
2. `medquad_ans_0001397-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
3. `medquad_ans_0001398-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
4. `medquad_ans_0001395-1` (score=0.0949) - Cockayne syndrome is a rare condition which causes short stature, premature aging (progeria), severe photosensitivity, and moderate to severe learning delay. This syndrome also includes failure to thrive in the newbor...
5. `medquad_ans_0000613-2` (score=0.0008) - Mabry syndrome is likely a rare condition, but its prevalence is unknown. More than 20 cases have been described in the scientific literature.

**Answer**:

Cockayne syndrome is a rare disorder characterized by short stature and an appearance of premature aging (progeria). It is associated with severe photosensitivity, failure to thrive, microcephaly, and impaired development of the nervous system. Other possible signs and symptoms include hearing loss, eye abnormalities, severe tooth decay, bone abnormalities, and changes in the brain visible on brain scans. The syndrome is divided into three subtypes: classical (type I or type A), which has an onset in early childhood (usually after age 1 year) and is sometimes called "moderate" or "classic" Cockayne syndrome; type II (type B), which is more severe and presents at birth with growth and developmental abnormalities, also known as cerebro-oculo-facio-skeletal (COFS) syndrome or Pena-Shokeir syndrome type II; and type III (type C), which is the mildest form and appears later in childhood. Cockayne

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 211 (`0000029-1__dup5`)

**Question**: What is (are) Gum (Periodontal) Disease ?

**Scores**: context_relevance=0.667, groundedness=0.894, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000904-1__dup2` (score=0.9924) - Your teeth are made of a hard, bonelike material. Inside the tooth are nerves and blood vessels. You need your teeth for many activities you may take for granted. These include eating, speaking and even smiling. But t...
2. `medquad_ans_0000903-1__dup2` (score=0.9919) - You call it a cavity. Your dentist calls it tooth decay or dental caries. They're all names for a hole in your tooth. The cause of tooth decay is plaque, a sticky substance in your mouth made up mostly of germs. Tooth...
3. `medquad_ans_0000029-9__dup1` (score=0.9863) - When gingivitis is not treated, it can advance to periodontitis (which means "inflammation around the tooth.") In periodontitis, gums pull away from the teeth and form "pockets" that become infected. The body's immune...
4. `medquad_ans_0000413-1__dup1` (score=0.9751) - If you have gum disease, you're not alone. Many U.S. adults currently have some form of the disease. It ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the t...
5. `medquad_ans_0000029-6__dup1` (score=0.9748) - Gum disease is an infection of the tissues that hold your teeth in place. In its early stages, it is usually painless, and many people are not aware that they have it. But in more advanced stages, gum disease can lead...

**Answer**:

Gum (periodontal) disease is a condition that ranges from simple gum inflammation, called gingivitis, to serious damage to the tissue and bone supporting the teeth. In gingivitis, the gums become red and swollen and can bleed easily. It is a mild form of gum disease that can usually be reversed with daily brushing and flossing and regular cleanings by a dentist or dental hygienist. If left untreated, gingivitis can progress to periodontitis, in which the gums pull away from the teeth and form infected pockets. The body's immune response to bacteria in plaque leads to the breakdown of bone and tissue that hold teeth in place. If not treated, this can result in the destruction of the gums, bone, and connective tissue supporting the teeth, potentially leading to tooth loss [medquad_ans_0000413-1__dup1]medquad_ans_000002

**Supported citations**: `medquad_ans_0000413-1__dup1`
**Unsupported citations**: _none_

### Example 212 (`0000494-2`)

**Question**: How many people are affected by hyperkalemic periodic paralysis ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.556

**Top docs**:

1. `medquad_ans_0000505-2` (score=0.9993) - Although its exact prevalence is unknown, hypokalemic periodic paralysis is estimated to affect 1 in 100,000 people. Men tend to experience symptoms of this condition more often than women.
2. `medquad_ans_0000494-2` (score=0.9992) - Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people.
3. `medquad_ans_0003059-2` (score=0.0001) - What are the signs and symptoms of Hypokalemic periodic paralysis? Hypokalemic periodic paralysis involves attacks of muscle weakness or loss of muscle movement (paralysis) that come and go. The weakness or paralysis ...
4. `medquad_ans_0003059-1` (score=0.0001) - Hypokalemic periodic paralysis is a condition that causes episodes of extreme muscle weakness typically beginning in childhood or adolescence. Most often, these episodes involve a temporary inability to move muscles i...
5. `medquad_ans_0002007-1` (score=0.0001) - Ohtahara syndrome is a neurological disorder characterized by seizures. The disorder affects newborns, usually within the first three months of life (most often within the first 10 days) in the form of epileptic seizu...

**Answer**:

Hyperkalemic periodic paralysis affects an estimated 1 in 200,000 people [medquad_ans_0000494-2].

**Supported citations**: `medquad_ans_0000494-2`
**Unsupported citations**: _none_

### Example 213 (`0005212-1`)

**Question**: What is (are) Pyle disease ?

**Scores**: context_relevance=0.800, groundedness=0.991, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0005212-1` (score=0.8399) - Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end o...
2. `medquad_ans_0004833-1` (score=0.0052) - Peutz-Jeghers syndrome (PJS) is an inherited condition that is associated with an increased risk of growths along the lining of the gastrointestinal tract (called hamartomatous polyps) and certain types of cancer. Mos...
3. `medquad_ans_0000054-7` (score=0.0009) - Peripheral arterial disease (P.A.D.) is a disease in which plaque (plak) builds up in the arteries that carry blood to your head, organs, and limbs. Plaque is made up of fat, cholesterol, calcium, fibrous tissue, and ...
4. `medquad_ans_0005160-1` (score=0.0003) - Pseudoxanthoma elasticum, PXE, is an inherited disorder that causes calcium and other minerals to accumulate in the elastic fibers of the skin, eyes, and blood vessels, and less frequently in other areas such as the d...
5. `medquad_ans_0000792-1` (score=0.0003) - Phenylketonuria (commonly known as PKU) is an inherited disorder that increases the levels of a substance called phenylalanine in the blood. Phenylalanine is a building block of proteins (an amino acid) that is obtain...

**Answer**:

Pyle disease is a bone disorder characterized by genu valgum (knock knees), Erlenmeyer flask deformity (where there is relative constriction of the diaphysis or shaft of the bone and flaring of the metaphysis or end of the bone), widening of the ribs and clavicles (collarbones), platyspondyly (flattening of the bones of the spine) and cortical thinning. Cranial involvement is minimal with some showing mild hyperostosis (excessive new bone formation) of the skull base and thickening of the frontal and occipital bones. Pyle disease is passed through families in an autosomal recessive manner [medquad_ans_0005212-1].

**Supported citations**: `medquad_ans_0005212-1`
**Unsupported citations**: _none_

### Example 214 (`0000051-3__dup2`)

**Question**: What causes Deep Vein Thrombosis ?

**Scores**: context_relevance=0.800, groundedness=0.618, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000051-3__dup2` (score=0.9954) - Blood clots can form in your body's deep veins if: A vein's inner lining is damaged. Injuries caused by physical, chemical, or biological factors can damage the veins. Such factors include surgery, serious injuries, i...
2. `medquad_ans_0000108-3__dup2` (score=0.0331) - Major Causes Pulmonary embolism (PE) usually begins as a blood clot in a deep vein of the leg. This condition is called deep vein thrombosis. The clot can break free, travel through the bloodstream to the lungs, and b...
3. `medquad_ans_0000108-4__dup2` (score=0.0002) - Pulmonary embolism (PE) occurs equally in men and women. The risk increases with age. For every 10 years after age 60, the risk of having PE doubles. Certain inherited conditions, such as factor V Leiden, increase the...
4. `medquad_ans_0000104-1__dup4` (score=0.0001) - Summary : Normally, if you get hurt, your body forms a blood clot to stop the bleeding. Some people get too many clots or their blood clots abnormally. Many conditions can cause the blood to clot too much or prevent b...
5. `medquad_ans_0000456-1` (score=0.0001) - Hereditary antithrombin deficiency is a disorder of blood clotting. People with this condition are at higher than average risk for developing abnormal blood clots, particularly a type of clot that occurs in the deep v...

**Answer**:

Deep vein thrombosis (DVT) can be caused by:

- Damage to a vein's inner lining due to injuries from physical, chemical, or biological factors such as surgery, serious injuries, inflammation, or immune responses [medquad_ans_0000051-3__dup2].
- Sluggish or slow blood flow, which may occur after surgery, when ill and bedridden for a long time, or during long periods of travel [medquad_ans_0000051-3__dup2; medquad_ans_0000108-3__dup2].
- Blood that is thicker or more likely to clot than normal, such as in inherited conditions like factor V Leiden, or due to hormone therapy or birth control pills [medquad_ans_0000051-3\_\_dup2; medquad_ans_00001

**Supported citations**: `medquad_ans_0000051-3__dup2`
**Unsupported citations**: `medquad_ans_0000051-3__dup2; medquad_ans_0000108-3__dup2`

### Example 215 (`0004081-1`)

**Question**: What is (are) Mitochondrial complex IV deficiency ?

**Scores**: context_relevance=0.571, groundedness=0.006, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0004086-1` (score=0.0234) - Mitochondrial genetic disorders refer to a group of conditions that affect the mitochondria (the structures in each cell of the body that are responsible for making energy). People with these conditions can present at...
2. `medquad_ans_0000665-1` (score=0.0088) - Mitochondrial trifunctional protein deficiency is a rare condition that prevents the body from converting certain fats to energy, particularly during periods without food (fasting). Signs and symptoms of mitochondrial...
3. `medquad_ans_0000196-1__dup2` (score=0.0076) - Mitochondrial myopathies are a group of neuromuscular diseases caused by damage to the mitochondriasmall, energy-producing structures that serve as the cells' "power plants." Nerve cells in the brain and muscles requi...
4. `medquad_ans_0003635-1` (score=0.0064) - Lipoic acid synthetase deficiency is a rare condition that affects the mitochondria. Mitochondria are tiny structures found in almost every cell of the body. They are responsible for creating most of the energy necess...
5. `medquad_ans_0000728-4` (score=0.0052) - This condition is inherited in a mitochondrial pattern, which is also known as maternal inheritance. This pattern of inheritance applies to genes contained in mtDNA. Because egg cells, but not sperm cells, contribute ...

**Answer**:

The provided context does not mention "Mitochondrial complex IV deficiency." Therefore, based on the available information, it is not possible to answer the question. [medquad_ans_0004086-1, medquad_ans_0000665-1, medquad_ans_0000196-1__dup2]

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0004086-1, medquad_ans_0000665-1, medquad_ans_0000196-1__dup2`

### Example 216 (`0000078-1`)

**Question**: What is (are) ataxia with oculomotor apraxia ?

**Scores**: context_relevance=0.857, groundedness=0.510, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000078-1` (score=0.4373) - Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement. The hallmark of this condition is difficulty coordinating movements (ataxia), which is often the first symptom. Most a...
2. `medquad_ans_0000495-1__dup1` (score=0.0007) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...
3. `medquad_ans_0000388-2` (score=0.0004) - Friedreich ataxia is estimated to affect 1 in 40,000 people. This condition is found in people with European, Middle Eastern, or North African ancestry. It is rarely identified in other ethnic groups.
4. `medquad_ans_0000097-1` (score=0.0004) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
5. `medquad_ans_0000079-2` (score=0.0003) - Ataxia with vitamin E deficiency is a rare condition; however, its prevalence is unknown.

**Answer**:

Ataxia with oculomotor apraxia is a condition characterized by progressive problems with movement, with the hallmark being difficulty coordinating movements (ataxia), which is often the first symptom. Most affected individuals also have oculomotor apraxia, making it difficult to move their eyes side-to-side, and thus must turn their head to see peripheral vision. There are multiple types of this condition, which are very similar but caused by mutations in different genes. The two most common types (types 1 and 2) share features including involuntary jerking movements (chorea), muscle twitches (myoclonus), and disturbances in nerve function (neuropathy). In type 1, ataxia begins around age 4, while in type 2, it begins around age 15. Chorea and myoclonus tend to disappear gradually in type 1 but persist throughout life in type 2. Individuals with

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 217 (`0000221-1`)

**Question**: What is (are) congenital deafness with labyrinthine aplasia, microtia, and microdontia ?

**Scores**: context_relevance=0.909, groundedness=0.543, answer_relevance=0.909

**Top docs**:

1. `medquad_ans_0000221-1` (score=0.9399) - Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects development of the ears and teeth. In people with this condition, the structures that fo...
2. `medquad_ans_0001158-1` (score=0.0010) - Cholesteatoma is a type of skin cyst located in the middle ear. It can be congenital (present from birth), but it more commonly occurs as a complication of chronic ear infection. The hallmark symptom is a painless dis...
3. `medquad_ans_0000273-1` (score=0.0008) - Deafness and myopia syndrome is a disorder that causes problems with both hearing and vision. People with this disorder have moderate to profound hearing loss in both ears that may worsen over time. The hearing loss m...
4. `medquad_ans_0001496-1` (score=0.0002) - Congenital deafness with vitiligo and achalasia is a syndrome characterized by deafness present from birth (congenital), associated with short stature, vitiligo, muscle wasting and achalasia (swallowing difficulties)....
5. `medquad_ans_0000623-1` (score=0.0002) - Mandibulofacial dysostosis with microcephaly (MFDM) is a disorder that causes abnormalities of the head and face. People with this disorder often have an unusually small head at birth, and the head does not grow at th...

**Answer**:

Congenital deafness with labyrinthine aplasia, microtia, and microdontia (also called LAMM syndrome) is a condition that affects the development of the ears and teeth. In people with this condition, the structures that form the inner ear are usually completely absent (labyrinthine aplasia). Rarely, affected individuals have some underdeveloped inner ear structures in one or both ears. The abnormalities of the inner ear cause a form of hearing loss called sensorineural deafness that is present from birth (congenital). Because the inner ear is important for balance as well as hearing, development of motor skills, such as sitting and crawling, may be delayed in affected infants. In addition, people with LAMM syndrome often have abnormally small outer ears (microtia) with narrow ear canals. They can also have unusually small, widely spaced teeth (microdontia) [medquad_ans_000

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 218 (`0000064-4`)

**Question**: Is Apert syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.000, answer_relevance=0.900

**Top docs**:

1. `medquad_ans_0000683-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000770-4` (score=0.9987) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000556-4` (score=0.9985) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...
4. `medquad_ans_0000031-4` (score=0.9984) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene. These cases oc...
5. `medquad_ans_0000283-4` (score=0.9984) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In approximately 45 percent of cases, an affected person inherit...

**Answer**:

The provided context does not specify whether Apert syndrome is inherited. While the context discusses autosomal dominant and autosomal recessive inheritance patterns, it does not mention Apert syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine if Apert syndrome is inherited.

[Note: The context lacks specific information about Apert syndrome.]

**Supported citations**: _none_
**Unsupported citations**: `Note: The context lacks specific information about Apert syndrome.`

### Example 219 (`0000341-3__dup2`)

**Question**: What is (are) Q Fever ?

**Scores**: context_relevance=0.750, groundedness=0.550, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0005229-1` (score=0.1319) - Q fever is a worldwide disease with acute and chronic stages caused by the bacteria known as Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs although a variety of species may be infected. Organi...
2. `medquad_ans_0000341-3__dup2` (score=0.0073) - More detailed information on the diagnosis, management, and treatment of Q fever is available in other sections of this web site and in the materials referenced in the section titled “Further Reading”. How to Contact ...
3. `medquad_ans_0000341-1__dup3` (score=0.0047) - Q fever can cause acute or chronic illness in humans, who usually acquire infection after contact with infected animals or exposure to contaminated environments. The acute symptoms caused by infection with Coxiella bu...
4. `medquad_ans_0000359-1__dup1` (score=0.0039) - A fever is a body temperature that is higher than normal. It is not an illness. It is part of your body's defense against infection. Most bacteria and viruses that cause infections do well at the body's normal tempera...
5. `medquad_ans_0000341-4__dup2` (score=0.0027) - In the United States, Q fever outbreaks have resulted mainly from occupational exposure involving veterinarians, meat processing plant workers, sheep and dairy workers, livestock farmers, and researchers at facilities...

**Answer**:

Q fever is a worldwide disease with acute and chronic stages caused by the bacteria Coxiella burnetii. Cattle, sheep, and goats are the primary reservoirs, although a variety of species may be infected. Organisms are excreted in birth fluids, milk, urine, and feces of infected animals and can survive for long periods in the environment. Infection of humans usually occurs by inhalation of these organisms from air that contains airborne barnyard dust contaminated by dried placental material, birth fluids, and excreta of infected animals. Other modes of transmission, including tick bites, ingestion of unpasteurized milk or dairy products, and human to human transmission, are rare. Humans are often very susceptible to the disease, and very few organisms may be required to cause infection. In less than 5% of cases, people with acute Q fever develop a chronic form of the disease. Treatment of the acute form is with antibiotics,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 220 (`0000038-5__dup1`)

**Question**: Who is at risk for Childhood Interstitial Lung Disease? ?

**Scores**: context_relevance=1.000, groundedness=0.939, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0000112-8` (score=0.9944) - Taking steps to ensure a healthy pregnancy might prevent your infant from being born before his or her lungs have fully developed. These steps include: Seeing your doctor regularly during your pregnancy Following a he...
2. `medquad_ans_0000038-5__dup1` (score=0.9936) - Childhood interstitial lung disease (chILD) is rare. Most children are not at risk for chILD. However, some factors increase the risk of developing chILD. These risk factors include: Having a family history of interst...
3. `medquad_ans_0000112-4__dup2` (score=0.5546) - Certain factors may increase the risk that your infant will have respiratory distress syndrome (RDS). These factors include: Premature delivery. The earlier your baby is born, the greater his or her risk for RDS. Most...
4. `medquad_ans_0000020-4__dup2` (score=0.0239) - People who have conditions that damage the lungs or increase the risk of lung infections are at risk for bronchiectasis. Such conditions include: Cystic fibrosis. This disease leads to almost half of the cases of bron...
5. `medquad_ans_0000038-4__dup2` (score=0.0159) - Researchers don't yet know all of the causes of childhood interstitial lung disease (chILD). Many times, these diseases have no clear cause. Some conditions and factors that may cause or lead to chILD include: Inherit...

**Answer**:

Children who have a family history of interstitial lung disease or childhood interstitial lung disease (chILD), inherited surfactant disorders, or a family history of such disorders are at increased risk. Other risk factors include problems with aspiration, immune system disorders, exposure to environmental lung irritants such as molds and chemicals, having systemic or autoimmune diseases (e.g., collagen vascular disease or inflammatory bowel disease), undergoing cancer treatments like radiation or chemotherapy, or having a bone marrow or lung transplant [medquad_ans_0000038-5__dup1].

**Supported citations**: `medquad_ans_0000038-5__dup1`
**Unsupported citations**: _none_

### Example 221 (`0003334-1`)

**Question**: What is (are) Juvenile temporal arteritis ?

**Scores**: context_relevance=1.000, groundedness=0.943, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0003334-1` (score=0.9011) - Juvenile temporal arteritis is a rare form of vasculitis, a group of conditions that cause inflammation of the blood vessels. Unlike the classic form of temporal arteritis, this condition is generally diagnosed in lat...
2. `medquad_ans_0000944-1__dup1` (score=0.0039) - Vasculitis is an inflammation of the blood vessels. It happens when the body's immune system attacks the blood vessel by mistake. It can happen because of an infection, a medicine, or another disease. The cause is oft...
3. `medquad_ans_0000533-1__dup1` (score=0.0009) - Juvenile arthritis (JA) is arthritis that happens in children. It causes joint swelling, pain, stiffness, and loss of motion. It can affect any joint, but is more common in the knees, hands, and feet. In some cases it...
4. `medquad_ans_0006001-1` (score=0.0006) - Takayasu arteritis is a condition that causes inflammation of the main blood vessel that carries blood from the heart to the rest of the body (aorta) and its associated branched blood vessels. As a result of the infla...
5. `medquad_ans_0000171-1__dup1` (score=0.0004) - Klippel-Trenaunay syndrome (KTS) is a rare congenital malformation involving blood and lymph vessels and abnormal growth of soft and bone tissue. Typical symptoms include hemangiomas (abnormal benign growths on the sk...

**Answer**:

Juvenile temporal arteritis is a rare form of vasculitis that causes inflammation of the blood vessels, specifically the temporal arteries located at the lower sides of the skull, directly underneath the temple. It is generally diagnosed in late childhood or early adulthood and often presents without signs or symptoms, with affected individuals sometimes having only a painless nodule or lump in the temporal region. The exact underlying cause is unknown, and the condition typically occurs sporadically with no family history. It is usually treated with surgical excision and rarely recurs [medquad_ans_0003334-1].

**Supported citations**: `medquad_ans_0003334-1`
**Unsupported citations**: _none_

### Example 222 (`0006577-1`)

**Question**: What is (are) 2-methylbutyryl-CoA dehydrogenase deficiency ?

**Scores**: context_relevance=0.833, groundedness=0.538, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0006577-1` (score=0.9565) - 2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of a particul...
2. `medquad_ans_0001082-1` (score=0.9285) - 2-methylbutyryl-CoA dehydrogenase deficiency is a type of organic acid disorder in which the body is unable to process proteins properly. Organic acid disorders lead to an abnormal buildup of particular acids known as...
3. `medquad_ans_0006567-1` (score=0.2868) - 2,4-Dienoyl-CoA reductase deficiency is associated with hypotonia and respiratory acidosis in infancy. This condition may be associated with the DECR1 gene and likely has an autosomal recessive pattern of inheritance.
4. `medquad_ans_0003501-1` (score=0.0118) - LCHAD deficiency, or long-chain 3-hydroxyacyl-CoA dehydrogenase deficiency, is a mitochondrial condition that prevents the body from converting certain fats to energy, particularly during periods without food (fasting...
5. `medquad_ans_0001859-1` (score=0.0059) - Dihydrolipoamide dehydrogenase (DLD) deficiency is a very rare condition that can vary in age of onset, symptoms and severity. The condition may be characterized by early-onset lactic acidosis and delayed development ...

**Answer**:

2-methylbutyryl-CoA dehydrogenase deficiency is a metabolic disorder in which individuals lack adequate levels of an enzyme called 2-methylbutyryl-CoA dehydrogenase. This enzyme assists in the processing of the amino acid isoleucine. The inability to process isoleucine correctly leads to the buildup of the amino acid in the body, which can cause a variety of health problems that vary widely from severe and life-threatening to mild or absent. Signs and symptoms can begin a few days after birth or later in childhood and often include poor feeding, lack of energy (lethargy), vomiting, and irritability. These symptoms may progress to serious medical problems such as difficulty breathing, seizures, and coma. Additional issues can include poor growth, vision problems, learning disabilities, muscle weakness, and delays in motor skills like standing and walking. Symptoms may be triggered by prolonged fasting, infections, or increased protein

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 223 (`0005840-2`)

**Question**: What are the symptoms of Spondyloepiphyseal dysplasia congenita ?

**Scores**: context_relevance=1.000, groundedness=0.992, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005840-2` (score=0.9994) - What are the signs and symptoms of Spondyloepiphyseal dysplasia congenita? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepiphyseal dysplasia congenita. If the information ...
2. `medquad_ans_0000947-1` (score=0.0001) - Spondyloepiphyseal dysplasia congenita is an inherited bone growth disorder that results in short stature (dwarfism), skeletal abnormalities, and problems with vision and hearing. This condition affects the bones of t...
3. `medquad_ans_0005840-1` (score=0.0001) - Spondyloepiphyseal dysplasia congenita is an inherited disorder of bone growth that affects the bones of the spine and ends of the long bones in the arms and legs. Features of this condition include short stature (dwa...
4. `medquad_ans_0000948-1` (score=0.0001) - Spondyloperipheral dysplasia is a disorder that impairs bone growth. This condition is characterized by flattened bones of the spine (platyspondyly) and unusually short fingers and toes (brachydactyly), with the excep...
5. `medquad_ans_0005831-1` (score=0.0001) - What are the signs and symptoms of Spondyloepimetaphyseal dysplasia Shohat type? The Human Phenotype Ontology provides the following list of signs and symptoms for Spondyloepimetaphyseal dysplasia Shohat type. If the ...

**Answer**:

The symptoms of Spondyloepiphyseal dysplasia congenita include abnormal form of the vertebral bodies, abnormality of epiphysis morphology, micromelia, narrow chest, short neck, short stature, short thorax, skeletal dysplasia, broad forehead, cleft palate, hyperlordosis, hypertelorism, malar flattening, osteoarthritis, talipes, cataract, glaucoma, hearing impairment, kyphosis, myopia, nystagmus, retinal detachment, and scoliosis [medquad_ans_0005840-2]. Additionally, other associated features may include barrel-shaped chest, cervical myelopathy, coxa vara, delayed calcaneal ossification, delayed pubic bone ossification, flat face, flattened epiphysis, hip dislocation, hypoplasia of the odontoid process, limitation of knee mobility, limited

**Supported citations**: `medquad_ans_0005840-2`
**Unsupported citations**: _none_

### Example 224 (`0005219-3`)

**Question**: What causes Pyridoxine-dependent epilepsy ?

**Scores**: context_relevance=1.000, groundedness=0.543, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0005219-3` (score=0.9912) - What causes pyridoxine-dependent epilepsy? Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each ...
2. `medquad_ans_0000094-3` (score=0.0018) - Mutations in the LGI1 gene cause ADPEAF. This gene provides instructions for making a protein called Lgi1 or epitempin, which is found primarily in nerve cells (neurons) in the brain. Although researchers have propose...
3. `medquad_ans_0000852-2` (score=0.0001) - Pyridoxal 5'-phosphate-dependent epilepsy is a rare condition; approximately 14 cases have been described in the scientific literature.
4. `medquad_ans_0000853-2` (score=0.0001) - Pyridoxine-dependent epilepsy occurs in 1 in 100,000 to 700,000 individuals. At least 100 cases have been reported worldwide.
5. `medquad_ans_0005219-4` (score=0.0001) - How might pyridoxine-dependent epilepsy be treated? Anticonvulsant drugs, which are usually given to control seizures, are ineffective in people with pyridoxine-dependent epilepsy. Instead, people with this type of se...

**Answer**:

Mutations in the ALDH7A1 gene cause pyridoxine-dependent epilepsy. This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The ALDH7A1 gene provides instructions for making an enzyme called α-aminoadipic semialdehyde (α-AASA) dehydrogenase, also known as antiquitin. This enzyme is involved in the breakdown of the protein building block (amino acid) lysine in the brain. When antiquitin is deficient, a molecule that interferes with vitamin B6 function builds up in various tissues. It is unclear how a lack of pyridoxine causes the seizures that are characteristic of this condition. Some individuals with pyridoxine-dependent epilepsy do not have identified mutations in the ALDH7A1 gene. In these cases, the cause of the condition is unknown [medquad_ans_00052

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 225 (`0000004_6-2`)

**Question**: Who is at risk for Childhood Hodgkin Lymphoma? ?

**Scores**: context_relevance=0.875, groundedness=0.918, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000004_6-2` (score=0.9949) - Epstein-Barr virus infection increases the risk of childhood Hodgkin lymphoma. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cance...
2. `medquad_ans_0000015_1-3` (score=0.0142) - Having certain inherited disorders can increase the risk of an extracranial germ cell tumor. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you ...
3. `medquad_ans_0000001_7-2` (score=0.0056) - The risk factors for childhood AML, childhood CML, JMML, and MDS are similar. Anything that increases your risk of getting a disease is called a risk factor. Having a risk factor does not mean that you will get cancer...
4. `medquad_ans_0000004_2-6` (score=0.0042) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The type of non-Hodgkin lymph...
5. `medquad_ans_0000004_6-5` (score=0.0027) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis (chance of recovery) and treatment options depend on the following: - The stage of the cancer. - The size of the tumor. - Whet...

**Answer**:

Children who are infected with the Epstein-Barr virus, those infected with the human immunodeficiency virus (HIV), children with certain diseases of the immune system, children with a personal history of mononucleosis ("mono"), and children who have a parent or sibling with a personal history of Hodgkin lymphoma are at increased risk for childhood Hodgkin lymphoma [medquad_ans_0000004_6-2]. Being exposed to common infections in early childhood may decrease the risk of Hodgkin lymphoma in children due to its effect on the immune system [medquad_ans_0000004_6-2].

**Supported citations**: `medquad_ans_0000004_6-2`
**Unsupported citations**: _none_

### Example 226 (`0001795-1`)

**Question**: What is (are) Dermatitis herpetiformis ?

**Scores**: context_relevance=1.000, groundedness=0.646, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0000095-1__dup4` (score=0.8953) - Dermatitis herpetiformis is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp. The face and groin can also be involved. A burning sensation ...
2. `medquad_ans_0001795-1` (score=0.7929) - Dermatitis herpetiformis is a rare, chronic, skin disorder characterized by groups of severely itchy blisters and raised skin lesions. These are more common on the knees, elbows, buttocks and shoulder blades. The slow...
3. `medquad_ans_0005124-1` (score=0.0012) - Prurigo nodularis is a skin condition characterized by hard crusty lumps that itch intensely. The exact cause of the condition is unknown. However, it can occur in isolation or as a result of repeated trauma to chroni...
4. `medquad_ans_0000470-1__dup1` (score=0.0012) - Hidradenitis suppurativa (HS) is a chronic skin disease. It can occur in one or multiple areas of your body. HS usually develops in your armpits, groin, and anal area. It causes long-term skin inflammation and can be ...
5. `medquad_ans_0000095-2__dup3` (score=0.0006) - Dermatitis herpetiformis is caused by the deposit of immunoglobulin A (IgA) in the skin, which triggers further immunologic reactions resulting in lesion formation. DH is an external manifestation of an abnormal immun...

**Answer**:

Dermatitis herpetiformis is a rare, chronic skin disorder characterized by groups of severely itchy blisters and raised skin lesions, which are more common on the knees, elbows, buttocks, and shoulder blades [medquad_ans_0001795-1]. It is characterized by small, clustered papules and vesicles that erupt symmetrically on the elbows, knees, buttocks, back, or scalp, and the face and groin can also be involved [medquad_ans_0000095-1]. A burning sensation may precede lesion formation, and lesions are usually scratched off by the time a patient comes in for a physical exam, appearing as erosions and excoriations [medquad_ans_0000095-1]. The condition is caused by the deposit of immunoglobulin A (IgA) in the skin, triggering immunologic reactions resulting in lesion formation

**Supported citations**: `medquad_ans_0001795-1`
**Unsupported citations**: `medquad_ans_0000095-1`

### Example 227 (`0000709-5`)

**Question**: What are the treatments for myotonia congenita ?

**Scores**: context_relevance=0.857, groundedness=0.596, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000210-2__dup2` (score=0.9956) - Treatment for myotonia may include mexiletine, quinine, phenytoin, and other anticonvulsant drugs. Physical therapy and other rehabilitative measures may help muscle function.
2. `medquad_ans_0000709-5` (score=0.9926) - These resources address the diagnosis or management of myotonia congenita: - Gene Review: Gene Review: Myotonia Congenita - Genetic Testing Registry: Congenital myotonia, autosomal dominant form - Genetic Testing Regi...
3. `medquad_ans_0000084-2__dup1` (score=0.9908) - The possibilities for treatment depend on the specific subtype of congenital myasthenia. Most treatments attempt to improve the signaling between nerve cell and muscle. These drugs include pyridostigmine, fluoxetine, ...
4. `medquad_ans_0000040-2__dup1` (score=0.9899) - Most people with myotonia congenita dont require special treatments. Stiff muscles usually resolve with exercise, or light movement, especially after resting. For individuals whose symptoms are more limiting, doctors ...
5. `medquad_ans_0000040-3__dup1` (score=0.9090) - Most individuals with myotonia congenita lead long, productive lives. Although muscle stiffness may interfere with walking, grasping, chewing, and swallowing, it is usually relieved with exercise.

**Answer**:

Treatments for myotonia congenita may include medications such as quinine, phenytoin, and mexiletine, which are anticonvulsant drugs. Physical therapy and other rehabilitative measures may also help improve muscle function. In most cases, stiff muscles resolve with exercise or light movement, especially after rest. For individuals with more limiting symptoms, these medications and rehabilitative therapies have shown some success [medquad_ans_0000210-2__dup2, medquad_ans_0000040-2__dup1]. However, there are no treatments to cure the underlying genetic abnormality [medquad_ans_0000084-2__dup1].

**Supported citations**: `medquad_ans_0000084-2__dup1`
**Unsupported citations**: `medquad_ans_0000210-2__dup2, medquad_ans_0000040-2__dup1`

### Example 228 (`0000133-13`)

**Question**: What to do for Wilson Disease ?

**Scores**: context_relevance=1.000, groundedness=0.463, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000133-13` (score=0.8860) - - Wilson disease is a genetic disease that prevents the body from removing extra copper. - Normally, the liver filters extra copper and releases it into bile. In Wilson disease, the liver does not filter copper correc...
2. `medquad_ans_0000133-11` (score=0.8679) - People with Wilson disease should reduce their dietary copper intake by avoiding foods that are high in copper, such as - shellfish - liver - mushrooms - nuts - chocolate People should not eat these foods during the i...
3. `medquad_ans_0000133-12` (score=0.0097) - A person cannot prevent Wilson disease; however, people with a family history of Wilson disease, especially those with an affected sibling or parent, should talk with a health care provider about testing. A health car...
4. `medquad_ans_0001037-5` (score=0.0008) - These resources address the diagnosis or management of Wilson disease: - Gene Review: Gene Review: Wilson Disease - Genetic Testing Registry: Wilson's disease - MedlinePlus Encyclopedia: Wilson's disease - National Hu...
5. `medquad_ans_0000133-8` (score=0.0006) - People who have Wilson disease that is not treated or diagnosed early can have serious complications, such as - cirrhosisscarring of the liver - kidney damageas liver function decreases, the kidneys may be damaged - p...

**Answer**:

For Wilson disease, treatment involves a lifelong effort to reduce and control copper levels in the body. This may include medications, changes in eating, diet, and nutrition, and in some cases, a liver transplant. People should reduce their dietary copper intake by avoiding high-copper foods such as shellfish, liver, mushrooms, nuts, and chocolate, especially during initial treatment; however, they should consult a health care provider to determine if these foods can be consumed in moderation during maintenance treatment. Additionally, individuals should check the copper levels in their tap water, as water from copper pipes or wells may contain copper residue; running water helps reduce copper levels. People should not use copper containers or cookware to store or prepare food or drinks. It is also important to discuss the use of complementary and alternative medical practices, including vitamins and dietary supplements, with a health care provider, as some supplements contain copper. If a supplement is recommended, a pharmacist can suggest copper-free options

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 229 (`0000592-3`)

**Question**: What are the genetic changes related to Lennox-Gastaut syndrome ?

**Scores**: context_relevance=0.889, groundedness=0.969, answer_relevance=0.622

**Top docs**:

1. `medquad_ans_0000592-3` (score=0.9980) - Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. About two-thirds of cases are described as symptomatic, which means that they are re...
2. `medquad_ans_0000814-4` (score=0.0001) - More than 99 percent of PPM-X syndrome cases occur in people with no history of the disorder in their family. Many of these cases result from new mutations in the MECP2 gene. A few families with more than one affected...
3. `medquad_ans_0000920-3` (score=0.0001) - Most people with Smith-Magenis syndrome have a deletion of genetic material from a specific region of chromosome 17. Although this region contains multiple genes, researchers believe that the loss of one particular ge...
4. `medquad_ans_0000794-4` (score=0.0001) - This condition is inherited in an X-linked recessive pattern. The PGK1 gene is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome), one altered copy of the g...
5. `medquad_ans_0000592-2` (score=0.0001) - Lennox-Gastaut syndrome affects an estimated 1 in 50,000 to 1 in 100,000 children. This condition accounts for about 4 percent of all cases of childhood epilepsy. For unknown reasons, it appears to be more common in m...

**Answer**:

Researchers have not identified any genes specific to Lennox-Gastaut syndrome, although the disorder likely has a genetic component. [medquad_ans_0000592-3]

**Supported citations**: `medquad_ans_0000592-3`
**Unsupported citations**: _none_

### Example 230 (`0000066-2`)

**Question**: How many people are affected by arginine:glycine amidinotransferase deficiency ?

**Scores**: context_relevance=0.600, groundedness=0.969, answer_relevance=0.440

**Top docs**:

1. `medquad_ans_0000065-2` (score=0.9995) - Arginase deficiency is a very rare disorder; it has been estimated to occur once in every 300,000 to 1,000,000 individuals.
2. `medquad_ans_0000066-2` (score=0.9994) - The prevalence of arginine:glycine amidinotransferase deficiency is unknown. The disorder has been identified in only a few families.
3. `medquad_ans_0000442-2` (score=0.9993) - Guanidinoacetate methyltransferase deficiency is a very rare disorder. About 80 affected individuals have been described in the medical literature. Of these, approximately one-third are of Portuguese origin.
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
2. `medquad_ans_0000379-1` (score=0.0001) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. About 90 percent of affected individuals are female. Males usually have milder signs and symptoms than females. ...
3. `medquad_ans_0002399-1` (score=0.0001) - Focal dermal hypoplasia is a genetic disorder that primarily affects the skin, skeleton, eyes, and face. The skin abnormalities are present from birth and can include streaks of very thin skin (dermal hypoplasia), cut...
4. `medquad_ans_0004506-2` (score=0.0001) - What are the signs and symptoms of Oculodentodigital dysplasia? Individuals with oculodentodigital dysplasia commonly have small eyes (microphthalmia) and other eye abnormalities that can lead to vision loss. They als...
5. `medquad_ans_0000379-2` (score=0.0001) - Focal dermal hypoplasia appears to be a rare condition, although its exact prevalence is unknown.

**Answer**:

The symptoms of Focal dermal hypoplasia include skin abnormalities such as streaks of very thin skin (dermal hypoplasia), yellowish-pink nodules of fat under the skin, areas where the top layers of skin are absent (cutis aplasia), telangiectases, and streaks of slightly darker or lighter skin. These can cause pain, itching, irritation, or lead to skin infections. With age, most develop wart-like growths (papillomas) around the nostrils, lips, anus, and female genitalia, and may also be present in the throat, causing problems with swallowing, breathing, or sleeping. Other features include small, ridged fingernails and toenails, sparse or brittle scalp hair, hand and foot abnormalities such as missing fingers or toes (oligodactyly), webbed or fused fingers or toes (syndactyly), and a deep split in

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 232 (`0000418-2__dup1`)

**Question**: Who is at risk for Parasites - Trichinellosis (also known as Trichinosis)? ?

**Scores**: context_relevance=0.909, groundedness=0.851, answer_relevance=0.909

**Top docs**:

1. `medquad_ans_0000418-1__dup2` (score=0.9876) - Trichinellosis, also called trichinosis, is caused by eating raw or undercooked meat of animals infected with the larvae of a species of worm called Trichinella. Infection occurs commonly in certain wild carnivorous (...
2. `medquad_ans_0000418-2__dup1` (score=0.9752) - People acquire trichinellosis by consuming raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Even tasting very small amounts of undercooked meat during preparation or...
3. `medquad_ans_0000415-1__dup2` (score=0.9549) - A single-celled parasite called Toxoplasma gondii causes a disease known as toxoplasmosis. While the parasite is found throughout the world, more than 60 million people in the United States may be infected with the To...
4. `medquad_ans_0000432-2__dup1` (score=0.9464) - Whipworm is a soil-transmitted helminth (STH) and is the third most common roundworm of humans. Whipworm causes an infection called trichuriasis and often occurs in areas where human feces is used as fertilizer or whe...
5. `medquad_ans_0000990-2` (score=0.5317) - Trichohepatoenteric syndrome is a rare condition with an estimated prevalence of about 1 in 1 million people. At least 44 cases have been reported in the medical literature.

**Answer**:

People at risk for trichinellosis (also known as trichinosis) include those who consume raw or undercooked meat infected with the Trichinella parasite, particularly wild game meat or pork. Eating undercooked wild game, especially bear meat, puts individuals at risk. Additionally, individuals who taste very small amounts of undercooked meat during preparation or cooking are also at risk. Pregnant women and those with compromised immune systems are not specifically mentioned as being at risk for trichinellosis in the provided context, though they are noted as being cautious with toxoplasmosis [medquad_ans_0000418-2__dup1].

The primary risk group identified in the context is consumers of raw or undercooked meat, especially from wild animals or domestic pigs [medquad_ans_0000418-1__dup2].

**Supported citations**: `medquad_ans_0000418-2__dup1`, `medquad_ans_0000418-1__dup2`
**Unsupported citations**: _none_

### Example 233 (`0001924-2`)

**Question**: What are the symptoms of Dravet syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.771

**Top docs**:

1. `medquad_ans_0001924-2` (score=0.9994) - What are the signs and symptoms of Dravet syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Dravet syndrome. If the information is available, the table below includes how oft...
2. `medquad_ans_0000145-3__dup3` (score=0.0005) - The symptoms of Goodpasture syndrome may initially include fatigue, nausea, vomiting, and weakness. The lungs are usually affected before or at the same time as the kidneys, and symptoms can include shortness of breat...
3. `medquad_ans_0003700-2` (score=0.0001) - What are the signs and symptoms of Lymphedema-distichiasis syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Lymphedema-distichiasis syndrome. If the information is available...
4. `medquad_ans_0005016-2` (score=0.0001) - What are the signs and symptoms of Prader-Willi syndrome? In infancy, Prader-Willi syndrome (PWS) is characterized by weak muscle tone (hypotonia), feeding difficulties, poor growth, and delayed development. In later ...
5. `medquad_ans_0002307-2` (score=0.0001) - What are the signs and symptoms of Fanconi Bickel syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Fanconi Bickel syndrome. If the information is available, the table below ...

**Answer**:

The signs and symptoms of Dravet syndrome include absence seizures, ataxia, autosomal dominant inheritance, cerebral atrophy, cortical visual impairment, epileptic encephalopathy, focal seizures with impairment of consciousness or awareness, generalized myoclonic seizures, hemiclonic seizures, infantile onset, mental deterioration, motor delay, postnatal microcephaly, and status epilepticus [medquad_ans_0001924-2].

**Supported citations**: `medquad_ans_0001924-2`
**Unsupported citations**: _none_

### Example 234 (`0000826-3`)

**Question**: What are the genetic changes related to prion disease ?

**Scores**: context_relevance=0.889, groundedness=0.954, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000826-3` (score=0.9974) - Between 10 and 15 percent of all cases of prion disease are caused by mutations in the PRNP gene. Because they can run in families, these forms of prion disease are classified as familial. Familial prion diseases, whi...
2. `medquad_ans_0000011-13` (score=0.0010) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
3. `medquad_ans_0000239-4__dup1` (score=0.0006) - The NINDS conducts and supports research on TSEs. This research is aimed at determining how abnormal prion proteins lead to disease, at finding better tests for diagnosing CJD and other disorders, and ultimately at fi...
4. `medquad_ans_0000031-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. Most cases result from new mutations in the gene. These cases oc...
5. `medquad_ans_0000556-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, an affected person inherits the mutation from one...

**Answer**:

Genetic changes related to prion disease include mutations in the PRNP gene, which occur in between 10 and 15 percent of all prion disease cases. These mutations result in the production of an abnormally shaped protein called PrPSc from one copy of the gene. This abnormal protein can bind to the normal prion protein (PrPC) and promote its transformation into PrPSc, leading to the accumulation of abnormal protein in the brain and neuronal damage [medquad_ans_0000826-3].

**Supported citations**: `medquad_ans_0000826-3`
**Unsupported citations**: _none_

### Example 235 (`0006430-1`)

**Question**: What are the symptoms of Weyers ulnar ray/oligodactyly syndrome ?

**Scores**: context_relevance=1.000, groundedness=0.985, answer_relevance=0.840

**Top docs**:

1. `medquad_ans_0006430-1` (score=0.9994) - What are the signs and symptoms of Weyers ulnar ray/oligodactyly syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Weyers ulnar ray/oligodactyly syndrome. If the information ...
2. `medquad_ans_0000420-1__dup1` (score=0.0001) - Apert syndrome is a disorder mainly characterized by craniosynostosis (premature fusion of skull bones, causing abnormalities in the shape of the head and face) and syndactyly (fusion or webbing or fingers and/or toes...
3. `medquad_ans_0005440-2` (score=0.0001) - What are the signs and symptoms of Russell-Silver syndrome? Signs and symptoms of Russell-Silver syndrome (RSS) can vary and may include: intrauterine growth restriction low birth weight poor growth short stature curv...
4. `medquad_ans_0000593-2__dup1` (score=0.0001) - What are the signs and symptoms of Axenfeld-Rieger syndrome type 1? The Human Phenotype Ontology provides the following list of signs and symptoms for Axenfeld-Rieger syndrome type 1. If the information is available, ...
5. `medquad_ans_0002911-2` (score=0.0001) - What are the signs and symptoms of Holt-Oram syndrome? People with Holt-Oram syndrome have abnormally developed bones in their upper limbs. At least one abnormality in the bones of the wrist (carpal bones) is present....

**Answer**:

The symptoms of Weyers ulnar ray/oligodactyly syndrome include absent thumb, aplasia/hypoplasia of the ulna, hypoplasia of the radius, oligodactyly (hands), proximal placement of thumb, proximal radial head dislocation, cleft palate, cleft upper lip, high palate, hydronephrosis, hypotelorism, long face, narrow face, mesomelia, and autosomal dominant inheritance [medquad_ans_0006430-1].

**Supported citations**: `medquad_ans_0006430-1`
**Unsupported citations**: _none_

### Example 236 (`0005029-1`)

**Question**: What are the symptoms of Premature aging Okamoto type ?

**Scores**: context_relevance=1.000, groundedness=0.931, answer_relevance=0.911

**Top docs**:

1. `medquad_ans_0005029-1` (score=0.9994) - What are the signs and symptoms of Premature aging Okamoto type? The Human Phenotype Ontology provides the following list of signs and symptoms for Premature aging Okamoto type. If the information is available, the ta...
2. `medquad_ans_0004534-2` (score=0.0003) - What are the signs and symptoms of Omenn syndrome? Infants with Omenn syndrome typically present shortly after birth, usually by 3 months of age. This is similar to other types of severe combined immunodeficiency (SCI...
3. `medquad_ans_0005916-1` (score=0.0001) - What are the signs and symptoms of Subaortic stenosis short stature syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Subaortic stenosis short stature syndrome. If the inform...
4. `medquad_ans_0002049-2` (score=0.0001) - What are the signs and symptoms of Ehlers-Danlos syndrome? There are six major types of Ehlers-Danlos syndrome (EDS). Although there is significant overlap in associated features, the subtypes are classified based on ...
5. `medquad_ans_0003546-2` (score=0.0001) - What are the signs and symptoms of Leprechaunism? The Human Phenotype Ontology provides the following list of signs and symptoms for Leprechaunism. If the information is available, the table below includes how often t...

**Answer**:

The symptoms of Premature aging Okamoto type include abnormality of blood and blood-forming tissues, abnormality of the hair, abnormality of the pinna, cataract, depressed nasal bridge, diabetes mellitus, growth abnormality, low-set ears, microcephaly, neoplasm, osteoporosis, osteosarcoma, and round face [medquad_ans_0005029-1]. The frequency of these symptoms is estimated based on the Human Phenotype Ontology, with data often derived from Orphanet, though specific percentages or fractions are not provided for all symptoms [medquad_ans_0005029-1].

**Supported citations**: `medquad_ans_0005029-1`
**Unsupported citations**: _none_

### Example 237 (`0000979-2`)

**Question**: How many people are affected by thrombocytopenia-absent radius syndrome ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0001065-2` (score=0.9994) - The estimated incidence of X-linked thrombocytopenia is between 1 and 10 per million males worldwide; this condition is rarer among females.
2. `medquad_ans_0001005-2` (score=0.9962) - The prevalence of TH deficiency is unknown.
3. `medquad_ans_0000307-2` (score=0.0257) - Dyserythropoietic anemia and thrombocytopenia is a rare condition; its prevalence is unknown. Occasionally, individuals with this disorder are mistakenly diagnosed as having more common blood disorders, making it even...
4. `medquad_ans_0000130-3__dup2` (score=0.0094) - People who are at highest risk for thrombocytopenia are those affected by one of the conditions or factors discussed in "What Causes Thrombocytopenia?" This includes people who: Have certain types of cancer, aplastic ...
5. `medquad_ans_0000329-2` (score=0.0007) - Essential thrombocythemia affects an estimated 1 to 24 per 1 million people worldwide.

**Answer**:

The provided context does not contain any information about thrombocytopenia-absent radius syndrome. Therefore, the number of people affected by this condition cannot be determined from the given text. [no relevant context]

**Supported citations**: _none_
**Unsupported citations**: `no relevant context`

### Example 238 (`0000265-4__dup1`)

**Question**: what research (or clinical trials) is being done for Tay-Sachs Disease ?

**Scores**: context_relevance=0.455, groundedness=0.860, answer_relevance=0.527

**Top docs**:

1. `medquad_ans_0000137-4__dup1` (score=0.9983) - The NINDS supports and conducts research on TSEs, including GSS. Much of this research is aimed at characterizing the agents that cause these disorders, clarifying the mechanisms underlying them, and, ultimately, find...
2. `medquad_ans_0000135-4__dup1` (score=0.9936) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
3. `medquad_ans_0000265-4__dup1` (score=0.9934) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system and to use that knowledge to reduce the burden of neurological dis...
4. `medquad_ans_0000130-4__dup1` (score=0.0353) - The NINDS supports research aimed at discovering new ways to diagnose, treat, and, ultimately, cure neuropathies such as Miller Fisher syndrome.
5. `medquad_ans_0000265-3__dup1` (score=0.0007) - Even with the best of care, children with Tay-Sachs disease usually die by age 4, from recurring infection.

**Answer**:

Research on Tay-Sachs disease is focused on expanding the use of virus-delivered gene therapy, which has been observed in an animal model of Tay-Sachs disease and is being explored for use in humans [medquad_ans_0000265-4__dup1]. Additionally, NINDS-funded research on gangliosidoses, which includes Tay-Sachs disease, involves using variations of magnetic resonance imaging to develop a biomarker for evaluating brain biochemistry and disease progression [medquad_ans_0000265-4__dup1]. This research aims to improve diagnosis and understand disease mechanisms. [medquad_ans_0000135-4__dup1]

**Supported citations**: `medquad_ans_0000265-4__dup1`, `medquad_ans_0000135-4__dup1`
**Unsupported citations**: _none_

### Example 239 (`0000008-1`)

**Question**: What is (are) achromatopsia ?

**Scores**: context_relevance=1.000, groundedness=0.544, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000008-1` (score=0.4357) - Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achro...
2. `medquad_ans_0000010-1__dup2` (score=0.0318) - Agnosia is a rare disorder characterized by an inability to recognize and identify objects or persons. People with agnosia may have difficulty recognizing the geometric features of an object or face or may be able to ...
3. `medquad_ans_0000008-2` (score=0.0004) - Achromatopsia affects an estimated 1 in 30,000 people worldwide. Complete achromatopsia is more common than incomplete achromatopsia. Complete achromatopsia occurs frequently among Pingelapese islanders, who live on o...
4. `medquad_ans_0000163-6` (score=0.0001) - The first step is to make sure the diagnosis is correct. For more information, patients should consult their health care provider. Health care providers may consult with CDC staff about the diagnosis and treatment of ...
5. `medquad_ans_0000025-1__dup1` (score=0.0001) - An acanthoma is a small, reddish bump that usually develops on the skin of an older adult. There are several types of acanthoma, including "acantholytic", "epidermolytic", "clear cell", and "melanoacanthoma". Though m...

**Answer**:

Achromatopsia is a condition characterized by a partial or total absence of color vision. People with complete achromatopsia cannot perceive any colors; they see only black, white, and shades of gray. Incomplete achromatopsia is a milder form that allows some color discrimination. Achromatopsia also involves other vision problems, including increased sensitivity to light and glare (photophobia), involuntary back-and-forth eye movements (nystagmus), and significantly reduced sharpness of vision (low visual acuity). Affected individuals can also have farsightedness (hyperopia) or, less commonly, nearsightedness (myopia). These vision problems develop in the first few months of life. Achromatopsia is different from common forms of color vision deficiency (color blindness), in which people can perceive color but have difficulty distinguishing between certain colors, such as red and green. Complete achromatops

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 240 (`0000039-11`)

**Question**: What are the treatments for Knee Replacement ?

**Scores**: context_relevance=0.857, groundedness=0.749, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000039-17` (score=0.9958) - A physical therapist will teach you exercises to help your recovery. You can expect some pain, discomfort, and stiffness as you begin therapy, but to get the best results from your new knee, it is important to do all ...
2. `medquad_ans_0000029-18` (score=0.9947) - Treatments may include deep cleaning, medications, surgery, and bone and tissue grafts.
3. `medquad_ans_0000049-18` (score=0.9923) - Warm towels, hot packs, or a warm bath or shower can provide temporary pain relief. Medications such as non-steroidal anti-inflammatory drugs, or NSAIDs, help reduce pain and inflammation that result from osteoarthrit...
4. `medquad_ans_0000049-22` (score=0.9918) - For many people, surgery helps relieve the pain and disability of osteoarthritis. Surgery may be performed to achieve one or more of the following. - Removal of loose pieces of bone and cartilage from the joint if the...
5. `medquad_ans_0000039-9` (score=0.9869) - The main reason to have knee replacement surgery is to ease pain and disability caused by arthritis or other joint problems, while preserving movement. Less commonly, it is used to correct some kinds of knee deformity.

**Answer**:

The provided context does not specify treatments for knee replacement surgery. It discusses treatments for osteoarthritis, such as deep cleaning, medications, surgery, bone and tissue grafts, and specific surgical procedures like arthroscopic debridement, osteotomy, and joint resurfacing. However, it does not explicitly list or define treatments specifically for knee replacement. Therefore, based on the given context, the treatments for knee replacement cannot be determined.

[medquad_ans_0000029-18]

**Supported citations**: `medquad_ans_0000029-18`
**Unsupported citations**: _none_

### Example 241 (`0000090-5__dup1`)

**Question**: What are the symptoms of Metabolic Syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.029, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000015-15` (score=0.9988) - Many people with diabetes experience one or more symptoms, including extreme thirst or hunger, a frequent need to urinate and/or fatigue. Some lose weight without trying. Additional signs include sores that heal slowl...
2. `medquad_ans_0000082-3__dup3` (score=0.9972) - The signs and symptoms of diabetes are - being very thirsty - urinating often - feeling very hungry - feeling very tired - losing weight without trying - sores that heal slowly - dry, itchy skin - feelings of pins and...
3. `medquad_ans_0000043-2__dup4` (score=0.9971) - Many people have no visible signs or symptoms of diabetes. Symptoms can also be so mild that you might not notice them. More than 5 million people in the United States have type 2 diabetes and do not know it. - increa...
4. `medquad_ans_0000015-4__dup2` (score=0.1396) - Diabetes is often called a "silent" disease because it can cause serious complications even before you have symptoms. Symptoms can also be so mild that you dont notice them. An estimated 8 million people in the United...
5. `medquad_ans_0000109-6` (score=0.0006) - Signs and symptoms of pulmonary hypertension (PH) may include: Shortness of breath during routine activity, such as climbing two flights of stairs Tiredness Chest pain A racing heartbeat Pain on the upper right side o...

**Answer**:

The provided context does not contain information about the symptoms of Metabolic Syndrome. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 242 (`0005211-1`)

**Question**: What are the symptoms of Pyknoachondrogenesis ?

**Scores**: context_relevance=1.000, groundedness=0.957, answer_relevance=0.867

**Top docs**:

1. `medquad_ans_0005211-1` (score=0.9994) - What are the signs and symptoms of Pyknoachondrogenesis? The Human Phenotype Ontology provides the following list of signs and symptoms for Pyknoachondrogenesis. If the information is available, the table below includ...
2. `medquad_ans_0002152-2` (score=0.0001) - What are the signs and symptoms of Erdheim-Chester disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Erdheim-Chester disease. If the information is available, the table below ...
3. `medquad_ans_0000986-2__dup1` (score=0.0001) - What are the signs and symptoms of Catamenial pneumothorax? The Human Phenotype Ontology provides the following list of signs and symptoms for Catamenial pneumothorax. If the information is available, the table below ...
4. `medquad_ans_0004689-1` (score=0.0001) - What are the signs and symptoms of Palmer Pagon syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Palmer Pagon syndrome. If the information is available, the table below incl...
5. `medquad_ans_0004117-1` (score=0.0001) - What are the signs and symptoms of Monomelic amyotrophy? The Human Phenotype Ontology provides the following list of signs and symptoms for Monomelic amyotrophy. If the information is available, the table below includ...

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
3. `medquad_ans_0003290-1` (score=0.0001) - What are the signs and symptoms of Jansen type metaphyseal chondrodysplasia? The Human Phenotype Ontology provides the following list of signs and symptoms for Jansen type metaphyseal chondrodysplasia. If the informat...
4. `medquad_ans_0005440-2` (score=0.0001) - What are the signs and symptoms of Russell-Silver syndrome? Signs and symptoms of Russell-Silver syndrome (RSS) can vary and may include: intrauterine growth restriction low birth weight poor growth short stature curv...
5. `medquad_ans_0000181-1__dup2` (score=0.0001) - Leukodystrophy refers to progressive degeneration of the white matter of the brain due to imperfect growth or development of the myelin sheath, the fatty covering that acts as an insulator around nerve fiber. Myelin, ...

**Answer**:

The symptoms of Metachromatic leukodystrophy include behavioral abnormality, cognitive impairment, decreased nerve conduction velocity, developmental regression, gait disturbance, genu recurvatum, incoordination, muscle weakness, neurological speech impairment, peripheral neuropathy, reduced consciousness/confusion, seizures, amaurosis fugax, hyperreflexia, hypertonia, limitation of joint mobility, muscular hypotonia, nystagmus, optic atrophy, aganglionic megacolon, ataxia, Babinski sign, bulbar palsy, cholecystitis, chorea, delusions, dysarthria, dystonia, EMG: neuropathic changes, emotional lability, gallbladder dysfunction, hallucinations, hyporeflexia, increased CSF protein, intellectual disability, loss of speech, mental deterioration, peripheral demyelination, progressive peripheral neuropathy, spastic tetraplegia, tetraplegia

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 244 (`0000186-4`)

**Question**: Is chronic granulomatous disease inherited ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.680

**Top docs**:

1. `medquad_ans_0000705-4` (score=0.9989) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0000797-4` (score=0.9989) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0000493-4` (score=0.9989) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000238-4` (score=0.9989) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000178-4` (score=0.9989) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not mention chronic granulomatous disease or its inheritance pattern. Therefore, it is not possible to determine from the given information whether chronic granulomatous disease is inherited.

Answer: Insufficient context [medquad_ans_0000705-4, medquad_ans_0000797-4, medquad_ans_0000493-4, medquad_ans_0000238-4, medquad_ans_0000178-4].

**Supported citations**: _none_
**Unsupported citations**: `medquad_ans_0000705-4, medquad_ans_0000797-4, medquad_ans_0000493-4, medquad_ans_0000238-4, medquad_ans_0000178-4`

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

**Scores**: context_relevance=0.800, groundedness=0.945, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000094-3__dup3` (score=0.9953) - The cause of cyclic vomiting syndrome is unknown. However, some experts believe that some possible problems with bodily functions may contribute to the cause, such as the following: - gastrointestinal motilitythe way ...
2. `medquad_ans_0000098-3__dup3` (score=0.0875) - Dumping syndrome is caused by problems with the storage of food particles in the stomach and emptying of particles into the duodenum. Early dumping syndrome results from rapid movement of fluid into the intestine foll...
3. `medquad_ans_0000094-12` (score=0.0025) - During the prodrome and vomiting phases of cyclic vomiting syndrome, a person will generally take in little or no nutrition by mouth. During the recovery phase, the person may be quite hungry as soon as the vomiting s...
4. `medquad_ans_0000107-8` (score=0.0006) - The problems of gastroparesis can include - severe dehydration due to persistent vomiting - gastroesophageal reflux disease (GERD), which is GER that occurs more than twice a week for a few weeks; GERD can lead to eso...
5. `medquad_ans_0000094-11` (score=0.0004) - A person should stay away from known triggers, especially during the well phase, as well as - get adequate sleep to prevent exhaustion - treat sinus problems or allergies - seek help on reducing stress and anxiety - a...

**Answer**:

The cause of cyclic vomiting syndrome is unknown. However, some experts believe that possible problems with bodily functions may contribute to the cause, such as gastrointestinal motility, central nervous system function, autonomic nervous system function, hormone imbalances, and in children, an abnormal inherited gene [medquad_ans_0000094-3__dup3].

**Supported citations**: `medquad_ans_0000094-3__dup3`
**Unsupported citations**: _none_

### Example 247 (`0000180-1__dup2`)

**Question**: Do you have information about Child Mental Health

**Scores**: context_relevance=0.750, groundedness=0.920, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000189-1__dup2` (score=0.9981) - Summary : Your child's health includes physical, mental and social well-being. Most parents know the basics of keeping children healthy, like offering them healthy foods, making sure they get enough sleep and exercise...
2. `medquad_ans_0000180-1__dup2` (score=0.6764) - Summary : It's important to recognize and treat mental illnesses in children early on. Once mental illness develops, it becomes a regular part of your child's behavior and is more difficult to treat. But it's not alwa...
3. `medquad_ans_0000177-1__dup2` (score=0.6129) - Summary : Children's healthy development depends on safe and positive experiences when they are very young. If you work or go to school, you want to know that your child is in good hands while you are away. You may ch...
4. `medquad_ans_0000603-1__dup1` (score=0.0020) - Summary : Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel and act as we cope with life. It also helps determine how we handle stress, relate to others, and mak...
5. `medquad_ans_0000046-5` (score=0.0001) - These resources address the diagnosis or management of alternating hemiplegia of childhood: - The Great Ormond Street Hospital - University of Utah School of Medicine These resources from MedlinePlus offer information...

**Answer**:

Yes, there is information about child mental health. The context indicates that it is important to recognize and treat mental illnesses in children early on, as they become a regular part of the child's behavior and are more difficult to treat later. Warning signs of a more serious mental health problem include problems in more than one setting, changes in appetite or sleep, social withdrawal or fear of things previously not feared, returning to younger behaviors like bedwetting, signs of being upset such as sadness or tearfulness, self-destructive behaviors like head-banging or sudden injuries, and repeated thoughts of death. Diagnosis involves evaluating the child's signs and symptoms, medical history, and family history. Treatments include medicines and talk therapy [medquad_ans_0000180-1__dup2].

**Supported citations**: `medquad_ans_0000180-1__dup2`
**Unsupported citations**: _none_

### Example 248 (`0001618-2`)

**Question**: What are the symptoms of Cowden syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.087, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000088-4__dup2` (score=0.9988) - A person may experience digestive signs and symptoms, or symptoms in other parts of the body. Digestive signs and symptoms are more common in children and can include - abdominal bloating - chronic diarrhea - constipa...
2. `medquad_ans_0000054-11` (score=0.9958) - People who have P.A.D. may have symptoms when walking or climbing stairs. These may include pain, numbness, aching, or heaviness in the leg muscles. Symptoms may also include cramping in the affected leg(s) and in the...
3. `medquad_ans_0000014-6` (score=0.0035) - Most people with primary hyperparathyroidism have no symptoms. When symptoms appear, they are often mild and nonspecific, such as - muscle weakness - fatigue and an increased need for sleep - feelings of depression - ...
4. `medquad_ans_0000088-6` (score=0.0017) - Signs and symptoms of celiac disease vary from person to person because of numerous factors, including - the length of time a person was breastfed as an infant; some studies have shown that the longer an infant was br...
5. `medquad_ans_0000093-5__dup1` (score=0.0013) - The most common signs and symptoms of Crohn's disease are - diarrhea - abdominal cramping and pain - weight loss Other general signs and symptoms include - feeling tired - nausea or loss of appetite - fever - anemiaa ...

**Answer**:

The provided context does not mention Cowden syndrome or its symptoms. Therefore, it is not possible to answer the question based on the given information. [doc_id]

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 249 (`0000395-5`)

**Question**: What are the treatments for Fukuyama congenital muscular dystrophy ?

**Scores**: context_relevance=0.778, groundedness=0.719, answer_relevance=0.778

**Top docs**:

1. `medquad_ans_0000395-5` (score=0.9929) - These resources address the diagnosis or management of Fukuyama congenital muscular dystrophy: - Gene Review: Gene Review: Congenital Muscular Dystrophy Overview - Gene Review: Gene Review: Fukuyama Congenital Muscula...
2. `medquad_ans_0000334-5` (score=0.0936) - These resources address the diagnosis or management of facioscapulohumeral muscular dystrophy: - Gene Review: Gene Review: Facioscapulohumeral Muscular Dystrophy - Genetic Testing Registry: Facioscapulohumeral muscula...
3. `medquad_ans_0000576-5` (score=0.0004) - These resources address the diagnosis or management of LAMA2-related muscular dystrophy: - Boston Children's Hospital: Treatment and Care for Muscular Dystrophy - Gene Review: Gene Review: LAMA2-Related Muscular Dystr...
4. `medquad_ans_0006267-3` (score=0.0003) - How might Ullrich muscular dystrophy be treated? Physical therapy, including early mobilization, regular stretching and splinting, is the main focus of supportive care. Respiratory support and night-time ventilation o...
5. `medquad_ans_0000666-5` (score=0.0002) - These resources address the diagnosis or management of Miyoshi myopathy: - Gene Review: Gene Review: ANO5-Related Muscle Diseases - Gene Review: Gene Review: Dysferlinopathy - Genetic Testing Registry: Miyoshi muscula...

**Answer**:

The provided context does not specify the treatments for Fukuyama congenital muscular dystrophy. While it lists resources that address the diagnosis or management of Fukuyama congenital muscular dystrophy, such as Gene Review and Genetic Testing Registry entries, no detailed information on specific treatments is included [medquad_ans_0000395-5]. Therefore, based on the available context, the treatments for Fukuyama congenital muscular dystrophy cannot be determined.

**Supported citations**: `medquad_ans_0000395-5`
**Unsupported citations**: _none_

### Example 250 (`0000004_6-4`)

**Question**: How to diagnose Childhood Hodgkin Lymphoma ?

**Scores**: context_relevance=1.000, groundedness=0.805, answer_relevance=1.000

**Top docs**:

1. `medquad_ans_0000004_7-3` (score=0.9978) - Tests that examine the body and lymph system are used to detect (find) and diagnose childhood non-Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to ...
2. `medquad_ans_0000004_6-4` (score=0.9976) - Tests that examine the lymph system are used to detect (find) and diagnose childhood Hodgkin lymphoma. The following tests and procedures may be used: - Physical exam and history : An exam of the body to check general...
3. `medquad_ans_0001059-5` (score=0.0004) - These resources address the diagnosis or management of XLP: - Children's Hospital of Philadelphia - Gene Review: Gene Review: Lymphoproliferative Disease, X-Linked - Genetic Testing Registry: Lymphoproliferative syndr...
4. `medquad_ans_0000004_7-2` (score=0.0001) - Signs of childhood non-Hodgkin lymphoma include breathing problems and swollen lymph nodes. These and other signs may be caused by childhood non-Hodgkin lymphoma or by other conditions. Check with a doctor if your chi...
5. `medquad_ans_0000004_6-3` (score=0.0001) - Signs of childhood Hodgkin lymphoma include swollen lymph nodes, fever, night sweats, and weight loss. These and other signs and symptoms may be caused by childhood Hodgkin lymphoma or by other conditions. Check with ...

**Answer**:

The provided context does not contain information on how to diagnose Childhood Hodgkin Lymphoma. It only discusses tests used to detect and diagnose childhood non-Hodgkin lymphoma [medquad_ans_0000004_7-3]. Therefore, based on the given context, the diagnosis of Childhood Hodgkin Lymphoma cannot be determined.

**Supported citations**: `medquad_ans_0000004_7-3`
**Unsupported citations**: _none_

### Example 251 (`0000556-4`)

**Question**: Is KBG syndrome inherited ?

**Scores**: context_relevance=0.500, groundedness=0.098, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001083-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000045-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000370-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000858-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0000122-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not mention KBG syndrome or its mode of inheritance. Therefore, based on the given information, it is not possible to determine whether KBG syndrome is inherited.

Answer: Insufficient context to determine if KBG syndrome is inherited.

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 252 (`0000012-1__dup2`)

**Question**: What is (are) Aicardi-Goutieres Syndrome Disorder ?

**Scores**: context_relevance=0.833, groundedness=0.539, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000012-1__dup2` (score=0.9683) - Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is...
2. `medquad_ans_0000197-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
3. `medquad_ans_0000196-1__dup1` (score=0.6838) - Aicardi-Goutieres syndrome is an inherited condition that mainly affects the brain, immune system, and skin. It is characterized by early-onset severe brain dysfunction (encephalopathy) that usually results in severe ...
4. `medquad_ans_0000029-1` (score=0.5035) - Aicardi-Goutieres syndrome is a disorder that mainly affects the brain, the immune system, and the skin. Most newborns with Aicardi-Goutieres syndrome do not show any signs or symptoms of the disorder at birth. Howeve...
5. `medquad_ans_0000028-1` (score=0.0742) - Aicardi syndrome is a disorder that occurs almost exclusively in females. It is characterized by three main features that occur together in most affected individuals. People with Aicardi syndrome have absent or underd...

**Answer**:

Aicardi-Goutieres syndrome (AGS) is an inherited encephalopathy that affects newborn infants and usually results in severe mental and physical handicap. There are two forms of the syndrome: an early-onset form that is severe and a late-onset form that has less impact upon neurological function. The early-onset form affects about 20 percent of all babies who have AGS and is characterized by neurological and liver abnormalities, such as enlargement of the liver and spleen and elevated liver enzymes. Jittery behavior and poor feeding ability mimic congenital viral infection. The later-onset form begins after the first weeks or months of normal development, with symptoms including progressive decline in head growth, weak or stiffened muscles (spasticity), cognitive and developmental delays, irritability, inconsolable crying, intermittent fever, seizures, loss of developmental skills, puffy swelling on the fingers, toes, and ears resembling chilblains,

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 253 (`0000026_1-8`)

**Question**: what research (or clinical trials) is being done for Renal Cell Cancer ?

**Scores**: context_relevance=0.750, groundedness=0.134, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000008-30` (score=0.9967) - Clinical trials are research studies on people to find out whether a new drug or treatment is both safe and effective. New therapies are tested on people only after laboratory and animal studies show promising results...
2. `medquad_ans_0000006_3-8` (score=0.9938) - New types of treatment are being tested in clinical trials. This summary section describes treatments that are being studied in clinical trials. It may not mention every new treatment being studied. Information about ...
3. `medquad_ans_0000022-4__dup1` (score=0.9931) - The mission of the National Institute of Neurological Disorders and Stroke (NINDS) is to seek fundamental knowledge about the brain and nervous system, and to use that knowledge to reduce the burden of neurological di...
4. `medquad_ans_0000040-23` (score=0.9891) - Clinical trials are research studies in which new treatments -- drugs, diagnostics, procedures, vaccines, and other therapies -- are tested in people to see if they are safe, effective, and better than the current sta...
5. `medquad_ans_0000038-33` (score=0.2934) - There are many researchers who are working on kidney disease. They are looking for ways to improve diagnosis, make treatments more effective, and make dialysis and transplantation work better. Several areas of researc...

**Answer**:

The provided context does not mention any research or clinical trials specifically for Renal Cell Cancer. Therefore, based on the given information, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 254 (`0000190-9`)

**Question**: What to do for Primary Biliary Cirrhosis ?

**Scores**: context_relevance=0.857, groundedness=0.923, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000183-9` (score=0.9931) - Infants with biliary atresia often have nutritional deficiencies and require special diets as they grow up. They may need a higher calorie diet, because biliary atresia leads to a faster metabolism. The disease also p...
2. `medquad_ans_0000191-8` (score=0.9736) - - Primary sclerosing cholangitis (PSC) inflames, scars, and blocks bile ducts inside and outside the liver. - When bile ducts become blocked, bile builds up in the liver and damages liver cells. - PSC can lead to vita...
3. `medquad_ans_0000190-9` (score=0.9146) - - Primary biliary cirrhosis is a chronic disease that causes the small bile ducts in the liver to become inflamed and damaged and ultimately disappear. - When chronic inflammation damages the bile ducts, bile and toxi...
4. `medquad_ans_0000184-9` (score=0.8577) - - Cirrhosis is a condition in which the liver slowly deteriorates and is unable to function normally due to chronic, or long lasting, injury. Scar tissue replaces healthy liver tissue and partially blocks the flow of ...
5. `medquad_ans_0000193-7` (score=0.5855) - To prevent cirrhosis, - see your doctor for treatment of your liver disease. Many of the causes of cirrhosis are treatable. Early treatment may prevent cirrhosis. - try to keep your weight in the normal range. Being o...

**Answer**:

For Primary Biliary Cirrhosis, health care providers may use a medical and family history, a physical exam, blood tests, imaging tests, and a liver biopsy to diagnose the condition. Treatment includes prescribing ursodiol (Actigall, Urso) to manage the disease [medquad_ans_0000190-9]. Most complications are related to cirrhosis and occur after the disease progresses to cirrhosis. The causes are unknown, but research suggests it is an autoimmune disease, and it is more common in individuals with a family history, particularly identical twins [medquad_ans_0000190-9].

**Supported citations**: `medquad_ans_0000190-9`
**Unsupported citations**: _none_

### Example 255 (`0000299-1__dup2`)

**Question**: What is (are) Drowning ?

**Scores**: context_relevance=0.750, groundedness=0.963, answer_relevance=0.550

**Top docs**:

1. `medquad_ans_0000193-1__dup3` (score=0.9860) - Food or small objects can cause choking if they get caught in your throat and block your airway. This keeps oxygen from getting to your lungs and brain. If your brain goes without oxygen for more than four minutes, yo...
2. `medquad_ans_0000299-1__dup2` (score=0.9778) - People drown when they get too much water in their lungs. You can drown in as little as an inch or two of water. Babies can drown in a sink or bathtub. Preschoolers are most likely to drown in a swimming pool. People ...
3. `medquad_ans_0000366-1__dup1` (score=0.0082) - Summary : Accidents happen. Someone chokes on an ice cube or gets stung by a bee. It is important to know when to call 9-1-1 -- it is for life-threatening emergencies. While waiting for help to arrive, you may be able...
4. `medquad_ans_0000872-1__dup2` (score=0.0059) - Summary : You may only think of tears as those salty drops that fall from your eyes when you cry. Actually, your tears clean your eyes every time you blink. Tears also keep your eyes moist, which is important for your...
5. `medquad_ans_0000967-1__dup1` (score=0.0016) - Summary : Playing in the water - whether swimming, boating or diving - can be fun. It can also be dangerous, especially for children. Being safe can help prevent injuries and drowning. To stay safe in the water - Avoi...

**Answer**:

Drowning occurs when a person gets too much water in their lungs. It can happen in as little as an inch or two of water. Babies can drown in a sink or bathtub, and preschoolers are most likely to drown in a swimming pool. People with seizure disorders are also at risk in the water. Drowning can happen quickly and silently [medquad_ans_0000299-1__dup2].

**Supported citations**: `medquad_ans_0000299-1__dup2`
**Unsupported citations**: _none_

### Example 256 (`0002058-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, kyphoscoliosis type ?

**Scores**: context_relevance=0.857, groundedness=0.991, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0002058-1` (score=0.9604) - Ehlers-Danlos syndrome (EDS), kyphoscoliosis type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. Common signs and symptoms include hyperextensible skin that is fragi...
2. `medquad_ans_0000318-1__dup1` (score=0.6639) - Ehlers-Danlos syndrome (EDS) is a group of inherited disorders that weaken connective tissues. Connective tissues are proteins that support skin, bones, blood vessels, and other organs. EDS usually affects your skin, ...
3. `medquad_ans_0000313-1` (score=0.2940) - Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and sympt...
4. `medquad_ans_0002063-1` (score=0.0005) - Ehlers-Danlos syndrome (EDS), vascular type is an inherited connective tissue disorder that is caused by defects in a protein called collagen. It is generally considered the most severe form of Ehlers-Danlos syndrome....
5. `medquad_ans_0000313-2` (score=0.0003) - Although it is difficult to estimate the overall frequency of Ehlers-Danlos syndrome, the combined prevalence of all types of this condition may be about 1 in 5,000 individuals worldwide. The hypermobility and classic...

**Answer**:

Ehlers-Danlos syndrome, kyphoscoliosis type is an inherited connective tissue disorder caused by defects in a protein called collagen. It is caused by changes (mutations) in the PLOD1 gene and is inherited in an autosomal recessive manner. Common signs and symptoms include hyperextensible skin that is fragile and bruises easily, joint hypermobility, severe hypotonia at birth, progressive kyphoscoliosis (kyphosis and scoliosis), and fragility of the sclera. Treatment is focused on preventing serious complications and relieving associated signs and symptoms [medquad_ans_0002058-1].

**Supported citations**: `medquad_ans_0002058-1`
**Unsupported citations**: _none_

### Example 257 (`0001860-2`)

**Question**: What are the symptoms of Dihydropteridine reductase deficiency ?

**Scores**: context_relevance=1.000, groundedness=0.986, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001860-2` (score=0.9994) - What are the signs and symptoms of Dihydropteridine reductase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropteridine reductase deficiency. If the information is...
2. `medquad_ans_0001862-2` (score=0.0008) - What are the signs and symptoms of Dihydropyrimidine dehydrogenase deficiency? The Human Phenotype Ontology provides the following list of signs and symptoms for Dihydropyrimidine dehydrogenase deficiency. If the info...
3. `medquad_ans_0000898-2` (score=0.0001) - Sepiapterin reductase deficiency appears to be a rare condition. At least 30 cases have been described in the scientific literature.
4. `medquad_ans_0000451-2__dup1` (score=0.0001) - What are the signs and symptoms of Aromatic L-amino acid decarboxylase deficiency? Symptoms, which typically present during the first year of life, include severe developmental delay, weak muscle tone (hypotonia), mus...
5. `medquad_ans_0002595-2` (score=0.0001) - What are the signs and symptoms of glucose-6-phosphate dehydrogenase (G6PD) deficiency? People with G6PD deficiency do not have signs of the disease unless their red blood cells are exposed to certain chemicals in foo...

**Answer**:

The symptoms of Dihydropteridine reductase deficiency include cognitive impairment (90%), microcephaly (90%), cerebral calcification, choreoathetosis, dysphagia, dystonia, episodic fever, excessive salivation, hyperphenylalaninemia, hypertonia, infantile onset, intellectual disability, irritability, muscular hypotonia, myoclonus, progressive neurologic deterioration, seizures, tremor, and variable expressivity [medquad_ans_0001860-2].

**Supported citations**: `medquad_ans_0001860-2`
**Unsupported citations**: _none_

### Example 258 (`0000669-2`)

**Question**: How many people are affected by monilethrix ?

**Scores**: context_relevance=0.286, groundedness=0.610, answer_relevance=0.514

**Top docs**:

1. `medquad_ans_0000989-2` (score=0.9994) - This condition affects an estimated 1 in 50,000 people.
2. `medquad_ans_0000484-2` (score=0.9994) - The exact incidence of this condition is unknown, but it is estimated to affect 1 in 87,000 people.
3. `medquad_ans_0000195-2` (score=0.9994) - The incidence of this condition is uncertain, but researchers estimate that the disorder affects 1 in 40,000 to 50,000 people.
4. `medquad_ans_0000091-2` (score=0.9993) - This condition is rare, affecting fewer than 1 per million people.
5. `medquad_ans_0000346-2` (score=0.9993) - This condition appears to be rare; only a few affected individuals have been reported worldwide.

**Answer**:

The provided context does not specify the number of people affected by monilethrix. Therefore, based on the given information, it is not possible to determine the exact incidence or prevalence of monilethrix. [medquad_ans_0000989-2][medquad_ans_0000484-2][medquad_ans_0000195-2][medquad_ans_0000091-2][medquad_ans_0000346-2]

**Supported citations**: `medquad_ans_0000989-2`, `medquad_ans_0000484-2`, `medquad_ans_0000195-2`, `medquad_ans_0000091-2`, `medquad_ans_0000346-2`
**Unsupported citations**: _none_

### Example 259 (`0002733-1`)

**Question**: What is (are) Hartnup disease ?

**Scores**: context_relevance=0.800, groundedness=0.531, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0002733-1` (score=0.3190) - Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system. It is a type of aminoaciduria. The condition may be diagnosed based on the ...
2. `medquad_ans_0000870-2` (score=0.0025) - RAMSVPS is a rare disorder. Only a small number of affected individuals and families, all from Saudi Arabia, have been described in the medical literature.
3. `medquad_ans_0000325-2` (score=0.0020) - EKVP is a rare disorder; its prevalence is unknown.
4. `medquad_ans_0000001-2` (score=0.0004) - Aarskog-Scott syndrome is believed to be a rare disorder; however, its prevalence is unknown because mildly affected people are often not diagnosed.
5. `medquad_ans_0001030-1__dup1` (score=0.0004) - Central core disease (CCD) is an inherited condition that involves muscle weakness, skeletal abnormalities, and an increased chance of having a severe reaction to some anesthesia medications. Muscle weakness ranges fr...

**Answer**:

Hartnup disease is a metabolic disorder characterized by abnormal transport of certain amino acids in the kidney and gastrointestinal system, making it a type of aminoaciduria. It is caused by mutations in the SLC6A19 gene and is inherited in an autosomal recessive manner. Most people with Hartnup disease have no symptoms (asymptomatic), though those who do show symptoms typically experience onset between the ages of 3 and 9, occasionally in adulthood. Signs and symptoms may include skin photosensitivity, neurologic findings, psychiatric symptoms, and ocular findings. Mental development is usually normal, though a few cases with intellectual impairment have been reported. Treatment may include a high-protein diet, protection from sunlight, vitamin supplementation, avoidance of certain drugs/medications, and in some cases, nicotinamide supplements and tryptophan ethyl ester [medquad_ans_0002733-

**Supported citations**: _none_
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

**Scores**: context_relevance=0.889, groundedness=0.813, answer_relevance=0.889

**Top docs**:

1. `medquad_ans_0000977-5` (score=0.9933) - These resources address the diagnosis or management of thiamine-responsive megaloblastic anemia syndrome: - Gene Review: Gene Review: Thiamine-Responsive Megaloblastic Anemia Syndrome - Genetic Testing Registry: Megal...
2. `medquad_ans_0002772-4` (score=0.0461) - How might hemoglobin E disease be treated? Treatment is usually not necessary. Folic acid supplements may be prescribed to help the body produce normal red blood cells and improve symptoms of anemia. People with hemog...
3. `medquad_ans_0000102-7` (score=0.0065) - Doctors treat pernicious anemia by replacing the missing vitamin B12 in the body. People who have pernicious anemia may need lifelong treatment. The goals of treating pernicious anemia include: Preventing or treating ...
4. `medquad_ans_0000002-7` (score=0.0045) - Treatment for anemia depends on the type, cause, and severity of the condition. Treatments may include dietary changes or supplements, medicines, procedures, or surgery to treat blood loss. Goals of Treatment The goal...
5. `medquad_ans_0000002-8` (score=0.0024) - You might be able to prevent repeat episodes of some types of anemia, especially those caused by lack of iron or vitamins. Dietary changes or supplements can prevent these types of anemia from occurring again. Treatin...

**Answer**:

The provided context does not contain specific information about the treatments for thiamine-responsive megaloblastic anemia syndrome. While it mentions resources related to the diagnosis and management of the condition, such as gene reviews, genetic testing, and MedlinePlus entries, no details on actual treatments are given [medquad_ans_0000977-5]. Therefore, based on the available context, the treatments for thiamine-responsive megaloblastic anemia syndrome cannot be determined.

**Supported citations**: `medquad_ans_0000977-5`
**Unsupported citations**: _none_

### Example 262 (`0000167-18`)

**Question**: What to do for Polycystic Kidney Disease ?

**Scores**: context_relevance=0.857, groundedness=0.732, answer_relevance=0.629

**Top docs**:

1. `medquad_ans_0000135-10` (score=0.9176) - - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. - Acquired cystic kidney disease occurs in children and adults who have - chronic kidney disease (CK...
2. `medquad_ans_0000167-18` (score=0.8975) - - Polycystic kidney disease (PKD) is a genetic disorder that causes numerous cysts to grow in the kidneys. - A gene mutation, or defect, causes polycystic kidney disease. - Autosomal dominant PKD is the most common fo...
3. `medquad_ans_0000281-1__dup2` (score=0.3016) - Summary : When your kidneys are healthy, they clean your blood. They also make hormones that keep your bones strong and your blood healthy. When your kidneys fail, you need treatment to replace the work your kidneys u...
4. `medquad_ans_0000038-20` (score=0.0042) - Treatments for early kidney disease include both diet and lifestyle changes and medications. Diet and lifestyle changes, such as eating heart healthy foods and exercising regularly to maintain a healthy weight, can he...
5. `medquad_ans_0000199-1__dup2` (score=0.0007) - You have two kidneys, each about the size of your fist. Their main job is to filter wastes and excess water out of your blood to make urine. They also keep the body's chemical balance, help control blood pressure, and...

**Answer**:

The provided context does not specify any treatments for polycystic kidney disease (PKD). It describes the symptoms, complications, and diagnostic methods (imaging tests and genetic testing) but does not outline specific therapeutic interventions. Therefore, based on the given information, no actions or treatments for PKD are explicitly stated.

[medquad_ans_0000167-18]

**Supported citations**: `medquad_ans_0000167-18`
**Unsupported citations**: _none_

### Example 263 (`0000061-5__dup2`)

**Question**: what research (or clinical trials) is being done for Rheumatoid Arthritis ?

**Scores**: context_relevance=0.818, groundedness=0.754, answer_relevance=0.818

**Top docs**:

1. `medquad_ans_0000253-4__dup1` (score=0.9979) - The goals of research on disorders such as Sjgren's syndrome focus on increasing knowledge and understanding of the disorder, improving diagnostic techniques, testing interventions, and finding ways to treat, prevent,...
2. `medquad_ans_0000018-14` (score=0.9978) - For information about the clinical trial on this topic, visit: http://www.clinicaltrials.gov/ct/show/NCT00372320?order=1. If you would like to read an interview with Dr. Bruce Baum, the study's principal investigator,...
3. `medquad_ans_0000090-4__dup1` (score=0.9973) - Much of the on-going research on RMDs is aimed at prevention and rehabilitation. The National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs.
4. `medquad_ans_0000008-30` (score=0.9969) - Clinical trials are research studies on people to find out whether a new drug or treatment is both safe and effective. New therapies are tested on people only after laboratory and animal studies show promising results...
5. `medquad_ans_0000207-1__dup2` (score=0.9962) - Summary : Clinical trials are research studies that test how well new medical approaches work in people. Each study answers scientific questions and tries to find better ways to prevent, screen for, diagnose, or treat...

**Answer**:

The provided context does not specify any research or clinical trials being conducted specifically for Rheumatoid Arthritis (RA). While it mentions that much of the ongoing research on rheumatic and musculoskeletal disorders (RMDs) is aimed at prevention and rehabilitation, and that the National Institute of Arthritis and Musculoskeletal and Skin Diseases (NIAMS) funds research on RMDs [medquad_ans_0000090-4__dup1], there is no direct information about specific research or clinical trials for Rheumatoid Arthritis. Therefore, based on the given context, the exact research or clinical trials for RA cannot be identified.

**Supported citations**: `medquad_ans_0000090-4__dup1`
**Unsupported citations**: _none_

### Example 264 (`0002561-4`)

**Question**: What are the treatments for Glass-Chapman-Hockley syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.988, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000199-2__dup1` (score=0.9954) - No cures or specific therapies for ML currently exists. Therapies are generally geared toward treating symptoms and providing supportive care to the child. For individuals with corneal clouding, surgery to remove the ...
2. `medquad_ans_0002561-4` (score=0.9770) - How might Glass-Chapman-Hockley syndrome be treated? Surgery is typically the treatment for craniosynostosis and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (an...
3. `medquad_ans_0003593-6` (score=0.9518) - How might Liddle syndrome be treated? Treatment for Liddle syndrome includes following a low sodium diet as well as taking potassium-sparing diuretics, which reduce blood pressure and correct hypokalemia and metabolic...
4. `medquad_ans_0006403-4` (score=0.9406) - How might Waterhouse-Friderichsen syndrome be treated? Treatment may include antibiotics and glucocorticoids. Other treatment is symptomatic and supportive.
5. `medquad_ans_0000860-2__dup1` (score=0.3046) - How might Brown-Sequard syndrome be treated?

**Answer**:

Surgery is typically the treatment for Glass-Chapman-Hockley syndrome and is based on the person's specific signs and symptoms. The goal is to increase the space in the front (anterior) part of the skull. The operation is usually performed when the person is between 9 to 12 months of age. If other sutures, other than the coronal suture, are involved, other surgeries may be performed [medquad_ans_0002561-4].

**Supported citations**: `medquad_ans_0002561-4`
**Unsupported citations**: _none_

### Example 265 (`0000868-5`)

**Question**: What are the treatments for Renpenning syndrome ?

**Scores**: context_relevance=0.571, groundedness=0.750, answer_relevance=0.457

**Top docs**:

1. `medquad_ans_0003615-3` (score=0.9334) - How might CREST syndrome be treated? Unfortunately, CREST syndrome has no known cure. The condition carries both physical and psychological consequences, so a holistic approach to management should be taken. Treatment...
2. `medquad_ans_0003593-6` (score=0.7537) - How might Liddle syndrome be treated? Treatment for Liddle syndrome includes following a low sodium diet as well as taking potassium-sparing diuretics, which reduce blood pressure and correct hypokalemia and metabolic...
3. `medquad_ans_0000247-2__dup2` (score=0.1648) - There is no cure for Rett syndrome. Treatment for the disorder is symptomatic, focusing on the management of symptoms, and supportive. Medication may be needed for breathing irregularities and motor difficulties, and ...
4. `medquad_ans_0000176-2__dup1` (score=0.0840) - Treatment for Wallenberg's syndrome is symptomatic. A feeding tube may be necessary if swallowing is very difficult. Speech/swallowing therapy may be beneficial. In some cases, medication may be used to reduce or elim...
5. `medquad_ans_0005119-5` (score=0.0784) - How might Proud syndrome be treated? The treatment of Proud syndrome is based on the signs and symptoms present in each person. For example, spasticity may be treated with a variety of therapies including medications ...

**Answer**:

The context provided does not contain any information about Renpenning syndrome. Therefore, it is not possible to provide a treatment based on the given context. [medquad_ans_0003615-3]

**Supported citations**: `medquad_ans_0003615-3`
**Unsupported citations**: _none_

### Example 266 (`0001105-1`)

**Question**: What are the symptoms of Charcot-Marie-Tooth disease type 2G ?

**Scores**: context_relevance=1.000, groundedness=0.988, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0001105-1` (score=0.9994) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2G? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2G. If the information is ava...
2. `medquad_ans_0001104-2` (score=0.0007) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2F? The subtypes of Charcot-Marie-Tooth type 2, including type 2F, have similar signs and symptoms. Affected individuals usually become symptomatic b...
3. `medquad_ans_0001088-2__dup1` (score=0.0003) - What are the signs and symptoms of Charcot-Marie-Tooth disease? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease. If the information is available, the tabl...
4. `medquad_ans_0001110-1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 2L? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 2L. If the information is ava...
5. `medquad_ans_0001092-1__dup1` (score=0.0001) - What are the signs and symptoms of Charcot-Marie-Tooth disease type 1C? The Human Phenotype Ontology provides the following list of signs and symptoms for Charcot-Marie-Tooth disease type 1C. If the information is ava...

**Answer**:

The symptoms of Charcot-Marie-Tooth disease type 2G include onion bulb formation (7.5%), areflexia, autosomal recessive inheritance, axonal degeneration/regeneration, decreased motor nerve conduction velocity, decreased number of peripheral myelinated nerve fibers, distal amyotrophy, distal muscle weakness, distal sensory impairment, flexion contracture, neonatal onset, pes cavus, spinal deformities, split hand, and vocal cord paresis [medquad_ans_0001105-1].

**Supported citations**: `medquad_ans_0001105-1`
**Unsupported citations**: _none_

### Example 267 (`0001393-2`)

**Question**: What are the symptoms of Coccygodynia ?

**Scores**: context_relevance=0.667, groundedness=0.029, answer_relevance=0.667

**Top docs**:

1. `medquad_ans_0000192-2__dup2` (score=0.9988) - Symptoms include - jaundice, which causes a yellowing of the skin and eyes - fatigue - abdominal pain - loss of appetite - nausea - vomiting - diarrhea - low grade fever - headache However, some people do not have sym...
2. `medquad_ans_0000011-5__dup2` (score=0.9916) - CJD does not cause the same symptoms in everyone. In most people who got CJD from pituitary hGH, the first signs they noticed were difficulty with walking and balance, dizziness, and/or clumsiness. Later, some began t...
3. `medquad_ans_0000012-10` (score=0.0004) - The most common symptoms of COPD are a cough that does not go away and coughing up a lot of sputum (mucus). These symptoms may occur years before lung damage has reduced the flow of air in and out of the lungs. Other ...
4. `medquad_ans_0000189-6` (score=0.0001) - Some people with porphyria-causing gene mutations have latent porphyria, meaning they have no symptoms of the disorder. Symptoms of cutaneous porphyrias include - oversensitivity to sunlight - blisters on exposed area...
5. `medquad_ans_0000148-1__dup3` (score=0.0001) - Carbon monoxide (CO) is a gas that has no odor or color. But it is very dangerous. It can cause sudden illness and death. CO is found in combustion fumes, such as those made by cars and trucks, lanterns, stoves, gas r...

**Answer**:

The provided context does not contain any information about the symptoms of coccygodynia. Therefore, it is not possible to answer the question based on the given text. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 268 (`0004025-1`)

**Question**: What are the symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate ?

**Scores**: context_relevance=1.000, groundedness=0.883, answer_relevance=0.933

**Top docs**:

1. `medquad_ans_0004025-1` (score=0.9994) - What are the signs and symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate? The Human Phenotype Ontology provides the following list of signs and symptoms for Microcephaly, corpus callosum dysgen...
2. `medquad_ans_0004549-2` (score=0.0001) - What are the signs and symptoms of Opitz G/BBB syndrome? Opitz G/BBB syndrome mainly affects structures along the midline of the body. The most common features of the condition are wide-spaced eyes (hypertelorism); de...
3. `medquad_ans_0004137-2` (score=0.0001) - What are the signs and symptoms of mosaic trisomy 9? The signs and symptoms of mosaic trisomy 9 vary but may include: Different degrees of developmental delay and intellectual disability Abnormal growth including low ...
4. `medquad_ans_0000623-1` (score=0.0001) - Mandibulofacial dysostosis with microcephaly (MFDM) is a disorder that causes abnormalities of the head and face. People with this disorder often have an unusually small head at birth, and the head does not grow at th...
5. `medquad_ans_0004682-1` (score=0.0001) - What are the signs and symptoms of Palant cleft palate syndrome? The Human Phenotype Ontology provides the following list of signs and symptoms for Palant cleft palate syndrome. If the information is available, the ta...

**Answer**:

The symptoms of Microcephaly, corpus callosum dysgenesis and cleft lip-palate include cleft palate, cleft upper lip, hypoplasia of the corpus callosum, and microcephaly. Preaxial hand polydactyly, proptosis, and seizures are also reported as symptoms. The frequency of these symptoms varies, with some being observed in a significant proportion of patients, while others may be rare or have limited data on prevalence [medquad_ans_0004025-1].

**Supported citations**: `medquad_ans_0004025-1`
**Unsupported citations**: _none_

### Example 269 (`0003402-5`)

**Question**: How to diagnose Klippel Feil syndrome ?

**Scores**: context_relevance=0.833, groundedness=0.985, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0003402-5` (score=0.9964) - How is Klippel Feil syndrome diagnosed? Klippel Feil syndrome (KFS) is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to d...
2. `medquad_ans_0000562-2` (score=0.0001) - Klippel-Feil syndrome is estimated to occur in 1 in 40,000 to 42,000 newborns worldwide. Females seem to be affected slightly more often than males.
3. `medquad_ans_0000170-3__dup1` (score=0.0001) - The prognosis for most individuals with Klippel-Feil Syndrome is good if the disorder is treated early and appropriately. Activities that can injure the neck should be avoided.
4. `medquad_ans_0000170-2__dup1` (score=0.0001) - Treatment for Klippel-Feil Syndrome is symptomatic and may include surgery to relieve cervical or craniocervical instability and constriction of the spinal cord, and to correct scoliosis. Physical therapy may also be ...
5. `medquad_ans_0003402-1` (score=0.0001) - Klippel Feil syndrome (KFS) is a congenital, musculoskeletal condition characterized by the fusion of at least two vertebrae of the neck. Common symptoms include a short neck, low hairline at the back of the head, and...

**Answer**:

Klippel Feil syndrome is typically diagnosed when X-rays or other imaging techniques show fusion of cervical vertebrae. X-rays of the entire spine should be performed to detect other spinal abnormalities, and additional imaging studies may be needed to assess the extent of the abnormality. Other initial exams are needed to detect additional physical abnormalities or underlying conditions, including examination of the chest to rule out involvement of the heart and lungs, examination of the chest wall to detect possible rib anomalies, MRI for spinal stenosis or neurological deficits, ultrasound of the kidneys for renal abnormalities, hearing evaluation due to high incidence of hearing loss, and various lab tests to assess organ function [medquad_ans_0003402-5].

**Supported citations**: `medquad_ans_0003402-5`
**Unsupported citations**: _none_

### Example 270 (`0000222-1`)

**Question**: What is (are) congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.833, groundedness=0.951, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0001497-1` (score=0.9537) - Congenital diaphragmatic hernia (CDH) is the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest cavity. It can range in severit...
2. `medquad_ans_0000464-1__dup1` (score=0.5711) - A hernia happens when part of an internal organ or tissue bulges through a weak area of muscle. Most hernias are in the abdomen. There are several types of hernias, including - Inguinal, in the groin. This is the the ...
3. `medquad_ans_0000222-1` (score=0.0661) - Congenital diaphragmatic hernia is a defect in the diaphragm. The diaphragm, which is composed of muscle and other fibrous tissue, separates the organs in the abdomen from those in the chest. Abnormal development of t...
4. `medquad_ans_0000002-1` (score=0.0366) - An abdominal wall defect is an opening in the abdomen through which various abdominal organs can protrude. This opening varies in size and can usually be diagnosed early in fetal development, typically between the ten...
5. `medquad_ans_0000222-2` (score=0.0004) - Congenital diaphragmatic hernia affects approximately 1 in 2,500 newborns.

**Answer**:

Congenital diaphragmatic hernia (CDH) is a birth defect characterized by the lack of development before birth of all or part of the diaphragm, which normally separates the organs in the abdomen from those in the chest. It ranges in severity from a thinned area in the diaphragm to its complete absence. This defect results in an abnormal opening that allows the stomach and intestines to move into the chest cavity, crowding the heart and lungs. This crowding can lead to underdevelopment of the lungs (pulmonary hypoplasia), potentially causing life-threatening breathing difficulties from birth. CDH may occur with other malformations and can present with symptoms such as breathing problems or abdominal pain, though in about 1 percent of cases, it is asymptomatic and may be detected incidentally during imaging for other reasons [medquad_ans_0001497-1]. It is often classified by position,

**Supported citations**: `medquad_ans_0001497-1`
**Unsupported citations**: _none_

### Example 271 (`0000173-1`)

**Question**: What is (are) Charcot-Marie-Tooth disease ?

**Scores**: context_relevance=0.800, groundedness=0.959, answer_relevance=0.640

**Top docs**:

1. `medquad_ans_0000168-1__dup2` (score=0.7742) - Charcot-Marie-Tooth disease (CMT) is a group of genetic nerve disorders. It is named after the three doctors who first identified it. In the United States, CMT affects about 1 in 2,500 people. CMT affects your periphe...
2. `medquad_ans_0001088-1__dup1` (score=0.0761) - Charcot-Marie-Tooth disease is a group of disorders that affect the peripheral nerves, the nerves running from outside the brain and spine. Defects in many different genes cause different forms of this disease. Common...
3. `medquad_ans_0000173-1` (score=0.0431) - Charcot-Marie-Tooth disease is a group of progressive disorders that affect the peripheral nerves. Peripheral nerves connect the brain and spinal cord to muscles and to sensory cells that detect sensations such as tou...
4. `medquad_ans_0001090-1__dup1` (score=0.0026) - Charcot-Marie-Tooth disease type 1A (CMT1A) is a type of inherited neurological disorder that affects the peripheral nerves. Affected individuals experience weakness and wasting (atrophy) of the muscles of the lower l...
5. `medquad_ans_0001094-1__dup1` (score=0.0020) - Charcot-Marie-Tooth disease type 1E (CMT1E) is a form of Charcot-Marie-Tooth disease, which is a group of rare conditions that affect the peripheral nerves. Signs and symptoms of CMT1E generally become apparent betwee...

**Answer**:

Charcot-Marie-Tooth disease (CMT) is a group of genetic nerve disorders that affect the peripheral nerves, which carry movement and sensation signals between the brain and spinal cord and the rest of the body [medquad_ans_0000168-1__dup2]. It is named after the three doctors who first identified it and is caused by defects in many different genes, resulting in over 40 types of the disease [medquad_ans_0001088-1__dup1]. CMT typically becomes apparent in adolescence or early adulthood, though onset can occur from early childhood through late adulthood [medquad_ans_0000173-1]. Symptoms may include foot drop, foot deformity, loss of lower leg muscle, numbness, slapping gait, and weakness in the hips, legs, or feet [medquad_ans_0001088-

**Supported citations**: `medquad_ans_0000168-1__dup2`, `medquad_ans_0001088-1__dup1`, `medquad_ans_0000173-1`
**Unsupported citations**: _none_

### Example 272 (`0000669-4`)

**Question**: Is monilethrix inherited ?

**Scores**: context_relevance=0.667, groundedness=0.000, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000784-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
2. `medquad_ans_0000998-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
3. `medquad_ans_0000898-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
4. `medquad_ans_0000676-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...
5. `medquad_ans_0001083-4` (score=0.9988) - This condition is inherited in an autosomal recessive pattern, which means both copies of the gene in each cell have mutations. The parents of an individual with an autosomal recessive condition each carry one copy of...

**Answer**:

The provided context does not specify whether monilethrix is inherited. While multiple entries describe autosomal recessive inheritance patterns, none explicitly mention monilethrix. Therefore, based on the given information, it is not possible to determine if monilethrix is inherited.

[doc_id] is not applicable due to insufficient context.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 273 (`0000215-1__dup3`)

**Question**: What is (are) Urinary Tract Infections in Children ?

**Scores**: context_relevance=0.875, groundedness=0.394, answer_relevance=0.775

**Top docs**:

1. `medquad_ans_0000216-3__dup1` (score=0.9759) - Before leaving the hospital, a baby with urine blockage may urinate only small amounts or may not urinate at all. As part of the routine newborn exam, the health care provider may feel an enlarged kidney or find a clo...
2. `medquad_ans_0000215-3__dup2` (score=0.0924) - Most UTIs are caused by bacteria that live in the bowel. The bacterium Escherichia coli (E. coli) causes the vast majority of UTIs. The urinary tract has several systems to prevent infection. The points where the uret...
3. `medquad_ans_0000215-9` (score=0.0738) - Most UTIs are caused by bacteria, which are treated with bacteria-fighting medications called antibiotics or antimicrobials. While a urine sample is sent to a laboratory, the health care provider may begin treatment w...
4. `medquad_ans_0000223-1__dup3` (score=0.0533) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscope. Bacteria \* are the most common cause of UTIs. Normally, bacteria that enter the urinary...
5. `medquad_ans_0000214-1__dup4` (score=0.0212) - A UTI is an infection in the urinary tract. Infections are caused by microbesorganisms too small to be seen without a microscopeincluding fungi, viruses, and bacteria. Bacteria are the most common cause of UTIs. Norma...

**Answer**:

Urinary Tract Infections (UTIs) in children are infections of the urinary tract caused primarily by bacteria, most commonly Escherichia coli (E. coli). These infections can present with symptoms such as fever, irritability, not eating, nausea, diarrhea, vomiting, cloudy or dark, bloody, or foul-smelling urine, and frequent urination. In infants under 2 months of age, a fever requires immediate evaluation by a health care provider. UTIs can occur despite the urinary tract's natural defenses, including one-way valves that prevent urine reflux and immune defenses, especially when bacteria can attach to the urinary tract lining. Risk factors include delaying urination, inadequate fluid intake, chronic constipation (fewer than two bowel movements per week), and the pressure from hard stool on the bladder, which can block urine flow and promote bacterial growth. Treatment typically involves antibiotics, with initial therapy often based on likely causative bacteria, and adjustments made

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 274 (`0000167-13`)

**Question**: What are the complications of Polycystic Kidney Disease ?

**Scores**: context_relevance=0.875, groundedness=0.796, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000135-6__dup1` (score=0.6068) - People with acquired cystic kidney disease may develop the following complications: - an infected cyst, which can cause fever and back pain. - blood in the urine, which can signal that a cyst in the kidney is bleeding...
2. `medquad_ans_0000135-5__dup2` (score=0.0328) - A person with acquired cystic kidney disease often has no symptoms. However, the complications of acquired cystic kidney disease can have signs and symptoms.
3. `medquad_ans_0000135-2__dup2` (score=0.0063) - Acquired cystic kidney disease differs from PKD in several ways. Unlike acquired cystic kidney disease, PKD is a genetic, or inherited, disorder that can cause complications such as high blood pressure and problems wi...
4. `medquad_ans_0000135-1__dup4` (score=0.0007) - Acquired cystic kidney disease happens when a person's kidneys develop fluid-filled sacs, called cysts, over time. Acquired cystic kidney disease is not the same as polycystic kidney disease (PKD), another disease tha...
5. `medquad_ans_0000804-5` (score=0.0005) - These resources address the diagnosis or management of polycystic kidney disease: - Gene Review: Gene Review: Polycystic Kidney Disease, Autosomal Dominant - Gene Review: Gene Review: Polycystic Kidney Disease, Autoso...

**Answer**:

The provided context does not contain information about the complications of Polycystic Kidney Disease (PKD). While it describes differences between acquired cystic kidney disease and PKD, such as PKD being a genetic disorder that can cause high blood pressure and problems with blood vessels in the brain and heart, the specific complications of PKD are not fully detailed in the given text. Therefore, based on the available context, the complications of PKD cannot be accurately answered.

[medquad_ans_0000135-2__dup2]

**Supported citations**: `medquad_ans_0000135-2__dup2`
**Unsupported citations**: _none_

### Example 275 (`0000649-1__dup1`)

**Question**: What is (are) Norovirus Infections ?

**Scores**: context_relevance=0.800, groundedness=0.932, answer_relevance=0.800

**Top docs**:

1. `medquad_ans_0000649-1__dup1` (score=0.8358) - Noroviruses are a group of related viruses. Infection with these viruses causes an illness called gastroenteritis, an inflammation of the stomach and intestines. It can spread from person to person, or through contami...
2. `medquad_ans_0000949-1__dup1` (score=0.0440) - Viruses are capsules with genetic material inside. They are very tiny, much smaller than bacteria. Viruses cause familiar infectious diseases such as the common cold, flu and warts. They also cause severe illnesses su...
3. `medquad_ans_0000123-1__dup3` (score=0.0038) - Viral gastroenteritis is inflammation of the lining of the stomach, small intestine, and large intestine. Several different viruses can cause viral gastroenteritis, which is highly contagious and extremely common. Vir...
4. `medquad_ans_0000123-4__dup2` (score=0.0034) - Four types of viruses cause most cases of viral gastroenteritis. Rotavirus Rotavirus is the leading cause of gastroenteritis among infants and young children. Rotavirus infections are most common in infants 3 to 15 mo...
5. `medquad_ans_0000228-5__dup1` (score=0.0031) - The drug ribavirin has been shown to be effective against the viruses in vitro, but the clinical usefulness of this drug is uncertain. A post-exposure therapy with a Nipah/Hendra neutralizing antibody, efficacious in ...

**Answer**:

Norovirus infections are a type of viral gastroenteritis caused by noroviruses, which are a group of related viruses. Infection with these viruses leads to an inflammation of the stomach and intestines, resulting in symptoms such as diarrhea, nausea and vomiting, and stomach pain. Other possible symptoms include fever, headache, or body aches. Norovirus can spread from person to person, through contaminated food or water, or by touching a contaminated surface. The infection can be serious, especially in young children and older adults. There is no specific medicine to treat norovirus infections; treatment focuses on bed rest and staying hydrated to prevent dehydration. Proper hand washing and safe food preparation can help prevent infections [medquad_ans_0000649-1__dup1]. Norovirus is the most common calicivirus and the most common cause of viral gastroenteritis in people of all ages [medquad_ans_0

**Supported citations**: `medquad_ans_0000649-1__dup1`
**Unsupported citations**: _none_

### Example 276 (`0000222-3`)

**Question**: What are the genetic changes related to congenital diaphragmatic hernia ?

**Scores**: context_relevance=0.400, groundedness=0.000, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000130-4__dup3` (score=0.9987) - Before birth, a childs nerve cells normally grow along the intestines in the direction of the anus. With HD, the nerve cells stop growing too soon. Why the nerve cells stop growing is unclear. Some HD is inherited, me...
2. `medquad_ans_0000013-9__dup1` (score=0.9931) - A heritable disease is caused by a mutation, or harmful change, in a gene inherited from a parent. Genes are small structures in your body's cells that determine how you look and tell your body how to work. Examples o...
3. `medquad_ans_0000167-2__dup3` (score=0.0014) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
4. `medquad_ans_0000160-3__dup2` (score=0.0011) - Scientists do not fully understand the cause of medullary sponge kidney or why cysts form in the tubules during fetal development. Even though medullary sponge kidney is present at birth, most cases do not appear to b...
5. `medquad_ans_0000615-4` (score=0.0007) - Maffucci syndrome is not inherited. The mutations that cause this disorder are somatic, which means they occur during a person's lifetime. A somatic mutation occurs in a single cell. As that cell continues to grow and...

**Answer**:

The provided context does not contain information about the genetic changes related to congenital diaphragmatic hernia. Therefore, based on the given text, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 277 (`0004534-3`)

**Question**: What causes Omenn syndrome ?

**Scores**: context_relevance=0.750, groundedness=0.108, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000238-4` (score=0.0027) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
2. `medquad_ans_0000928-4` (score=0.0027) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
3. `medquad_ans_0000947-4` (score=0.0027) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000011-13` (score=0.0019) - Although CJD is a rare disorder, some of the world's leading researchers are working hard to learn more about this disease. About 10 percent of the people who get CJD have the inherited type. Some people have gotten C...
5. `medquad_ans_0005067-3` (score=0.0005) - What causes primrose syndrome? The cause of primrose syndrome is currently unknown. Cases of affected males and a affected female have been reported in the literature. All cases seem to be sporadic. Sporadic refers to...

**Answer**:

The provided context does not contain any information about Omenn syndrome or its causes. Therefore, it is not possible to answer the question based on the given context. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 278 (`0000097-1`)

**Question**: What is (are) autosomal recessive cerebellar ataxia type 1 ?

**Scores**: context_relevance=0.875, groundedness=0.993, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000097-1` (score=0.9864) - Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the ce...
2. `medquad_ans_0000032-4__dup1` (score=0.0027) - The NINDS supports and conducts a broad range of basic and clinical research on cerebellar and spinocerebellar degeneration, including work aimed at finding the cause(s) of ataxias and ways to treat, cure, and, ultima...
3. `medquad_ans_0005761-1` (score=0.0026) - Spinocerebellar ataxia type 11 (SCA11) is characterized by progressive cerebellar ataxia (difficulty walking and balance) and abnormal eye signs (jerky pursuit, horizontal and vertical movements (nystagmus), pyramidal...
4. `medquad_ans_0005792-1` (score=0.0016) - Spinocerebellar ataxia autosomal recessive 7, also called SCAR7, is a slowly progressive hereditary form of spinocerebellar ataxia. Symptoms of SCAR7 can include difficulty walking and writing, speech difficulties (dy...
5. `medquad_ans_0000495-1__dup1` (score=0.0014) - Ataxia telangiectasia (A-T) is rare condition that affects the nervous system, the immune system, and many other parts of the body. Signs and symptoms of the condition usually begin in early childhood, often before ag...

**Answer**:

Autosomal recessive cerebellar ataxia type 1 (ARCA1) is a condition characterized by progressive problems with movement due to a loss (atrophy) of nerve cells in the part of the brain that coordinates movement (the cerebellum). Signs and symptoms of the disorder first appear in early to mid-adulthood. People with this condition initially experience impaired speech (dysarthria), problems with coordination and balance (ataxia), or both. They may also have difficulty with movements that involve judging distance or scale (dysmetria). Other features of ARCA1 include abnormal eye movements (nystagmus) and problems following the movements of objects with the eyes. The movement problems are slowly progressive, often resulting in the need for a cane, walker, or wheelchair [medquad_ans_0000097-1].

**Supported citations**: `medquad_ans_0000097-1`
**Unsupported citations**: _none_

### Example 279 (`0000050-2__dup1`)

**Question**: What are the treatments for Brachial Plexus Injuries ?

**Scores**: context_relevance=0.250, groundedness=0.892, answer_relevance=0.650

**Top docs**:

1. `medquad_ans_0000233-2__dup1` (score=0.9965) - Generally, treatment for the disorder begins with stretching exercises and massage. Anti-inflammatory drugs may be prescribed. Cessation of running, bicycling, or similar activities may be advised. A corticosteroid in...
2. `medquad_ans_0000238-2__dup1` (score=0.9954) - Treatment for individuals with PLS is symptomatic. Muscle relaxants such as baclofen, tizanidine, and the benzodiazepines may reduce spasticity. Other drugs may relieve pain and antidepressants can help treat depressi...
3. `medquad_ans_0000267-2__dup1` (score=0.9954) - Treatment begins with exercise programs and physical therapy to strengthen chest muscles, restore normal posture, and relieve compression by increasing the space of the area the nerve passes through. Doctors will ofte...
4. `medquad_ans_0000255-2__dup1` (score=0.9947) - Treatment may include such medications as baclofen, diazepam, tizanidine or clonazepam. Physical therapy regimens may include muscle stretching and range of motion exercises to help prevent shrinkage or shortening of ...
5. `medquad_ans_0000090-2__dup1` (score=0.9936) - Treatment for RMDs usually includes reducing or stopping the motions that cause symptoms. Options include taking breaks to give the affected area time to rest, and adopting stretching and relaxation exercises. Applyin...

**Answer**:

The treatments for Brachial Plexus Injuries include symptomatic management with muscle relaxants such as baclofen, tizanidine, and benzodiazepines to reduce spasticity [medquad_ans_0000238-2__dup1]. Pain relief can be achieved through medications such as non-steroidal anti-inflammatory drugs (e.g., naproxen or ibuprofen), corticosteroids, or other analgesics [medquad_ans_0000267-2__dup1]. Physical therapy, including muscle stretching and range of motion exercises, may help prevent muscle shrinkage and improve function [medquad_ans_0000238-2\_\_dup1][medquad_ans_0000255-2__dup1]. In some cases, targeted injection of botulinum toxin into affected muscles can help weaken overactive muscles and improve

**Supported citations**: `medquad_ans_0000238-2__dup1`, `medquad_ans_0000267-2__dup1`, `medquad_ans_0000255-2__dup1`
**Unsupported citations**: _none_

### Example 280 (`0000216-2__dup1`)

**Question**: What causes Urine Blockage in Newborns ?

**Scores**: context_relevance=0.833, groundedness=0.944, answer_relevance=0.700

**Top docs**:

1. `medquad_ans_0000218-4__dup2` (score=0.9972) - Urine leakage has many possible causes.
2. `medquad_ans_0000216-8` (score=0.9936) - - Many types of defects in the urinary tract can cause urine blockage: - vesicoureteral reflux (VUR) - ureteropelvic junction (UPJ) obstruction - bladder outlet obstruction (BOO), such as posterior urethral valves (PU...
3. `medquad_ans_0000216-7` (score=0.0299) - Researchers have not found that a mothers eating, diet, and nutrition play a role in causing or preventing urine blockage in newborns.
4. `medquad_ans_0000216-2__dup1` (score=0.0293) - Many types of defects in the urinary tract can cause urine blockage: - Vesicoureteral reflux (VUR). Most children with VUR are born with a ureter that did not grow long enough during development in the womb. The valve...
5. `medquad_ans_0000216-4__dup1` (score=0.0033) - When a defect in the urinary tract blocks the flow of urine, the urine backs up and causes the ureters to swell, called hydroureter, and hydronephrosis. Hydronephrosis is the most common problem found during prenatal ...

**Answer**:

Urine blockage in newborns can be caused by various defects in the urinary tract, including vesicoureteral reflux (VUR), ureteropelvic junction (UPJ) obstruction, bladder outlet obstruction (BOO) such as posterior urethral valves (PUV), ureterocele, and genetic conditions like prune belly syndrome (PBS), esophageal atresia (EA), and congenital heart defects. Additionally, spina bifida and other birth defects affecting the spinal cord can lead to urine blockage. These defects may result in urine backing up, causing hydroureter and hydronephrosis. Urine blockage may not be apparent until a child develops symptoms of a urinary tract infection (UTI) or is detected during a routine newborn exam with signs such as an enlarged kidney or closed urethra [medquad_ans_0000216-8]. Prenatal tests like

**Supported citations**: `medquad_ans_0000216-8`
**Unsupported citations**: _none_

### Example 281 (`0000526-2`)

**Question**: How many people are affected by intranuclear rod myopathy ?

**Scores**: context_relevance=0.333, groundedness=0.738, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000165-2` (score=0.9994) - Centronuclear myopathy is a rare condition; its exact prevalence is unknown.
2. `medquad_ans_0000526-2` (score=0.9993) - Intranuclear rod myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
3. `medquad_ans_0000575-2` (score=0.9993) - Although Laing distal myopathy is thought to be rare, its prevalence is unknown. Several families with the condition have been identified worldwide.
4. `medquad_ans_0000148-2` (score=0.9992) - Cap myopathy is a rare disorder that has been identified in only a small number of individuals. Its exact prevalence is unknown.
5. `medquad_ans_0000705-2` (score=0.9977) - The prevalence of myofibrillar myopathy is unknown.

**Answer**:

The exact prevalence of intranuclear rod myopathy is unknown [medquad_ans_0000526-2]. Therefore, the number of people affected by intranuclear rod myopathy cannot be determined from the provided context.

**Supported citations**: `medquad_ans_0000526-2`
**Unsupported citations**: _none_

### Example 282 (`0000036-28`)

**Question**: What is (are) High Blood Pressure ?

**Scores**: context_relevance=1.000, groundedness=0.902, answer_relevance=0.733

**Top docs**:

1. `medquad_ans_0000058-3__dup4` (score=0.9849) - Blood pressure is the force of blood flow inside your blood vessels. Blood pressure is written with two numbers separated by a slash. For example, a blood pressure result of 130/80 is said as 130 over 80. The first nu...
2. `medquad_ans_0000036-12` (score=0.9806) - Abnormal blood pressure is higher than 120/80 mmHg. If either your systolic or diastolic blood pressure is higher than normal (120/80) but not high enough to be considered high blood pressure (140/90), you have pre-hy...
3. `medquad_ans_0000036-21` (score=0.9779) - In most cases, the goal is probably to keep your blood pressure below 140/90 mmHg (130/80 if you have diabetes or chronic kidney disease). Normal blood pressure is less than 120/80. Ask your doctor what your blood pre...
4. `medquad_ans_0000036-10` (score=0.9690) - Normal blood pressure for adults is defined as a systolic pressure below 120 mmHg and a diastolic pressure below 80 mmHg. It is normal for blood pressures to change when you sleep, wake up, or are excited or nervous. ...
5. `medquad_ans_0000565-1__dup1` (score=0.9669) - You've probably heard that high blood pressure is a problem. Sometimes blood pressure that is too low can also cause problems. Blood pressure is the force of your blood pushing against the walls of your arteries. Each...

**Answer**:

High blood pressure is defined as a blood pressure reading of 140/90 mmHg or higher. It is characterized by increased pressure in the blood vessels, which forces the heart to work harder and can lead to strain on the heart, damage to blood vessels, and an increased risk of heart attack, stroke, eye problems, and kidney problems. The goal for most adults is to keep blood pressure below 140/90 mmHg, and for individuals with diabetes or chronic kidney disease, the target is below 130/80 mmHg [medquad_ans_0000036-21]. Abnormal blood pressure is higher than 120/80 mmHg, and if the pressure is above normal but not yet in the high blood pressure range, it is classified as pre-hypertension [medquad_ans_0000036-

**Supported citations**: `medquad_ans_0000036-21`
**Unsupported citations**: _none_

### Example 283 (`0001051-4`)

**Question**: Is X-linked dystonia-parkinsonism inherited ?

**Scores**: context_relevance=0.750, groundedness=0.275, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000570-4` (score=0.9961) - This condition is inherited in an X-linked recessive pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In males (who have only one X chromosome),...
2. `medquad_ans_0001065-4` (score=0.9958) - This condition is inherited in an X-linked pattern. The gene associated with this condition is located on the X chromosome, which is one of the two sex chromosomes. In females (who have two X chromosomes), a mutation ...
3. `medquad_ans_0001717-4` (score=0.3935) - How is Danon disease inherited? Dannon disease is inherited in an X-linked fashion. Click here to visit the Centre for Genetics Education Web site to learn more about X linked inheritance.
4. `medquad_ans_0004758-4` (score=0.0003) - Is Partington syndrome inherited? Partington syndrome is inherited in an X-linked recessive manner. A condition is considered X-linked if the mutated gene that causes the condition is located on the X chromosome, one ...
5. `medquad_ans_0000052-31` (score=0.0002) - Genetics is one of the most exciting areas of Parkinson's disease research. Studying the genes responsible for inherited cases can help researchers understand both inherited and sporadic cases of the disease. Sporadic...

**Answer**:

The provided context does not contain information about the inheritance pattern of X-linked dystonia-parkinsonism. Therefore, it is not possible to determine based on the given text whether X-linked dystonia-parkinsonism is inherited in a specific manner.

Answer: Insufficient context.

**Supported citations**: _none_
**Unsupported citations**: _none_

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
3. `medquad_ans_0000006_8-6` (score=0.2247) - The process used to find out if cancer has spread within the brain or to other parts of the body is called staging. There is no standard system for staging childhood craniopharyngioma. Craniopharyngioma is described a...
4. `medquad_ans_0000722-2` (score=0.0994) - Neuroblastoma is the most common cancer in infants younger than 1 year. It occurs in 1 in 100,000 children and is diagnosed in about 650 children each year in the United States.
5. `medquad_ans_0000006_5-6` (score=0.0401) - Certain factors affect prognosis (chance of recovery) and treatment options. The prognosis and treatment options depend on: - The type of brain stem glioma. - Where the tumor is found in the brain and if it has spread...

**Answer**:

There is no standard staging system for childhood brain stem glioma [medquad_ans_0000006_5-7].

**Supported citations**: `medquad_ans_0000006_5-7`
**Unsupported citations**: _none_

### Example 286 (`0000266-1__dup2`)

**Question**: What is (are) Tethered Spinal Cord Syndrome ?

**Scores**: context_relevance=0.857, groundedness=0.063, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000267-1__dup1` (score=0.9332) - TOS is an umbrella term that encompasses three related syndromes that involve compression of the nerves, arteries, and veins in the lower neck and upper chest area and cause pain in the arm, shoulder, and neck. Most d...
2. `medquad_ans_0000266-1__dup2` (score=0.3112) - Tethered spinal cord syndrome is a neurological disorder caused by tissue attachments that limit the movement of the spinal cord within the spinal column. Attachments may occur congenitally at the base of the spinal c...
3. `medquad_ans_0000838-1__dup2` (score=0.2233) - Your spinal cord is a bundle of nerves that runs down the middle of your back. It carries signals back and forth between your body and your brain. It is protected by your vertebrae, which are the bone disks that make ...
4. `medquad_ans_0000266-2__dup1` (score=0.0036) - MRI imaging is often used to evaluate individuals with these symptoms, and can be used to diagnose the location of the tethering, lower than normal position of the conus medullaris, or presence of a tumor or fatty mas...
5. `medquad_ans_0000054-4__dup1` (score=0.0036) - The NINDS supports and conducts a wide range of research on spinal cord disorders such as BSS. The goal of this research is to find ways to prevent, treat, and, ultimately, cure these disorders.

**Answer**:

The provided context does not contain information about Tethered Spinal Cord Syndrome. Therefore, it is not possible to answer the question based on the given context. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 287 (`0000058-3__dup1`)

**Question**: What is the outlook for Carpal Tunnel Syndrome ?

**Scores**: context_relevance=0.875, groundedness=0.858, answer_relevance=0.875

**Top docs**:

1. `medquad_ans_0000058-3__dup1` (score=0.9958) - In general, carpal tunnel syndrome responds well to treatment, but less than half of individuals report their hand(s) feeling completely normal following surgery. Some residual numbness or weakness is common. At work,...
2. `medquad_ans_0000136-5__dup2` (score=0.0001) - The symptoms of dialysis-related amyloidosis may include - pain, stiffness, and fluid in the joints. - abnormal, fluid-containing sacs, called cysts, in some bones. - carpal tunnel syndrome, caused by unusual buildup ...
3. `medquad_ans_0000058-1__dup1` (score=0.0001) - Carpal tunnel syndrome (CTS) occurs when the median nerve, which runs from the forearm into the palm of the hand, becomes pressed or squeezed at the wrist. The carpal tunnel is a narrow, rigid passageway of ligament a...
4. `medquad_ans_0000419-1__dup1` (score=0.0001) - No matter how old you are or what you do for a living, you are always using your hands. When there is something wrong with them, you may not be able to do your regular activities. Hand problems include - Carpal tunnel...
5. `medquad_ans_0000978-1__dup1` (score=0.0001) - Your wrist is made up of eight small bones known as carpals. They support a tube that runs through your wrist. That tube, called the carpal tunnel, has tendons and a nerve inside. It is covered by a ligament, which ho...

**Answer**:

The outlook for carpal tunnel syndrome is generally positive, as it responds well to treatment. However, less than half of individuals report their hand(s) feeling completely normal following surgery, and some residual numbness or weakness is common [medquad_ans_0000058-3__dup1]. Symptoms typically begin gradually and may include numbness, tingling, weakness, and sometimes pain in the hand and wrist, which can interfere with daily activities such as driving or reading [medquad_ans_0000058-1__dup1]. While treatment can alleviate symptoms, complete recovery is not guaranteed for all patients.

**Supported citations**: `medquad_ans_0000058-3__dup1`, `medquad_ans_0000058-1__dup1`
**Unsupported citations**: _none_

### Example 288 (`0000296-5`)

**Question**: What are the treatments for Donnai-Barrow syndrome ?

**Scores**: context_relevance=0.714, groundedness=0.000, answer_relevance=0.714

**Top docs**:

1. `medquad_ans_0000033-2__dup1` (score=0.9951) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
2. `medquad_ans_0000220-2__dup1` (score=0.9946) - There is no specific treatmentfor OPCA. Physicians may try different medications to treat the ataxia, tremor, and rigidity that are associated with the disorder. Other treatments are directed at specific symptoms. Sti...
3. `medquad_ans_0000188-2__dup1` (score=0.9909) - MJD is incurable, but some symptoms of the disease can be treated. For those individuals who show parkinsonian features, levodopa therapy can help for many years. Treatment with antispasmodic drugs, such as baclofen, ...
4. `medquad_ans_0000200-2__dup2` (score=0.9899) - Currently there is no cure for these disease syndromes.Medical care is directed at treating systemic conditions and improving the person's quality of life. Physical therapy and daily exercise may delay joint problems ...
5. `medquad_ans_0003593-6` (score=0.6287) - How might Liddle syndrome be treated? Treatment for Liddle syndrome includes following a low sodium diet as well as taking potassium-sparing diuretics, which reduce blood pressure and correct hypokalemia and metabolic...

**Answer**:

The provided context does not contain any information about the treatments for Donnai-Barrow syndrome. Therefore, it is not possible to answer the question based on the given text. [no_context]

**Supported citations**: _none_
**Unsupported citations**: `no_context`

### Example 289 (`0000811-1`)

**Question**: What is (are) porphyria ?

**Scores**: context_relevance=0.750, groundedness=0.968, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000189-1__dup3` (score=0.8104) - Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. These disorders are usually inherited, meaning they are caused by abnormalities in genes passed from parents to...
2. `medquad_ans_0000189-3__dup2` (score=0.6440) - Each of the eight types of porphyria corresponds to low levels of a specific enzyme in the heme biosynthetic pathway. Experts often classify porphyrias as acute or cutaneous based on the symptoms a person experiences:...
3. `medquad_ans_0000729-1__dup1` (score=0.4636) - Porphyrias are a group of genetic disorders caused by problems with how your body makes a substance called heme. Heme is found throughout the body, especially in your blood and bone marrow, where it carries oxygen. Th...
4. `medquad_ans_0000811-1` (score=0.3049) - Porphyria is a group of disorders caused by abnormalities in the chemical steps that lead to heme production. Heme is a vital molecule for all of the body's organs, although it is most abundant in the blood, bone marr...
5. `medquad_ans_0000189-5__dup1` (score=0.0841) - Most porphyrias are inherited disorders. Scientists have identified genes for all eight enzymes in the heme biosynthetic pathway. Most porphyrias result from inheriting an abnormal gene, also called a gene mutation, f...

**Answer**:

Porphyrias are rare disorders that affect mainly the skin or nervous system and may cause abdominal pain. These disorders are usually inherited and are caused by abnormalities in genes passed from parents to children. When a person has a porphyria, cells fail to change body chemicals called porphyrins and porphyrin precursors into heme, the substance that gives blood its red color. The body makes heme mainly in the bone marrow and liver through a process called the heme biosynthetic pathway, which is controlled by one of eight enzymes. A deficiency in any of these enzymes leads to a buildup of porphyrins and porphyrin precursors in the body, causing illness [medquad_ans_0000189-1__dup3]. Each of the eight types of porphyria corresponds to a low level of a specific enzyme in this pathway, and porphyrias are classified as acute or cutaneous based

**Supported citations**: `medquad_ans_0000189-1__dup3`
**Unsupported citations**: _none_

### Example 290 (`0000585-1__dup1`)

**Question**: What is (are) Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy ?

**Scores**: context_relevance=0.917, groundedness=0.985, answer_relevance=0.850

**Top docs**:

1. `medquad_ans_0000167-11` (score=0.9791) - Autosomal recessive PKD is a rare genetic disorder that affects the liver as well as the kidneys. The signs of autosomal recessive PKD frequently appear in the earliest months of life, even in the womb, so health care...
2. `medquad_ans_0001900-1` (score=0.9606) - Dominant optic atrophy (DOA) is an inherited optic nerve disorder characterized by degeneration of the optic nerves. It typically starts during the first decade of life. Affected people usually develop moderate visual...
3. `medquad_ans_0000585-1__dup1` (score=0.9521) - Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition described by Iwashita et al. in 1969 in a Korean brother and sister. This condition is characterized by variable d...
4. `medquad_ans_0000934-1__dup1` (score=0.7000) - Usher syndrome is an inherited disease that causes serious hearing loss and retinitis pigmentosa, an eye disorder that causes your vision to get worse over time. It is the most common condition that affects both heari...
5. `medquad_ans_0004797-1` (score=0.0041) - Pendred syndrome is a condition usually characterized by sensorineural hearing loss in both ears (bilateral) and euthyroid goiter (enlargement of the thyroid gland with normal thyroid gland function). The amount of he...

**Answer**:

Autosomal recessive optic atrophy, hearing loss, and peripheral neuropathy is a neurological condition characterized by variable degrees of hearing loss, distal weakness and loss of muscle tissue (atrophy) in the upper limbs, variable degrees of weakness and atrophy of the lower limbs, and optic atrophy with or without visual impairment. Autosomal recessive inheritance has been suggested [medquad_ans_0000585-1__dup1].

**Supported citations**: `medquad_ans_0000585-1__dup1`
**Unsupported citations**: _none_

### Example 291 (`0000794-3`)

**Question**: What are the genetic changes related to phosphoglycerate kinase deficiency ?

**Scores**: context_relevance=0.500, groundedness=0.029, answer_relevance=0.740

**Top docs**:

1. `medquad_ans_0000808-3` (score=0.9872) - Mutations in the GAA gene cause Pompe disease. The GAA gene provides instructions for producing an enzyme called acid alpha-glucosidase (also known as acid maltase). This enzyme is active in lysosomes, which are struc...
2. `medquad_ans_0000167-2__dup3` (score=0.0019) - A gene mutation, or defect, causes polycystic kidney disease. Genes provide instructions for making proteins in the body. A gene mutation is a permanent change in the deoxyribonucleic acid (DNA) sequence that makes up...
3. `medquad_ans_0000495-3` (score=0.0018) - Mutations in the AASS gene cause hyperlysinemia. The AASS gene provides instructions for making an enzyme called aminoadipic semialdehyde synthase. This enzyme performs two functions in the breakdown of lysine. First,...
4. `medquad_ans_0000553-3` (score=0.0006) - Kabuki syndrome is caused by mutations in the KMT2D gene (also known as MLL2) or the KDM6A gene. Between 55 and 80 percent of cases of Kabuki syndrome are caused by mutations in the KMT2D gene. This gene provides inst...
5. `medquad_ans_0000465-3` (score=0.0004) - Mutations in the FH gene cause hereditary leiomyomatosis and renal cell cancer. The FH gene provides instructions for making an enzyme called fumarase (also known as fumarate hydratase). This enzyme participates in an...

**Answer**:

The provided context does not contain information about the genetic changes related to phosphoglycerate kinase deficiency. Therefore, no answer can be given based on the available information. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 292 (`0000850-3`)

**Question**: What are the genetic changes related to pulmonary veno-occlusive disease ?

**Scores**: context_relevance=0.400, groundedness=0.903, answer_relevance=0.720

**Top docs**:

1. `medquad_ans_0000047-3` (score=0.9990) - ACD/MPV can be caused by mutations in the FOXF1 gene. The protein produced from the FOXF1 gene is a transcription factor, which means that it attaches (binds) to specific regions of DNA and helps control the activity ...
2. `medquad_ans_0000095-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
3. `medquad_ans_0000362-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
4. `medquad_ans_0000585-4` (score=0.0002) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In most cases, an affected person has one parent with the condit...
5. `medquad_ans_0000367-4` (score=0.0001) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The genetic changes related to pulmonary veno-occlusive disease (ACD/MPV) include mutations in the FOXF1 gene and deletions of genetic material on the long arm of chromosome 16 in the 16q24.1 region. Mutations in the FOXF1 gene result in an inactive protein that cannot regulate the development of pulmonary blood vessels and the gastrointestinal tract, leading to abnormal formation. Deletion of one copy of the FOXF1 gene reduces the production of the FOXF1 protein, affecting pulmonary blood vessel development and causing the main features of ACD/MPV. Additionally, loss of other genes in the 16q24.1 region may contribute to additional abnormalities such as heart defects. In about 60 percent of affected infants, the genetic cause is unknown [medquad_ans_0000047-3]. The condition is inherited in an autosomal dominant pattern,

**Supported citations**: `medquad_ans_0000047-3`
**Unsupported citations**: _none_

### Example 293 (`0001006-3`)

**Question**: What are the genetic changes related to tyrosinemia ?

**Scores**: context_relevance=0.750, groundedness=0.075, answer_relevance=0.750

**Top docs**:

1. `medquad_ans_0000482-3` (score=0.8696) - Histidinemia is caused by mutations in the HAL gene, which provides instructions for making an enzyme called histidase. Histidase breaks down histidine to a molecule called urocanic acid. Histidase is active (expresse...
2. `medquad_ans_0000664-3` (score=0.0057) - Mutations in the TYMP gene (previously known as ECGF1) cause MNGIE disease. This gene provides instructions for making an enzyme called thymidine phosphorylase. Thymidine is a molecule known as a nucleoside, which (af...
3. `medquad_ans_0001005-3` (score=0.0021) - Mutations in the TH gene cause TH deficiency. The TH gene provides instructions for making the enzyme tyrosine hydroxylase, which is important for normal functioning of the nervous system. Tyrosine hydroxylase takes p...
4. `medquad_ans_0000229-3` (score=0.0002) - Congenital hyperinsulinism is caused by mutations in genes that regulate the release (secretion) of insulin, which is produced by beta cells in the pancreas. Insulin clears excess sugar (in the form of glucose) from t...
5. `medquad_ans_0001006-2` (score=0.0001) - Worldwide, tyrosinemia type I affects about 1 in 100,000 individuals. This type is more common in Norway where 1 in 60,000 to 74,000 individuals are affected. Tyrosinemia type I is even more common in Quebec, Canada w...

**Answer**:

The provided context does not mention tyrosinemia or any genetic changes related to it. Therefore, based on the given information, it is not possible to answer the question. [no context]

**Supported citations**: _none_
**Unsupported citations**: `no context`

### Example 294 (`0000004-6__dup2`)

**Question**: What are the treatments for Graves' Disease ?

**Scores**: context_relevance=0.857, groundedness=0.856, answer_relevance=0.743

**Top docs**:

1. `medquad_ans_0000209-2` (score=0.9953) - Treatment involves restoring normal levels of thyroid hormone and may include thyroid drugs, radioactive iodine, and sometimes partial or complete surgical removal of the thyroid.
2. `medquad_ans_0000436-5` (score=0.9887) - These resources address the diagnosis or management of Graves disease: - American Thyroid Association: Thyroid Function Tests - Genetic Testing Registry: Graves disease 2 - Genetic Testing Registry: Graves disease 3 -...
3. `medquad_ans_0000005-6__dup2` (score=0.9680) - Treatment generally depends on whether the thyroid is damaged enough to cause hypothyroidism. In the absence of hypothyroidism, some health care providers treat Hashimotos disease to reduce the size of the goiter. Oth...
4. `medquad_ans_0000008-8__dup2` (score=0.9435) - Health care providers treat hyperthyroidism with medications, radioiodine therapy, or thyroid surgery. The aim of treatment is to bring thyroid hormone levels to a normal state, thus preventing long-term complications...
5. `medquad_ans_0000004-6__dup2` (score=0.9344) - People with Graves disease have three treatment options: radioiodine therapy, medications, and thyroid surgery. Radioiodine therapy is the most common treatment for Graves disease in the United States. Graves disease ...

**Answer**:

Treatments for Graves' Disease include medications, radioactive iodine therapy, and sometimes surgical removal of the thyroid. The goal of treatment is to restore normal levels of thyroid hormone and relieve uncomfortable symptoms. Treatment options may vary based on the cause of hyperthyroidism, its severity, and individual patient factors such as age, allergies, and other medical conditions [medquad_ans_0000436-5]. Specific resources mention treatment options provided by the Graves' Disease & Thyroid Foundation and the Thyroid Disease Manager, which cover diagnosis and management of Graves' disease [medquad_ans_0000436-5]. Additionally, treatment may involve thyroid drugs, radioactive iodine, or partial or complete surgical removal of the thyroid [medquad_ans_0000209-2].

**Supported citations**: `medquad_ans_0000436-5`, `medquad_ans_0000209-2`
**Unsupported citations**: _none_

### Example 295 (`0000039-1__dup2`)

**Question**: What is (are) Batten Disease ?

**Scores**: context_relevance=0.800, groundedness=0.497, answer_relevance=0.600

**Top docs**:

1. `medquad_ans_0000039-1__dup2` (score=0.2434) - Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. In some cases, the early signs are subtle, taking the form of personality and behavior changes, slow learning, clumsiness, ...
2. `medquad_ans_0000039-3__dup1` (score=0.0037) - Over time, affected children suffer cognitive impairment, worsening seizures, and progressive loss of sight and motor skills. Eventually, children with Batten disease become blind, bedridden, and demented. Batten dise...
3. `medquad_ans_0000205-2` (score=0.0024) - CMAMMA appears to be a rare disease. Approximately a dozen cases have been reported in the scientific literature.
4. `medquad_ans_0000039-4__dup1` (score=0.0015) - The biochemical defects that underlie several NCLs have recently been discovered. An enzyme called palmitoyl-protein thioesterase has been shown to be insufficiently active in the infantile form of Batten disease (thi...
5. `medquad_ans_0000955-2` (score=0.0009) - The prevalence of this condition is unknown. Only a few affected individuals have been described in the medical literature.

**Answer**:

Batten disease is a fatal, inherited disorder of the nervous system that begins in childhood. It is characterized by a buildup of substances called lipopigments in the body's tissues, which are made up of fats and proteins. Early signs may include personality and behavior changes, slow learning, clumsiness, or stumbling, and vision loss is often an early symptom that may lead to suspicion during an eye exam. Over time, affected children experience cognitive impairment, worsening seizures, and progressive loss of sight and motor skills, eventually becoming blind, bedridden, and demented. The disease is often fatal by the late teens or twenties. Biochemical defects underlying Batten disease include insufficient activity of the enzyme palmitoyl-protein thioesterase in the infantile form (CLN1), a deficiency of an acid protease in the late infantile form (CLN2), and a mutated gene in juvenile Batten disease (CLN3),

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 296 (`0002061-1`)

**Question**: What is (are) Ehlers-Danlos syndrome, progeroid type ?

**Scores**: context_relevance=0.571, groundedness=0.000, answer_relevance=0.686

**Top docs**:

1. `medquad_ans_0000318-1__dup1` (score=0.2245) - Ehlers-Danlos syndrome (EDS) is a group of inherited disorders that weaken connective tissues. Connective tissues are proteins that support skin, bones, blood vessels, and other organs. EDS usually affects your skin, ...
2. `medquad_ans_0000673-4` (score=0.0016) - This condition is usually inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder. In some cases, the inheritance pattern is unknown.
3. `medquad_ans_0000797-4` (score=0.0013) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
4. `medquad_ans_0000498-4` (score=0.0013) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.
5. `medquad_ans_0000493-4` (score=0.0013) - This condition is inherited in an autosomal dominant pattern, which means one copy of the altered gene in each cell is sufficient to cause the disorder.

**Answer**:

The provided context does not contain information about Ehlers-Danlos syndrome, progeroid type. Therefore, a specific answer cannot be generated based on the given text. [doc_id] is not applicable.

**Supported citations**: _none_
**Unsupported citations**: `doc_id`

### Example 297 (`0000416-3`)

**Question**: What are the genetic changes related to glutamate formiminotransferase deficiency ?

**Scores**: context_relevance=0.900, groundedness=0.962, answer_relevance=0.660

**Top docs**:

1. `medquad_ans_0000416-3` (score=0.9991) - Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. The FTCD gene provides instructions for making the enzyme formiminotransferase cyclodeaminase. This enzyme is involved in the last two steps ...
2. `medquad_ans_0000396-1__dup2` (score=0.0005) - A genetic brain disorder is caused by a variation or a mutation in a gene. A variation is a different form of a gene. A mutation is a change in a gene. Genetic brain disorders affect the development and function of th...
3. `medquad_ans_0000923-3` (score=0.0004) - Mutations in the NSD1 gene are the primary cause of Sotos syndrome, accounting for up to 90 percent of cases. Other genetic causes of this condition have not been identified. The NSD1 gene provides instructions for ma...
4. `medquad_ans_0000442-3` (score=0.0003) - Mutations in the GAMT gene cause guanidinoacetate methyltransferase deficiency. The GAMT gene provides instructions for making the enzyme guanidinoacetate methyltransferase. This enzyme participates in the two-step pr...
5. `medquad_ans_0000631-3` (score=0.0001) - McCune-Albright syndrome is caused by a mutation in the GNAS gene. The GNAS gene provides instructions for making one part of a protein complex called a guanine nucleotide-binding protein, or a G protein. In a process...

**Answer**:

Mutations in the FTCD gene cause glutamate formiminotransferase deficiency. These mutations reduce or eliminate the function of the enzyme formiminotransferase cyclodeaminase, which is involved in the metabolism of the amino acid histidine and in the production of folate [medquad_ans_0000416-3]. However, the specific relationship between these genetic changes and the health problems associated with the mild and severe forms of the disorder is unclear [medquad_ans_0000416-3].

**Supported citations**: `medquad_ans_0000416-3`
**Unsupported citations**: _none_

### Example 298 (`0001169-2`)

**Question**: What are the symptoms of Chondrodysplasia punctata 1, X-linked recessive ?

**Scores**: context_relevance=1.000, groundedness=0.978, answer_relevance=0.822

**Top docs**:

1. `medquad_ans_0001169-2` (score=0.9994) - What are the signs and symptoms of Chondrodysplasia punctata 1, X-linked recessive? The Human Phenotype Ontology provides the following list of signs and symptoms for Chondrodysplasia punctata 1, X-linked recessive. I...
2. `medquad_ans_0001169-1` (score=0.0001) - Chondrodysplasia punctata 1, X-linked recessive (CDPX1) is a genetic disorder present from birth that affects bone and cartilage development. On x-ray, affected infants have characteristic spots at the ends of their b...
3. `medquad_ans_0001047-1` (score=0.0001) - X-linked chondrodysplasia punctata 1 is a disorder of cartilage and bone development that occurs almost exclusively in males. Chondrodysplasia punctata is an abnormality that appears on x-rays as spots (stippling) nea...
4. `medquad_ans_0001163-5` (score=0.0001) - How is chondrocalcinosis 2 diagnosed? A diagnosis of chondrocalcinosis 2 is often suspected based on characteristic signs and symptoms. Specialized testing, such as synovial fluid analysis, can then be ordered to conf...
5. `medquad_ans_0000876-2` (score=0.0001) - Rhizomelic chondrodysplasia punctata affects fewer than 1 in 100,000 people worldwide. RCDP1 is more common than RCDP2 or RCDP3.

**Answer**:

The symptoms of Chondrodysplasia punctata 1, X-linked recessive include abnormality of the vertebral column, anosmia, cataract, depressed nasal bridge, epiphyseal stippling, hearing impairment, hypogonadism, ichthyosis, microcephaly, short distal phalanx of finger, short nasal septum, short nose, short stature, and X-linked recessive inheritance [medquad_ans_0001169-2]. Additional common features include shortened fingers, a flat nose, breathing abnormalities, hearing loss, abnormalities of the spinal bones in the neck, and delayed intellectual development [medquad_ans_0001169-1].

**Supported citations**: `medquad_ans_0001169-2`, `medquad_ans_0001169-1`
**Unsupported citations**: _none_

### Example 299 (`0000132-8`)

**Question**: What are the treatments for Whipple Disease ?

**Scores**: context_relevance=0.714, groundedness=0.456, answer_relevance=0.829

**Top docs**:

1. `medquad_ans_0000033-2__dup1` (score=0.9944) - There is no cure for A-T and, currently, no way to slow the progression of the disease. Treatment is symptomatic and supportive. Physical and occupational therapy help to maintain flexibility. Speech therapy is import...
2. `medquad_ans_0000188-2__dup1` (score=0.9887) - MJD is incurable, but some symptoms of the disease can be treated. For those individuals who show parkinsonian features, levodopa therapy can help for many years. Treatment with antispasmodic drugs, such as baclofen, ...
3. `medquad_ans_0000132-10` (score=0.9879) - A person with Whipple disease and malabsorption may need - a diet high in calories and protein - vitamins - nutritional supplements
4. `medquad_ans_0000132-8` (score=0.9844) - The health care provider prescribes antibiotics to destroy the T. whipplei bacteria and treat Whipple disease. Health care providers choose antibiotics that treat the infection in the small intestine and cross the blo...
5. `medquad_ans_0000165-2__dup1` (score=0.9735) - The standard treatment for Whipple's disease is a prolonged course of antibiotics (up to two years), including penicillin and cefriaxone or doxycycline with hydroxychloroquine. Sulfa drugs (sulfonamides) such as sulfa...

**Answer**:

The treatments for Whipple disease include antibiotics to destroy the T. whipplei bacteria. Intravenous (IV) antibiotics are typically prescribed for the first 2 weeks of treatment and may include ceftriaxone (Rocephin), meropenem (Merrem I.V.), penicillin G (Pfizerpen), and streptomycin (Streptomycin). After completing the IV antibiotics, patients are prescribed long-term oral antibiotics for at least 1 to 2 years to ensure the infection is cured throughout the body. Oral antibiotics may include trimethoprim/sulfamethoxazole (Septra, Bactrim) or doxycycline (Vibramycin). Patients must finish the prescribed course of antibiotics to ensure all T. whipplei bacteria are eliminated. In cases of malabsorption, a diet high in calories and protein, along with vitamins and nutritional supplements, may be necessary [medquad

**Supported citations**: _none_
**Unsupported citations**: _none_

### Example 300 (`0000042-2__dup2`)

**Question**: What are the treatments for Bell's Palsy ?

**Scores**: context_relevance=0.857, groundedness=0.984, answer_relevance=0.857

**Top docs**:

1. `medquad_ans_0000255-2__dup1` (score=0.9944) - Treatment may include such medications as baclofen, diazepam, tizanidine or clonazepam. Physical therapy regimens may include muscle stretching and range of motion exercises to help prevent shrinkage or shortening of ...
2. `medquad_ans_0000074-2__dup1` (score=0.9930) - There is no cure for CMT, but physical therapy, occupational therapy, braces and other orthopedic devices, and orthopedic surgery can help people cope with the disabling symptoms of the disease. In addition, pain-kill...
3. `medquad_ans_0000085-2__dup1` (score=0.9929) - Currently, only central core disease has an effective treatment (see above). There are no known cures for any of these disorders. Supportive treatment may involve orthopedic treatments, as well as physical, occupation...
4. `medquad_ans_0000131-2__dup1` (score=0.9914) - Treatment begins with a thorough diagnostic evaluation, usually performed by a neurologist, including an assessment of motor and sensory skills, balance and coordination, mental status, reflexes, and functioning of th...
5. `medquad_ans_0000042-2__dup2` (score=0.9856) - Steroids such as prednisone -- used to reduce inflammation and swelling -- are an effective treatment for Bell's palsy. Antiviral drugs may have some benefit in shortening the course of the disease. Analgesics such as...

**Answer**:

Treatments for Bell's palsy include steroids such as prednisone, which are used to reduce inflammation and swelling, and antiviral drugs, which may have some benefit in shortening the course of the disease. Analgesics such as aspirin, acetaminophen, or ibuprofen may relieve pain. Keeping the eye moist and protecting it from debris and injury, especially at night, is important, and lubricating eye drops can help. Other therapies such as physical therapy, facial massage, or acupuncture may provide a potential small improvement in facial nerve function and pain [medquad_ans_0000042-2__dup2].

**Supported citations**: `medquad_ans_0000042-2__dup2`
**Unsupported citations**: _none_
